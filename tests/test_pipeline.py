"""Exercise retrieval, strict vision JSON, disk caching and experiment evaluation offline."""

import json
from functools import partial

import httpx
import numpy as np
import pytest
from PIL import Image
from streamlit.testing.v1 import AppTest

import config
from catalog import load_photos
from embedding_client import OpenRouterEmbeddings, normalize
from evaluation import load_runs, save_run, summarize
from recognition import Settings, recognize
from schemas import CallStats, Candidate, Photo, Prediction, Recognition, VisionDecision
from storage import file_digest, write_json
from vector_index import VectorIndex, build_index, index_path, load_index


@pytest.fixture
def dataset(tmp_path):
    photos = []
    for index, (model, color) in enumerate([("A", "red"), ("A", "blue"), ("B", "green")]):
        name = f"{index}.png"
        Image.new("RGB", (12, 8), color).save(tmp_path / name)
        photos.append(Photo(name_file=name, model=model, color="Красный", type="Угловой"))
    source = tmp_path / "query.png"
    Image.new("RGB", (10, 10), "yellow").save(source)
    return photos, source


def transport(monkeypatch, decision=None):
    requests = []

    def handler(request):
        payload = json.loads(request.content)
        requests.append(payload)
        if "messages" not in payload:
            data = {"data": [{"embedding": [3, 4]}], "usage": {"cost": 0.001}}
        else:
            value = decision or {
                "prediction": {
                    "model": "A",
                    "color": "Жёлтый",
                    "configuration": "Прямой",
                    "orientation": None,
                },
                "ranked_models": ["A", "B"],
            }
            data = {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": json.dumps(value, ensure_ascii=False)},
                    }
                ],
                "usage": {"cost": 0.02},
            }
        return httpx.Response(200, json=data)

    monkeypatch.setattr(
        httpx, "Client", partial(httpx.Client, transport=httpx.MockTransport(handler))
    )
    return requests


def test_unique_models_and_duplicate_exclusion():
    index = VectorIndex(
        [
            Photo(name_file="a.png", model="A"),
            Photo(name_file="b.png", model="A"),
            Photo(name_file="c.png", model="B"),
        ],
        np.array([[1, 0], [0.9, 0.1], [0, 1]]),
        ["original", "other", "third"],
    )
    result = index.search(np.array([1, 0]), exclude_hash="original")
    assert [c.model for c in result] == ["A", "B"]
    assert result[0].name_file == "b.png"
    with pytest.raises(ValueError, match="Размерность"):
        index.search(np.ones(3))


@pytest.mark.parametrize("vector", [[0, 0], [float("nan"), 1], [[1, 2]], [], [float("inf")]])
def test_invalid_vectors(vector):
    with pytest.raises(ValueError):
        normalize(vector)


def test_cache_separates_query_document_and_model(monkeypatch, tmp_path, dataset):
    requests = transport(monkeypatch)
    _, source = dataset
    with OpenRouterEmbeddings("test/model", tmp_path / "cache") as client:
        _, first = client.embed_details(source)
        _, cached = client.embed_details(source)
        client.embed_details(source, query=True)
    with OpenRouterEmbeddings("test/other", tmp_path / "cache") as client:
        client.embed(source)
    assert len(requests) == 3
    assert not first.cached and cached.cached and cached.cost_usd == 0
    assert requests[1]["input_type"] == "search_query"


def test_index_roundtrip_stale_catalog_and_resume(monkeypatch, tmp_path, dataset):
    requests = transport(monkeypatch)
    photos, _ = dataset
    options = {"directory": tmp_path / "indexes", "cache_dir": tmp_path / "cache"}
    build_index(photos, tmp_path, "Cohere", "test/model", **options)
    build_index(photos, tmp_path, "Cohere", "test/model", **options)
    assert len(requests) == 3
    index = load_index(photos, tmp_path, "Cohere", "test/model", options["directory"])
    assert len(index.search(np.array([3, 4]))) == 2
    assert index_path("Voyage", "test/model", options["directory"]) != index_path(
        "Cohere", "test/model", options["directory"]
    )
    Image.new("RGB", (3, 3), "black").save(tmp_path / photos[0].name_file)
    with pytest.raises(ValueError, match="изменились"):
        load_index(photos, tmp_path, "Cohere", "test/model", options["directory"])


@pytest.mark.parametrize(
    "mode",
    ["Vision Only", "Cohere Embeddings", "Voyage Embeddings", "Cohere + Vision", "Voyage + Vision"],
)
def test_five_modes_and_vision_cache(monkeypatch, tmp_path, dataset, mode):
    requests = transport(monkeypatch)
    photos, source = dataset
    options = Settings(mode=mode, embedding_model="test/embed", vision_model="test/vision")
    if options.provider:
        build_index(
            photos,
            tmp_path,
            options.provider,
            options.embedding_model,
            directory=tmp_path / "index",
            cache_dir=tmp_path / "cache",
        )
    kwargs = {"cache_dir": tmp_path / "cache", "index_dir": tmp_path / "index"}
    result = recognize(source, photos, tmp_path, options, **kwargs)
    count = len(requests)
    cached = recognize(source, photos, tmp_path, options, **kwargs)
    assert len(requests) == count
    assert all(c.cached for c in cached.calls)
    assert cached.cost_usd == 0
    assert result.prediction.model == "A"
    assert len({c.model for c in result.candidates}) == len(result.candidates)
    if options.vision:
        assert result.prediction.color == "Жёлтый"
        payload = next(p for p in requests if "messages" in p)
        assert payload["response_format"]["json_schema"]["strict"]
        text = json.dumps(payload["messages"], ensure_ascii=False)
        assert "Красный" not in text  # no ground truth sent to vision
        assert "query.png" not in text
        assert (
            len([part for part in payload["messages"][1]["content"] if part["type"] == "image_url"])
            == 3
        )
    else:
        assert result.prediction.color is None and result.prediction.configuration is None


def test_threshold_abstention(monkeypatch, tmp_path, dataset):
    transport(monkeypatch)
    photos, source = dataset
    build_index(
        photos,
        tmp_path,
        "Cohere",
        "test/embed",
        directory=tmp_path / "index",
        cache_dir=tmp_path / "cache",
    )
    # Opposite normalized query has negative cosine against all documents.
    monkeypatch.setattr(
        OpenRouterEmbeddings,
        "embed_details",
        lambda *a, **kw: (np.array([-0.6, -0.8]), CallStats()),
    )
    result = recognize(
        source,
        photos,
        tmp_path,
        Settings(mode="Cohere Embeddings", embedding_model="test/embed", threshold=0),
        cache_dir=tmp_path / "cache",
        index_dir=tmp_path / "index",
    )
    assert result.prediction.model is None
    assert result.candidates


def test_hallucinated_model_is_not_cached(monkeypatch, tmp_path, dataset):
    decision = {
        "prediction": {
            "model": "Invented",
            "color": None,
            "configuration": None,
            "orientation": None,
        },
        "ranked_models": ["A"],
    }
    transport(monkeypatch, decision)
    photos, source = dataset
    with pytest.raises(ValueError, match="вне списка"):
        recognize(source, photos, tmp_path, Settings(), cache_dir=tmp_path / "cache")
    assert not list((tmp_path / "cache").glob("*.json"))


def test_nulls_schema_and_orientation():
    prediction = Prediction(model=None, color=None, configuration="Прямой", orientation="Левая")
    assert prediction.orientation is None
    schema = VisionDecision.model_json_schema()
    assert schema["additionalProperties"] is False
    assert set(schema["$defs"]["Prediction"]["required"]) == {
        "model",
        "color",
        "configuration",
        "orientation",
    }


def test_manifest_paths_and_optional_ground_truth(tmp_path):
    Image.new("RGB", (2, 2)).save(tmp_path / "photo.png")
    assert load_photos(tmp_path / "test.json")[0].model is None
    write_json(tmp_path / "test.json", [{"name_file": "../secret.png"}])
    with pytest.raises(ValueError, match="filename"):
        load_photos(tmp_path / "test.json")


def test_evaluation_deduplicates_and_counts_unknown_cost(monkeypatch, tmp_path, dataset):
    transport(monkeypatch)
    photos, source = dataset
    settings = Settings()
    result = recognize(source, photos, tmp_path, settings, cache_dir=tmp_path / "cache")
    photo = Photo(name_file=source.name, model="A", color="Wrong")
    save_run(photo, source, settings, result, tmp_path / "runs")
    cached = recognize(source, photos, tmp_path, settings, cache_dir=tmp_path / "cache")
    save_run(photo, source, settings, cached, tmp_path / "runs")
    row = summarize(load_runs(tmp_path / "runs"))[0]
    assert row["Фото"] == 1 and row["model accuracy"] == 1 and row["color accuracy"] == 0
    assert row["orientation N"] == 0 and row["orientation accuracy"] is None
    assert row["API запусков"] == 1 and row["Известные расходы, USD"] == 0.02
    result.calls[0].cost_usd = None
    assert result.cost_usd is None
    result.calls[0].cached = True
    assert result.cost_usd == 0


def test_streamlit_empty_test_set(monkeypatch, tmp_path, dataset):
    photos, _ = dataset
    manifest = tmp_path / "catalog.json"
    write_json(manifest, [p.model_dump(by_alias=True) for p in photos])
    monkeypatch.setattr(config, "CATALOG_FILE", manifest)
    monkeypatch.setattr(config, "TEST_FILE", tmp_path / "empty/test.json")
    monkeypatch.setattr(config, "INDEX_DIR", tmp_path / "indexes")
    monkeypatch.setattr("ui.load_runs", lambda: [])
    app = AppTest.from_string("from ui import render\nrender()")
    app.run(timeout=20)
    assert not app.exception
    assert any("Добавьте фотографии" in item.value for item in app.info)


def test_streamlit_index_status_survives_restart_and_detects_stale_index(
    monkeypatch, tmp_path, dataset
):
    requests = transport(monkeypatch)
    photos, _ = dataset
    manifest = tmp_path / "catalog.json"
    write_json(manifest, [p.model_dump(by_alias=True) for p in photos])
    directory = tmp_path / "indexes"
    monkeypatch.setattr(config, "CATALOG_FILE", manifest)
    monkeypatch.setattr(config, "TEST_FILE", tmp_path / "empty/test.json")
    monkeypatch.setattr(config, "INDEX_DIR", directory)
    monkeypatch.setattr("ui.load_runs", lambda: [])
    app = AppTest.from_string("from ui import render\nrender()").run(timeout=20)
    assert not app.exception
    assert any("Cohere: индекс отсутствует" in item.value for item in app.info)
    assert not requests  # status inspection never sends an embedding request

    build_index(
        photos,
        tmp_path,
        "Cohere",
        config.COHERE_MODEL,
        directory=directory,
        cache_dir=tmp_path / "cache",
    )
    request_count = len(requests)
    for _ in range(2):
        app = AppTest.from_string("from ui import render\nrender()").run(timeout=20)
        assert not app.exception
        assert any("Cohere: индекс готов · 3 фото" in item.value for item in app.success)
        assert any("Voyage: индекс отсутствует" in item.value for item in app.info)

    Image.new("RGB", (3, 3), "black").save(tmp_path / photos[0].name_file)
    app.run()
    assert not app.exception
    assert any("Cohere: индекс требует пересборки" in item.value for item in app.warning)
    assert any("Каталог или настройки изменились" in item.value for item in app.caption)

    index_path("Cohere", config.COHERE_MODEL, directory).write_bytes(b"broken archive")
    app.run()
    assert not app.exception
    assert any("Cohere: индекс требует пересборки" in item.value for item in app.warning)
    assert len(requests) == request_count


def test_content_hash_excludes_copies(monkeypatch, tmp_path, dataset):
    photos, source = dataset
    source.write_bytes((tmp_path / photos[2].name_file).read_bytes())
    decision = {
        "prediction": {"model": "A", "color": None, "configuration": None, "orientation": None},
        "ranked_models": ["A"],
    }
    requests = transport(monkeypatch, decision)
    result = recognize(source, photos, tmp_path, Settings(), cache_dir=tmp_path / "cache")
    assert file_digest(source) == file_digest(tmp_path / photos[2].name_file)
    content = requests[0]["messages"][1]["content"]
    assert len([p for p in content if p["type"] == "image_url"]) == 2
    assert result.prediction.model == "A"


def test_streamlit_navigation_automatically_recognizes(monkeypatch, tmp_path, dataset):
    photos, _ = dataset
    manifest = tmp_path / "catalog.json"
    write_json(manifest, [p.model_dump(by_alias=True) for p in photos])
    monkeypatch.setattr(config, "CATALOG_FILE", manifest)
    monkeypatch.setattr(config, "TEST_FILE", manifest)
    monkeypatch.setattr(config, "INDEX_DIR", tmp_path / "indexes")
    monkeypatch.setattr("ui.load_runs", lambda: [])
    recognized = []

    def fake_recognize(source, *args, **kwargs):
        recognized.append(source.name)
        return Recognition(
            prediction=Prediction(model="A", color=None, configuration=None, orientation=None),
            candidates=[Candidate(model="A", name_file=photos[1].name_file)],
            calls=[CallStats(cached=True, cost_usd=0)],
            seconds=0.1,
            experiment="offline",
        )

    monkeypatch.setattr("ui.recognize", fake_recognize)
    monkeypatch.setattr("ui.save_run", lambda *a: None)
    app = AppTest.from_string("from ui import render\nrender()")
    app.run(timeout=20)
    assert not app.exception and recognized == [photos[0].name_file]
    next(b for b in app.button if b.label == "Далее →").click().run()
    assert not app.exception and recognized[-1] == photos[1].name_file
    app.run()
    assert len(recognized) == 2  # unrelated rerender does not issue another request
    next(b for b in app.button if b.label == "← Назад").click().run()
    assert not app.exception and recognized[-1] == photos[0].name_file


def test_summary_common_samples_only(monkeypatch, tmp_path, dataset):
    transport(monkeypatch)
    photos, source = dataset
    result = recognize(source, photos, tmp_path, Settings(), cache_dir=tmp_path / "cache")
    save_run(Photo(name_file=source.name, model="A"), source, Settings(), result, tmp_path / "runs")
    first = load_runs(tmp_path / "runs")[0]
    import copy

    second = copy.deepcopy(first)
    second["result"]["experiment"] = "different"
    second["image_hash"] = "different_photo"
    assert summarize([first, second], common_only=True) == []
    second["image_hash"] = first["image_hash"]
    assert len(summarize([first, second], common_only=True)) == 2
