"""Local experiment workbench. API calls occur only in recognition/index actions."""

import json
import os
import re
from dataclasses import replace

import httpx
import streamlit as st
from loguru import logger
from pydantic import ValidationError

import config
from catalog import load_photos
from evaluation import comparisons, load_runs, save_run, summarize
from images import EXTENSIONS, open_rgb
from recognition import Settings, experiment_key, recognize
from schemas import MODES, Photo, Recognition
from storage import digest, file_digest, write_json
from vector_index import build_index, index_path, load_index


def error_message(error: Exception) -> str:
    # Never display response bodies, headers or base64 request content.
    if isinstance(error, httpx.HTTPStatusError):
        stage = "Embeddings" if error.request.url.path.endswith("/embeddings") else "Vision"
        message = (
            f"OpenRouter {stage}: HTTP {error.response.status_code}. Проверьте ключ, баланс, "
            "модель и ограничения провайдера."
        )
    elif isinstance(error, httpx.RequestError):
        message = f"OpenRouter: {type(error).__name__}: ошибка соединения или таймаут."
    elif isinstance(error, ValidationError):
        fields = {"prediction", "model", "color", "configuration", "orientation", "ranked_models"}
        issues = []
        for issue in error.errors(include_input=False, include_context=False, include_url=False)[
            :5
        ]:
            location = ".".join(
                str(p) if p in fields or isinstance(p, int) else "?" for p in issue["loc"]
            )
            issues.append(f"{location or 'JSON'}: {issue['type']}")
        message = "Ошибка валидации: " + "; ".join(issues)
    elif isinstance(error, (ValueError, FileNotFoundError)):
        message = str(error)
    else:
        message = "Неожиданная ошибка обработки данных."
    if key := os.environ.get("OPENROUTER_API_KEY"):
        message = message.replace(key, "[API key скрыт]")
    message = re.sub(r"data:image/[^;\s]+;base64,[A-Za-z0-9+/=]+", "[изображение скрыто]", message)
    message = re.sub(r"(?i)Bearer\s+\S+|sk-or-v1-[A-Za-z0-9_-]+", "[ключ скрыт]", message)
    return f"{type(error).__name__}: {' '.join(message.split())[:1000]}"


def show_error(error: Exception, context: str = "Операция") -> None:
    message = error_message(error)
    st.error(message)
    logger.warning("{}: {}", context, message)


def sidebar() -> tuple[Settings, dict[str, str], bool]:
    with st.sidebar:
        st.title("Эксперимент")
        with st.form("settings"):
            mode = st.selectbox("Режим", MODES)
            vision = st.selectbox("Vision-модель", config.VISION_MODELS)
            custom_vision = st.text_input("Другая Vision-модель (ID OpenRouter)")
            cohere = st.text_input("Cohere model ID", config.COHERE_MODEL)
            voyage = st.text_input("Voyage model ID", config.VOYAGE_MODEL)
            threshold = st.slider("Порог cosine для Embeddings", -1.0, 1.0, 0.0, 0.01)
            prompt = st.text_area("Промпт", config.DEFAULT_PROMPT, height=300)
            auto = st.checkbox("Распознавать при переключении фото", value=True)
            st.form_submit_button("Применить настройки")
        st.caption(
            "Cosine — сходство, не вероятность. Порог нужно подобрать на тестах. "
            "В Embeddings цвет, конфигурация и ориентация остаются null."
        )
        st.caption("Ориентация указана со стороны наблюдателя перед диваном.")
    models = {"Cohere": cohere.strip(), "Voyage": voyage.strip()}
    provider = next((p for p in models if mode.startswith(p)), "Voyage")
    return (
        Settings(mode, models[provider], custom_vision.strip() or vision, prompt, threshold),
        models,
        auto,
    )


def show_result(result: Recognition, photo: Photo, source, catalog, root) -> None:
    left, right = st.columns(2)
    with left:
        st.subheader("Исходное фото")
        with open_rgb(source) as image:
            st.image(image, caption=photo.name_file, width="stretch")
    with right:
        st.subheader("Распознанная модель")
        match = next(
            (
                p
                for p in catalog
                if p.model == result.prediction.model
                and file_digest(root / p.name_file) != file_digest(source)
            ),
            None,
        )
        chosen = next((c for c in result.candidates if c.model == result.prediction.model), None)
        if chosen:
            match = next(p for p in catalog if p.name_file == chosen.name_file)
        if match:
            with open_rgb(root / match.name_file) as image:
                st.image(image, caption=f"{match.model} · {match.name_file}", width="stretch")
        else:
            st.info("Модель не определена; подходящий эталон не выбран.")
    st.json(result.prediction.model_dump())
    cols = st.columns(3)
    cols[0].metric("Время запроса", f"{result.seconds:.2f} с")
    cols[1].metric(
        "Стоимость этого запуска",
        "Неизвестна" if result.cost_usd is None else f"${result.cost_usd:.6f}",
    )
    cols[2].metric("Ответов из кэша", f"{sum(c.cached for c in result.calls)}/{len(result.calls)}")
    st.caption(
        "Цена берётся из usage.cost. Отсутствующая цена не считается нулём. "
        "Стоимость индексации учитывается отдельно."
    )
    st.subheader("TOP-5 кандидатов")
    st.dataframe([c.model_dump() for c in result.candidates], hide_index=True, width="stretch")
    if result.candidates:
        with st.expander("Фотографии кандидатов"):
            for column, candidate in zip(
                st.columns(len(result.candidates)), result.candidates, strict=True
            ):
                with column, open_rgb(root / candidate.name_file) as image:
                    st.image(image, caption=candidate.model, width="stretch")
    if result.candidates and result.candidates[0].similarity is None:
        st.caption("Vision Only: кандидаты ранжированы LLM, cosine не вычислялся.")
    st.subheader("Сравнение с известными параметрами")
    st.dataframe(comparisons(photo, result), hide_index=True, width="stretch")
    st.caption(
        "Пустая разметка не участвует в оценке. null при известной разметке — пропуск ответа."
    )
    st.download_button(
        "Скачать результат JSON",
        result.model_dump_json(indent=2),
        file_name="recognition.json",
        mime="application/json",
    )
    with st.expander("Время, токены и стоимость по этапам"):
        st.json([c.model_dump() for c in result.calls])


def render() -> None:
    st.set_page_config(page_title="Sofa Recognition MVP", layout="wide")
    st.title("Sofa Recognition MVP")
    st.caption("Сравнение методов по точности, времени и стоимости")
    settings, models, auto = sidebar()
    try:
        photos = load_photos(config.CATALOG_FILE, references=True)
        if not photos:
            st.warning("Добавьте эталоны в каталог.")
            return
    except Exception as error:
        show_error(error)
        return
    root = config.CATALOG_FILE.parent
    with st.expander(
        f"Индексы · {len(photos)} эталонов · {len({p.model for p in photos})} моделей",
        expanded=True,
    ):
        provider = st.selectbox("Индекс провайдера", list(models))
        st.caption(
            "Построение отправляет все эталоны в OpenRouter. Уже рассчитанные embeddings "
            "берутся из кэша; после сбоя можно продолжить повторным запуском."
        )
        if st.button("Построить индекс"):
            bar = st.progress(0.0)
            try:
                calls = build_index(
                    photos,
                    root,
                    provider,
                    models[provider],
                    progress=lambda i, total: bar.progress(i / total),
                )
                report = {
                    "provider": provider,
                    "model": models[provider],
                    "calls": [c.model_dump() for c in calls],
                }
                write_json(
                    index_path(provider, models[provider]).with_suffix(".build.json"), report
                )
                st.success(
                    f"Индекс готов. API: {sum(not c.cached for c in calls)}, "
                    f"из кэша: {sum(c.cached for c in calls)}."
                )
                st.json(report)
            except Exception as error:
                show_error(error, f"Индексация {provider}")
        for column, (name, model) in zip(st.columns(len(models)), models.items(), strict=True):
            with column:
                st.markdown(f"**{name}** · `{model}`")
                if not index_path(name, model, config.INDEX_DIR).exists():
                    st.info(f"{name}: индекс отсутствует")
                    continue
                try:
                    index = load_index(photos, root, name, model, config.INDEX_DIR)
                except Exception as error:
                    st.warning(f"{name}: индекс требует пересборки")
                    st.caption(error_message(error))
                else:
                    st.success(f"{name}: индекс готов · {len(index.photos)} фото")
    source_kind = st.radio(
        "Источник", ("Тестовый набор", "Загрузить фото", "Каталог: самопроверка"), horizontal=True
    )
    dataset, test_root = [], config.TEST_FILE.parent
    try:
        if source_kind == "Тестовый набор":
            dataset = load_photos(config.TEST_FILE)
            if dataset:
                st.info(
                    "Переключайте фотографии тестового набора стрелками или списком. "
                    "Известные параметры из test.json используются для оценки точности. "
                    "При включённом автораспознавании новое фото обрабатывается автоматически."
                )
        elif source_kind == "Каталог: самопроверка":
            dataset, test_root = photos, root
            st.info(
                "Выберите фото каталога стрелками или списком. Самопроверка исключает "
                "точные копии исходного фото из эталонов. Для независимой оценки качества "
                "используйте тестовый набор новых фотографий."
            )
        else:
            uploaded = st.file_uploader(
                "Фото дивана", type=[s.lstrip(".") for s in sorted(EXTENSIONS)]
            )
            if uploaded is not None:
                import hashlib

                test_root = config.ROOT / "data/uploads"
                test_root.mkdir(parents=True, exist_ok=True)
                suffix = "." + uploaded.name.rsplit(".", 1)[-1].lower()
                path = test_root / (hashlib.sha256(uploaded.getvalue()).hexdigest() + suffix)
                if not path.exists():
                    path.write_bytes(uploaded.getvalue())
                dataset = [Photo(name_file=path.name)]
    except Exception as error:
        show_error(error)
    if dataset:
        dataset_id = digest([source_kind, [p.model_dump() for p in dataset]])
        selection_key = "slide_" + dataset_id
        st.session_state.setdefault(selection_key, 0)
        previous, selector, following = st.columns([1, 5, 1])
        if previous.button("← Назад"):
            st.session_state[selection_key] = (st.session_state[selection_key] - 1) % len(dataset)
        if following.button("Далее →"):
            st.session_state[selection_key] = (st.session_state[selection_key] + 1) % len(dataset)
        selected = selector.selectbox(
            "Фотография",
            range(len(dataset)),
            key=selection_key,
            format_func=lambda i: f"{i + 1}/{len(dataset)} · {dataset[i].name_file}",
        )
        photo = dataset[selected]
        source = test_root / photo.name_file
        try:
            identity = digest(
                [experiment_key(settings, photos, root), file_digest(source), photo.truth()]
            )
            manual = st.button("Распознать / повторить без кэша")
            if manual or (auto and st.session_state.get("attempt") != identity):
                st.session_state["attempt"] = identity
                st.session_state.pop("current", None)
                with st.spinner("Сравниваю с эталонами…"):
                    result = recognize(source, photos, root, settings, force=manual)
                    save_run(photo, source, settings, result)
                    st.session_state["current"] = (identity, result)
            current = st.session_state.get("current")
            if current and current[0] == identity:
                show_result(current[1], photo, source, photos, root)
            else:
                with open_rgb(source) as image:
                    st.image(image, caption=photo.name_file, width=650)
        except Exception as error:
            show_error(error, f"{photo.name_file}: {settings.mode}")
        with st.expander("Пакетное сравнение методов"):
            modes = st.multiselect("Методы", MODES, default=list(MODES))
            limit = st.number_input(
                "Первые N фото набора",
                min_value=1,
                max_value=len(dataset),
                value=min(5, len(dataset)),
            )
            st.caption(
                "Используются выбранная Vision-модель, промпт и оба embedding ID. "
                "Для embedding-режимов сначала постройте оба индекса. Кэш включён."
            )
            fresh_requests = st.checkbox("Обойти кэш для замера скорости и стоимости", value=False)
            if st.button("Запустить сравнение", disabled=not modes):
                failures = []
                completed = 0
                bar = st.progress(0.0)
                for item in dataset[: int(limit)]:
                    for mode in modes:
                        provider = next((p for p in models if mode.startswith(p)), "Voyage")
                        options = replace(settings, mode=mode, embedding_model=models[provider])
                        try:
                            result = recognize(
                                test_root / item.name_file,
                                photos,
                                root,
                                options,
                                force=fresh_requests,
                            )
                            save_run(item, test_root / item.name_file, options, result)
                        except Exception as error:
                            message = f"{item.name_file}: {mode}: {error_message(error)}"
                            failures.append(message)
                            logger.warning("Пакетное распознавание: {}", message)
                        completed += 1
                        bar.progress(completed / (int(limit) * len(modes)))
                st.success(f"Завершено попыток: {completed}; ошибок: {len(failures)}")
                if failures:
                    st.warning("\n\n".join(failures))
    elif source_kind == "Тестовый набор":
        st.info(
            "Добавьте фотографии в data/test/. Необязательный test.json задаёт известные "
            "параметры для оценки точности."
        )
    elif source_kind == "Загрузить фото":
        st.info(
            "Загрузите своё фото дивана в формате JPEG, PNG или WebP через поле выше. "
            "При включённом автораспознавании обработка начнётся после загрузки; иначе "
            "нажмите «Распознать / повторить без кэша». Известные параметры не заданы, "
            "поэтому точность по этому фото не оценивается."
        )
    with st.expander("Журнал и сводка экспериментов", expanded=True):
        runs = load_runs()
        if runs:
            experiments = sorted({r["result"]["experiment"] for r in runs})
            selected_experiments = st.multiselect(
                "Эксперименты в сводке",
                experiments,
                default=experiments,
                format_func=lambda value: value[:12],
            )
            common_only = st.checkbox(
                "Сравнивать только общие фото выбранных экспериментов", value=True
            )
            selected_runs = [r for r in runs if r["result"]["experiment"] in selected_experiments]
            summary = summarize(selected_runs, common_only=common_only)
            if summary:
                st.dataframe(summary, hide_index=True, width="stretch")
            else:
                st.info("У выбранных экспериментов пока нет общих фото.")
            st.caption(
                "Оценка использует последний результат каждого уникального фото. "
                "Accuracy включает отказы, coverage — доля ответов. "
                "Сравнивайте методы на одинаковом наборе фото; число размеченных примеров "
                "показано в колонках N. В среднее время и цену API входят только запуски "
                "полностью без кэша. Расходы включают все успешные запуски выбранных фото."
            )
            st.download_button(
                "Скачать журнал JSON",
                json.dumps(runs, ensure_ascii=False, indent=2),
                file_name="experiments.json",
                mime="application/json",
            )
        else:
            st.caption("Результатов ещё нет.")
