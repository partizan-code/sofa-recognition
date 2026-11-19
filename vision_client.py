"""Compare the source with labeled visual references using strict JSON output."""

import json
from pathlib import Path

import config
from api_client import OpenRouterClient
from images import image_url, reference_url
from schemas import CallStats, VisionDecision


class OpenRouterVision(OpenRouterClient):
    def analyze(
        self, source: Path, references: dict[str, list[Path]], model: str, prompt: str
    ) -> tuple[VisionDecision, CallStats]:
        content = [
            {"type": "text", "text": "ИСХОДНОЕ ФОТО. Анализируй его параметры."},
            {"type": "image_url", "image_url": {"url": image_url(source)}},
        ]
        for name, paths in references.items():
            content.extend(
                [
                    {
                        "type": "text",
                        "text": "ЭТАЛОН МОДЕЛИ: " + json.dumps(name, ensure_ascii=False),
                    },
                    {"type": "image_url", "image_url": {"url": reference_url(paths)}},
                ]
            )
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": prompt + "\nВыбирай модель только из подписанных "
                    "эталонов либо null. ranked_models — ранжирование эталонов, не уверенность.",
                },
                {"role": "user", "content": content},
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "sofa_recognition",
                    "strict": True,
                    "schema": VisionDecision.model_json_schema(),
                },
            },
            "provider": {"require_parameters": True},
            "max_tokens": config.MAX_OUTPUT_TOKENS,
        }
        data, stats = self.post(config.OPENROUTER_CHAT_URL, payload)
        choice = data["choices"][0]
        if choice.get("finish_reason") != "stop":
            raise ValueError(
                f"Vision не завершил ответ: finish_reason={choice.get('finish_reason')!r}; "
                "возможен лимит токенов или фильтрация; кэш не сохранён"
            )
        decision = VisionDecision.model_validate_json(choice["message"]["content"])
        if decision.prediction.model is not None and decision.prediction.model not in references:
            raise ValueError(
                f"Vision вернул модель вне списка эталонов: {decision.prediction.model!r}; "
                f"эталоны: {list(references)!r}"
            )
        if len(set(decision.ranked_models)) != len(decision.ranked_models) or any(
            name not in references for name in decision.ranked_models
        ):
            raise ValueError(
                f"Vision вернул недопустимый список кандидатов: {decision.ranked_models!r}; "
                f"эталоны: {list(references)!r}"
            )
        self.save_response(config.OPENROUTER_CHAT_URL, payload, data)
        return decision, stats
