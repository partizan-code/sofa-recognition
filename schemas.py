"""Validated catalog records and recognition results."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

import config

Configuration = Literal["Прямой", "Угловой", "П-образный"]
MODES = (
    "Vision Only",
    "Cohere Embeddings",
    "Voyage Embeddings",
    "Cohere + Vision",
    "Voyage + Vision",
)


class Prediction(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str | None
    color: str | None = Field(json_schema_extra={"enum": [*config.COLOR_PALETTE, None]})
    configuration: Configuration | None

    @field_validator("color")
    @classmethod
    def palette_color(cls, value: str | None) -> str | None:
        if value is not None and value not in config.COLOR_PALETTE:
            raise ValueError("color must be a palette name or null")
        return value


class VisionDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")
    prediction: Prediction
    ranked_models: list[str] = Field(max_length=5)


class Photo(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)
    name_file: str
    model: str | None = None
    color: str | None = None
    configuration: Configuration | None = Field(default=None, alias="type")

    @field_validator("name_file")
    @classmethod
    def plain_filename(cls, value: str) -> str:
        if not value or value in {".", ".."} or any(c in value for c in "/\\:"):
            raise ValueError("name_file must be a filename without a path")
        return value

    def truth(self) -> dict:
        return self.model_dump(exclude={"name_file"})


class Candidate(BaseModel):
    model: str
    name_file: str
    similarity: float | None = None


class CallStats(BaseModel):
    seconds: float = 0.0
    cost_usd: float | None = None
    cached: bool = False
    usage: dict = Field(default_factory=dict)


class Recognition(BaseModel):
    prediction: Prediction
    candidates: list[Candidate]
    calls: list[CallStats]
    seconds: float
    experiment: str

    @property
    def cost_usd(self) -> float | None:
        if any(call.cost_usd is None and not call.cached for call in self.calls):
            return None
        return sum(call.cost_usd or 0 for call in self.calls if not call.cached)
