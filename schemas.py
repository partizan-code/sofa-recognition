"""Validated catalog records and recognition results."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Configuration = Literal["Прямой", "Угловой", "П-образный"]
Orientation = Literal["Левая", "Правая"]
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
    color: str | None
    configuration: Configuration | None
    orientation: Orientation | None

    @model_validator(mode="after")
    def applicable_orientation(self) -> "Prediction":
        if self.configuration != "Угловой":
            self.orientation = None
        return self


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
    orientation: Orientation | None = None

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
