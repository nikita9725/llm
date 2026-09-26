"""Validated output models for the text-processing pipeline."""

from pydantic import BaseModel, ConfigDict, Field, field_validator


class TextAnalysis(BaseModel):
    """Structured result returned by the LLM pipeline."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    summary: str = Field(min_length=1, max_length=300)
    key_points: list[str] = Field(min_length=3, max_length=3)
    helpful_response: str = Field(min_length=1, max_length=200)

    @field_validator("key_points")
    @classmethod
    def key_points_must_be_non_empty(cls, value: list[str]) -> list[str]:
        normalized = [point.strip() for point in value]
        if any(not point for point in normalized):
            raise ValueError("key points must not be empty")
        return normalized
