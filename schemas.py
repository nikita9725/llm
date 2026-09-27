"""Validated output models for the text-processing pipeline."""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Category(StrEnum):
    """Supported intent categories for analyzed text."""

    QUESTION = "question"
    REQUEST = "request"
    FEEDBACK = "feedback"
    PROBLEM = "problem"
    INFORMATIONAL = "informational"
    OTHER = "other"


class Sentiment(StrEnum):
    """Supported sentiment labels for analyzed text."""

    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"
    MIXED = "mixed"


class TextAnalysis(BaseModel):
    """Structured result returned by the LLM pipeline."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    summary: str = Field(min_length=1, max_length=300, strict=True)
    category: Category
    sentiment: Sentiment
    key_points: list[str] = Field(min_length=3, max_length=3, strict=True)
    final_answer: str = Field(min_length=1, max_length=200, strict=True)

    @field_validator("key_points")
    @classmethod
    def key_points_must_be_non_empty(cls, value: list[str]) -> list[str]:
        if any(not isinstance(point, str) for point in value):
            raise ValueError("key points must be strings")
        normalized = [point.strip() for point in value]
        if any(not point for point in normalized):
            raise ValueError("key points must not be empty")
        return normalized
