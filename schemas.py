"""Validated output models for the text-processing pipeline."""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Category(StrEnum):
    """Business routes supported by the pipeline."""

    SUPPORT = "support"
    FEEDBACK = "feedback"
    COMPLAINT = "complaint"
    SALES = "sales"
    GENERAL_QUESTION = "general_question"


class Sentiment(StrEnum):
    """Supported sentiment labels for analyzed text."""

    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"
    MIXED = "mixed"


class StrictOutputModel(BaseModel):
    """Base configuration shared by all model-produced payloads."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class Classification(StrictOutputModel):
    """Validated result of the classification stage."""

    category: Category
    intent: str = Field(min_length=1, max_length=200, strict=True)


class RoutedResponse(StrictOutputModel):
    """Validated content produced after selecting a category route."""

    summary: str = Field(min_length=1, max_length=300, strict=True)
    sentiment: Sentiment
    key_points: list[str] = Field(min_length=3, max_length=3, strict=True)
    final_answer: str = Field(min_length=1, max_length=1000, strict=True)

    @field_validator("key_points")
    @classmethod
    def key_points_must_be_non_empty(cls, value: list[str]) -> list[str]:
        if any(not isinstance(point, str) for point in value):
            raise ValueError("key points must be strings")
        normalized = [point.strip() for point in value]
        if any(not point for point in normalized):
            raise ValueError("key points must not be empty")
        return normalized


class TextAnalysis(RoutedResponse):
    """Combined structured result returned by the complete pipeline."""

    category: Category
    intent: str = Field(min_length=1, max_length=200, strict=True)
