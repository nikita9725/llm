"""Validated data contracts shared by every application layer."""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Category(StrEnum):
    SUPPORT = "support"
    FEEDBACK = "feedback"
    COMPLAINT = "complaint"
    SALES = "sales"
    GENERAL_QUESTION = "general_question"


class Sentiment(StrEnum):
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"
    MIXED = "mixed"


class StrictModel(BaseModel):
    """Reject surprising fields and normalize surrounding whitespace."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class MeaningExtraction(StrictModel):
    core_meaning: str = Field(min_length=1, max_length=500, strict=True)
    user_goal: str = Field(min_length=1, max_length=300, strict=True)
    important_details: list[str] = Field(min_length=1, max_length=10, strict=True)

    @field_validator("important_details")
    @classmethod
    def details_must_be_non_empty(cls, value: list[str]) -> list[str]:
        normalized = [item.strip() for item in value]
        if any(not item for item in normalized):
            raise ValueError("important details must not be empty")
        return normalized


class Classification(StrictModel):
    category: Category
    intent: str = Field(min_length=1, max_length=200, strict=True)


class StructuredFields(StrictModel):
    summary: str = Field(min_length=1, max_length=300, strict=True)
    sentiment: Sentiment
    key_points: list[str] = Field(min_length=3, max_length=3, strict=True)

    @field_validator("key_points")
    @classmethod
    def key_points_must_be_non_empty(cls, value: list[str]) -> list[str]:
        normalized = [point.strip() for point in value]
        if any(not point for point in normalized):
            raise ValueError("key points must not be empty")
        return normalized


class FinalAnswer(StrictModel):
    text: str = Field(min_length=1, max_length=1000, strict=True)


class SelfCheck(StrictModel):
    is_consistent: bool
    details_preserved: bool
    issues: list[str] = Field(max_length=10, strict=True)

    @field_validator("issues")
    @classmethod
    def issues_must_be_non_empty(cls, value: list[str]) -> list[str]:
        normalized = [issue.strip() for issue in value]
        if any(not issue for issue in normalized):
            raise ValueError("issues must not contain empty strings")
        return normalized

    @property
    def passed(self) -> bool:
        return self.is_consistent and self.details_preserved and not self.issues


# Each input contract makes the dependency chain visible to readers and mypy.
class MeaningInput(StrictModel):
    source_text: str = Field(min_length=1, strict=True)


class ClassificationInput(MeaningInput):
    meaning: MeaningExtraction


class StructuredFieldsInput(ClassificationInput):
    classification: Classification


class FinalAnswerInput(StructuredFieldsInput):
    structured_fields: StructuredFields


class SelfCheckInput(FinalAnswerInput):
    final_answer: FinalAnswer


class PipelineResult(SelfCheckInput):
    self_check: SelfCheck


# Day 2's historical flat contract is intentionally kept separate.
class RoutedResponse(StructuredFields):
    final_answer: str = Field(min_length=1, max_length=1000, strict=True)


class TextAnalysis(RoutedResponse):
    category: Category
    intent: str = Field(min_length=1, max_length=200, strict=True)


StrictOutputModel = StrictModel
