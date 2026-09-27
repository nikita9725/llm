import pytest
from pydantic import ValidationError

from schemas import Category, Sentiment, TextAnalysis

VALID_PAYLOAD = {
    "summary": "Summary",
    "category": "request",
    "sentiment": "neutral",
    "key_points": ["One", "Two", "Three"],
    "final_answer": "Response",
}


def test_text_analysis_accepts_exactly_three_non_empty_points() -> None:
    result = TextAnalysis(
        summary="Summary",
        category=Category.REQUEST,
        sentiment=Sentiment.NEUTRAL,
        key_points=[" One ", "Two", "Three"],
        final_answer="Response",
    )

    assert result.key_points == ["One", "Two", "Three"]


@pytest.mark.parametrize(
    "points", [[], ["one", "two"], ["one", "two", "three", "four"]]
)
def test_text_analysis_rejects_wrong_number_of_points(points: list[str]) -> None:
    with pytest.raises(ValidationError):
        TextAnalysis.model_validate({**VALID_PAYLOAD, "key_points": points})


def test_text_analysis_rejects_blank_point() -> None:
    with pytest.raises(ValidationError):
        TextAnalysis.model_validate(
            {**VALID_PAYLOAD, "key_points": ["one", " ", "three"]}
        )


@pytest.mark.parametrize(
    ("field", "length"),
    [("summary", 301), ("final_answer", 201)],
)
def test_text_analysis_rejects_overlong_strings(field: str, length: int) -> None:
    payload = VALID_PAYLOAD.copy()
    payload[field] = "x" * length

    with pytest.raises(ValidationError):
        TextAnalysis.model_validate(payload)


def test_text_analysis_accepts_length_boundaries() -> None:
    result = TextAnalysis(
        summary="s" * 300,
        category=Category.INFORMATIONAL,
        sentiment=Sentiment.POSITIVE,
        key_points=["one", "two", "three"],
        final_answer="r" * 200,
    )

    assert len(result.summary) == 300
    assert len(result.final_answer) == 200


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("category", "unknown"),
        ("sentiment", "uncertain"),
        ("summary", 42),
        ("key_points", ["one", 2, "three"]),
        ("key_points", ("one", "two", "three")),
        ("final_answer", ["not", "a", "string"]),
    ],
)
def test_text_analysis_rejects_invalid_enum_or_type(field: str, value: object) -> None:
    with pytest.raises(ValidationError):
        TextAnalysis.model_validate({**VALID_PAYLOAD, field: value})


@pytest.mark.parametrize("field", ["summary", "category", "sentiment", "final_answer"])
def test_text_analysis_rejects_missing_required_field(field: str) -> None:
    payload = VALID_PAYLOAD.copy()
    del payload[field]

    with pytest.raises(ValidationError):
        TextAnalysis.model_validate(payload)


def test_text_analysis_rejects_extra_field() -> None:
    with pytest.raises(ValidationError):
        TextAnalysis.model_validate({**VALID_PAYLOAD, "unexpected": True})
