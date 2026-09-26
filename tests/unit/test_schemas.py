import pytest
from pydantic import ValidationError

from schemas import TextAnalysis


def test_text_analysis_accepts_exactly_three_non_empty_points() -> None:
    result = TextAnalysis(
        summary="Summary",
        key_points=[" One ", "Two", "Three"],
        helpful_response="Response",
    )

    assert result.key_points == ["One", "Two", "Three"]


@pytest.mark.parametrize(
    "points", [[], ["one", "two"], ["one", "two", "three", "four"]]
)
def test_text_analysis_rejects_wrong_number_of_points(points: list[str]) -> None:
    with pytest.raises(ValidationError):
        TextAnalysis(summary="Summary", key_points=points, helpful_response="Response")


def test_text_analysis_rejects_blank_point() -> None:
    with pytest.raises(ValidationError):
        TextAnalysis(
            summary="Summary",
            key_points=["one", " ", "three"],
            helpful_response="Response",
        )


@pytest.mark.parametrize(
    ("field", "length"),
    [("summary", 301), ("helpful_response", 201)],
)
def test_text_analysis_rejects_overlong_strings(field: str, length: int) -> None:
    payload = {
        "summary": "s",
        "key_points": ["one", "two", "three"],
        "helpful_response": "r",
    }
    payload[field] = "x" * length

    with pytest.raises(ValidationError):
        TextAnalysis.model_validate(payload)


def test_text_analysis_accepts_length_boundaries() -> None:
    result = TextAnalysis(
        summary="s" * 300,
        key_points=["one", "two", "three"],
        helpful_response="r" * 200,
    )

    assert len(result.summary) == 300
    assert len(result.helpful_response) == 200
