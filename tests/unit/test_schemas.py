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
