import pytest
from pydantic import ValidationError

from schemas import (
    Category,
    MeaningExtraction,
    PipelineResult,
    SelfCheck,
    StructuredFields,
)
from tests.fakes import make_result


def test_pipeline_result_contains_the_complete_trace() -> None:
    result = make_result()

    payload = result.model_dump(mode="json")

    assert set(payload) == {
        "source_text",
        "meaning",
        "classification",
        "structured_fields",
        "final_answer",
        "self_check",
    }
    assert payload["classification"]["category"] == "support"


@pytest.mark.parametrize("details", [[], ["valid", " "]])
def test_meaning_requires_non_empty_details(details: list[str]) -> None:
    with pytest.raises(ValidationError):
        MeaningExtraction(
            core_meaning="Meaning", user_goal="Goal", important_details=details
        )


@pytest.mark.parametrize(
    "points", [[], ["one", "two"], ["one", "two", "three", "four"]]
)
def test_structured_fields_requires_exactly_three_points(
    points: list[str],
) -> None:
    with pytest.raises(ValidationError):
        StructuredFields(summary="Summary", sentiment="neutral", key_points=points)


def test_self_check_passed_is_derived_from_all_fields() -> None:
    assert SelfCheck(is_consistent=True, details_preserved=True, issues=[]).passed
    assert not SelfCheck(
        is_consistent=True,
        details_preserved=True,
        issues=["Missing detail"],
    ).passed


def test_models_reject_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        SelfCheck.model_validate(
            {
                "is_consistent": True,
                "details_preserved": True,
                "issues": [],
                "verdict": "pass",
            }
        )


def test_pipeline_result_rejects_invalid_category() -> None:
    payload = make_result().model_dump()
    payload["classification"]["category"] = "unknown"

    with pytest.raises(ValidationError):
        PipelineResult.model_validate(payload)

    assert Category.SUPPORT.value == "support"
