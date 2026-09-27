import pytest

from prompts import (
    ROUTE_INSTRUCTIONS,
    ClassificationPromptStrategy,
    FinalAnswerPromptStrategy,
    MeaningPromptStrategy,
    SelfCheckPromptStrategy,
    StructuredFieldsPromptStrategy,
)
from schemas import (
    Category,
    Classification,
    ClassificationInput,
    FinalAnswer,
    FinalAnswerInput,
    MeaningExtraction,
    MeaningInput,
    SelfCheckInput,
    StructuredFields,
    StructuredFieldsInput,
)


def inputs() -> tuple[
    MeaningInput,
    ClassificationInput,
    StructuredFieldsInput,
    FinalAnswerInput,
    SelfCheckInput,
]:
    meaning_input = MeaningInput(source_text="Unique source")
    meaning = MeaningExtraction(
        core_meaning="Meaning", user_goal="Goal", important_details=["Detail"]
    )
    classification_input = ClassificationInput(
        source_text="Unique source", meaning=meaning
    )
    classification = Classification(category=Category.SUPPORT, intent="Get help")
    structured_input = StructuredFieldsInput(
        source_text="Unique source",
        meaning=meaning,
        classification=classification,
    )
    fields = StructuredFields(
        summary="Summary",
        sentiment="neutral",
        key_points=["One", "Two", "Three"],
    )
    final_input = FinalAnswerInput(
        **structured_input.model_dump(), structured_fields=fields
    )
    check_input = SelfCheckInput(
        **final_input.model_dump(), final_answer=FinalAnswer(text="Answer")
    )
    return (
        meaning_input,
        classification_input,
        structured_input,
        final_input,
        check_input,
    )


def test_every_strategy_has_a_distinct_stage_and_includes_source() -> None:
    strategies_and_inputs = zip(
        (
            MeaningPromptStrategy(),
            ClassificationPromptStrategy(),
            StructuredFieldsPromptStrategy(),
            FinalAnswerPromptStrategy(),
            SelfCheckPromptStrategy(),
        ),
        inputs(),
        strict=True,
    )

    stages = []
    for strategy, data in strategies_and_inputs:
        stages.append(strategy.stage_name)
        assert "Unique source" in strategy.build_user_prompt(data)
        assert "JSON" in strategy.build_system_prompt(data)

    assert len(set(stages)) == 5


def test_each_stage_receives_the_previous_result() -> None:
    _, classification, structured, final, check = inputs()

    assert "Meaning" in ClassificationPromptStrategy().build_user_prompt(classification)
    assert "Get help" in StructuredFieldsPromptStrategy().build_user_prompt(structured)
    assert "Summary" in FinalAnswerPromptStrategy().build_user_prompt(final)
    assert "Answer" in SelfCheckPromptStrategy().build_user_prompt(check)


@pytest.mark.parametrize("category", Category)
def test_final_answer_strategy_selects_route(category: Category) -> None:
    _, _, structured, _, _ = inputs()
    data = FinalAnswerInput(
        **{
            **structured.model_dump(),
            "classification": Classification(category=category, intent="Intent"),
        },
        structured_fields=StructuredFields(
            summary="Summary",
            sentiment="neutral",
            key_points=["One", "Two", "Three"],
        ),
    )

    prompt = FinalAnswerPromptStrategy().build_system_prompt(data)

    assert f"Route: {category.value}" in prompt
    assert ROUTE_INSTRUCTIONS[category] in prompt
