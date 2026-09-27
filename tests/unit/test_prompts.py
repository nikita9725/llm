import pytest

from prompts import (
    CLASSIFICATION_SYSTEM_PROMPT,
    DEFAULT_PROMPT_VARIANT,
    PROMPT_VARIANTS,
    ROUTE_INSTRUCTIONS,
    PromptVariant,
    build_classification_user_prompt,
    build_routed_system_prompt,
    build_user_prompt,
)
from schemas import Category


def test_three_named_prompt_variants_are_available() -> None:
    assert [variant.name for variant in PROMPT_VARIANTS] == [
        "minimal",
        "format_focused",
        "structured",
    ]
    assert DEFAULT_PROMPT_VARIANT in PROMPT_VARIANTS


@pytest.mark.parametrize("variant", PROMPT_VARIANTS)
def test_prompt_variant_inserts_source_text_exactly_once(
    variant: PromptVariant,
) -> None:
    source_text = "Unique {source} text"

    rendered = variant.build_user_prompt(source_text)

    assert rendered.count(source_text) == 1


def test_build_user_prompt_uses_default_variant() -> None:
    assert build_user_prompt("Text") == DEFAULT_PROMPT_VARIANT.build_user_prompt("Text")


@pytest.mark.parametrize("variant", PROMPT_VARIANTS)
def test_every_prompt_requests_complete_structured_contract(
    variant: PromptVariant,
) -> None:
    for field in (
        "summary",
        "category",
        "intent",
        "sentiment",
        "key_points",
        "final_answer",
    ):
        assert field in variant.system_prompt
    for value in (
        "support",
        "feedback",
        "complaint",
        "sales",
        "general_question",
        "positive",
        "neutral",
        "negative",
        "mixed",
    ):
        assert value in variant.system_prompt


def test_classifier_prompt_contains_all_categories_and_source_once() -> None:
    source = "Unique source text"

    assert build_classification_user_prompt(source).count(source) == 1
    for category in Category:
        assert category.value in CLASSIFICATION_SYSTEM_PROMPT


def test_every_category_has_a_distinct_explicit_route_instruction() -> None:
    assert set(ROUTE_INSTRUCTIONS) == set(Category)
    prompts = [build_routed_system_prompt(category) for category in Category]

    assert len(set(prompts)) == len(Category)
    for category, prompt in zip(Category, prompts, strict=True):
        assert f"Route: {category.value}" in prompt
        assert ROUTE_INSTRUCTIONS[category] in prompt
