import pytest

from prompts import (
    DEFAULT_PROMPT_VARIANT,
    PROMPT_VARIANTS,
    PromptVariant,
    build_user_prompt,
)


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
