import json

import pytest

from llm_client import LLMAPIError
from pipeline import PipelineError, build_default_pipeline
from schemas import Category
from tests.fakes import FakeLLMGateway, stage_responses


def test_pipeline_runs_five_stages_and_returns_full_trace() -> None:
    gateway = FakeLLMGateway(stage_responses())

    result = build_default_pipeline(gateway).run("  Исходный текст  ")

    assert len(gateway.calls) == 5
    assert result.source_text == "Исходный текст"
    assert result.classification.category is Category.SUPPORT
    assert result.final_answer.text
    assert result.self_check.passed


def test_outputs_flow_into_the_next_prompt() -> None:
    gateway = FakeLLMGateway(stage_responses())

    build_default_pipeline(gateway).run("Исходный текст")

    assert "Пользователь просит помощь" in gateway.calls[1][1]
    assert "Получить помощь" in gateway.calls[2][1]
    assert "Нужна помощь" in gateway.calls[3][1]
    assert "Выполните диагностические шаги" in gateway.calls[4][1]


@pytest.mark.parametrize("category", Category)
def test_pipeline_routes_final_answer(category: Category) -> None:
    gateway = FakeLLMGateway(stage_responses(category.value))

    result = build_default_pipeline(gateway).run("Text")

    assert result.classification.category is category
    assert f"Route: {category.value}" in gateway.calls[3][0]


@pytest.mark.parametrize("failed_stage", range(5))
def test_pipeline_identifies_malformed_stage(failed_stage: int) -> None:
    responses = stage_responses()
    responses[failed_stage] = "{broken"
    gateway = FakeLLMGateway(responses)

    with pytest.raises(PipelineError, match="не является корректным JSON"):
        build_default_pipeline(gateway).run("Text")

    assert len(gateway.calls) == failed_stage + 1


def test_pipeline_wraps_gateway_error_with_stage_name() -> None:
    gateway = FakeLLMGateway([LLMAPIError("offline")])

    with pytest.raises(PipelineError, match="extract meaning.*offline"):
        build_default_pipeline(gateway).run("Text")


def test_pipeline_rejects_empty_input_before_calling_gateway() -> None:
    gateway = FakeLLMGateway([])

    with pytest.raises(PipelineError, match="must not be empty"):
        build_default_pipeline(gateway).run(" ")

    assert gateway.calls == []


def test_failed_self_check_is_a_valid_pipeline_result() -> None:
    responses = stage_responses()
    responses[-1] = json.dumps(
        {
            "is_consistent": False,
            "details_preserved": False,
            "issues": ["Ответ потерял версию приложения"],
        },
        ensure_ascii=False,
    )

    result = build_default_pipeline(FakeLLMGateway(responses)).run("Text")

    assert not result.self_check.passed
    assert result.self_check.issues
