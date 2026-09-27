import json

import pytest

from llm_client import EmptyLLMResponseError, LLMAPIError
from pipeline import MAX_RAW_RESPONSE_CHARS, PipelineError, build_default_pipeline
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
    responses[failed_stage : failed_stage + 1] = ["{broken", "{still broken"]
    gateway = FakeLLMGateway(responses)

    with pytest.raises(PipelineError, match="не является корректным JSON"):
        build_default_pipeline(gateway).run("Text")

    assert len(gateway.calls) == failed_stage + 2


def test_pipeline_repairs_invalid_json_once() -> None:
    valid = stage_responses()
    responses = ["not JSON", valid[0], *valid[1:]]
    gateway = FakeLLMGateway(responses)

    result = build_default_pipeline(gateway).run("Text")

    assert result.meaning.core_meaning
    assert len(gateway.calls) == 6
    repair_system, repair_user = gateway.calls[1]
    assert "previous response was rejected" in repair_system
    assert "not JSON" in repair_user


def test_pipeline_repairs_response_with_missing_keys() -> None:
    valid = stage_responses()
    responses = ['{"core_meaning":"Only one field"}', valid[0], *valid[1:]]

    result = build_default_pipeline(FakeLLMGateway(responses)).run("Text")

    assert result.meaning.user_goal


def test_pipeline_repairs_empty_response() -> None:
    valid = stage_responses()
    responses = [EmptyLLMResponseError("empty"), valid[0], *valid[1:]]

    result = build_default_pipeline(FakeLLMGateway(responses)).run("Text")

    assert result.meaning.important_details


def test_pipeline_repairs_too_long_field() -> None:
    valid = stage_responses()
    too_long = json.dumps(
        {
            "core_meaning": "x" * 501,
            "user_goal": "Goal",
            "important_details": ["Detail"],
        }
    )
    responses = [too_long, valid[0], *valid[1:]]

    result = build_default_pipeline(FakeLLMGateway(responses)).run("Text")

    assert len(result.meaning.core_meaning) <= 500


def test_pipeline_limits_raw_response_in_repair_prompt() -> None:
    valid = stage_responses()
    oversized = "x" * (MAX_RAW_RESPONSE_CHARS + 1)
    gateway = FakeLLMGateway([oversized, valid[0], *valid[1:]])

    build_default_pipeline(gateway).run("Text")

    repair_user = gateway.calls[1][1]
    assert "слишком длинный" in repair_user
    assert oversized not in repair_user
    assert "x" * 2_000 in repair_user


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
