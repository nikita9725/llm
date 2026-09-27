import json
import logging
from pathlib import Path
from unittest.mock import Mock

import pytest

import main as app
from examples import ROUTING_EXAMPLES
from llm_client import LLMAPIError
from main import PipelineError, parse_analysis_response, process_text
from prompts import ROUTE_INSTRUCTIONS
from schemas import Category, Sentiment

VALID_CLASSIFICATION = json.dumps(
    {"category": "support", "intent": "Восстановить доступ"},
    ensure_ascii=False,
)
VALID_ROUTED_RESPONSE = json.dumps(
    {
        "summary": "Кратко",
        "sentiment": "neutral",
        "key_points": ["Первое", "Второе", "Третье"],
        "final_answer": "Полезный ответ",
    },
    ensure_ascii=False,
)
VALID_RESPONSE = json.dumps(
    {
        "summary": "Кратко",
        "category": "support",
        "intent": "Восстановить доступ",
        "sentiment": "neutral",
        "key_points": ["Первое", "Второе", "Третье"],
        "final_answer": "Полезный ответ",
    },
    ensure_ascii=False,
)


def test_process_text_classifies_routes_and_returns_combined_model() -> None:
    client = Mock()
    client.complete.side_effect = [VALID_CLASSIFICATION, VALID_ROUTED_RESPONSE]

    result = process_text(" Исходный текст ", client)

    assert result.summary == "Кратко"
    assert result.category is Category.SUPPORT
    assert result.intent == "Восстановить доступ"
    assert result.sentiment is Sentiment.NEUTRAL
    assert len(result.key_points) == 3
    assert client.complete.call_count == 2
    first_call, second_call = client.complete.call_args_list
    assert "Исходный текст" in first_call.kwargs["user_prompt"]
    assert "Исходный текст" in second_call.kwargs["user_prompt"]
    assert "Восстановить доступ" in second_call.kwargs["user_prompt"]
    assert "Route: support" in second_call.kwargs["system_prompt"]


@pytest.mark.parametrize("category", Category)
def test_process_text_selects_explicit_instruction_for_category(
    category: Category,
) -> None:
    client = Mock()
    client.complete.side_effect = [
        json.dumps({"category": category.value, "intent": "Handle request"}),
        VALID_ROUTED_RESPONSE,
    ]

    result = process_text("Text", client)

    assert result.category is category
    routed_prompt = client.complete.call_args_list[1].kwargs["system_prompt"]
    assert ROUTE_INSTRUCTIONS[category] in routed_prompt


def test_process_text_rejects_empty_input_without_api_call() -> None:
    client = Mock()

    with pytest.raises(PipelineError, match="must not be empty"):
        process_text("  ", client)

    client.complete.assert_not_called()


@pytest.mark.parametrize(
    "response",
    [
        '{"category": "support"}',
        '{"category": "request", "intent": "Help"}',
        '{"category": "support", "intent": "Help", "extra": 1}',
    ],
)
def test_process_text_rejects_invalid_classification(response: str) -> None:
    client = Mock()
    client.complete.return_value = response

    with pytest.raises(PipelineError, match="этапе классификации.*проверку схемы"):
        process_text("Text", client)

    assert client.complete.call_count == 1


def test_process_text_rejects_invalid_routed_response() -> None:
    client = Mock()
    client.complete.side_effect = [VALID_CLASSIFICATION, '{"summary": "x"}']

    with pytest.raises(PipelineError, match="этапе генерации.*проверку схемы"):
        process_text("Text", client)

    assert client.complete.call_count == 2


@pytest.mark.parametrize(
    ("responses", "stage"),
    [
        ([LLMAPIError("failed")], "классификации"),
        ([VALID_CLASSIFICATION, LLMAPIError("failed")], "генерации"),
    ],
)
def test_process_text_identifies_stage_for_provider_errors(
    responses: list[str | LLMAPIError], stage: str
) -> None:
    client = Mock()
    client.complete.side_effect = responses

    with pytest.raises(PipelineError, match=f"этапе {stage}"):
        process_text("Text", client)


def test_process_text_reports_malformed_json_location() -> None:
    client = Mock()
    client.complete.return_value = '{"category": }'

    with pytest.raises(
        PipelineError,
        match=r"этапе классификации.*корректным JSON.*строка 1, столбец",
    ):
        process_text("Text", client)


def test_parse_analysis_response_returns_validated_model() -> None:
    assert parse_analysis_response(VALID_RESPONSE).summary == "Кратко"


def test_cli_text_prints_and_saves_single_json(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    client = Mock()
    client.complete.side_effect = [VALID_CLASSIFICATION, VALID_ROUTED_RESPONSE]
    monkeypatch.setattr(app.LLMClient, "from_env", lambda: client)
    output = tmp_path / "result.json"

    exit_code = app.main(["--text", "Текст", "--output", str(output)])

    assert exit_code == 0
    console = capsys.readouterr().out
    assert "Краткое резюме: Кратко" in console
    assert "Намерение: Восстановить доступ" in console
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["summary"] == "Кратко"
    assert payload["category"] == "support"
    assert payload["intent"] == "Восстановить доступ"
    assert payload["sentiment"] == "neutral"
    assert payload["final_answer"] == "Полезный ответ"


def test_cli_reads_utf8_input_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    client = Mock()
    client.complete.side_effect = [VALID_CLASSIFICATION, VALID_ROUTED_RESPONSE]
    monkeypatch.setattr(app.LLMClient, "from_env", lambda: client)
    source = tmp_path / "input.txt"
    source.write_text("Текст из файла", encoding="utf-8")

    assert app.main(["--input-file", str(source)]) == 0
    assert all(
        "Текст из файла" in call.kwargs["user_prompt"]
        for call in client.complete.call_args_list
    )


def test_cli_handles_malformed_json_without_crashing(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    client = Mock()
    client.complete.return_value = "this is not JSON"
    monkeypatch.setattr(app.LLMClient, "from_env", lambda: client)

    with caplog.at_level(logging.ERROR):
        exit_code = app.main(["--text", "Текст"])

    assert exit_code == 1
    assert "этапе классификации" in caplog.text
    assert "не является корректным JSON" in caplog.text
    assert "строка 1, столбец 1" in caplog.text


def test_demo_continues_after_one_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    client = Mock()
    successful_calls = [
        response
        for example in ROUTING_EXAMPLES[1:]
        for response in (
            json.dumps(
                {
                    "category": example.expected_category.value,
                    "intent": "Handle request",
                }
            ),
            VALID_ROUTED_RESPONSE,
        )
    ]
    client.complete.side_effect = [LLMAPIError("temporary failure"), *successful_calls]
    monkeypatch.setattr(app.LLMClient, "from_env", lambda: client)

    assert app.main([]) == 1
    assert client.complete.call_count == 19


def test_demo_reports_categories_and_classification_accuracy(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    client = Mock()
    client.complete.side_effect = [
        response
        for example in ROUTING_EXAMPLES
        for response in (
            json.dumps(
                {
                    "category": example.expected_category.value,
                    "intent": "Handle request",
                }
            ),
            VALID_ROUTED_RESPONSE,
        )
    ]
    monkeypatch.setattr(app.LLMClient, "from_env", lambda: client)

    assert app.main([]) == 0

    output = capsys.readouterr().out
    assert "=== Сводка по примерам ===" in output
    assert "support: 2" in output
    assert "complaint: 2" in output
    assert "general_question: 2" in output
    assert "Точность классификации: 10/10" in output
    assert output.count("  OK ") == 10
    assert client.complete.call_count == 20


def test_cli_rejects_empty_text_before_loading_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    factory = Mock()
    monkeypatch.setattr(app.LLMClient, "from_env", factory)

    assert app.main(["--text", " "]) == 1
    factory.assert_not_called()
