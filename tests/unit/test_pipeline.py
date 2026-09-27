import json
import logging
from pathlib import Path
from unittest.mock import Mock

import pytest

import main as app
from llm_client import LLMAPIError
from main import PipelineError, parse_analysis_response, process_text
from prompts import MINIMAL_PROMPT
from schemas import Category, Sentiment

VALID_RESPONSE = json.dumps(
    {
        "summary": "Кратко",
        "category": "request",
        "sentiment": "neutral",
        "key_points": ["Первое", "Второе", "Третье"],
        "final_answer": "Полезный ответ",
    },
    ensure_ascii=False,
)


def test_process_text_returns_validated_model() -> None:
    client = Mock()
    client.complete.return_value = VALID_RESPONSE

    result = process_text(" Исходный текст ", client)

    assert result.summary == "Кратко"
    assert result.category is Category.REQUEST
    assert result.sentiment is Sentiment.NEUTRAL
    assert len(result.key_points) == 3
    assert "Исходный текст" in client.complete.call_args.kwargs["user_prompt"]


def test_process_text_uses_selected_prompt_variant() -> None:
    client = Mock()
    client.complete.return_value = VALID_RESPONSE

    process_text("Text", client, MINIMAL_PROMPT)

    client.complete.assert_called_once_with(
        system_prompt=MINIMAL_PROMPT.system_prompt,
        user_prompt=MINIMAL_PROMPT.build_user_prompt("Text"),
    )


def test_process_text_rejects_empty_input_without_api_call() -> None:
    client = Mock()

    with pytest.raises(PipelineError, match="must not be empty"):
        process_text("  ", client)

    client.complete.assert_not_called()


@pytest.mark.parametrize(
    "response",
    [
        '{"summary": "x"}',
        (
            '{"summary":"x","category":"request","sentiment":"neutral",'
            '"key_points":["1","2"],"final_answer":"y"}'
        ),
        (
            '{"summary":"x","category":"unknown","sentiment":"neutral",'
            '"key_points":["1","2","3"],"final_answer":"y"}'
        ),
        (
            '{"summary":"x","category":"request","sentiment":"neutral",'
            '"key_points":["1","2","3"],"final_answer":"y","extra":1}'
        ),
    ],
)
def test_process_text_rejects_schema_violation(response: str) -> None:
    client = Mock()
    client.complete.return_value = response

    with pytest.raises(PipelineError, match="не прошёл проверку схемы"):
        process_text("Text", client)


def test_process_text_reports_malformed_json_location() -> None:
    client = Mock()
    client.complete.return_value = '{"summary": }'

    with pytest.raises(PipelineError, match=r"корректным JSON.*строка 1, столбец"):
        process_text("Text", client)


def test_parse_analysis_response_returns_validated_model() -> None:
    assert parse_analysis_response(VALID_RESPONSE).summary == "Кратко"


def test_cli_text_prints_and_saves_single_json(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    client = Mock()
    client.complete.return_value = VALID_RESPONSE
    monkeypatch.setattr(app.LLMClient, "from_env", lambda: client)
    output = tmp_path / "result.json"

    exit_code = app.main(["--text", "Текст", "--output", str(output)])

    assert exit_code == 0
    assert "Краткое резюме: Кратко" in capsys.readouterr().out
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["summary"] == "Кратко"
    assert payload["category"] == "request"
    assert payload["sentiment"] == "neutral"
    assert payload["final_answer"] == "Полезный ответ"


def test_cli_reads_utf8_input_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    client = Mock()
    client.complete.return_value = VALID_RESPONSE
    monkeypatch.setattr(app.LLMClient, "from_env", lambda: client)
    source = tmp_path / "input.txt"
    source.write_text("Текст из файла", encoding="utf-8")

    assert app.main(["--input-file", str(source)]) == 0
    assert "Текст из файла" in client.complete.call_args.kwargs["user_prompt"]


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
    assert "не является корректным JSON" in caplog.text
    assert "строка 1, столбец 1" in caplog.text


def test_demo_continues_after_one_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    client = Mock()
    client.complete.side_effect = [
        LLMAPIError("temporary failure"),
        VALID_RESPONSE,
        VALID_RESPONSE,
        VALID_RESPONSE,
        VALID_RESPONSE,
    ]
    monkeypatch.setattr(app.LLMClient, "from_env", lambda: client)

    assert app.main([]) == 1
    assert client.complete.call_count == 5


def test_demo_uses_category_and_sentiment_in_summary(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    client = Mock()
    payloads = [
        {
            "summary": f"Summary {index}",
            "category": category,
            "sentiment": sentiment,
            "key_points": ["One", "Two", "Three"],
            "final_answer": "Answer",
        }
        for index, (category, sentiment) in enumerate(
            [
                ("request", "neutral"),
                ("feedback", "mixed"),
                ("request", "positive"),
                ("problem", "negative"),
                ("informational", "neutral"),
            ],
            start=1,
        )
    ]
    client.complete.side_effect = [json.dumps(payload) for payload in payloads]
    monkeypatch.setattr(app.LLMClient, "from_env", lambda: client)

    assert app.main([]) == 0

    output = capsys.readouterr().out
    assert "=== Сводка по примерам ===" in output
    assert "request: 2" in output
    assert "neutral: 2" in output
    assert "Требуют внимания: Обратная связь, Проблема с доставкой" in output


def test_cli_rejects_empty_text_before_loading_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    factory = Mock()
    monkeypatch.setattr(app.LLMClient, "from_env", factory)

    assert app.main(["--text", " "]) == 1
    factory.assert_not_called()
