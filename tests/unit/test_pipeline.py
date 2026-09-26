import json
from pathlib import Path
from unittest.mock import Mock

import pytest

import main as app
from llm_client import LLMAPIError
from main import PipelineError, parse_analysis_response, process_text
from prompts import MINIMAL_PROMPT

VALID_RESPONSE = json.dumps(
    {
        "summary": "Кратко",
        "key_points": ["Первое", "Второе", "Третье"],
        "helpful_response": "Полезный ответ",
    },
    ensure_ascii=False,
)


def test_process_text_returns_validated_model() -> None:
    client = Mock()
    client.complete.return_value = VALID_RESPONSE

    result = process_text(" Исходный текст ", client)

    assert result.summary == "Кратко"
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
        "not json",
        '{"summary": "x"}',
        '{"summary":"x","key_points":["1","2"],"helpful_response":"y"}',
        '{"summary":"x","key_points":["1","2","3"],"helpful_response":"y","extra":1}',
    ],
)
def test_process_text_rejects_invalid_structured_response(response: str) -> None:
    client = Mock()
    client.complete.return_value = response

    with pytest.raises(PipelineError, match="invalid structured JSON"):
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
    assert json.loads(output.read_text(encoding="utf-8"))["summary"] == "Кратко"


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


def test_demo_continues_after_one_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    client = Mock()
    client.complete.side_effect = [
        LLMAPIError("temporary failure"),
        VALID_RESPONSE,
        VALID_RESPONSE,
    ]
    monkeypatch.setattr(app.LLMClient, "from_env", lambda: client)

    assert app.main([]) == 1
    assert client.complete.call_count == 3


def test_cli_rejects_empty_text_before_loading_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    factory = Mock()
    monkeypatch.setattr(app.LLMClient, "from_env", factory)

    assert app.main(["--text", " "]) == 1
    factory.assert_not_called()
