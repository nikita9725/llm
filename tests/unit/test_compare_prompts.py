import json
from pathlib import Path
from unittest.mock import Mock

import pytest

import compare_prompts as app
from compare_prompts import compare_prompts
from llm_client import LLMAPIError
from prompts import FORMAT_FOCUSED_PROMPT, MINIMAL_PROMPT, STRUCTURED_PROMPT


def response(summary: str = "Summary") -> str:
    return json.dumps(
        {
            "summary": summary,
            "category": "support",
            "intent": "Get help",
            "sentiment": "neutral",
            "key_points": ["One", "Two", "Three"],
            "final_answer": "Response",
        }
    )


def test_compare_prompts_runs_each_variant_on_each_input() -> None:
    client = Mock()
    client.complete.return_value = response()
    inputs = [("First", "Text one"), ("Second", "Text two")]
    variants = [MINIMAL_PROMPT, FORMAT_FOCUSED_PROMPT, STRUCTURED_PROMPT]

    report = compare_prompts(client, inputs=inputs, variants=variants)

    assert client.complete.call_count == 6
    assert len(report["runs"]) == 6
    assert report["metrics"] == [
        {
            "variant": variant.name,
            "valid_responses": 2,
            "total_inputs": 2,
            "average_output_characters": 48,
        }
        for variant in variants
    ]


def test_compare_prompts_records_error_and_continues() -> None:
    client = Mock()
    client.complete.side_effect = [LLMAPIError("failed"), response()]

    report = compare_prompts(
        client,
        inputs=[("First", "Text one"), ("Second", "Text two")],
        variants=[MINIMAL_PROMPT],
    )

    assert client.complete.call_count == 2
    assert report["runs"][0] == {
        "variant": "minimal",
        "input": "First",
        "status": "error",
        "error": "failed",
    }
    assert report["metrics"][0]["valid_responses"] == 1


def test_compare_prompts_preserves_invalid_raw_response() -> None:
    client = Mock()
    client.complete.return_value = '{"unexpected": true}'

    report = compare_prompts(
        client,
        inputs=[("Input", "Text")],
        variants=[MINIMAL_PROMPT],
    )

    assert report["runs"][0]["raw_response"] == '{"unexpected": true}'


def test_cli_returns_failure_when_no_response_is_valid(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    client = Mock()
    client.complete.side_effect = LLMAPIError("failed")
    monkeypatch.setattr(app.LLMClient, "from_env", lambda: client)
    output = tmp_path / "comparison.json"

    exit_code = app.main(["--output", str(output)])

    assert exit_code == 1
    assert output.is_file()
    assert all(
        metric["valid_responses"] == 0
        for metric in json.loads(output.read_text(encoding="utf-8"))["metrics"]
    )
