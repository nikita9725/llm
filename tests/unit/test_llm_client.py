from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from openai import APIConnectionError

import llm_client
from llm_client import (
    EmptyLLMResponseError,
    LLMAPIError,
    LLMClient,
    LLMConfig,
    LLMConfigurationError,
)


def make_response(content: str | None, *, choices: bool = True) -> SimpleNamespace:
    values = (
        [SimpleNamespace(message=SimpleNamespace(content=content))] if choices else []
    )
    return SimpleNamespace(choices=values)


def test_config_reads_neutral_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(llm_client, "load_dotenv", lambda: False)
    monkeypatch.setenv("LLM_API_KEY", "secret")
    monkeypatch.setenv("LLM_BASE_URL", "https://llm.example/v1")
    monkeypatch.setenv("LLM_MODEL", "example-model")

    config = LLMConfig.from_env()

    assert config == LLMConfig("secret", "https://llm.example/v1", "example-model")


def test_config_lists_missing_variables(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(llm_client, "load_dotenv", lambda: False)
    for name in ("LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL"):
        monkeypatch.delenv(name, raising=False)

    with pytest.raises(LLMConfigurationError) as error:
        LLMConfig.from_env()

    assert "LLM_API_KEY" in str(error.value)
    assert "LLM_BASE_URL" in str(error.value)
    assert "LLM_MODEL" in str(error.value)


def test_complete_sends_openai_compatible_json_request() -> None:
    sdk = Mock()
    sdk.chat.completions.create.return_value = make_response('{"ok": true}')
    client = LLMClient(
        LLMConfig("secret", "https://llm.example/v1", "example-model"),
        sdk_client=sdk,
    )

    result = client.complete("system", "user")

    assert result == '{"ok": true}'
    sdk.chat.completions.create.assert_called_once_with(
        model="example-model",
        messages=[
            {"role": "system", "content": "system"},
            {"role": "user", "content": "user"},
        ],
        response_format={"type": "json_object"},
        stream=False,
    )


@pytest.mark.parametrize(
    "response",
    [make_response(None), make_response("  "), make_response("x", choices=False)],
)
def test_complete_rejects_empty_response(response: SimpleNamespace) -> None:
    sdk = Mock()
    sdk.chat.completions.create.return_value = response
    client = LLMClient(
        LLMConfig("key", "https://example.test", "model"), sdk_client=sdk
    )

    with pytest.raises(EmptyLLMResponseError):
        client.complete("system", "user")


def test_complete_wraps_provider_api_errors() -> None:
    sdk = Mock()
    sdk.chat.completions.create.side_effect = APIConnectionError(request=Mock())
    client = LLMClient(
        LLMConfig("key", "https://example.test", "model"), sdk_client=sdk
    )

    with pytest.raises(LLMAPIError, match="provider request failed"):
        client.complete("system", "user")
