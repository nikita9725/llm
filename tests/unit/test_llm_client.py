from types import SimpleNamespace

import pytest
from openai import APIConnectionError

from llm_client import (
    EmptyLLMResponseError,
    LLMAPIError,
    LLMClient,
    LLMConfig,
    LLMConfigurationError,
)


class FakeCompletions:
    def __init__(self, response: object = None, error: Exception | None = None) -> None:
        self.response = response
        self.error = error
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return self.response


class FakeSDK:
    def __init__(self, completions: FakeCompletions) -> None:
        self.chat = SimpleNamespace(completions=completions)


def response(content: str | None, *, choices: bool = True) -> SimpleNamespace:
    values = (
        [SimpleNamespace(message=SimpleNamespace(content=content))] if choices else []
    )
    return SimpleNamespace(choices=values)


def test_config_is_created_from_an_explicit_mapping() -> None:
    config = LLMConfig.from_mapping(
        {
            "LLM_API_KEY": "secret",
            "LLM_BASE_URL": "https://llm.example/v1",
            "LLM_MODEL": "example-model",
        }
    )

    assert config.model == "example-model"


def test_config_lists_missing_values() -> None:
    with pytest.raises(LLMConfigurationError) as error:
        LLMConfig.from_mapping({})

    assert "LLM_API_KEY" in str(error.value)
    assert "LLM_BASE_URL" in str(error.value)
    assert "LLM_MODEL" in str(error.value)


def test_client_uses_injected_sdk() -> None:
    completions = FakeCompletions(response('{"ok": true}'))
    client = LLMClient(
        LLMConfig("secret", "https://example.test", "model"),
        sdk_client=FakeSDK(completions),
    )

    assert client.complete("system", "user") == '{"ok": true}'
    assert completions.calls[0]["response_format"] == {"type": "json_object"}


@pytest.mark.parametrize(
    "provider_response", [response(None), response(" "), response("x", choices=False)]
)
def test_client_rejects_empty_provider_response(provider_response: object) -> None:
    client = LLMClient(
        LLMConfig("secret", "https://example.test", "model"),
        sdk_client=FakeSDK(FakeCompletions(provider_response)),
    )

    with pytest.raises(EmptyLLMResponseError):
        client.complete("system", "user")


def test_client_wraps_provider_errors() -> None:
    completions = FakeCompletions(error=APIConnectionError(request=object()))
    client = LLMClient(
        LLMConfig("secret", "https://example.test", "model"),
        sdk_client=FakeSDK(completions),
    )

    with pytest.raises(LLMAPIError):
        client.complete("system", "user")
