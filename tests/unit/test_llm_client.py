import logging
from types import SimpleNamespace

import httpx
import pytest
from openai import APIConnectionError, APIStatusError

from llm_client import (
    EmptyLLMResponseError,
    LLMAPIError,
    LLMClient,
    LLMConfig,
    LLMConfigurationError,
    RetryPolicy,
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


class ScriptedCompletions:
    def __init__(self, events: list[object | Exception]) -> None:
        self._events = iter(events)
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        event = next(self._events)
        if isinstance(event, Exception):
            raise event
        return event


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
        retry_policy=RetryPolicy(max_attempts=1),
    )

    with pytest.raises(LLMAPIError):
        client.complete("system", "user")


def test_client_retries_transient_errors_with_exponential_backoff(
    caplog: pytest.LogCaptureFixture,
) -> None:
    completions = ScriptedCompletions(
        [
            APIConnectionError(request=httpx.Request("POST", "https://example.test")),
            APIConnectionError(request=httpx.Request("POST", "https://example.test")),
            response('{"ok": true}'),
        ]
    )
    delays: list[float] = []
    client = LLMClient(
        LLMConfig("secret", "https://example.test", "model"),
        sdk_client=FakeSDK(completions),
        sleeper=delays.append,
    )

    with caplog.at_level(logging.INFO, logger="llm_pipeline"):
        result = client.complete("system", "user")

    assert result == '{"ok": true}'
    assert len(completions.calls) == 3
    assert delays == [0.5, 1.0]
    assert "request recovered" in caplog.text


def test_client_stops_after_retry_budget_is_exhausted() -> None:
    error = APIConnectionError(request=httpx.Request("POST", "https://example.test"))
    completions = ScriptedCompletions([error, error, error])
    delays: list[float] = []
    client = LLMClient(
        LLMConfig("secret", "https://example.test", "model"),
        sdk_client=FakeSDK(completions),
        sleeper=delays.append,
    )

    with pytest.raises(LLMAPIError, match="after 3 attempt"):
        client.complete("system", "user")

    assert len(completions.calls) == 3
    assert delays == [0.5, 1.0]


def test_client_does_not_retry_permanent_api_error() -> None:
    request = httpx.Request("POST", "https://example.test")
    error = APIStatusError(
        "Bad request",
        response=httpx.Response(400, request=request),
        body=None,
    )
    completions = ScriptedCompletions([error])
    delays: list[float] = []
    client = LLMClient(
        LLMConfig("secret", "https://example.test", "model"),
        sdk_client=FakeSDK(completions),
        sleeper=delays.append,
    )

    with pytest.raises(LLMAPIError, match="after 1 attempt"):
        client.complete("system", "user")

    assert len(completions.calls) == 1
    assert delays == []


def test_production_sdk_retries_are_disabled() -> None:
    client = LLMClient(LLMConfig("secret", "https://example.test", "model"))

    assert client._client.max_retries == 0

    client._client.close()


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"max_attempts": 0}, "max_attempts"),
        ({"base_delay_seconds": -0.1}, "base_delay_seconds"),
    ],
)
def test_retry_policy_rejects_invalid_values(
    kwargs: dict[str, int | float], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        RetryPolicy(**kwargs)
