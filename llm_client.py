"""OpenAI-compatible implementation of the LLM gateway interface."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from openai import APIConnectionError, APIError, APIStatusError, OpenAI

LOGGER = logging.getLogger("llm_pipeline")


class LLMError(RuntimeError):
    """Base error for failures at the provider boundary."""


class LLMConfigurationError(LLMError):
    pass


class LLMAPIError(LLMError):
    pass


class EmptyLLMResponseError(LLMError):
    pass


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    """Explicit retry settings for transient provider failures."""

    max_attempts: int = 3
    base_delay_seconds: float = 0.5

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        if self.base_delay_seconds < 0:
            raise ValueError("base_delay_seconds must not be negative")

    def delay_after(self, failed_attempt: int) -> float:
        return self.base_delay_seconds * 2.0 ** (failed_attempt - 1)


@dataclass(frozen=True, slots=True)
class LLMConfig:
    api_key: str
    base_url: str
    model: str

    @classmethod
    def from_mapping(cls, source: Mapping[str, str | None]) -> LLMConfig:
        """Create configuration without reading global process state."""

        values = {
            name: (source.get(name) or "").strip()
            for name in ("LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL")
        }
        missing = [name for name, value in values.items() if not value]
        if missing:
            raise LLMConfigurationError(
                f"Missing required environment variables: {', '.join(missing)}"
            )
        return cls(
            api_key=values["LLM_API_KEY"],
            base_url=values["LLM_BASE_URL"],
            model=values["LLM_MODEL"],
        )


class LLMClient:
    """Small infrastructure adapter; orchestration lives elsewhere."""

    def __init__(
        self,
        config: LLMConfig,
        *,
        sdk_client: Any | None = None,
        retry_policy: RetryPolicy | None = None,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        self.config = config
        self._retry_policy = retry_policy or RetryPolicy()
        self._sleeper = sleeper
        self._client = sdk_client or OpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
            timeout=60.0,
            # Retries are implemented below so their policy and logs stay explicit.
            max_retries=0,
        )

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        for attempt in range(1, self._retry_policy.max_attempts + 1):
            try:
                response = self._client.chat.completions.create(
                    model=self.config.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    response_format={"type": "json_object"},
                    stream=False,
                )
            except APIError as error:
                can_retry = self._is_transient(error)
                if not can_retry or attempt == self._retry_policy.max_attempts:
                    LOGGER.error(
                        "LLM API request failed | attempt=%d/%d | transient=%s",
                        attempt,
                        self._retry_policy.max_attempts,
                        can_retry,
                    )
                    raise LLMAPIError(
                        f"The LLM provider request failed after {attempt} attempt(s)"
                    ) from error

                delay = self._retry_policy.delay_after(attempt)
                LOGGER.warning(
                    "Transient LLM API error; retrying | attempt=%d/%d | "
                    "delay=%.1fs | error=%s",
                    attempt,
                    self._retry_policy.max_attempts,
                    delay,
                    type(error).__name__,
                )
                self._sleeper(delay)
                continue

            content = self._response_content(response)
            if attempt > 1:
                LOGGER.info("LLM API request recovered | attempt=%d", attempt)
            return content

        raise AssertionError("retry loop must return or raise")

    @staticmethod
    def _is_transient(error: APIError) -> bool:
        if isinstance(error, APIConnectionError):
            return True
        if isinstance(error, APIStatusError):
            return error.status_code in {408, 409, 429} or error.status_code >= 500
        return False

    @staticmethod
    def _response_content(response: Any) -> str:
        content = response.choices[0].message.content if response.choices else None
        if not isinstance(content, str) or not content.strip():
            raise EmptyLLMResponseError("The LLM provider returned an empty response")
        return content.strip()
