"""OpenAI-compatible implementation of the LLM gateway interface."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from openai import APIError, OpenAI


class LLMError(RuntimeError):
    """Base error for failures at the provider boundary."""


class LLMConfigurationError(LLMError):
    pass


class LLMAPIError(LLMError):
    pass


class EmptyLLMResponseError(LLMError):
    pass


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

    def __init__(self, config: LLMConfig, *, sdk_client: Any | None = None) -> None:
        self.config = config
        self._client = sdk_client or OpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
            timeout=60.0,
        )

    def complete(self, system_prompt: str, user_prompt: str) -> str:
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
            raise LLMAPIError("The LLM provider request failed") from error

        content = response.choices[0].message.content if response.choices else None
        if not content or not content.strip():
            raise EmptyLLMResponseError("The LLM provider returned an empty response")
        return content.strip()
