"""Provider-neutral client for OpenAI-compatible chat completion APIs."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from dotenv import load_dotenv
from openai import APIError, OpenAI


class LLMError(RuntimeError):
    """Base error for failures while calling an LLM provider."""


class LLMConfigurationError(LLMError):
    """Raised when required provider configuration is missing."""


class LLMAPIError(LLMError):
    """Raised when the provider rejects or cannot complete a request."""


class EmptyLLMResponseError(LLMError):
    """Raised when the provider returns no usable message content."""


@dataclass(frozen=True, slots=True)
class LLMConfig:
    """Runtime configuration for an OpenAI-compatible endpoint."""

    api_key: str
    base_url: str
    model: str

    @classmethod
    def from_env(cls) -> LLMConfig:
        """Load configuration from .env and the process environment."""

        load_dotenv()
        values = {
            "LLM_API_KEY": os.getenv("LLM_API_KEY", "").strip(),
            "LLM_BASE_URL": os.getenv("LLM_BASE_URL", "").strip(),
            "LLM_MODEL": os.getenv("LLM_MODEL", "").strip(),
        }
        missing = [name for name, value in values.items() if not value]
        if missing:
            joined = ", ".join(missing)
            raise LLMConfigurationError(
                f"Missing required environment variables: {joined}"
            )
        return cls(
            api_key=values["LLM_API_KEY"],
            base_url=values["LLM_BASE_URL"],
            model=values["LLM_MODEL"],
        )


class LLMClient:
    """Small adapter around the provider SDK; contains API-call logic only."""

    def __init__(
        self,
        config: LLMConfig,
        *,
        sdk_client: Any | None = None,
    ) -> None:
        self.config = config
        self._client = sdk_client or OpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
            timeout=60.0,
        )

    @classmethod
    def from_env(cls) -> LLMClient:
        return cls(LLMConfig.from_env())

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        """Request one JSON chat completion and return its textual content."""

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
