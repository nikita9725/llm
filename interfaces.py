"""Explicit interfaces between the application's object-oriented layers."""

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

from schemas import Category, PipelineResult


class LLMGateway(Protocol):
    def complete(self, system_prompt: str, user_prompt: str) -> str: ...


class PipelineRunner(Protocol):
    def run(self, text: str) -> PipelineResult: ...


class ResultPresenter(Protocol):
    def present_result(self, title: str, result: PipelineResult) -> None: ...

    def present_errors(self, errors: Sequence[tuple[str, str]]) -> None: ...

    def present_summary(
        self,
        results: Sequence[tuple[str, PipelineResult]],
        expected_categories: Mapping[str, Category],
    ) -> None: ...

    def present_comparison(self, report: dict[str, Any], output: Path) -> None: ...


class ResultWriter(Protocol):
    def write(self, path: Path, payload: object) -> None: ...
