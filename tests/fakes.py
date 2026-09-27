"""Small interface implementations shared by unit and integration tests."""

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from pipeline import PipelineError
from schemas import (
    Category,
    Classification,
    FinalAnswer,
    MeaningExtraction,
    PipelineResult,
    SelfCheck,
    Sentiment,
    StructuredFields,
)


def stage_responses(category: str = "support") -> list[str]:
    return [
        json.dumps(
            {
                "core_meaning": "Пользователь просит помощь",
                "user_goal": "Решить проблему",
                "important_details": ["Ошибка возникает при входе"],
            },
            ensure_ascii=False,
        ),
        json.dumps(
            {"category": category, "intent": "Получить помощь"},
            ensure_ascii=False,
        ),
        json.dumps(
            {
                "summary": "Нужна помощь",
                "sentiment": "neutral",
                "key_points": ["Проблема", "Контекст", "Цель"],
            },
            ensure_ascii=False,
        ),
        json.dumps({"text": "Выполните диагностические шаги."}, ensure_ascii=False),
        json.dumps(
            {
                "is_consistent": True,
                "details_preserved": True,
                "issues": [],
            }
        ),
    ]


class FakeLLMGateway:
    def __init__(self, responses: Sequence[str | Exception]) -> None:
        self._responses = iter(responses)
        self.calls: list[tuple[str, str]] = []

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        self.calls.append((system_prompt, user_prompt))
        response = next(self._responses)
        if isinstance(response, Exception):
            raise response
        return response


def make_result(
    category: Category = Category.SUPPORT, *, passed: bool = True
) -> PipelineResult:
    return PipelineResult(
        source_text="Исходный текст",
        meaning=MeaningExtraction(
            core_meaning="Смысл",
            user_goal="Цель",
            important_details=["Деталь"],
        ),
        classification=Classification(category=category, intent="Намерение"),
        structured_fields=StructuredFields(
            summary="Резюме",
            sentiment=Sentiment.NEUTRAL,
            key_points=["Один", "Два", "Три"],
        ),
        final_answer=FinalAnswer(text="Ответ"),
        self_check=SelfCheck(
            is_consistent=passed,
            details_preserved=passed,
            issues=[] if passed else ["Потеряна деталь"],
        ),
    )


class FakePipeline:
    def __init__(self, results: Sequence[PipelineResult | PipelineError]) -> None:
        self._results = iter(results)
        self.inputs: list[str] = []

    def run(self, text: str) -> PipelineResult:
        self.inputs.append(text)
        result = next(self._results)
        if isinstance(result, PipelineError):
            raise result
        return result


class SpyPresenter:
    def __init__(self) -> None:
        self.results: list[tuple[str, PipelineResult]] = []
        self.summaries: list[
            tuple[Sequence[tuple[str, PipelineResult]], Mapping[str, Category]]
        ] = []
        self.comparisons: list[tuple[dict[str, Any], Path]] = []
        self.errors: list[Sequence[tuple[str, str]]] = []

    def present_result(self, title: str, result: PipelineResult) -> None:
        self.results.append((title, result))

    def present_errors(self, errors: Sequence[tuple[str, str]]) -> None:
        self.errors.append(errors)

    def present_summary(
        self,
        results: Sequence[tuple[str, PipelineResult]],
        expected_categories: Mapping[str, Category],
    ) -> None:
        self.summaries.append((results, expected_categories))

    def present_comparison(self, report: dict[str, Any], output: Path) -> None:
        self.comparisons.append((report, output))


class InMemoryResultWriter:
    def __init__(self) -> None:
        self.writes: list[tuple[Path, object]] = []

    def write(self, path: Path, payload: object) -> None:
        self.writes.append((path, payload))
