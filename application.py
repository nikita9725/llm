"""Use cases and output adapters, independent from argparse and the SDK."""

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

from pydantic import BaseModel

from examples import SAMPLE_INPUTS
from interfaces import LLMGateway, PipelineRunner, ResultPresenter, ResultWriter
from llm_client import LLMError
from pipeline import PipelineError, parse_analysis_response
from prompts import PROMPT_VARIANTS, PromptVariant
from schemas import Category, PipelineResult, Sentiment


class ApplicationError(RuntimeError):
    """An expected use-case level failure suitable for a short CLI message."""


@dataclass(frozen=True, slots=True)
class ApplicationReport:
    results: list[tuple[str, PipelineResult]]
    errors: list[tuple[str, str]]


class AnalysisApplication:
    def __init__(
        self,
        pipeline: PipelineRunner,
        presenter: ResultPresenter,
        writer: ResultWriter,
    ) -> None:
        self._pipeline = pipeline
        self._presenter = presenter
        self._writer = writer

    def analyze(
        self,
        inputs: Sequence[tuple[str, str]],
        *,
        expected_categories: Mapping[str, Category] | None = None,
        output: Path | None = None,
    ) -> ApplicationReport:
        results: list[tuple[str, PipelineResult]] = []
        errors: list[tuple[str, str]] = []

        for title, text in inputs:
            try:
                result = self._pipeline.run(text)
            except PipelineError as error:
                errors.append((title, str(error)))
                continue
            results.append((title, result))
            self._presenter.present_result(title, result)

        if expected_categories is not None:
            self._presenter.present_summary(results, expected_categories)

        report = ApplicationReport(results=results, errors=errors)
        if output is not None and not errors:
            payload: object
            if expected_categories is None:
                payload = results[0][1] if results else []
            else:
                payload = [result for _, result in results]
            self._writer.write(output, payload)

        if errors:
            details = "; ".join(f"{title}: {message}" for title, message in errors)
            raise ApplicationError(f"Analysis failed: {details}")
        return report


class PromptComparisonApplication:
    def __init__(
        self,
        gateway: LLMGateway,
        presenter: ResultPresenter,
        writer: ResultWriter,
    ) -> None:
        self._gateway = gateway
        self._presenter = presenter
        self._writer = writer

    def run(
        self,
        inputs: Sequence[tuple[str, str]],
        output: Path,
        variants: Sequence[PromptVariant] = PROMPT_VARIANTS,
    ) -> dict[str, Any]:
        report = compare_prompts(self._gateway, inputs=inputs, variants=variants)
        self._writer.write(output, report)
        self._presenter.present_comparison(report, output)
        if not any(metric["valid_responses"] for metric in report["metrics"]):
            raise ApplicationError("Prompt comparison produced no valid responses")
        return report


def compare_prompts(
    gateway: LLMGateway,
    *,
    inputs: Sequence[tuple[str, str]] = SAMPLE_INPUTS,
    variants: Sequence[PromptVariant] = PROMPT_VARIANTS,
) -> dict[str, Any]:
    """Historical Day 2 experiment, now depending only on the gateway interface."""

    runs: list[dict[str, Any]] = []
    metrics: list[dict[str, int | str]] = []
    for variant in variants:
        valid = 0
        characters = 0
        for title, source_text in inputs:
            raw: str | None = None
            try:
                raw = gateway.complete(
                    variant.system_prompt,
                    variant.build_user_prompt(source_text.strip()),
                )
                result = parse_analysis_response(raw)
            except (PipelineError, LLMError) as error:
                failure: dict[str, Any] = {
                    "variant": variant.name,
                    "input": title,
                    "status": "error",
                    "error": str(error),
                }
                if raw is not None:
                    failure["raw_response"] = raw
                runs.append(failure)
                continue

            payload = result.model_dump(mode="json")
            valid += 1
            characters += sum(
                len(value) if isinstance(value, str) else sum(map(len, value))
                for value in payload.values()
            )
            runs.append(
                {
                    "variant": variant.name,
                    "input": title,
                    "status": "valid",
                    "result": payload,
                }
            )
        metrics.append(
            {
                "variant": variant.name,
                "valid_responses": valid,
                "total_inputs": len(inputs),
                "average_output_characters": round(characters / valid) if valid else 0,
            }
        )
    return {"metrics": metrics, "runs": runs}


class JsonResultWriter:
    def write(self, path: Path, payload: object) -> None:
        def serializable(value: object) -> object:
            if isinstance(value, BaseModel):
                return value.model_dump(mode="json")
            if isinstance(value, list):
                return [serializable(item) for item in value]
            return value

        try:
            path.write_text(
                json.dumps(serializable(payload), ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        except OSError as error:
            raise ApplicationError(f"Could not write output file: {path}") from error


class ConsolePresenter:
    def __init__(self, stream: TextIO) -> None:
        self._stream = stream

    def _line(self, value: str = "") -> None:
        self._stream.write(value + "\n")

    def present_result(self, title: str, result: PipelineResult) -> None:
        self._line(f"\n=== {title} ===")
        self._line("1. EXTRACT MEANING")
        self._line(f"   Смысл: {result.meaning.core_meaning}")
        self._line(f"   Цель: {result.meaning.user_goal}")
        self._line(f"   Детали: {', '.join(result.meaning.important_details)}")
        self._line("2. CLASSIFY REQUEST")
        self._line(f"   Категория: {result.classification.category.value}")
        self._line(f"   Намерение: {result.classification.intent}")
        self._line("3. BUILD STRUCTURED FIELDS")
        self._line(f"   Резюме: {result.structured_fields.summary}")
        self._line(f"   Тональность: {result.structured_fields.sentiment.value}")
        for index, point in enumerate(result.structured_fields.key_points, start=1):
            self._line(f"   {index}) {point}")
        self._line("4. GENERATE FINAL ANSWER")
        self._line(f"   {result.final_answer.text}")
        status = "PASS" if result.self_check.passed else "FAIL"
        self._line(f"5. SELF-CHECK: {status}")
        for issue in result.self_check.issues:
            self._line(f"   - {issue}")

    def present_summary(
        self,
        results: Sequence[tuple[str, PipelineResult]],
        expected_categories: Mapping[str, Category],
    ) -> None:
        self._line("\n=== Сводка по примерам ===")
        category_counts = Counter(
            result.classification.category for _, result in results
        )
        sentiment_counts = Counter(
            result.structured_fields.sentiment for _, result in results
        )
        self._line("Категории:")
        for category in Category:
            if count := category_counts[category]:
                self._line(f"  {category.value}: {count}")
        self._line("Тональности:")
        for sentiment in Sentiment:
            if count := sentiment_counts[sentiment]:
                self._line(f"  {sentiment.value}: {count}")
        passed = sum(result.self_check.passed for _, result in results)
        self._line(f"Self-check PASS/FAIL: {passed}/{len(results) - passed}")
        correct = sum(
            result.classification.category is expected_categories[title]
            for title, result in results
        )
        self._line(f"Точность классификации: {correct}/{len(expected_categories)}")

    def present_comparison(self, report: dict[str, Any], output: Path) -> None:
        self._line("Prompt comparison:")
        for metric in report["metrics"]:
            self._line(
                f"- {metric['variant']}: {metric['valid_responses']}/"
                f"{metric['total_inputs']} valid, "
                f"{metric['average_output_characters']} average output characters"
            )
        self._line(f"Saved detailed report to {output}")
