"""Command-line entry point and orchestration for the text pipeline."""

import argparse
import json
import logging
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel, ValidationError

from examples import ROUTING_EXAMPLES
from llm_client import LLMClient, LLMError
from prompts import (
    CLASSIFICATION_SYSTEM_PROMPT,
    build_classification_user_prompt,
    build_routed_system_prompt,
    build_routed_user_prompt,
)
from schemas import Category, Classification, RoutedResponse, Sentiment, TextAnalysis

LOGGER = logging.getLogger("llm_pipeline")


class PipelineError(RuntimeError):
    """Raised when pipeline input or model output is invalid."""


def _parse_model_response[OutputModel: BaseModel](
    raw_response: str,
    model_type: type[OutputModel],
    *,
    stage: str,
) -> OutputModel:
    """Parse one JSON response and validate it for a named pipeline stage."""

    try:
        payload = json.loads(raw_response)
    except json.JSONDecodeError as error:
        message = (
            f"Ответ модели на этапе {stage} не является корректным JSON "
            f"(строка {error.lineno}, столбец {error.colno})"
        )
        raise PipelineError(message) from error

    try:
        return model_type.model_validate(payload)
    except ValidationError as error:
        issues = "; ".join(
            f"{'.'.join(map(str, issue['loc'])) or '<root>'}: {issue['msg']}"
            for issue in error.errors()
        )
        raise PipelineError(
            f"Ответ модели на этапе {stage} не прошёл проверку схемы: {issues}"
        ) from error


def parse_analysis_response(raw_response: str) -> TextAnalysis:
    """Parse a complete response used by the historical prompt comparison."""

    return _parse_model_response(raw_response, TextAnalysis, stage="анализа")


def parse_classification_response(raw_response: str) -> Classification:
    """Parse the first-stage classification response."""

    return _parse_model_response(raw_response, Classification, stage="классификации")


def parse_routed_response(raw_response: str) -> RoutedResponse:
    """Parse the second-stage routed response."""

    return _parse_model_response(raw_response, RoutedResponse, stage="генерации")


def process_text(
    text: str,
    client: LLMClient,
) -> TextAnalysis:
    """Classify one text, route it in code, and validate the generated response."""

    normalized_text = text.strip()
    if not normalized_text:
        raise PipelineError("Input text must not be empty")

    LOGGER.info("Starting text classification")
    try:
        raw_classification = client.complete(
            system_prompt=CLASSIFICATION_SYSTEM_PROMPT,
            user_prompt=build_classification_user_prompt(normalized_text),
        )
    except LLMError as error:
        raise PipelineError(f"Ошибка LLM на этапе классификации: {error}") from error
    classification = parse_classification_response(raw_classification)

    LOGGER.info("Selected response route: %s", classification.category.value)
    try:
        raw_response = client.complete(
            system_prompt=build_routed_system_prompt(classification.category),
            user_prompt=build_routed_user_prompt(normalized_text, classification),
        )
    except LLMError as error:
        raise PipelineError(f"Ошибка LLM на этапе генерации: {error}") from error
    response = parse_routed_response(raw_response)
    result = TextAnalysis(
        **response.model_dump(),
        category=classification.category,
        intent=classification.intent,
    )

    LOGGER.info("Text analysis completed")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Summarize and analyze text using an OpenAI-compatible LLM."
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--text", help="Text to analyze")
    source.add_argument("--input-file", type=Path, help="UTF-8 text file to analyze")
    parser.add_argument("--output", type=Path, help="Write successful results as JSON")
    return parser


def _print_result(title: str, result: TextAnalysis) -> None:
    print(f"\n=== {title} ===")
    print(f"Краткое резюме: {result.summary}")
    print(f"Категория: {result.category.value}")
    print(f"Намерение: {result.intent}")
    print(f"Тональность: {result.sentiment.value}")
    print("Ключевые мысли:")
    for index, point in enumerate(result.key_points, start=1):
        print(f"  {index}. {point}")
    print(f"Итоговый ответ: {result.final_answer}")


def _print_demo_summary(
    results: Sequence[tuple[str, TextAnalysis]],
    expected_categories: dict[str, Category],
) -> None:
    """Print aggregate counts and highlight results requiring attention."""

    category_counts = Counter(result.category for _, result in results)
    sentiment_counts = Counter(result.sentiment for _, result in results)
    attention_titles = [
        title
        for title, result in results
        if result.sentiment in {Sentiment.NEGATIVE, Sentiment.MIXED}
    ]

    print("\n=== Сводка по примерам ===")
    print("Категории:")
    for category in Category:
        if count := category_counts[category]:
            print(f"  {category.value}: {count}")
    print("Тональности:")
    for sentiment in Sentiment:
        if count := sentiment_counts[sentiment]:
            print(f"  {sentiment.value}: {count}")
    if attention_titles:
        print(f"Требуют внимания: {', '.join(attention_titles)}")
    else:
        print("Требуют внимания: нет")

    correct = sum(
        result.category is expected_categories[title] for title, result in results
    )
    total = len(expected_categories)
    print(f"Точность классификации: {correct}/{total}")
    for title, result in results:
        expected = expected_categories[title]
        marker = "OK" if result.category is expected else "FAIL"
        print(
            f"  {marker} {title}: ожидалась {expected.value}, "
            f"получена {result.category.value}"
        )


def _write_json(path: Path, results: TextAnalysis | list[TextAnalysis]) -> None:
    payload: dict[str, object] | list[dict[str, object]]
    if isinstance(results, list):
        payload = [result.model_dump() for result in results]
    else:
        payload = results.model_dump()
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    LOGGER.info("Saved structured output to %s", path)


def _read_input_file(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError as error:
        raise PipelineError(f"Could not read input file: {path}") from error


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.text is not None:
        inputs = [("Результат", args.text)]
        demo_mode = False
    elif args.input_file is not None:
        try:
            file_text = _read_input_file(args.input_file)
        except PipelineError as error:
            LOGGER.error("%s", error)
            return 1
        inputs = [(args.input_file.name, file_text)]
        demo_mode = False
    else:
        inputs = [(example.title, example.text) for example in ROUTING_EXAMPLES]
        demo_mode = True

    if any(not text.strip() for _, text in inputs):
        LOGGER.error("Input text must not be empty")
        return 1

    try:
        client = LLMClient.from_env()
    except LLMError as error:
        LOGGER.error("%s", error)
        return 1

    titled_results: list[tuple[str, TextAnalysis]] = []
    had_errors = False
    for title, text in inputs:
        try:
            result = process_text(text, client)
        except (PipelineError, LLMError) as error:
            LOGGER.error("Analysis failed for %s: %s", title, error)
            had_errors = True
            if not demo_mode:
                break
            continue
        titled_results.append((title, result))
        _print_result(title, result)

    if demo_mode and titled_results:
        expected_categories = {
            example.title: example.expected_category for example in ROUTING_EXAMPLES
        }
        _print_demo_summary(titled_results, expected_categories)

    if args.output and not had_errors:
        try:
            results = [result for _, result in titled_results]
            output: TextAnalysis | list[TextAnalysis]
            output = results if demo_mode else results[0]
            _write_json(args.output, output)
        except OSError as error:
            LOGGER.error("Could not write output file: %s", error)
            return 1

    return 1 if had_errors else 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    raise SystemExit(main())
