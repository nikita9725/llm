"""Command-line entry point and orchestration for the text pipeline."""

import argparse
import json
import logging
from collections.abc import Sequence
from pathlib import Path

from pydantic import ValidationError

from examples import SAMPLE_INPUTS
from llm_client import LLMClient, LLMError
from prompts import DEFAULT_PROMPT_VARIANT, PromptVariant
from schemas import TextAnalysis

LOGGER = logging.getLogger("llm_pipeline")


class PipelineError(RuntimeError):
    """Raised when pipeline input or model output is invalid."""


def parse_analysis_response(raw_response: str) -> TextAnalysis:
    """Parse and validate one structured model response."""

    try:
        payload = json.loads(raw_response)
        return TextAnalysis.model_validate(payload)
    except (json.JSONDecodeError, ValidationError) as error:
        LOGGER.error("The model returned an invalid structured response")
        raise PipelineError("The model returned invalid structured JSON") from error


def process_text(
    text: str,
    client: LLMClient,
    prompt_variant: PromptVariant = DEFAULT_PROMPT_VARIANT,
) -> TextAnalysis:
    """Analyze one text and validate the provider's structured response."""

    normalized_text = text.strip()
    if not normalized_text:
        raise PipelineError("Input text must not be empty")

    LOGGER.info("Starting text analysis")
    raw_response = client.complete(
        system_prompt=prompt_variant.system_prompt,
        user_prompt=prompt_variant.build_user_prompt(normalized_text),
    )
    result = parse_analysis_response(raw_response)

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
    print("Ключевые мысли:")
    for index, point in enumerate(result.key_points, start=1):
        print(f"  {index}. {point}")
    print(f"Полезный ответ: {result.helpful_response}")


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
        inputs = SAMPLE_INPUTS
        demo_mode = True

    if any(not text.strip() for _, text in inputs):
        LOGGER.error("Input text must not be empty")
        return 1

    try:
        client = LLMClient.from_env()
    except LLMError as error:
        LOGGER.error("%s", error)
        return 1

    results: list[TextAnalysis] = []
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
        results.append(result)
        _print_result(title, result)

    if args.output and not had_errors:
        try:
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
