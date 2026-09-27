"""The application's single command-line dispatcher and composition root."""

import argparse
import logging
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Protocol, TextIO

from dotenv import dotenv_values

from application import (
    AnalysisApplication,
    ApplicationError,
    ConsolePresenter,
    JsonResultWriter,
    PromptComparisonApplication,
)
from examples import ROUTING_EXAMPLES, SAMPLE_INPUTS
from llm_client import LLMClient, LLMConfig, LLMError
from pipeline import PipelineError, build_default_pipeline
from schemas import Category


class CompositionRoot(Protocol):
    """Factory interface that keeps concrete infrastructure out of the CLI."""

    def create_analysis_application(self) -> AnalysisApplication: ...

    def create_comparison_application(self) -> PromptComparisonApplication: ...


class ProductionCompositionRoot:
    def __init__(self, config: LLMConfig, stream: TextIO) -> None:
        self._gateway = LLMClient(config)
        self._presenter = ConsolePresenter(stream)
        self._writer = JsonResultWriter()

    def create_analysis_application(self) -> AnalysisApplication:
        return AnalysisApplication(
            build_default_pipeline(self._gateway), self._presenter, self._writer
        )

    def create_comparison_application(self) -> PromptComparisonApplication:
        return PromptComparisonApplication(self._gateway, self._presenter, self._writer)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llm-pipeline",
        description="Five-stage text analysis using an OpenAI-compatible LLM.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    analyze = commands.add_parser("analyze", help="Run the five-stage pipeline")
    source = analyze.add_mutually_exclusive_group()
    source.add_argument("--text", help="Text to analyze")
    source.add_argument("--input-file", type=Path, help="UTF-8 input file")
    analyze.add_argument("--output", type=Path, help="Write the complete JSON trace")

    compare = commands.add_parser(
        "compare-prompts", help="Run the historical Day 2 experiment"
    )
    compare.add_argument(
        "--output",
        type=Path,
        default=Path("prompt_comparison.json"),
        help="JSON report path",
    )
    return parser


def _configuration() -> LLMConfig:
    values: dict[str, str | None] = dict(dotenv_values())
    values.update(os.environ)
    return LLMConfig.from_mapping(values)


def _analysis_inputs(
    text: str | None, input_file: Path | None
) -> tuple[list[tuple[str, str]], Mapping[str, Category] | None]:
    if text is not None:
        if not text.strip():
            raise ApplicationError("Input text must not be empty")
        return [("Результат", text)], None
    if input_file is not None:
        try:
            content = input_file.read_text(encoding="utf-8")
        except OSError as error:
            raise ApplicationError(
                f"Could not read input file: {input_file}"
            ) from error
        if not content.strip():
            raise ApplicationError("Input text must not be empty")
        return [(input_file.name, content)], None
    expected = {
        example.title: example.expected_category for example in ROUTING_EXAMPLES
    }
    return [(example.title, example.text) for example in ROUTING_EXAMPLES], expected


def run_cli(
    argv: Sequence[str] | None = None,
    *,
    root: CompositionRoot | None = None,
    stream: TextIO | None = None,
) -> None:
    """Dispatch one command. Optional dependencies make integration tests patch-free."""

    args = build_parser().parse_args(argv)
    output_stream = stream or sys.stdout
    try:
        active_root = root or ProductionCompositionRoot(_configuration(), output_stream)
        if args.command == "analyze":
            inputs, expected = _analysis_inputs(args.text, args.input_file)
            active_root.create_analysis_application().analyze(
                inputs,
                expected_categories=expected,
                output=args.output,
            )
        else:
            active_root.create_comparison_application().run(SAMPLE_INPUTS, args.output)
    except (ApplicationError, PipelineError, LLMError) as error:
        raise SystemExit(str(error)) from error


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    run_cli()
