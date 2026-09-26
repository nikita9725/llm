"""Run the Day 2 prompt comparison on the same demonstration inputs."""

import argparse
import json
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from examples import SAMPLE_INPUTS
from llm_client import LLMClient, LLMError
from main import PipelineError, parse_analysis_response
from prompts import PROMPT_VARIANTS, PromptVariant

LOGGER = logging.getLogger("prompt_comparison")


def compare_prompts(
    client: LLMClient,
    *,
    inputs: Sequence[tuple[str, str]] = SAMPLE_INPUTS,
    variants: Sequence[PromptVariant] = PROMPT_VARIANTS,
) -> dict[str, Any]:
    """Run every prompt variant on every input and collect comparable results."""

    runs: list[dict[str, Any]] = []
    metrics: list[dict[str, int | str]] = []

    for variant in variants:
        valid_responses = 0
        total_output_characters = 0
        for title, source_text in inputs:
            raw_response: str | None = None
            try:
                raw_response = client.complete(
                    system_prompt=variant.system_prompt,
                    user_prompt=variant.build_user_prompt(source_text.strip()),
                )
                result = parse_analysis_response(raw_response)
            except (PipelineError, LLMError) as error:
                failed_run = {
                    "variant": variant.name,
                    "input": title,
                    "status": "error",
                    "error": str(error),
                }
                if raw_response is not None:
                    failed_run["raw_response"] = raw_response
                runs.append(failed_run)
                continue

            payload = result.model_dump()
            valid_responses += 1
            total_output_characters += sum(
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
                "valid_responses": valid_responses,
                "total_inputs": len(inputs),
                "average_output_characters": (
                    round(total_output_characters / valid_responses)
                    if valid_responses
                    else 0
                ),
            }
        )

    return {"metrics": metrics, "runs": runs}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare all prompt variants on the same sample texts."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("prompt_comparison.json"),
        help="JSON report path (default: prompt_comparison.json)",
    )
    return parser


def _print_metrics(report: dict[str, Any]) -> None:
    print("Prompt comparison:")
    for metric in report["metrics"]:
        print(
            f"- {metric['variant']}: "
            f"{metric['valid_responses']}/{metric['total_inputs']} valid, "
            f"{metric['average_output_characters']} average output characters"
        )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        client = LLMClient.from_env()
        report = compare_prompts(client)
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    except (LLMError, OSError) as error:
        LOGGER.error("Prompt comparison failed: %s", error)
        return 1

    _print_metrics(report)
    print(f"Saved detailed report to {args.output}")
    has_valid_response = any(
        metric["valid_responses"] > 0 for metric in report["metrics"]
    )
    return 0 if has_valid_response else 1


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    raise SystemExit(main())
