from pathlib import Path
from typing import Any

import pytest

from cli import build_parser, run_cli


class RecordingAnalysisApplication:
    def __init__(self) -> None:
        self.calls: list[tuple[object, object, object]] = []

    def analyze(
        self,
        inputs: object,
        *,
        expected_categories: object = None,
        output: object = None,
    ) -> None:
        self.calls.append((inputs, expected_categories, output))


class RecordingComparisonApplication:
    def __init__(self) -> None:
        self.calls: list[tuple[object, object]] = []

    def run(self, inputs: object, output: object) -> dict[str, Any]:
        self.calls.append((inputs, output))
        return {}


class FakeRoot:
    def __init__(self) -> None:
        self.analysis = RecordingAnalysisApplication()
        self.comparison = RecordingComparisonApplication()

    def create_analysis_application(self) -> Any:
        return self.analysis

    def create_comparison_application(self) -> Any:
        return self.comparison


def test_cli_dispatches_analyze_without_global_patching() -> None:
    root = FakeRoot()

    run_cli(
        ["analyze", "--text", "Text", "--output", "result.json"],
        root=root,
    )

    inputs, expected, output = root.analysis.calls[0]
    assert inputs == [("Результат", "Text")]
    assert expected is None
    assert output == Path("result.json")


def test_cli_dispatches_prompt_comparison() -> None:
    root = FakeRoot()

    run_cli(["compare-prompts", "--output", "report.json"], root=root)

    assert root.comparison.calls[0][1] == Path("report.json")


def test_cli_rejects_empty_text_before_building_application() -> None:
    root = FakeRoot()

    with pytest.raises(SystemExit, match="must not be empty"):
        run_cli(["analyze", "--text", " "], root=root)

    assert root.analysis.calls == []


def test_parser_requires_one_subcommand() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args([])
