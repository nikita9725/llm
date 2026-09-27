import json
from io import StringIO
from pathlib import Path

import pytest

from application import (
    AnalysisApplication,
    ConsolePresenter,
    JsonResultWriter,
    PromptComparisonApplication,
)
from cli import run_cli
from examples import ROUTING_EXAMPLES
from pipeline import build_default_pipeline
from schemas import PipelineResult
from tests.fakes import FakeLLMGateway, stage_responses


class TestCompositionRoot:
    __test__ = False

    def __init__(self, gateway: FakeLLMGateway, stream: StringIO) -> None:
        self.gateway = gateway
        self.presenter = ConsolePresenter(stream)
        self.writer = JsonResultWriter()

    def create_analysis_application(self) -> AnalysisApplication:
        return AnalysisApplication(
            build_default_pipeline(self.gateway), self.presenter, self.writer
        )

    def create_comparison_application(self) -> PromptComparisonApplication:
        return PromptComparisonApplication(self.gateway, self.presenter, self.writer)


@pytest.mark.integration
def test_unified_cli_runs_complete_analysis_and_writes_trace(
    tmp_path: Path,
) -> None:
    gateway = FakeLLMGateway(stage_responses())
    stream = StringIO()
    output = tmp_path / "result.json"
    root = TestCompositionRoot(gateway, stream)

    run_cli(
        ["analyze", "--text", "Исходный текст", "--output", str(output)],
        root=root,
        stream=stream,
    )

    assert len(gateway.calls) == 5
    assert "1. EXTRACT MEANING" in stream.getvalue()
    assert "5. SELF-CHECK: PASS" in stream.getvalue()
    saved = PipelineResult.model_validate_json(output.read_text(encoding="utf-8"))
    assert saved.meaning.core_meaning
    assert saved.self_check.passed


@pytest.mark.integration
def test_demo_integrates_ten_complete_chains() -> None:
    responses = [
        response
        for example in ROUTING_EXAMPLES
        for response in stage_responses(example.expected_category.value)
    ]
    gateway = FakeLLMGateway(responses)
    stream = StringIO()

    run_cli(["analyze"], root=TestCompositionRoot(gateway, stream), stream=stream)

    assert len(gateway.calls) == 50
    assert stream.getvalue().count("5. SELF-CHECK: PASS") == 10
    assert "Точность классификации: 10/10" in stream.getvalue()


@pytest.mark.integration
def test_compare_prompts_uses_the_same_cli_entry(tmp_path: Path) -> None:
    flat = json.dumps(
        {
            "summary": "Summary",
            "category": "support",
            "intent": "Get help",
            "sentiment": "neutral",
            "key_points": ["One", "Two", "Three"],
            "final_answer": "Response",
        }
    )
    gateway = FakeLLMGateway([flat] * 30)
    stream = StringIO()
    output = tmp_path / "comparison.json"

    run_cli(
        ["compare-prompts", "--output", str(output)],
        root=TestCompositionRoot(gateway, stream),
        stream=stream,
    )

    assert len(gateway.calls) == 30
    assert output.is_file()
    assert "Prompt comparison:" in stream.getvalue()
