import os
import subprocess
import sys
from pathlib import Path

import pytest
from dotenv import dotenv_values

from examples import ROUTING_EXAMPLES
from llm_client import LLMClient, LLMConfig, LLMConfigurationError
from pipeline import process_text
from schemas import PipelineResult

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def feature_enabled(name: str) -> bool:
    """Read an opt-in flag from the process first and then from local .env."""

    file_values = dotenv_values(PROJECT_ROOT / ".env")
    value = os.getenv(name, file_values.get(name) or "")
    return value.strip().lower() in {"1", "true", "yes", "on"}


def real_client() -> LLMClient:
    values: dict[str, str | None] = dict(dotenv_values(PROJECT_ROOT / ".env"))
    values.update(os.environ)
    return LLMClient(LLMConfig.from_mapping(values))


@pytest.mark.e2e
def test_unified_cli_returns_complete_trace(tmp_path: Path) -> None:
    if not feature_enabled("RUN_E2E"):
        pytest.skip("Set RUN_E2E=true to enable real API tests")
    try:
        real_client()
    except LLMConfigurationError as error:
        pytest.skip(str(error))

    output = tmp_path / "result.json"
    completed = subprocess.run(
        [
            sys.executable,
            "main.py",
            "analyze",
            "--text",
            "Приложение падает при загрузке PDF на Android 15. Как это исправить?",
            "--output",
            str(output),
        ],
        cwd=PROJECT_ROOT,
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        timeout=180,
    )

    assert completed.returncode == 0, completed.stderr
    result = PipelineResult.model_validate_json(output.read_text(encoding="utf-8"))
    assert result.final_answer.text
    assert "5. SELF-CHECK:" in completed.stdout


@pytest.mark.e2e
def test_routing_accuracy_on_all_examples() -> None:
    if not feature_enabled("RUN_ROUTING_E2E"):
        pytest.skip("Set RUN_ROUTING_E2E=true to enable the 50-call routing test")
    try:
        client = real_client()
    except LLMConfigurationError as error:
        pytest.skip(str(error))

    results = [
        (example, process_text(example.text, client)) for example in ROUTING_EXAMPLES
    ]
    correct = sum(
        result.classification.category is example.expected_category
        for example, result in results
    )

    assert correct >= 8
    assert all(result.final_answer.text for _, result in results)
