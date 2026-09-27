import os
import subprocess
import sys
from pathlib import Path

import pytest

from examples import ROUTING_EXAMPLES
from llm_client import LLMClient, LLMConfig, LLMConfigurationError
from main import process_text
from schemas import TextAnalysis

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.e2e
def test_cli_returns_valid_analysis(tmp_path: Path) -> None:
    if os.getenv("RUN_E2E") != "1":
        pytest.skip("Set RUN_E2E=1 to enable real API tests")

    try:
        LLMConfig.from_env()
    except LLMConfigurationError as error:
        pytest.skip(str(error))

    output_path = tmp_path / "result.json"
    completed = subprocess.run(
        [
            sys.executable,
            "main.py",
            "--text",
            (
                "Команда выпустила обновление раньше срока. Пользователи довольны "
                "скоростью, но просят улучшить документацию."
            ),
            "--output",
            str(output_path),
        ],
        cwd=PROJECT_ROOT,
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        timeout=90,
    )

    assert completed.returncode == 0, (
        f"CLI failed with exit code {completed.returncode}\n"
        f"stdout:\n{completed.stdout}\n"
        f"stderr:\n{completed.stderr}"
    )
    assert "Краткое резюме:" in completed.stdout
    assert "Категория:" in completed.stdout
    assert "Намерение:" in completed.stdout
    assert "Тональность:" in completed.stdout
    assert "Ключевые мысли:" in completed.stdout
    assert "Итоговый ответ:" in completed.stdout
    assert output_path.is_file()

    result = TextAnalysis.model_validate_json(output_path.read_text(encoding="utf-8"))
    assert result.summary
    assert result.category
    assert result.intent
    assert result.sentiment
    assert len(result.key_points) == 3
    assert all(result.key_points)
    assert result.final_answer


@pytest.mark.e2e
def test_routing_accuracy_on_labeled_examples() -> None:
    if os.getenv("RUN_ROUTING_E2E") != "1":
        pytest.skip("Set RUN_ROUTING_E2E=1 to enable the 20-call routing test")

    try:
        client = LLMClient.from_env()
    except LLMConfigurationError as error:
        pytest.skip(str(error))

    results = [
        (example, process_text(example.text, client)) for example in ROUTING_EXAMPLES
    ]
    correct = sum(
        result.category is example.expected_category for example, result in results
    )

    assert correct >= 8, "\n".join(
        f"{example.title}: expected={example.expected_category.value}, "
        f"actual={result.category.value}"
        for example, result in results
    )
    assert all(result.intent and result.final_answer for _, result in results)
