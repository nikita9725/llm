import os
import subprocess
import sys
from pathlib import Path

import pytest

from llm_client import LLMConfig, LLMConfigurationError
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
    assert "Ключевые мысли:" in completed.stdout
    assert "Полезный ответ:" in completed.stdout
    assert output_path.is_file()

    result = TextAnalysis.model_validate_json(output_path.read_text(encoding="utf-8"))
    assert result.summary
    assert len(result.key_points) == 3
    assert all(result.key_points)
    assert result.helpful_response
