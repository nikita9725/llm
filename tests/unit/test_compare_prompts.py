import json

from application import compare_prompts
from llm_client import LLMAPIError
from prompts import MINIMAL_PROMPT
from tests.fakes import FakeLLMGateway


def flat_response(summary: str = "Summary") -> str:
    return json.dumps(
        {
            "summary": summary,
            "category": "support",
            "intent": "Get help",
            "sentiment": "neutral",
            "key_points": ["One", "Two", "Three"],
            "final_answer": "Response",
        }
    )


def test_comparison_runs_each_variant_on_each_input() -> None:
    gateway = FakeLLMGateway([flat_response(), flat_response()])

    report = compare_prompts(
        gateway,
        inputs=[("First", "Text one"), ("Second", "Text two")],
        variants=[MINIMAL_PROMPT],
    )

    assert len(gateway.calls) == 2
    assert report["metrics"][0]["valid_responses"] == 2


def test_comparison_records_errors_and_raw_response() -> None:
    gateway = FakeLLMGateway([LLMAPIError("offline"), '{"unexpected": true}'])

    report = compare_prompts(
        gateway,
        inputs=[("First", "Text one"), ("Second", "Text two")],
        variants=[MINIMAL_PROMPT],
    )

    assert report["runs"][0]["status"] == "error"
    assert report["runs"][1]["raw_response"] == '{"unexpected": true}'
