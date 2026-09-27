from pathlib import Path

from examples import ROUTING_EXAMPLES
from schemas import Category

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SAMPLE_INPUTS_DIR = PROJECT_ROOT / "sample_inputs"


def test_demo_files_match_routing_examples() -> None:
    files = sorted(SAMPLE_INPUTS_DIR.glob("*.txt"))

    assert len(files) == len(ROUTING_EXAMPLES) == 10
    assert {file.name for file in files} == {
        example.filename for example in ROUTING_EXAMPLES
    }
    for example in ROUTING_EXAMPLES:
        content = (SAMPLE_INPUTS_DIR / example.filename).read_text(encoding="utf-8")
        assert content.strip() == example.text


def test_demo_set_covers_every_route_twice() -> None:
    categories = [example.expected_category for example in ROUTING_EXAMPLES]

    assert {category: categories.count(category) for category in Category} == {
        category: 2 for category in Category
    }
