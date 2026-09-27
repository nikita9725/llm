from pathlib import Path

import pytest

from application import AnalysisApplication, ApplicationError
from pipeline import PipelineError
from schemas import Category
from tests.fakes import (
    FakePipeline,
    InMemoryResultWriter,
    SpyPresenter,
    make_result,
)


def test_application_coordinates_dependencies() -> None:
    presenter = SpyPresenter()
    writer = InMemoryResultWriter()
    pipeline = FakePipeline([make_result()])
    app = AnalysisApplication(pipeline, presenter, writer)
    output = Path("result.json")

    report = app.analyze([("Example", "Text")], output=output)

    assert pipeline.inputs == ["Text"]
    assert presenter.results[0][0] == "Example"
    assert writer.writes == [(output, report.results[0][1])]


def test_application_builds_demo_summary() -> None:
    presenter = SpyPresenter()
    result = make_result(Category.SUPPORT)
    app = AnalysisApplication(FakePipeline([result]), presenter, InMemoryResultWriter())

    app.analyze(
        [("Example", "Text")],
        expected_categories={"Example": Category.SUPPORT},
    )

    assert len(presenter.summaries) == 1


def test_application_returns_partial_batch_success() -> None:
    presenter = SpyPresenter()
    pipeline = FakePipeline(
        [PipelineError("first failed"), make_result(Category.FEEDBACK)]
    )
    app = AnalysisApplication(pipeline, presenter, InMemoryResultWriter())

    report = app.analyze(
        [("First", "one"), ("Second", "two")],
        expected_categories={
            "First": Category.SUPPORT,
            "Second": Category.FEEDBACK,
        },
    )

    assert pipeline.inputs == ["one", "two"]
    assert [title for title, _ in presenter.results] == ["Second"]
    assert report.errors == [("First", "first failed")]
    assert presenter.errors == [[("First", "first failed")]]


def test_application_raises_when_every_input_fails() -> None:
    app = AnalysisApplication(
        FakePipeline([PipelineError("offline")]),
        SpyPresenter(),
        InMemoryResultWriter(),
    )

    with pytest.raises(ApplicationError, match="Only: offline"):
        app.analyze([("Only", "text")])


def test_failed_batch_writes_error_envelope_before_raising() -> None:
    writer = InMemoryResultWriter()
    app = AnalysisApplication(
        FakePipeline([PipelineError("first"), PipelineError("second")]),
        SpyPresenter(),
        writer,
    )
    output = Path("report.json")

    with pytest.raises(ApplicationError):
        app.analyze([("One", "one"), ("Two", "two")], output=output)

    assert writer.writes == [
        (
            output,
            {
                "results": [],
                "errors": [
                    {"title": "One", "message": "first"},
                    {"title": "Two", "message": "second"},
                ],
            },
        )
    ]
