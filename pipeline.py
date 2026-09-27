"""Type-safe orchestration of the five sequential LLM stages."""

import json
import logging
from dataclasses import dataclass

from pydantic import BaseModel, ValidationError

from interfaces import LLMGateway
from llm_client import LLMError
from prompts import (
    ClassificationPromptStrategy,
    FinalAnswerPromptStrategy,
    MeaningPromptStrategy,
    PromptStrategy,
    SelfCheckPromptStrategy,
    StructuredFieldsPromptStrategy,
)
from schemas import (
    Classification,
    ClassificationInput,
    FinalAnswer,
    FinalAnswerInput,
    MeaningExtraction,
    MeaningInput,
    PipelineResult,
    SelfCheck,
    SelfCheckInput,
    StructuredFields,
    StructuredFieldsInput,
    TextAnalysis,
)

LOGGER = logging.getLogger("llm_pipeline")


class PipelineError(RuntimeError):
    """A stage could not produce its promised validated output."""


def parse_model_response[OutputT: BaseModel](
    raw_response: str,
    model_type: type[OutputT],
    *,
    stage: str,
) -> OutputT:
    try:
        payload = json.loads(raw_response)
    except json.JSONDecodeError as error:
        raise PipelineError(
            f"Ответ модели на этапе {stage} не является корректным JSON "
            f"(строка {error.lineno}, столбец {error.colno})"
        ) from error

    try:
        return model_type.model_validate(payload)
    except ValidationError as error:
        issues = "; ".join(
            f"{'.'.join(map(str, issue['loc'])) or '<root>'}: {issue['msg']}"
            for issue in error.errors()
        )
        raise PipelineError(
            f"Ответ модели на этапе {stage} не прошёл проверку схемы: {issues}"
        ) from error


@dataclass(frozen=True, slots=True)
class LLMStep[InputT, OutputT: BaseModel]:
    """Template method shared by every LLM-backed stage."""

    gateway: LLMGateway
    prompt: PromptStrategy[InputT]
    output_model: type[OutputT]

    def run(self, data: InputT) -> OutputT:
        stage = self.prompt.stage_name
        LOGGER.info("Stage started: %s", stage)
        try:
            raw_response = self.gateway.complete(
                system_prompt=self.prompt.build_system_prompt(data),
                user_prompt=self.prompt.build_user_prompt(data),
            )
        except LLMError as error:
            raise PipelineError(f"Ошибка LLM на этапе {stage}: {error}") from error

        result = parse_model_response(raw_response, self.output_model, stage=stage)
        LOGGER.info("Stage completed: %s | %s", stage, result.model_dump_json())
        return result


@dataclass(frozen=True, slots=True)
class MultiStepPipeline:
    meaning_step: LLMStep[MeaningInput, MeaningExtraction]
    classification_step: LLMStep[ClassificationInput, Classification]
    structured_fields_step: LLMStep[StructuredFieldsInput, StructuredFields]
    final_answer_step: LLMStep[FinalAnswerInput, FinalAnswer]
    self_check_step: LLMStep[SelfCheckInput, SelfCheck]

    def run(self, text: str) -> PipelineResult:
        source_text = text.strip()
        if not source_text:
            raise PipelineError("Input text must not be empty")

        meaning = self.meaning_step.run(MeaningInput(source_text=source_text))
        classification = self.classification_step.run(
            ClassificationInput(source_text=source_text, meaning=meaning)
        )
        structured_fields = self.structured_fields_step.run(
            StructuredFieldsInput(
                source_text=source_text,
                meaning=meaning,
                classification=classification,
            )
        )
        final_answer = self.final_answer_step.run(
            FinalAnswerInput(
                source_text=source_text,
                meaning=meaning,
                classification=classification,
                structured_fields=structured_fields,
            )
        )
        self_check = self.self_check_step.run(
            SelfCheckInput(
                source_text=source_text,
                meaning=meaning,
                classification=classification,
                structured_fields=structured_fields,
                final_answer=final_answer,
            )
        )
        return PipelineResult(
            source_text=source_text,
            meaning=meaning,
            classification=classification,
            structured_fields=structured_fields,
            final_answer=final_answer,
            self_check=self_check,
        )


def build_default_pipeline(gateway: LLMGateway) -> MultiStepPipeline:
    """Composition helper: all default policies are visible in one place."""

    return MultiStepPipeline(
        meaning_step=LLMStep(gateway, MeaningPromptStrategy(), MeaningExtraction),
        classification_step=LLMStep(
            gateway, ClassificationPromptStrategy(), Classification
        ),
        structured_fields_step=LLMStep(
            gateway, StructuredFieldsPromptStrategy(), StructuredFields
        ),
        final_answer_step=LLMStep(gateway, FinalAnswerPromptStrategy(), FinalAnswer),
        self_check_step=LLMStep(gateway, SelfCheckPromptStrategy(), SelfCheck),
    )


def process_text(text: str, gateway: LLMGateway) -> PipelineResult:
    """Convenient library facade used by examples and E2E tests."""

    return build_default_pipeline(gateway).run(text)


def parse_analysis_response(raw_response: str) -> TextAnalysis:
    """Parse the historical flat response used by compare-prompts."""

    return parse_model_response(raw_response, TextAnalysis, stage="prompt comparison")
