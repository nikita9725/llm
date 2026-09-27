"""Type-safe orchestration of the five sequential LLM stages."""

import json
import logging
from dataclasses import dataclass

from pydantic import BaseModel, ValidationError

from interfaces import LLMGateway
from llm_client import EmptyLLMResponseError, LLMError
from prompts import (
    ClassificationPromptStrategy,
    FinalAnswerPromptStrategy,
    MeaningPromptStrategy,
    PromptStrategy,
    SelfCheckPromptStrategy,
    StructuredFieldsPromptStrategy,
    normalize_prompt,
    render_prompt,
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
MAX_RAW_RESPONSE_CHARS = 10_000
MAX_REPAIR_RESPONSE_CHARS = 2_000


class PipelineError(RuntimeError):
    """A stage could not produce its promised validated output."""


class ResponseValidationError(PipelineError):
    """A provider response cannot be accepted as a stage result."""

    def __init__(self, message: str, *, raw_response: str) -> None:
        super().__init__(message)
        self.raw_response = raw_response


def parse_model_response[OutputT: BaseModel](
    raw_response: str,
    model_type: type[OutputT],
    *,
    stage: str,
) -> OutputT:
    if len(raw_response) > MAX_RAW_RESPONSE_CHARS:
        raise ResponseValidationError(
            f"Ответ модели на этапе {stage} слишком длинный: "
            f"{len(raw_response)} символов при лимите {MAX_RAW_RESPONSE_CHARS}",
            raw_response=raw_response,
        )

    try:
        payload = json.loads(raw_response)
    except json.JSONDecodeError as error:
        raise ResponseValidationError(
            f"Ответ модели на этапе {stage} не является корректным JSON "
            f"(строка {error.lineno}, столбец {error.colno})",
            raw_response=raw_response,
        ) from error

    try:
        return model_type.model_validate(payload)
    except ValidationError as error:
        issues = "; ".join(
            f"{'.'.join(map(str, issue['loc'])) or '<root>'}: {issue['msg']}"
            for issue in error.errors()
        )
        raise ResponseValidationError(
            f"Ответ модели на этапе {stage} не прошёл проверку схемы: {issues}",
            raw_response=raw_response,
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
        system_prompt = self.prompt.build_system_prompt(data)
        user_prompt = self.prompt.build_user_prompt(data)

        try:
            result = self._complete_and_validate(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                stage=stage,
            )
        except ResponseValidationError as first_error:
            LOGGER.warning(
                "Stage response rejected; using repair prompt | stage=%s | error=%s",
                stage,
                first_error,
            )
            repair_system, repair_user = self._build_repair_prompts(
                system_prompt,
                user_prompt,
                first_error,
            )
            try:
                result = self._complete_and_validate(
                    system_prompt=repair_system,
                    user_prompt=repair_user,
                    stage=stage,
                )
            except PipelineError as repair_error:
                LOGGER.error("Stage repair failed | stage=%s", stage)
                raise PipelineError(
                    f"Этап {stage} не восстановлен после fallback: "
                    f"первая ошибка: {first_error}; "
                    f"ошибка repair: {repair_error}"
                ) from repair_error
            LOGGER.info("Stage repaired: %s", stage)

        LOGGER.info("Stage completed: %s", stage)
        return result

    def _complete_and_validate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        stage: str,
    ) -> OutputT:
        try:
            raw_response = self.gateway.complete(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            )
        except EmptyLLMResponseError as error:
            raise ResponseValidationError(
                f"Ответ модели на этапе {stage} пуст",
                raw_response="",
            ) from error
        except LLMError as error:
            raise PipelineError(f"Ошибка LLM на этапе {stage}: {error}") from error

        return parse_model_response(raw_response, self.output_model, stage=stage)

    @staticmethod
    def _build_repair_prompts(
        system_prompt: str,
        user_prompt: str,
        error: ResponseValidationError,
    ) -> tuple[str, str]:
        invalid_response = error.raw_response[:MAX_REPAIR_RESPONSE_CHARS] or "<empty>"
        repair_instructions = normalize_prompt(
            """
            The previous response was rejected. Correct the response and return
            exactly one JSON object matching the requested schema. Do not include
            Markdown fences, commentary, or fields outside the schema.
            """
        )
        repair_system = f"{system_prompt}\n\n{repair_instructions}"

        repair_context = render_prompt(
            """
            <validation_error>
            {validation_error}
            </validation_error>
            <invalid_response>
            {invalid_response}
            </invalid_response>
            Return only the corrected JSON object.
            """,
            validation_error=error,
            invalid_response=invalid_response,
        )
        repair_user = f"{user_prompt}\n\n{repair_context}"

        return repair_system, repair_user


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
