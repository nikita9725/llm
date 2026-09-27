# Multi-step LLM pipeline — День 5

Учебное приложение показывает, как построить не один большой prompt, а прозрачную
цепочку из пяти LLM-вызовов. Результат каждого шага валидируется Pydantic и только
после этого передаётся дальше.

## Что построено

```text
CLI
 └── AnalysisApplication
      └── MultiStepPipeline
           ├── 1. extract meaning
           ├── 2. classify request
           ├── 3. build structured fields
           ├── 4. generate final answer
           └── 5. self-check result
```

Один текст требует пяти API-вызовов. Demo из десяти примеров — пятидесяти.

Этапы выполняются последовательно:

1. **Meaning extraction** выделяет смысл, цель пользователя и важные детали.
2. **Classification** получает исходный текст и результат первого этапа, затем
   выбирает одну из категорий: `support`, `feedback`, `complaint`, `sales`
   или `general_question`.
3. **Structured fields** строит резюме, тональность и три ключевые мысли.
4. **Final answer** использует все предыдущие результаты и инструкцию выбранного
   маршрута.
5. **Self-check** сравнивает ответ с исходным текстом, ищет противоречия и
   потерянные детали.

Self-check — аудитор, а не автор ответа. Если он возвращает `FAIL`, pipeline всё
равно успешно завершён, а обнаруженные проблемы сохраняются в `issues`.

## Почему код разделён на объекты

- `interfaces.py` содержит Protocol-интерфейсы. Бизнес-логика зависит от
  `LLMGateway`, а не от OpenAI SDK.
- `LLMStep` реализует Template Method: построить prompts, вызвать gateway,
  разобрать JSON, проверить схему и записать лог.
- Каждый этап имеет собственную `PromptStrategy`. Стратегию можно заменить без
  изменения orchestration.
- `MultiStepPipeline` показывает порядок выполнения без скрытой магии.
- `AnalysisApplication` координирует pipeline, вывод и сохранение.
- `ProductionCompositionRoot` — единственное место, где создаются конкретные
  production-объекты.
- `main.py` только запускает единую CLI.

Это не классическая Chain of Responsibility. В ней запрос обычно обрабатывает один
из подходящих обработчиков, а здесь обязательны все пять звеньев, поэтому явный
pipeline точнее отражает задачу.

## Установка

Требуются Python 3.14, [uv](https://docs.astral.sh/uv/) и API, совместимый с
OpenAI Chat Completions JSON mode.

```bash
uv python install 3.14
uv sync
cp .env.example .env
```

Настройте `.env`:

```dotenv
LLM_API_KEY=your-secret-key
LLM_BASE_URL=https://your-provider.example/v1
LLM_MODEL=your-model-name
RUN_E2E=false
```

Конфигурация создаётся через `LLMConfig.from_mapping()`. Чтение `.env` и
окружения выполняется только в composition root, поэтому unit-тестам не нужно
менять глобальное окружение.

## Единая точка входа

У приложения одна команда и две подкоманды.

Анализ одного текста:

```bash
uv run llm-pipeline analyze --text "Не могу войти после смены телефона"
```

Чтение UTF-8 файла и сохранение полной трассы:

```bash
uv run llm-pipeline analyze --input-file sample.txt --output result.json
```

Demo на десяти размеченных примерах:

```bash
uv run llm-pipeline analyze
```

Историческое сравнение промптов Дня 2:

```bash
uv run llm-pipeline compare-prompts --output prompt_comparison.json
```

`uv run python main.py ...` остаётся совместимым launcher, но вызывает тот же
CLI-dispatcher. У `compare_prompts.py` больше нет отдельного запуска.

При ожидаемой ошибке CLI показывает короткое сообщение и поднимает `SystemExit`.
Ручных `return 0` и `return 1` нет. Неожиданная ошибка сохраняет traceback,
что полезно при обучении и отладке.

## Итоговый JSON

```json
{
  "source_text": "Не могу войти после смены телефона",
  "meaning": {
    "core_meaning": "Пользователь потерял доступ",
    "user_goal": "Восстановить вход",
    "important_details": ["Проблема появилась после смены телефона"]
  },
  "classification": {
    "category": "support",
    "intent": "восстановить доступ"
  },
  "structured_fields": {
    "summary": "Нужна помощь со входом",
    "sentiment": "neutral",
    "key_points": ["Нет доступа", "Телефон изменён", "Нужно восстановление"]
  },
  "final_answer": {
    "text": "Проверьте доступ к привязанной почте..."
  },
  "self_check": {
    "is_consistent": true,
    "details_preserved": true,
    "issues": []
  }
}
```

## Тестирование

Тесты разделены по уровню:

- **unit** изолируют одну схему, стратегию или класс;
- **integration** собирают настоящие application, pipeline, пять шагов, presenter
  и JSON writer, но используют `FakeLLMGateway` без сети;
- **E2E** обращаются к реальному провайдеру только по явному флагу.

```bash
uv run pytest tests/unit
uv run pytest tests/integration
uv run pytest
uv run ruff format --check .
uv run ruff check .
uv run mypy
```

Реальный пятишаговый E2E:

```bash
RUN_E2E=true uv run pytest -m e2e
```

Полный routing E2E из 50 запросов:

```bash
RUN_ROUTING_E2E=true uv run pytest -m e2e \
  tests/e2e/test_pipeline_e2e.py::test_routing_accuracy_on_all_examples
```

Fake, spy и in-memory реализации передаются через конструкторы. В test suite нет
`monkeypatch`: зависимости заменяются через интерфейсы.

## Структура

- `main.py` — тонкий launcher;
- `cli.py` — единый parser, подкоманды и production composition root;
- `interfaces.py` — интерфейсы внешних границ;
- `application.py` — use cases, presenter и JSON writer;
- `pipeline.py` — `LLMStep` и пятишаговый orchestration;
- `prompts.py` — prompt strategies и исторические варианты Дня 2;
- `schemas.py` — входные, промежуточные и итоговые Pydantic-модели;
- `llm_client.py` — OpenAI-compatible реализация `LLMGateway`;
- `examples.py` — десять размеченных примеров;
- `tests/unit`, `tests/integration`, `tests/e2e` — три уровня тестов.
