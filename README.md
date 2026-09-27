# LLM pipeline

Учебный provider-neutral pipeline: принимает текст, вызывает любой OpenAI-compatible
LLM API и возвращает строго валидированный JSON с резюме, классификацией,
тональностью, ключевыми мыслями и итоговым ответом.

## Требования

- Python 3.14
- [`uv`](https://docs.astral.sh/uv/)
- доступ к API, совместимому с OpenAI Chat Completions и JSON mode

## Установка

```bash
uv python install 3.14
uv sync
cp .env.example .env
```

Заполните `.env` параметрами своего провайдера:

```dotenv
LLM_API_KEY=your-secret-key
LLM_BASE_URL=https://your-provider.example/v1
LLM_MODEL=your-model-name
```

Файл `.env` исключён из Git. Ключ нельзя добавлять в `.env.example` или исходный код.

## Запуск

Один текст:

```bash
uv run python main.py --text "Текст для анализа"
```

Текст из UTF-8 файла:

```bash
uv run python main.py --input-file sample.txt
```

Без аргументов программа последовательно обработает пять встроенных примеров:

```bash
uv run python main.py
```

После результатов demo-режим выводит сводку по категориям и тональностям, а также
отмечает тексты с `negative` или `mixed` sentiment. Так структурированные поля
используются downstream-логикой, а не только отображаются пользователю.

Результат можно одновременно сохранить в JSON:

```bash
uv run python main.py --text "Текст" --output output.json
```

## Сравнение промптов (День 2)

В `prompts.py` определены три независимые пары system prompt и user template:

- `minimal` — короткая инструкция без детального контракта;
- `format_focused` — явная JSON-схема и измеримые ограничения;
- `structured` — приоритетные правила, границы исходного текста и критерии качества.

Одинаковый набор из пяти демонстрационных текстов можно прогнать через все варианты
одной командой (15 API-запросов):

```bash
uv run python compare_prompts.py --output prompt_comparison.json
```

Команда сохраняет успешные результаты, невалидные сырые ответы, ошибки и сводные
метрики. `structured` остаётся основным вариантом по умолчанию. После изменения
контракта дня 3 сравнительные метрики нужно получить заново реальным запуском;
старые результаты для трёх полей к новому формату неприменимы.

## Structured output и валидация (День 3)

Форма результата:

```json
{
  "summary": "Краткое резюме",
  "category": "request",
  "sentiment": "neutral",
  "key_points": ["Первая мысль", "Вторая мысль", "Третья мысль"],
  "final_answer": "Итоговый ответ"
}
```

Допустимые `category`: `question`, `request`, `feedback`, `problem`,
`informational`, `other`. Допустимые `sentiment`: `positive`, `neutral`,
`negative`, `mixed`.

`summary` ограничен 300 символами, `final_answer` — 200 символами, а `key_points`
всегда содержит ровно три непустых строки. Pydantic-схема запрещает пропущенные и
лишние поля, неизвестные enum-значения и неправильные типы данных.

Сломанный JSON и нарушение схемы обрабатываются отдельно. CLI возвращает код `1`
и сообщает либо строку и столбец синтаксической ошибки, либо конкретные поля,
которые не прошли проверку; необработанный traceback пользователю не показывается.

## Тесты

Unit-тесты не обращаются к сети и не расходуют API quota:

```bash
uv run pytest
```

Статические проверки и форматирование:

```bash
uv run ruff format --check .
uv run ruff check .
uv run mypy
```

Установка и ручной запуск Git hooks:

```bash
uv run pre-commit install
uv run pre-commit run --all-files
```

E2E-тест запускает приложение как отдельную CLI-команду, выполняет один реальный,
потенциально платный запрос и проверяет консольный вывод и сохранённый JSON. Тест
запускается только явно и использует конфигурацию из `.env`:

```bash
RUN_E2E=1 uv run pytest -m e2e
```

Без `RUN_E2E=1` или необходимых `LLM_*` переменных e2e-тест будет пропущен.

## Структура

- `llm_client.py` — только конфигурация и вызов API;
- `prompts.py` — шаблоны промптов;
- `compare_prompts.py` — воспроизводимое сравнение вариантов;
- `schemas.py` — Pydantic-схема результата;
- `main.py` — orchestration и CLI;
- `examples.py` — пять демонстрационных текстов;
- `tests/unit/` — изолированные unit-тесты без сетевых запросов;
- `tests/e2e/` — opt-in тесты с реальным LLM API.

На текущем этапе не реализованы routing, streaming, retries и fallback: они будут
добавляться в следующие дни проекта.
