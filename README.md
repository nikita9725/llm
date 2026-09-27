# LLM pipeline

Учебный provider-neutral pipeline: принимает текст, вызывает любой OpenAI-compatible
LLM API, классифицирует запрос, явно выбирает маршрут ответа в Python и возвращает
строго валидированный JSON.

## Требования и установка

- Python 3.14
- [`uv`](https://docs.astral.sh/uv/)
- API, совместимый с OpenAI Chat Completions и JSON mode

```bash
uv python install 3.14
uv sync
cp .env.example .env
```

Заполните `.env` параметрами провайдера:

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

Текст из UTF-8 файла и сохранение результата:

```bash
uv run python main.py --input-file sample.txt --output output.json
```

Без аргументов программа последовательно обрабатывает десять размеченных примеров:

```bash
uv run python main.py
```

Для каждого примера выполняются классификация и генерация, поэтому demo-режим
делает 20 API-запросов. После результатов он выводит ожидаемые и полученные
категории и итоговую точность классификации.

## Сравнение промптов (День 2)

В `prompts.py` сохранены три независимых монолитных варианта: `minimal`,
`format_focused` и `structured`. Набор из десяти текстов можно прогнать через них
одной командой (30 API-запросов):

```bash
uv run python compare_prompts.py --output prompt_comparison.json
```

Это исторический эксперимент дня 2, а не основной routed pipeline. После изменения
контракта прежние сравнительные результаты необходимо получить заново.

## Structured output и валидация (День 3)

Итоговая форма результата:

```json
{
  "summary": "Краткое резюме",
  "sentiment": "neutral",
  "key_points": ["Первая мысль", "Вторая мысль", "Третья мысль"],
  "final_answer": "Итоговый ответ",
  "category": "support",
  "intent": "восстановить доступ к аккаунту"
}
```

Допустимые `category`: `support`, `feedback`, `complaint`, `sales`,
`general_question`. Допустимые `sentiment`: `positive`, `neutral`, `negative`,
`mixed`.

`summary` ограничен 300 символами, `intent` — 200, `final_answer` — 1000, а
`key_points` всегда содержит ровно три непустых строки. Pydantic-схемы запрещают
пропущенные и лишние поля, неизвестные enum-значения и неправильные типы данных.
Ошибки JSON и схемы содержат название этапа: классификация или генерация.

## Классификация и routing (День 4)

Основной pipeline использует два независимых LLM-вызова:

1. Классификатор возвращает только валидированные `category` и `intent`.
2. Python-код выбирает инструкцию из явного mapping по `Category`.
3. Генератор формирует остальные поля, после чего результаты объединяются.

Маршруты задают разное поведение:

- `support` — структурированные технические шаги;
- `feedback` — признание предложения и следующий шаг без ложных обещаний;
- `complaint` — эмпатичный план решения или эскалации;
- `sales` — краткое ценностное предложение и призыв к действию;
- `general_question` — прямой информативный ответ.

При сочетании технической проблемы с явной претензией приоритет получает
`complaint`. `intent` и остальные естественные поля используют язык входного текста.

## Тесты

Unit-тесты не обращаются к сети и не расходуют API quota:

```bash
uv run pytest
uv run ruff format --check .
uv run ruff check .
uv run mypy
```

Базовый E2E-тест выполняет реальный двухшаговый pipeline и запускается только явно:

```bash
RUN_E2E=1 uv run pytest -m e2e
```

Расширенная проверка routing обрабатывает все десять размеченных примеров (20
API-вызовов) и требует минимум 8 правильных категорий:

```bash
RUN_ROUTING_E2E=1 uv run pytest -m e2e \
  tests/e2e/test_pipeline_e2e.py::test_routing_accuracy_on_labeled_examples
```

Без флага запуска или необходимых `LLM_*` переменных E2E-тесты пропускаются.

## Структура

- `llm_client.py` — конфигурация и вызов API;
- `prompts.py` — промпты классификации и mapping инструкций маршрутов;
- `compare_prompts.py` — воспроизводимое сравнение вариантов дня 2;
- `schemas.py` — Pydantic-схемы этапов и итогового результата;
- `main.py` — двухшаговый orchestration и CLI;
- `examples.py` — десять размеченных текстов, по два на каждый маршрут;
- `tests/unit/` — изолированные unit-тесты;
- `tests/e2e/` — opt-in тесты с реальным LLM API.

Streaming, retries и fallback пока не реализованы.
