# LLM pipeline

Учебный provider-neutral pipeline: принимает текст, вызывает любой OpenAI-compatible
LLM API и возвращает валидированный JSON с кратким резюме, тремя ключевыми мыслями
и полезным ответом.

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

Без аргументов программа последовательно обработает три встроенных примера:

```bash
uv run python main.py
```

Результат можно одновременно сохранить в JSON:

```bash
uv run python main.py --text "Текст" --output output.json
```

Форма результата:

```json
{
  "summary": "Краткое резюме",
  "key_points": ["Первая мысль", "Вторая мысль", "Третья мысль"],
  "helpful_response": "Полезный ответ"
}
```

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
- `schemas.py` — Pydantic-схема результата;
- `main.py` — orchestration и CLI;
- `examples.py` — три демонстрационных текста;
- `tests/unit/` — изолированные unit-тесты без сетевых запросов;
- `tests/e2e/` — opt-in тесты с реальным LLM API.

На первом этапе не реализованы routing, streaming, retries и fallback: они будут
добавляться в следующие дни проекта.
