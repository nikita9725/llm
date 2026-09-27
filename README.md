# Надёжный multi-step LLM pipeline

Учебное приложение показывает, как построить прозрачную цепочку LLM-вызовов и
защитить её от сетевых сбоев и некорректных ответов модели. Результат каждого шага
проходит ограничения по размеру, JSON-декодирование и Pydantic-валидацию. Только
после этого данные передаются следующему этапу.

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

Вокруг каждого вызова работают два независимых механизма восстановления:

```text
LLMStep
 ├── LLMClient: transport retry для временной ошибки API
 ├── guardrails: размер → JSON → Pydantic schema
 └── repair prompt: один повтор для плохого ответа модели
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

## Почему retry разделён на два уровня

Временная недоступность API и невалидный ответ модели — разные виды ошибок.

`LLMClient` повторяет запрос, когда тот не был надёжно выполнен: при connection
error, timeout, HTTP 408/409/429 или 5xx. По умолчанию разрешены три попытки с
задержками 0.5 и 1.0 секунды. Ошибки вроде HTTP 400 не повторяются: тот же запрос
с теми же параметрами снова завершится так же.

Встроенные retry OpenAI SDK отключены через `max_retries=0`. Благодаря этому число
попыток не умножается скрыто, а каждая задержка и причина повтора видны в логах.
`RetryPolicy` передаётся в клиент явно, а функция ожидания заменяется в unit-тестах,
поэтому тесты backoff не спят по-настоящему.

`LLMStep` отвечает за semantic repair. Если API вернул пустую строку, текст вместо
JSON, объект с неправильными ключами или слишком длинное значение, этап один раз
отправляет repair prompt. В нём есть исходное задание, причина отклонения и не более
2000 символов плохого ответа. Если исправленный ответ тоже не проходит guardrails,
возвращается читаемый `PipelineError`.

Один этап может сделать не более шести физических API-запросов: до трёх transport
попыток исходного запроса и до трёх попыток единственного repair-запроса. Цикла
repair → repair нет.

## Guardrails

Проверки выполняются от дешёвых к более содержательным:

1. Пустой ответ отклоняется на границе `LLMClient`.
2. Сырой ответ ограничен 10 000 символами до разбора JSON.
3. `json.loads` запрещает текст, Markdown fences и синтаксически неверный JSON.
4. Строгие Pydantic-схемы запрещают лишние и отсутствующие поля, неверные типы,
   неизвестные enum-значения и превышение длины полей.

Например, финальный `text` ограничен 1000 символами, `summary` — 300, а
`key_points` должен содержать ровно три непустые строки.

## Как оформлены prompts

Многострочные prompts записаны с обычными отступами Python-кода, но перед отправкой
проходят через `normalize_prompt()`, основанный на `textwrap.dedent().strip()`.
Динамические многострочные значения вставляются функцией `render_prompt()` уже
после нормализации шаблона. Это важно: подстановка JSON внутрь f-string до `dedent`
мешала бы корректно вычислить общий отступ.

В результате код методов не теряет табуляцию, а в API не уходят случайные пробелы,
начальная пустая строка или завершающий перевод строки. XML-теги располагаются на
отдельных строках; вложенные отступы остаются только внутри JSON.

## Почему код разделён на объекты

- `interfaces.py` содержит Protocol-интерфейсы. Бизнес-логика зависит от
  `LLMGateway`, а не от OpenAI SDK.
- `LLMStep` реализует Template Method: построить prompts, вызвать gateway,
  применить fallback, разобрать JSON, проверить схему и записать лог.
- Каждый этап имеет собственную `PromptStrategy`. Стратегию можно заменить без
  изменения orchestration.
- `MultiStepPipeline` показывает порядок выполнения без скрытой магии.
- `AnalysisApplication` координирует pipeline, вывод и сохранение.
- `ProductionCompositionRoot` — единственное место, где создаются конкретные
  production-объекты.
- `main.py` только запускает единую CLI.
- Методы prompt-стратегий и внутренние helpers, которым не нужно состояние объекта,
  оформлены как `@staticmethod`. Фабрики и Pydantic-валидаторы остаются
  `@classmethod`, потому что используют `cls` по назначению.

Это не классическая Chain of Responsibility. В ней запрос обычно обрабатывает один
из подходящих обработчиков, а здесь обязательны все пять звеньев, поэтому явный
pipeline точнее отражает задачу.

## Быстрый старт с нуля

Требуются Python 3.12–3.14, [uv](https://docs.astral.sh/uv/) и API, совместимый
с OpenAI Chat Completions JSON mode. Команды ниже создают окружение строго по
зафиксированному `uv.lock`.

```bash
git clone <repository-url>
cd llm
uv python install 3.12
uv sync --locked
cp .env.example .env
```

Настройте `.env`:

```dotenv
LLM_API_KEY=your-secret-key
LLM_BASE_URL=https://your-provider.example/v1
LLM_MODEL=your-model-name
RUN_E2E=false
```

Проверьте установку без обращения к внешнему API:

```bash
uv run pytest
uv run llm-pipeline --help
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
uv run llm-pipeline analyze \
  --input-file sample_inputs/01_support_password_reset.txt \
  --output result.json
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

Пример сокращённого консольного результата:

```text
1. EXTRACT MEANING
   Смысл: Пользователь потерял доступ
2. CLASSIFY REQUEST
   Категория: support
3. BUILD STRUCTURED FIELDS
   Резюме: Нужна помощь со входом
4. GENERATE FINAL ANSWER
   Проверьте доступ к привязанной почте...
5. SELF-CHECK: PASS
```

При ожидаемой ошибке единственного текста CLI показывает короткое сообщение и
поднимает `SystemExit`. В пакетном demo ошибка одного текста не отменяет остальные:
программа показывает успешные результаты и отдельную секцию ошибок. Ненулевое
завершение происходит только тогда, когда не обработан ни один вход.

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

Для пакетного запуска `--output` содержит и успехи, и ошибки:

```json
{
  "results": [
    {"title": "Пример", "result": {"source_text": "..."}}
  ],
  "errors": [
    {"title": "Другой пример", "message": "Этап ... не восстановлен ..."}
  ]
}
```

Успешный одиночный запуск сохраняет прежний формат — непосредственно
`PipelineResult`, без дополнительной обёртки.

## Демонстрационные сценарии

Каталог `sample_inputs/` содержит самостоятельные UTF-8 входы, каждый из которых
можно передать через `--input-file`. Те же тексты входят во встроенный batch-demo;
unit-тест не позволяет двум наборам разойтись.

| Файл | Сценарий | Ожидаемый route |
| --- | --- | --- |
| `01_support_password_reset.txt` | восстановление доступа | `support` |
| `02_support_pdf_crash.txt` | сбой приложения | `support` |
| `03_feedback_search.txt` | отзыв о поиске | `feedback` |
| `04_feedback_dark_theme.txt` | предложение тёмной темы | `feedback` |
| `05_complaint_delayed_order.txt` | задержка заказа | `complaint` |
| `06_complaint_double_charge.txt` | двойное списание | `complaint` |
| `07_sales_demo.txt` | запрос демонстрации | `sales` |
| `08_sales_licenses.txt` | покупка лицензий | `sales` |
| `09_general_python.txt` | вопрос о Python | `general_question` |
| `10_general_planning.txt` | планирование встречи | `general_question` |

## Логирование

Логи показывают начало и завершение этапа, номер API-попытки, backoff, переход к
repair prompt, окончательную ошибку и частичный успех batch. API-ключи, полные
prompts и полные ответы модели не логируются. Это оставляет сообщения полезными для
диагностики и не раздувает вывод пользовательскими данными.

## Тестирование

Тесты разделены по уровню:

- **unit** изолируют одну схему, стратегию или класс;
- **integration** собирают настоящие application, pipeline, пять шагов, presenter
  и JSON writer, но используют `FakeLLMGateway` без сети;
- **E2E** обращаются к реальному провайдеру только по явному флагу.

Негативные тесты детерминированно проверяют текст вместо JSON, отсутствующие ключи,
пустой и слишком длинный ответ, исчерпание repair, временную недоступность API,
backoff, постоянную 4xx-ошибку и частично успешный batch. Реальные E2E оставлены для
happy path: внешний провайдер нельзя надёжно заставить вернуть конкретную ошибку.

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
- `examples.py` и `sample_inputs/` — десять размеченных demo-сценариев;
- `tests/unit`, `tests/integration`, `tests/e2e` — три уровня тестов.

## Соответствие финальному заданию

| Требование | Реализация |
| --- | --- |
| Принимать произвольный текст | `--text` и `--input-file` |
| Summary и key points | этап `StructuredFields`, ровно три key points |
| Категория и intent | этап `Classification`, пять допустимых routes |
| Structured JSON | строгие Pydantic-схемы и `--output` |
| Routing влияет на ответ | отдельная инструкция ответа для каждой категории |
| Multi-step workflow | пять последовательных этапов с типизированными входами |
| Guardrails | лимиты, strict JSON, запрет лишних полей, schema validation |
| Retries и fallback | transport retry и один semantic repair на этап |
| Логирование | начало/конец этапов, retry, repair и ошибки без секретов |
| Проверяемость | unit, integration, opt-in E2E и 10 demo-входов |

Таким образом, pipeline выполняет полный цикл: принимает сырой текст, извлекает
смысл, классифицирует запрос, строит структурированные поля, генерирует ответ по
выбранному маршруту и возвращает проверенную JSON-трассу. Это намеренно один
учебный workflow; RAG, Agents, web-интерфейс и production-инфраструктура остаются
за рамками mini-product.
