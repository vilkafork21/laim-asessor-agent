# Автоасессор

Система автоматической оценки ответов LLM-агентов на основе подхода "LLM As A Judge" с использованием RAG для few-shot обучения и Pydantic structured output.

Для `scoring.method=accuracy` эталонные `prediction` и `target` используются
только в `laim-baskets-adapter`, который материализует `main_metric`. Ассесор
калибруется на этой итоговой метке и на monitoring оценивает наблюдаемое
prediction без требования `GT` в трейсах. `acc_auto` — точность ассесора на
holdout эталонной корзины, а не метрика live-трейсов.

## Возможности

- Автоматическая разметка данных по заданным критериям
- **Pydantic Structured Output** - строго типизированные ответы от LLM
- Калибровка на размеченной выборке с расчетом метрик (Cohen's kappa, Krippendorff alpha, корреляция Спирмана)
- Поддержка нескольких асессоров с голосованием большинства
- Гибридный поиск (FAISS + BM25) для релевантных примеров
- Загрузка доменных знаний из документов (PDF, DOCX, TXT)
- Три режима работы: `scoring`, `monitoring`, `combined`

## Быстрый старт

### Установка

```bash
pip install -r requirements.txt
```

### Настройка окружения

Создайте файл `.env` со следующими параметрами:

```bash
# Модель
MODEL=GigaChat

# Аутентификация (Sigma контур)
CREDENTIALS=your_credentials
AUTH_URL=https://ngw.devices.sberbank.ru:9443/api/v2/oauth
BASE_URL=https://gigachat.devices.sberbank.ru/api/v1
SCOPE=GIGACHAT_API_PERS

# Или через AI Gateway (SDS контур)
# AI_GATEWAY_URL=https://ai-gateway.example.com

# Параметры модели
TEMPERATURE=0.001
TOP_P=0.001
TIMEOUT=300
VERIFY_SSL_CERTS=True
STREAMING=False
```

### Запуск

```python
from main import main
import json
import pandas as pd
from pathlib import Path

# Загрузка данных
rag_dataset = pd.read_csv("data/rag_dataset.csv", sep=";")
monitoring_umr = pd.read_parquet("data/monitoring_umr.parquet")
monitoring_metric = json.loads(Path("data/monitoring_metric.json").read_text())

# Запуск оценки
result = main(
    reference_umr=rag_dataset,
    monitoring_metric=monitoring_metric,
    assessor_instruction=Path("path/to/instruction.docx"),
    monitoring_umr=monitoring_umr,
    stage="combined",     # "scoring", "monitoring" или "combined"
    num_assessors=3,     # количество параллельных асессоров
)

# Получение результатов
scored_data = result["scored_data"]
accuracy = result["acc_auto"]
```

## Структура проекта

```
├── main.py                      # Главная функция, точка входа
├── run.py                       # Пример использования
├── test_run.py                  # Тестовый запуск
├── utils.py                     # Утилиты (работа с файлами, голосование)
├── agent/
│   ├── asessor_agent.py         # Основной класс Asessor
│   ├── config.py                # Конфигурация моделей (Sigma/SDS контуры)
│   ├── prompts.py               # Промпты для LLM
│   ├── score_results.py         # Обработка и оценка результатов
│   └── pydantic_output.py       # Динамическое создание Pydantic моделей
├── retriever/
│   ├── retriever.py             # RAG система (FAISS + BM25 гибридный поиск)
│   └── doc_loaders.py           # Загрузка документов (PDF, DOCX, TXT, CSV)
├── tests/
│   ├── test_pydantic_output.py  # Тесты Pydantic модуля
│   └── test_utils.py            # Тесты утилит
├── processed_data/              # Выходные данные
└── data/                        # Входные данные
```

## Основные модули

### agent/pydantic_output.py

Динамическое создание Pydantic моделей для structured output:

```python
from agent.pydantic_output import create_simple_output_model

# Создание модели на основе колонок из датасета
model = create_simple_output_model(
    answer_columns=["target", "cheked"],
    dataset=rag_dataset,
    model_name="AssessmentOutput",
)
```

Модель автоматически определяет типы полей (int, float, bool, str) и добавляет описания для GigaChat.

### agent/asessor_agent.py

Основной класс асессора с интеграцией Pydantic:

```python
from agent.asessor_agent import Asessor

asessor = Asessor(
    llm=llm,
    embedding_model=embedding_model,
    dataset=train_dataset,
    context_columns=["history"],
    answer_columns=["target", "cheked"],
    instruction=instruction_text,
)

# Асинхронный запуск разметки
results = await asessor.run(monitoring_umr)
```

### retriever/retriever.py

Гибридный поиск с RRF нормализацией:

- **FAISS** - плотный поиск по эмбеддингам
- **BM25** - разреженный поиск по ключевым словам
- **RRF** - объединение результатов

## Режимы работы (stage)

| Режим | Описание |
|-------|----------|
| `scoring` | Только калибровка на тестовой выборке (без мониторинга) |
| `monitoring` | Только разметка новых данных (без калибровки) |
| `combined` | Оба этапа последовательно |

## Многоколоночный асессор

При `num_assessors > 1` запускаются несколько параллельных асессоров. Результаты объединяются через голосование большинством:

```python
# 3 асессора
result = main(..., num_assessors=3)

# Результат содержит:
# agent_0_target, agent_1_target, agent_2_target - индивидуальные голоса
# agent_target - результат голосования большинства
```

## Метрики

Считаются на calibration-holdout между судьёй и человеческой разметкой,
печатаются в лог калибровки и уходят в `assessment_result.calibration_metrics`
(`acc_auto` остаётся отдельным float-портом):

- **Точность** (`acc_auto`): доля точных совпадений оценок
- **Baseline по моде** (`baseline_mode_accuracy`): точность константного ответа модой разметки
- **Cohen's kappa** (`cohen_kappa`): согласие судьи с человеком с поправкой на случайность
- **Krippendorff alpha** (`krippendorff_alpha`): надёжность согласия (nominal, по кодам меток)
- **Корреляция Спирмана** (`spearman_correlation`): связь между оценками агента и человека

## Входные данные

### Обязательные

- **reference_umr** (pd.DataFrame): Выборка с асессорской разметкой для few-shot примеров
  - Контекст строится централизованно из `assessment_mode`
  - Должна содержать колонки с критериями оценки (target, checked и др.)
- **monitoring_metric** (dict): контракт `laim-monitoring-metric.v2`

### Опциональные

- **assessor_instruction** (Path): DOCX или UTF-8 TXT с инструкцией
- **monitoring_umr** (pd.DataFrame): flat либо packed dialogue `laim-umr.v2`
- **domain_rag_files_zip** (Path): ZIP архив с доменными знаниями

## Архитектура

### Основные этапы (combined mode)

1. **Валидация данных**: Проверка наличия всех необходимых колонок
2. **Инициализация LLM**: Настройка GigaChat (sigma или sds контур)
3. **Обработка инструкции**: Структурирование в Markdown, суммаризация
4. **Создание Pydantic модели**: Динамическая генерация на основе answer_columns
5. **RAG инициализация**: Создание индекса для few-shot примеров
6. **Калибровка** (scoring): Оценка на 20% тестовой выборки
7. **Разметка** (monitoring): Оценка новых трейсов
8. **Голосование**: Объединение результатов нескольких асессоров

Единица оценки берётся из `monitoring_metric.assessment_mode`. В `qa`
оценивается один `current_turn`; в `turn_with_history` история нужна только для
понимания текущего хода; в `dialogue` весь упорядоченный список `turns`
получает одну сессионную оценку. Для `dialogue` один judge предсказывает сразу
итоговый `assessment_score` по эталонному `main_metric`; исходные голоса
ассессоров используются только при построении эталона и не становятся полями
LLM-ответа. Выход разворачивается в turn-строки только как транспорт для
downstream-нод: общий `assessment_unit_id` не позволяет KM посчитать длинный
диалог несколько раз. LLM-разметка помечается `assessor_id="judge"`.

### Pydantic Structured Output

Интеграция с GigaChat для строго типизированного вывода:

```python
# В Asessor используется:
self.agent_chain = self.printing_chain | self.llm.with_structured_output(self._output_model)
```

Это обеспечивает:
- Валидацию ответов на уровне LLM
- Автоматический парсинг без JSON ошибок
- Соответствие типов данным из `reference_umr`

## Требования

- Python 3.12+
- GigaChat API (Sberbank)
- Зависимости см. в `requirements.txt`
