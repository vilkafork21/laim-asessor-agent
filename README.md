# laim-asessor-agent

Нода-судья мониторингового контура LAIM. Принимает **эталонную корзину**
(`reference_umr`), **контракт метрики** (`monitoring_metric`), **инструкцию
ассессора** и **трейсы мониторинга** (`monitoring_umr`); LLM-судья размечает
трейсы по инструкции с few-shot примерами из корзины и отдаёт в контур:

1. `scored_data` — трейсы с оценкой судьи в `main_metric`;
2. `acc_auto` — точность судьи на holdout эталонной корзины;
3. `assessment_result` — машиночитаемый статус и покрытие расчёта.

## Зачем нода нужна

КМ контура определена на человеческой разметке корзины, а живые трейсы никто
не размечает. Нода закрывает этот разрыв: судья калибруется на оценках самой
корзины и переносит её политику разметки на трейсы, так что вниз по контуру
уходит тот же `main_metric`, что и в эталоне. Ключевые решения:

- **Судья предсказывает итоговую оценку, а не голоса разметчиков.** Для
  `scoring.method=accuracy` и для режима `dialogue` контракт переписывается в
  `identity` по `main_metric`: эталонные `prediction`/`target` и голоса трёх
  ассессоров используются только адаптером при построении `main_metric`, а
  судья учится на итоговой метке и на мониторинге не требует `GT` в трейсах.
- **Судья видит только UMR.** Снижать оценку он вправе лишь за нарушения,
  названные в инструкции и видимые в данных; то, что проверяется только
  знанием предметной области, он не угадывает. Оценки на нижней границе шкалы
  проходят второй проход-«адвокат» (`REVIEW_PROMPT`).
- **Калибровка обязательна и не приукрашивает.** Помимо `acc_auto`
  публикуются baseline по моде, каппа Коэна, альфа Криппендорфа и полнота/
  точность на дефектах — на перекошенной разметке судья, зачитывающий всё
  подряд, виден только по ним.
- **Падение там, где измерение стало бы нерепрезентативным:** нет инструкции,
  отказ судьи больше чем на 10% строк, эталон без разметки. Деградация —
  только явная: `not_computable` от адаптера, отключённый доменный RAG,
  единичные отказы судьи.

## Место в контуре

```text
laim-baskets-adapter.reference_umr ─────────────────┐
laim-kriteria-selector.validated_monitoring_metric ─┤
laim-traces-dataset-converter.monitoring_umr ───────┤
assessor_instruction (источник данных workflow) ────┤
domain_rag_files_zip (в port_wiring не подключён) ──┤
                                                    ▼
                                           laim-asessor-agent
        │
        ├──► scored_data       ─► laim-global-drift-test.monitoring_umr, laim-km-dynamic-test.scored_df
        ├──► acc_auto          ─► laim-km-dynamic-test.acc_auto, laim-agg.assessor_accuracy
        └──► assessment_result ─► laim-km-dynamic-test.assessment_result, laim-agg.assessment_result
```

## Порты и настройки

### Входы

| Порт | Обязателен | Что приходит с платформы |
|---|---|---|
| `reference_umr` | да | dataframe: корзина в формате тестового датасета (flat или packed dialogue `laim-umr.v2`) с `main_metric` и колонками источников контракта |
| `monitoring_metric` | да | контракт `laim-monitoring-metric.v2`; `v1` с `evaluation_unit=turn` поднимается до v2 автоматически, `v1` dialogue отклоняется |
| `monitoring_umr` | да в `descriptor.json`; код требует его при `stage` `monitoring`/`combined` | dataframe выхода TDC: packed dialogue или flat с `session_id`; принимаются также parquet bytes и путь к parquet/xlsx/csv |
| `assessor_instruction` | да, `getPortAsLocalPath` | локальный путь к DOCX или UTF-8 TXT: файл либо каталог артефакта ровно с одним таким файлом; файл без расширения распознаётся по сигнатуре ZIP |
| `domain_rag_files_zip` | нет, `getPortAsLocalPath` | ZIP с документами домена: pdf, docx, doc, txt, md, csv, xlsx, xls |

### Выходы

| Порт | Тип | Что отдаёт |
|---|---|---|
| `scored_data` | dataframe | строки `monitoring_umr` после нормализации (при `stage=scoring` — строки `reference_umr`) плюс `main_metric`, `assessment_unit_id`, `assessor_id`, `agent_<source_id>` |
| `acc_auto` | default | float — доля точных совпадений судьи с разметкой на holdout; `None` при `stage=monitoring` и при `not_computable` |
| `assessment_result` | default | `laim-assessment-result.v1`: статус, режим, покрытие, `calibration_metrics` |

### Настройки

| Настройка | По умолчанию | Зачем менять |
|---|---|---|
| `scoring_rag_train_size` | `0.8` | Доля эталона в train (few-shot база); остальное — holdout для `acc_auto`. Меньше — устойчивее метрики калибровки, беднее примеры |
| `model_id` | `giga` | Маршрут судьи: `giga` — GigaChat через `langchain-gigachat`; `minimax-m2.5`, `qwen3-coder-next` — AI Gateway (нужен `AI_GATEWAY_URL`) |
| `llm_model` | `GigaChat-3-Ultra` | Точная модель GigaChat; действует только для `giga`; переменная окружения `MODEL` приоритетнее |
| `num_assessors` | `1` | Сколько раз судья независимо оценивает каждую единицу; при `>1` итог — голос большинства |
| `instruction_llm_preprocessing` | `false` | Включает LLM-структурирование в Markdown и суммаризацию инструкции; по умолчанию инструкция подаётся дословно |
| `stage` | `combined` | `scoring` — только калибровка; `monitoring` — только разметка трейсов (без `acc_auto`); `combined` — оба |
| `min_holdout_units` | `20` | Допуск судьи (тест 6.3.3): меньше единиц в holdout — `admission_status = not_assessed` |
| `min_holdout_defect_units` | `4` | Меньше единиц критичного класса (минимальная оценка) в holdout — `not_assessed`: профиль ошибок не измерим |
| `weak_holdout_defect_units` | `10` | От `min_holdout_defect_units` до этого числа дефектов — допуск `amber` (профиль ошибок измерен грубо) |
| `min_defect_recall` | `0.5` | Полнота судьи на критичном классе; ниже — `amber`, вместе с низкой каппой — `red` |
| `min_kappa` | `0.2` | Каппа Коэна судьи против разметчиков; ниже — `amber`, вместе с низкой полнотой — `red` (судья не лучше моды) |
| `max_invalid_share` | `0.2` | Допустимая доля отказов судьи: выше — `red` на калибровке и `not_computable/judge_refusals` на мониторинге |

Пороги допуска — временные параметры мониторинга: калиброваны на пилотной
корзине CI09997554 (283 единицы, около 7 % дефектов, каппа 0.40, полнота 0.75)
и подлежат пересмотру по мере накопления реальных прогонов.

## Как проходит прогон

```text
1. Контракт      validate_monitoring_metric: v1→v2, статус; not_computable → короткий выход
2. Нормализация  normalize_umr: packed dialogue → плоские строки с reference_group_id/turn_index
3. Цель судьи    _assessment_contract: accuracy и dialogue → identity по main_metric
4. Единицы       unitize по assessment_mode; эталон — только размеченные единицы
5. Модели        судья по model_id; эмбеддер GigaChat с чанкованием
6. Калибровка    стратифицированный сплит → судья на holdout → acc_auto и метрики согласия
7. Мониторинг    судья с few-shot из всего размеченного эталона → оценка трейсов
8. Публикация    оценки на turn-строки, assessment_result
```

**1–2.** Контракт проверяется с `require_computed=False`: статус
`not_computable` от адаптера не роняет ноду, а уходит наружу тем же статусом.
Обе таблицы приводятся к плоской форме: packed `dialogue` разворачивается в
строки с `reference_group_id`/`turn_index`, flat с `session_id` получает группу
из него; смешанная форма и пустой `query_id`/`session_id` — ошибка контракта.

**3.** Для `accuracy` и `dialogue` источники контракта заменяются одним:
`source_id=assessment_score`, `column_name=main_metric`, `role=final_score`.
Для `accuracy` в эталоне `output_answer` подменяется колонкой `prediction`,
чтобы контекст судьи совпадал с тем, что он увидит в трейсах. К инструкции
дописывается перечень полей ответа (`_source_instruction`).

**4.** Единица оценки — по `assessment_mode`: `qa` — строка (`current_turn`),
`turn_with_history` — строка с упорядоченной `history`, `dialogue` — сессия с
полным списком `turns` (оценка и источники внутри сессии обязаны быть
константны). Контекст сериализуется в JSON `assessment_context` — единственная
context-колонка судьи. Из эталона остаются только единицы с непустой
разметкой по всем источникам (без записи в лог); эталон без единой размеченной
единицы — `MonitoringContractError`.

**5.** Судья один на весь прогон (см. «Внешние сервисы»). Эмбеддер —
`BoundedGigaChatEmbeddings`: текст режется на куски по `EMBEDDING_MAX_CHARS=1000`
символов, отправляется пакетами по `EMBEDDING_BATCH_SIZE=100`, вектор единицы —
среднее кусков (эмбеддер GigaChat принимает не более 514 токенов, а единица
несёт полный диалог).

**6.** Сплит `_split_units`: стратификация по метке первого источника
(для `dialogue` — по группе с минимальной меткой в ней), перемешивание с
`random_state=42384`, в train попадает `int(n * scoring_rag_train_size)` единиц
каждого класса, но не меньше одной и не все; дефектов в корзинах 5–12%, и без
стратификации holdout остаётся без них. На train строится `Asessor`: шкала
`answer_columns_values_set` из меток; Pydantic-модель `AssessmentOutput` с
`Literal` наблюдавшейся шкалы (до 50 меток; для функции GigaChat метки
кодируются строками и после валидации возвращаются исходными); индекс примеров
`QuestionAnswerRetriever` (FAISS + BM25 c RRF-нормализацией, `k=10`, вес BM25
0.8) и отдельный индекс примеров-нарушений (метка на минимуме шкалы), из
которого в few-shot гарантированно добавляются `DEFECT_EXAMPLES_QUOTA=3`
похожих нарушения; при доменном ZIP — `DomainRetriever` (`k=5`, чанки 1000/200).
Судья обходит holdout через `process_with_rate_limit`: не больше
`MAX_INFLIGHT_REQUESTS=4` запросов в полёте; HTTP 429 придерживает весь узел
(backoff 2…60 с, до 6 попыток); прочая ошибка — один повтор через 3 с; строка
без ответа получает `None`. Затем `_review_lowest`: вердикты на нижней
границе шкалы уходят на второй проход `REVIEW_PROMPT`, ответ заменяет первый.
По отвеченным единицам считаются `acc_auto` (точное совпадение),
`baseline_mode_accuracy`, `cohen_kappa`, `krippendorff_alpha` (nominal),
`spearman_correlation`, `defect_recall`/`defect_precision`, объём
`holdout_units`/`holdout_defect_units`; меньше 10 дефектных единиц — warning.
При `num_assessors>1` единица оценивается N раз, голоса ложатся в
`agent_<i>_<source_id>`, мода — в `agent_<source_id>`.

**7.** В `combined` мониторинг использует тот же объект `Asessor`, который
оценивал holdout: неизменны train/RAG и подготовленная инструкция. Holdout
не добавляется в RAG после калибровки. В отдельном `monitoring` судья
строится на всём размеченном эталоне; допуска калибровки у этого запуска нет. Для `accuracy` без колонки `prediction` в трейсах
судья оценивает сам `output_answer` (сообщение в stdout); с колонкой —
подмена как в эталоне. Обход трейсов — тот же, что в п. 6.

**8.** `_score_predictions` прогоняет поля ответа судьи через `score_units`
контракта; единица без ответа получает `NaN` без применения
`missing_policy`. `broadcast_scores` пишет `main_metric` и `assessment_unit_id`
в каждую строку единицы, добавляет `assessor_id="judge"` и колонки
`agent_*`; собирается `assessment_result`.

### Пример лога прогона

Пример прогона на корзине CI09997554, режим
`dialogue` (на стенде судья и эмбеддер подменены локальными, так что цифры
иллюстрируют формат и объёмы, а не качество GigaChat):

```text
INFO agent.asessor_agent: СУММАРИЗАЦИЯ ИНСТРУКЦИИ ОТКЛЮЧЕНА
INFO agent.asessor_agent: Возможные ответы агента асессора:
{'assessment_score': {np.float64(0.0), np.float64(1.0)}}
INFO QuestionAnswerRetriever: Количество примеров: 226
INFO agent.asessor_agent: Примеров с минимальной оценкой в базе: 16
INFO QuestionAnswerRetriever: Количество примеров: 16
INFO agent.asessor_agent: Создана Pydantic модель: AssessmentOutput
INFO agent.asessor_agent: Всего трейсов: 57
INFO agent.asessor_agent: Перепроверка сниженных оценок: 14 из 57
INFO QuestionAnswerRetriever: Количество примеров: 283
INFO agent.asessor_agent: Примеров с минимальной оценкой в базе: 20
INFO agent.asessor_agent: Всего трейсов: 94
INFO agent.asessor_agent: Перепроверка сниженных оценок: 22 из 94
```

Читается сверху вниз: 283 размеченных диалога эталона → 226 в train и 57 в
holdout, 14 сниженных вердиктов перепроверены; затем индекс на всех 283, 94
сессии мониторинга, 22 перепроверки. Итого 187 вызовов судьи за 931 с. Строки
калибровки идут через `print` в stdout и в `node.log` стенда не попали; их
формат из `main.py` со значениями `assessment_result.json` того прогона:

```text
calibration: в holdout 4 единиц с минимальной оценкой — каппа и альфа на таком объёме неустойчивы
calibration: acc_auto=0.789, baseline по моде=0.930, каппа Коэна=0.162, альфа Криппендорфа=0.135, корреляция Спирмана=0.195, полнота на дефектах=0.500, точность на дефектах=0.167
```

Warning-строки `utils.py` при проблемах с судьёй (форматы дословно):

```text
WARNING utils: Квота провайдера исчерпана (попытка %d/%d): придерживаю узел на %.1fs
WARNING utils: Судья не обработал %d из %d строк (допустимо %d); эти единицы исключены из оценки. input <i>: <тип>: <сообщение>
```

## Форматы выхода и контракты

`scored_data` — исходные строки нормализованного UMR (для packed dialogue —
по одной на реплику: `session_id`, `query_id`, `input_query`, `output_answer`,
`reference_group_id`, `turn_index`, `input_query_count`) плюс:

- `main_metric` — оценка единицы по контракту (float, `NaN` — судья не ответил);
- `assessment_unit_id` — `query_id` строки в `qa`/`turn_with_history`,
  идентификатор сессии в `dialogue`; по нему потребители не считают длинный
  диалог несколько раз;
- `assessor_id` — всегда `judge`;
- `agent_<source_id>` — сырой ответ судьи по полю (`agent_assessment_score`
  для `accuracy` и `dialogue`; при `num_assessors>1` ещё `agent_<i>_<source_id>`).

| Режим контракта | Единица наблюдения | Что видит судья |
|---|---|---|
| `qa` | строка | `current_turn` |
| `turn_with_history` | строка | `current_turn` + упорядоченная `history` сессии |
| `dialogue` | сессия (`reference_group_id`) | весь список `turns`; одна оценка повторяется во всех строках сессии |

`assessment_result` (`laim-assessment-result.v1`), значения того же прогона:

```json
{"contract_version": "laim-assessment-result.v1", "status": "computed",
 "assessment_mode": "dialogue", "total_units": 94, "scored_units": 94,
 "refused_units": 0, "refused_share": 0.0,
 "calibration_metrics": {"acc_auto": 0.877, "holdout_units": 57,
   "holdout_defect_units": 4, "invalid_share": 0.0,
   "baseline_mode_accuracy": 0.930, "cohen_kappa": 0.404,
   "krippendorff_alpha": 0.398, "spearman_correlation": 0.446,
   "defect_recall": 0.75, "defect_precision": 0.333,
   "bias_mean": -0.088, "bias_ci_lower": -0.177, "bias_ci_upper": 0.001,
   "bias_units": 57,
   "admission_status": "amber", "admission_reason_code": "few_critical_units",
   "admission_reason": "единиц критичного класса в holdout 4 меньше 10: …"}}
```

- `refused_units` / `refused_share` — единицы мониторинга без ответа судьи;
  при `refused_share > max_invalid_share` статус `not_computable`,
  `reason_code = judge_refusals`, счётчики и `calibration_metrics` сохраняются.
- `invalid_share` — доля holdout без ответа судьи на калибровке.
- `bias_mean`, `bias_ci_lower`, `bias_ci_upper`, `bias_units` — смещение
  судьи относительно разметчиков на шкале КМ (средняя парная разность
  «судья − человек» по holdout с 95 % интервалом); `null`, если пар меньше двух.
  Потребитель — `laim-km-dynamic-test`: поправка КМ мониторинга.
- `admission_status` — допуск судьи по тесту 6.3.3: `green`, `amber`
  (усиленный контроль), `red` (прокси-оценки непригодны), `not_assessed`
  (holdout мал или критичный класс не представлен); `admission_reason_code`
  ∈ `admitted`, `few_critical_units`, `weak_agreement`,
  `no_better_than_baseline`, `judge_refusals`, `holdout_too_small`,
  `critical_class_underrepresented`. Потребители — km и агрегатор.
- `cohen_kappa` / `krippendorff_alpha` — `null`, если согласие невычислимо
  (не подменяются нулём); каппа считается по кодам меток, поэтому дробные
  шкалы допустимы.

`calibration_metrics` присутствует только при `stage` `scoring`/`combined`.
При `not_computable` входного контракта: `status`, `total_units: null`,
`scored_units: 0`, `reason` и `reason_code` адаптера, `assessment_mode` — если есть.

## Падение против деградации

Нода умирает исключением; собственных `reason_code` у падений нет — причина
в тексте исключения. `reason_code` наружу уходит только в `assessment_result`
при `not_computable`, и это код адаптера.

| Причина | Исключение |
|---|---|
| Неизвестная версия контракта, `v1` dialogue, битые `scoring`/`aggregation`/`baseline` | `MonitoringContractError` |
| UMR пуст, не flat и не packed, смешан, пустой `query_id`/`session_id` в контекстном режиме | `MonitoringContractError` |
| `stage` `monitoring`/`combined` без `monitoring_umr`; эталон без источников контракта или без `prediction` при `accuracy`; ни одной размеченной единицы | `MonitoringContractError` |
| Пустая инструкция (`LLM-оценка требует непустую инструкцию в assessor_instruction`) | `MonitoringContractError` |
| Артефакт инструкции не найден; в каталоге не ровно один DOCX/TXT; не UTF-8; не DOCX/TXT | `FileNotFoundError`, `ValueError` |
| Калибровка невозможна: меньше 2 единиц или групп, сплит пуст, судья не ответил ни на одну единицу | `MonitoringContractError` |
| Неизвестный `stage`; `num_assessors < 1`; не-`giga` модель без `AI_GATEWAY_URL` | `ValueError` |
| Эмбеддер вернул векторы разной размерности или не по числу чанков | `MonitoringContractError` |

| Событие | Реакция |
|---|---|
| Контракт `status=not_computable` | `assessment_result.status=not_computable` с `reason`/`reason_code` адаптера; `scored_data` — `monitoring_umr` с `NaN` в `main_metric`; `acc_auto=None` |
| `domain_rag_files_zip` пуст, недоступен, не распаковался или без поддерживаемых файлов | сообщение в stdout, доменный RAG отключён |
| Документы домена дали 0 чанков | `WARNING Domain RAG: 0 документов после разбора, доменный контекст отключён` |
| Отказ судьи на строках мониторинга | `WARNING Судья не обработал…`; `main_metric=NaN`, `refused_units` растёт; доля выше `max_invalid_share` — `assessment_result.status=not_computable`, `reason_code=judge_refusals`, нода не падает |
| Отказ судьи на строках holdout | единицы исключены из калибровки, `invalid_share` растёт; выше `max_invalid_share` — `admission_status=red` |
| Допуск судьи `red` или `not_assessed` | `assessment_result` публикуется как `computed`; решение о непригодности прокси-оценок принимают km (`judge_not_admitted`) и агрегатор |
| HTTP 429 | общий тормоз узла, `WARNING Квота провайдера исчерпана…`, повтор |
| Второй проход не ответил | остаётся вердикт первого прохода |
| Меньше `weak_holdout_defect_units` дефектных единиц в holdout | `admission_status=amber`, `few_critical_units` |
| `accuracy` без `prediction` в трейсах | `monitoring: prediction … недоступен в UMR, судья оценивает output_answer` |
| Неразмеченные единицы в эталоне | молча исключаются из few-shot и шкалы |

## Внешние сервисы

- **Судья `giga`** — `GigaChat` из `langchain-gigachat` с нативным function
  calling (`with_structured_output`). Контур определяется окружением: есть
  `AI_GATEWAY_URL` — контур `sds`, адрес нормализуется до `…/api/v1`; иначе
  контур `sigma` с `CREDENTIALS`, `AUTH_URL`, `BASE_URL`, `SCOPE`,
  `VERIFY_SSL_CERTS`. Модель: `MODEL` → `llm_model` → `GigaChat-3-Ultra`.
  Сэмплирование `TEMPERATURE=0.001`, `TOP_P=0.001`, `TIMEOUT`, `STREAMING`.
- **Судья `minimax-m2.5` / `qwen3-coder-next`** — `SdsChatModel`: POST
  `{AI_GATEWAY_URL}/api/v1/chat/completions`, `Authorization: Bearer
  AI_GATEWAY_API_KEY`, `max_tokens=16384`, `top_k=1`; `temperature`/`top_p`
  отправляются только если заданы в окружении; таймаут соединения 10 с,
  ответа — `TIMEOUT` (по умолчанию 300 с). Structured output — контракт
  плоского JSON в промпте плюс разбор последнего JSON-объекта ответа (блок
  `<think>` отбрасывается, обёртка `answer` принимается) и проверка
  Pydantic-схемой. Автоматического фолбэка между судьями нет.
- **Эмбеддер** — всегда `GigaChatEmbeddings` с конфигом контура (в `sds`
  через тот же шлюз); альтернативы нет.
- Переменные читаются и из файла `.env` рядом с кодом (`python-dotenv`).
- **Ретраи**: квота — до 6 попыток с backoff 2…60 с или `Retry-After` из
  текста ошибки; прочие ошибки судьи — один повтор; LLM-обработка инструкции
  (если включена) — 3 попытки через 5 с.
- **Детерминированность**: сплит фиксирован (`random_state=42384`), индексы
  FAISS точные; вердикты судьи при почти нулевой температуре воспроизводимы не
  строго.
- **Недоступность**: шлюз недоступен — `RuntimeError` по превышению допуска
  отказов; эмбеддер недоступен — исключение при построении индекса. HDFS не
  используется.

## Наблюдаемость

- Лог платформы: логгеры `agent.asessor_agent` (шкала, размер индексов,
  число трейсов, перепроверки), `QuestionAnswerRetriever`/`ContentRetriever`,
  `utils` (квота, отказы судьи), `agent.sds_chat_model` (по каждому вызову
  шлюза: `model`, `http`, `elapsed`, `request_id`, `finish_reason`, токены),
  `retriever.doc_loaders` (файлы и чанки домена).
- stdout через `print`: строки `calibration:`, `monitoring:`,
  `domain_rag_files_zip:`. Прогресс-бары `tqdm` уходят в stderr.
- Отдельного порта журнала нет: машиночитаемый итог — `assessment_result`.
  Триаж на сотне прогонов — по `status`, `scored_units/total_units`,
  `calibration_metrics.acc_auto` против `baseline_mode_accuracy`,
  `defect_recall` и `holdout_defect_units`.
- На уровне DEBUG в лог попадают сериализованные входы судьи и скоры чанков
  ретривера.

## Карта кода

```text
main.py                     порты платформы, контракт судьи, сплит, калибровка, broadcast
utils.py                    чтение инструкции, распаковка ZIP, rate limit и квота, голосование
agent/asessor_agent.py      Asessor: инструкция, индексы примеров и нарушений, цепочки, второй проход
agent/config.py             ModelsConfig: контур sigma/sds из окружения, параметры GigaChat
agent/sds_chat_model.py     SdsChatModel: AI Gateway chat/completions, разбор JSON-ответа
agent/pydantic_output.py    Pydantic-модель ответа со шкалой из данных
agent/score_results.py      разбор ответов судьи, acc_auto, каппа, альфа, Спирман, дефекты
agent/prompts.py            SYSTEM_PROMPT, REVIEW_PROMPT, промпты обработки инструкции
retriever/retriever.py      QuestionAnswerRetriever, DomainRetriever: FAISS + BM25 с RRF
retriever/doc_loaders.py    загрузка документов домена и разбиение на чанки
laim_monitoring/core.py     вендорная копия контракта: валидация, normalize_umr, unitize, score_units
tests/                      13 файлов pytest: контракт диалога и accuracy, сплит, метрики, маршруты, промпт
descriptor.json             порты, настройки, sourceFiles
.github/workflows/ci.yml    ruff check и pytest на Python 3.12
```

## Что делать, если

- **`acc_auto` высокий, а `defect_recall` около нуля** — судья не видит
  класс нарушений (например, недостоверные условия продукта, которые по UMR
  не проверить). Это не дефект расчёта: смотрите инструкцию и доменный RAG.
- **`holdout_defect_units` меньше 10** — каппа и альфа неустойчивы; судите по
  `acc_auto` против `baseline_mode_accuracy` и по `defect_recall`, либо
  уменьшите `scoring_rag_train_size`.
- **Нода упала на `Судья не обработал N из M строк`** — шлюз или квота:
  проверьте `AI_GATEWAY_URL`/`AI_GATEWAY_API_KEY` (или креды `sigma`) и
  перезапустите; единичные отказы нода переживает сама.
- **`Domain RAG: 0 документов`, хотя ZIP не пустой** — проверка архива ищет
  файлы рекурсивно, а загрузчик читает только корень: документы во вложенных
  папках дают ноль чанков молча. Положите файлы в корень архива. Файлы
  `xlsx`/`xls` требуют пакет `unstructured`, которого нет в
  `requirements.txt`: они логируются как ошибка загрузки и пропускаются.
- **`assessment_result.status=not_computable`** — контракт пришёл невычислимым
  от адаптера/селектора; чинить здесь нечего, смотрите `reason_code`.
- **Выбран не-`giga` судья и нода упала на `не задан AI_GATEWAY_URL`** — нода
  в контуре `sigma`; выберите `giga` или задайте адрес шлюза.

## Деплой

Нода самодостаточна: никаких импортов из соседних каталогов и из
`monitoring/shared/`; `laim_monitoring/` — вендорная копия. База —
`py312-simple`, синтаксис и stdlib новее Python 3.12 не используются. Точка
входа — функция `main` в `main.py`. `descriptor.json` перечисляет 14 файлов в
`script.runConfiguration.sourceFiles`: `main.py`, `utils.py`, пакеты `agent/`
(7 файлов с `__init__.py`), `retriever/` (3) и `laim_monitoring/` (2); теста
соответствия списка диску в ноде нет — CI (`.github/workflows/ci.yml`)
запускает `ruff check .` и `python -m pytest -q`. Зависимости —
`requirements.txt`: `langchain`, `langchain-core`, `langchain-community`,
`langchain-gigachat`, `langchain-text-splitters`, `faiss-cpu`, `rank-bm25`,
`pandas`, `numpy`, `pyarrow`, `fastparquet`, `scikit-learn`, `scipy`,
`krippendorff`, `pydantic`, `python-dotenv`, `python-docx`, `docx2txt`,
`pypdf`, `pymupdf`, `openpyxl`, `requests`, `tqdm`. Сборочного скрипта нет:
ZIP модуля — файлы из `sourceFiles` плюс `descriptor.json` и
`requirements.txt` из головы ветки `dev` (нода контура
`laim-asessor-agent-dev-test`).
## Глоссарий

- **Судья** — LLM, размечающая единицы по инструкции; в выходе `assessor_id=judge`.
- **Эталон / reference** — размеченная корзина первичной валидации в формате
  тестового датасета (`laim-umr.v2`, flat или packed dialogue).
- **Единица оценки** — то, что получает один `main_metric`: строка в `qa` и
  `turn_with_history`, сессия в `dialogue`.
- **Калибровка** — прогон судьи на holdout эталона с расчётом `acc_auto` и
  метрик согласия с человеком.
- **Holdout** — часть эталона, не вошедшая в few-shot базу.
- **Дефект** — единица с оценкой на нижней границе шкалы; `defect_recall`
  показывает, какую долю таких единиц судья находит.
- **Few-shot** — похожие размеченные единицы эталона в промпте судьи.
- **Доменный RAG** — чанки документов из `domain_rag_files_zip` в промпте.
- **Structured output** — ответ судьи строго по Pydantic-схеме со шкалой из
  данных эталона.
