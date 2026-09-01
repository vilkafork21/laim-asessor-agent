"""
Утилиты для работы Автоасессора.

Основные функции:
- remove_directory: удаление директории
- read_docx: чтение инструкции из DOCX или UTF-8 TXT
- extract_zip: распаковка ZIP архивов
- process_with_rate_limit: асинхронная обработка с rate limiting
- add_voting_columns: голосование между несколькими асессорами
"""

import asyncio
import logging
import os
import re
import shutil
import time
import zipfile
from pathlib import Path
from collections import Counter
from typing import Any, List, Optional

import pandas as pd
from tqdm.asyncio import tqdm

utils_logger = logging.getLogger(__name__)

# =====================================================
# CONSTANTS
# =====================================================

# Words to filter from text
BLACKLIST_WORDS: list[str] = ["президент", "сбер"]

# Default delay between API requests (seconds)
DEFAULT_DELAY_SECONDS: float = 5.0

# Потолок одновременных запросов судьи на узел. Квота у провайдера общая на всю
# ноду, поэтому держать в полёте столько запросов, сколько строк, — это
# гарантированный HTTP 429 независимо от того, как разнесены их старты.
MAX_INFLIGHT_REQUESTS: int = 4

# Отказ по квоте не является ошибкой строки: её надо дожать, а не потерять.
MAX_QUOTA_ATTEMPTS: int = 6
QUOTA_BASE_BACKOFF_SECONDS: float = 2.0
QUOTA_MAX_BACKOFF_SECONDS: float = 60.0

# Судья вправе потерять малую долю строк: такие единицы исключаются из оценки,
# а не роняют узел. Смерть узла оставлена массовому отказу — метрика по
# оставшимся строкам была бы уже нерепрезентативной. Нижняя граница в одну
# строку не даёт единственному сбою убить маленький батч.
JUDGE_FAILURE_TOLERANCE: float = 0.1


# =====================================================
# FILE SYSTEM OPERATIONS
# =====================================================


def remove_directory(directory_path: str) -> bool:
    """
    Удаляет директорию и все её содержимое, если она существует.

    Returns:
        True if directory was removed, False otherwise.
    """
    if os.path.exists(directory_path):
        try:
            shutil.rmtree(directory_path)
            utils_logger.debug(f"Директория '{directory_path}' успешно удалена")
            return True
        except Exception as e:
            utils_logger.error(
                f"Ошибка при удалении директории '{directory_path}': {e}"
            )
            return False
    else:
        utils_logger.error(f"Директория '{directory_path}' не существует")
        return False


def read_docx(artifact_path: str) -> str:
    """Читает единственную инструкцию DOCX или UTF-8 TXT из artifact."""
    artifact = Path(artifact_path)
    if not artifact.exists():
        raise FileNotFoundError(f"Artifact инструкции не найден: '{artifact}'")
    if artifact.is_dir():
        files = sorted(
            item
            for item in artifact.iterdir()
            if item.is_file()
            and not item.name.startswith((".", "~$"))
            and item.suffix.casefold() in {".docx", ".txt"}
        )
        if not files:
            raise FileNotFoundError(
                f"В директории '{artifact}' нет инструкции DOCX или UTF-8 TXT"
            )
        if len(files) != 1:
            raise ValueError(
                f"В директории '{artifact}' найдено {len(files)} инструкций DOCX/TXT; "
                "ожидается ровно одна"
            )
        source_file = files[0]
    elif artifact.is_file():
        source_file = artifact
    else:
        raise ValueError(
            f"Artifact инструкции не является файлом или директорией: '{artifact}'"
        )

    suffix = source_file.suffix.casefold()
    try:
        with source_file.open("rb") as stream:
            signature = stream.read(4)
    except OSError as exc:
        raise ValueError(
            f"Не удалось прочитать инструкцию '{source_file}': {exc}"
        ) from exc

    extensionless_docx = not suffix and signature == b"PK\x03\x04"
    if suffix == ".txt" or (not suffix and not extensionless_docx):
        try:
            text = source_file.read_text(encoding="utf-8-sig")
        except UnicodeDecodeError as exc:
            raise ValueError(
                f"TXT-инструкция должна быть в кодировке UTF-8: '{source_file}': {exc}"
            ) from exc
        except OSError as exc:
            raise ValueError(
                f"Не удалось прочитать TXT-инструкцию '{source_file}': {exc}"
            ) from exc
        if not text.strip():
            raise ValueError(
                f"TXT-инструкция '{source_file}' не содержит непустого текста"
            )
        return text
    if suffix != ".docx" and not extensionless_docx:
        raise ValueError(
            f"Неподдерживаемый формат инструкции '{source_file}': "
            "ожидается DOCX или UTF-8 TXT"
        )

    from docx import Document
    from docx.oxml.ns import qn
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    try:
        doc = Document(str(source_file))
    except Exception as e:
        raise ValueError(f"Не удалось прочитать DOCX '{source_file}': {e}") from e

    paragraph_tag = qn("w:p")
    table_tag = qn("w:tbl")
    sdt_tag = qn("w:sdt")
    sdt_content_tag = qn("w:sdtContent")

    def iter_blocks(parent):
        for child in parent.iterchildren():
            if child.tag in {paragraph_tag, table_tag}:
                yield child
            elif child.tag == sdt_tag:
                for sdt_child in child.iterchildren():
                    if sdt_child.tag == sdt_content_tag:
                        yield from iter_blocks(sdt_child)

    parts: list[str] = []
    for element in iter_blocks(doc.element.body):
        if element.tag == paragraph_tag:
            parts.append(Paragraph(element, doc).text)
        elif element.tag == table_tag:
            table = Table(element, doc)
            parts.extend("\t".join(cell.text for cell in row.cells) for row in table.rows)

    text = "\n".join(parts)
    if not text.strip():
        raise ValueError(f"DOCX '{source_file}' не содержит непустого текста")
    return text


def extract_zip(zip_path: str, extract_to: str | None = None) -> str:
    """Распаковывает ZIP архив в указанную директорию."""
    if not os.path.exists(zip_path):
        utils_logger.error(f"Архив не найден: {zip_path}")
    if extract_to is None:
        archive_name = os.path.splitext(os.path.basename(zip_path))[0]
        extract_to = os.path.join(os.path.dirname(zip_path), archive_name)

    os.makedirs(extract_to, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)
        extracted_files = zip_ref.namelist()

    utils_logger.info(f"Архив успешно распакован в: {extract_to}")
    utils_logger.info(f"Распаковано файлов: {len(extracted_files)}")

    return extract_to


# =====================================================
# ASYNC PROCESSING
# =====================================================


def _is_quota_error(error: BaseException) -> bool:
    """Отказ по общей квоте провайдера, а не дефект конкретной строки."""
    text = str(error).lower()
    return "429" in text or "too many requests" in text or "rate limit" in text


def _retry_after_seconds(error: BaseException) -> float | None:
    """Пауза, которую провайдер назвал сам; она точнее любого нашего backoff."""
    found = re.search(r"retry[-_\s]?after\D{0,4}(\d+(?:\.\d+)?)", str(error), re.IGNORECASE)
    return float(found.group(1)) if found else None


class _QuotaGate:
    """Общий на узел тормоз: отказ по квоте придерживает все строки, а не одну.

    Одиночный backoff внутри строки бесполезен, пока соседние строки продолжают
    выбирать ту же квоту, — повтор приходится ровно на тот же отказ.
    """

    def __init__(self) -> None:
        self._until = 0.0

    async def wait(self) -> None:
        while True:
            remaining = self._until - time.monotonic()
            if remaining <= 0:
                return
            await asyncio.sleep(remaining)

    def penalize(self, seconds: float) -> None:
        self._until = max(self._until, time.monotonic() + seconds)


async def process_with_rate_limit(
    printing_chain,
    main_chain,
    batch_inputs: List[dict],
    delay_seconds: float = DEFAULT_DELAY_SECONDS,
) -> List[Any]:
    """
    Асинхронный вызов Гиги с прогресс баром.

    Темп задают потолок одновременных запросов и общий на узел тормоз по квоте;
    `delay_seconds` осталась паузой перед повтором обычной транзиентной ошибки.
    """
    results = [None] * len(batch_inputs)
    errors: list[tuple[int, str]] = []
    print("")
    # Создаем три прогресс-бара
    with (
        tqdm(
            total=len(batch_inputs),
            desc="📤 Started",
            position=0,
            bar_format="{desc}: {n_fmt}/{total_fmt}",
        ) as started_bar,
        tqdm(
            total=len(batch_inputs),
            desc="✅ Finished",
            position=1,
            bar_format="{desc}: {n_fmt}/{total_fmt}",
        ) as finished_bar,
        tqdm(
            total=len(batch_inputs),
            desc="❌ Errors",
            position=2,
            bar_format="{desc}: {n_fmt}/{total_fmt}",
            colour="red",
        ) as error_bar,
    ):
        # Для отслеживания запущенных и завершенных задач
        started_count = 0
        finished_count = 0
        error_count = 0

        # Lock для безопасного обновления счетчиков
        lock = asyncio.Lock()
        gate = _QuotaGate()
        semaphore = asyncio.Semaphore(MAX_INFLIGHT_REQUESTS)

        async def worker(i, inp):
            nonlocal started_count, finished_count, error_count

            last_error = None
            started = False
            quota_attempts = 0
            other_attempts = 0
            while True:
                await gate.wait()
                async with semaphore:
                    try:
                        if not started:
                            async with lock:
                                started_count += 1
                                started_bar.n = started_count
                                started_bar.refresh()
                            started = True
                            await printing_chain.ainvoke(inp)
                        results[i] = await main_chain.ainvoke(inp)
                    except Exception as error:
                        last_error = error
                    else:
                        # Обновляем счетчик завершенных задач
                        async with lock:
                            finished_count += 1
                            finished_bar.n = finished_count
                            finished_bar.refresh()
                        return

                if _is_quota_error(last_error):
                    quota_attempts += 1
                    if quota_attempts >= MAX_QUOTA_ATTEMPTS:
                        break
                    pause = _retry_after_seconds(last_error) or min(
                        QUOTA_BASE_BACKOFF_SECONDS * 2 ** (quota_attempts - 1),
                        QUOTA_MAX_BACKOFF_SECONDS,
                    )
                    utils_logger.warning(
                        "Квота провайдера исчерпана (попытка %d/%d): придерживаю узел на %.1fs",
                        quota_attempts, MAX_QUOTA_ATTEMPTS, pause,
                    )
                    gate.penalize(pause)
                    continue

                other_attempts += 1
                if other_attempts >= 2:
                    break
                await asyncio.sleep(delay_seconds)

            response = f"   ❌ Error on input {i}: {last_error}"
            utils_logger.info(f"	📌AGENT OUTPUT {i} \n{response}")
            results[i] = None

            # Обновляем счетчик ошибок
            async with lock:
                errors.append((i, f"{type(last_error).__name__}: {last_error}"))
                error_count += 1
                error_bar.n = error_count
                error_bar.refresh()

        # Создаем и запускаем все задачи
        tasks = [worker(i, inp) for i, inp in enumerate(batch_inputs)]
        await asyncio.gather(*tasks)

        # Все запросы стартовали; finished отражает только успешные ответы.
        started_bar.n = len(batch_inputs)
        started_bar.refresh()
        finished_bar.refresh()

    if errors:
        examples = "; ".join(
            f"input {index}: {message[:300]}" for index, message in errors[:3]
        )
        tolerated = max(1, int(len(batch_inputs) * JUDGE_FAILURE_TOLERANCE))
        if len(errors) > tolerated:
            raise RuntimeError(
                f"Судья не обработал {len(errors)} из {len(batch_inputs)} строк "
                f"(допустимо {tolerated}). {examples}"
            )
        utils_logger.warning(
            "Судья не обработал %d из %d строк (допустимо %d); "
            "эти единицы исключены из оценки. %s",
            len(errors), len(batch_inputs), tolerated, examples,
        )
    return results


# =====================================================
# TEXT PROCESSING
# =====================================================


def blacklist_remover(text: str) -> str:
    """
    Простая версия удаления стоп-слов.
    Внимание: может удалять части слов (например, "президентский" -> "ский").
    """
    result = text
    for word in BLACKLIST_WORDS:
        # Заменяем все вхождения слова, независимо от регистра
        result = result.replace(word, "")
        result = result.replace(word.capitalize(), "")
        result = result.replace(word.upper(), "")

    # Удаляем множественные пробелы
    while "  " in result:
        result = result.replace("  ", " ")

    return result.strip()


# =====================================================
# DATA TYPE CONVERSION
# =====================================================


# =====================================================
# VOTING AND CONSENSUS
# =====================================================



def _to_hashable(item: Any) -> Any:
    """Convert item to hashable type for use in Counter."""
    try:
        hash(item)
        return item
    except TypeError:
        if isinstance(item, list):
            return tuple(item)
        elif isinstance(item, dict):
            return tuple(sorted(item.items()))
        return item


def get_mode_safe(values: List[Any]) -> Optional[Any]:
    """
    Находит моду (самый частый элемент), поддерживая нехешируемые типы (list, dict).
    """
    if not values:
        return None

    counter = Counter(_to_hashable(item) for item in values)

    if not counter:
        return None
    most_common_key, _ = counter.most_common(1)[0]
    for item in values:
        if _to_hashable(item) == most_common_key:
            return item

    return values[0]


def _get_assessor_columns(combined_results: pd.DataFrame, col_pattern: str) -> list[str]:
    """Колонки ответов асессоров для критерия: agent_<i>_<критерий> или agent_<критерий>.

    Точный матч имени критерия обязателен: подстрочный поиск захватывал бы
    source_10 по образцу source_1 и ломал голосование.
    """
    pattern = re.compile(rf"^agent_(\d+_)?{re.escape(col_pattern)}$")
    return [c for c in combined_results.columns if pattern.match(c)]


def add_voting_columns(
    combined_results: pd.DataFrame,
    answer_columns: list[str],
    mode: str = "combined",
) -> pd.DataFrame:
    """Добавляет колонки с результатами голосования большинства."""
    for col_pattern in answer_columns:
        agent_cols = _get_assessor_columns(combined_results, col_pattern)

        if not agent_cols:
            continue

        ans_col = ("agent_" if mode == "scoring" else "") + col_pattern

        combined_results[ans_col] = combined_results[agent_cols].apply(
            lambda row: get_mode_safe(row.dropna().tolist()), axis=1
        )

    return combined_results
