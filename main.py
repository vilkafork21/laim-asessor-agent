"""Sber DS entrypoint автоассесора для канонического UMR."""

from __future__ import annotations

import asyncio
import io
import os
from pathlib import Path
from typing import Annotated

import pandas as pd
from langchain_core.embeddings import Embeddings
from langchain_gigachat.chat_models import GigaChat
from langchain_gigachat.embeddings.gigachat import GigaChatEmbeddings

from agent.asessor_agent import Asessor
from agent.config import ModelsConfig
from agent.sds_chat_model import SdsChatModel
from agent.score_results import AnswersProcessor, ResultsScorer
from assessment_plan import (
    JudgePlan,
    apply_judge_labels,
    build_judge_plan,
    input_observed,
    judge_instruction,
    score_judge_predictions,
)
import laim_monitoring
from laim_monitoring import (
    MonitoringContractError,
    agent_inputs,
    broadcast_scores,
    normalize_umr,
    unitize,
    validate_monitoring_metric,
)
from utils import add_voting_columns, extract_zip, read_docx, remove_directory


def _load_df(value) -> pd.DataFrame | None:
    if value is None or isinstance(value, pd.DataFrame):
        return value
    if isinstance(value, (bytes, bytearray)):
        try:
            return pd.read_parquet(io.BytesIO(value))
        except Exception as exc:
            raise ValueError("DataFrame bytes должны содержать parquet") from exc
    path = Path(value)
    if path.is_dir():
        candidates = sorted(
            item for item in path.iterdir()
            if item.is_file() and item.suffix.lower() in {".parquet", ".xlsx", ".csv"}
        )
        if len(candidates) != 1:
            raise ValueError(f"Ожидался один DataFrame artifact, найдено {candidates}")
        path = candidates[0]
    readers = {
        ".parquet": pd.read_parquet,
        ".xlsx": pd.read_excel,
        ".csv": pd.read_csv,
    }
    if path.suffix.lower() not in readers:
        raise ValueError(f"Неподдерживаемый DataFrame artifact: {path}")
    return readers[path.suffix.lower()](path)


def _load_instruction(value) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        text = value.get("text")
        if not isinstance(text, str):
            raise ValueError("instruction dict должен содержать строковый text")
        if not text.strip():
            raise ValueError("instruction dict содержит пустой text")
        return text
    return read_docx(str(value))


def _domain_path(value) -> str | None:
    """Порт domain_rag_files_zip опционален: пустой/битый/бесполезный вход
    отключает доменный RAG, а не роняет оценку (FAISS падает на 0 документов)."""
    if value is None or not str(value).strip():
        return None
    source = Path(str(value))
    if not source.exists() or (source.is_file() and source.stat().st_size == 0):
        print(f"domain_rag_files_zip: путь недоступен, доменный RAG отключён: {source}")
        return None
    destination = "/tmp/laim-domain-rag"
    remove_directory(directory_path=destination)
    try:
        extracted = extract_zip(zip_path=str(source), extract_to=destination)
    except Exception as exc:
        print(f"domain_rag_files_zip: не распакован ({exc}), доменный RAG отключён")
        return None
    supported = {".pdf", ".docx", ".doc", ".txt", ".md", ".csv", ".xlsx", ".xls"}
    has_documents = any(
        item.suffix.lower() in supported
        for item in Path(extracted).rglob("*")
        if item.is_file()
    )
    if not has_documents:
        print(
            "domain_rag_files_zip: в архиве нет поддерживаемых документов "
            "(pdf/docx/txt/md/csv/xlsx), доменный RAG отключён"
        )
        return None
    return extracted


def _assessment_frame(frame: pd.DataFrame, contract: dict) -> pd.DataFrame:
    """Судья видит как ответ агента его наблюдаемый вход (класс/маршрут), если он один."""
    agent = agent_inputs(contract)
    if len(agent) != 1:
        return frame
    column = agent[0]["column"]
    if not input_observed(frame, agent[0]):
        raise MonitoringContractError(f"UMR не содержит наблюдаемый ответ агента: {column}")
    result = frame.copy()
    result["output_answer"] = result[column]
    return result


def _assessor_units(frame: pd.DataFrame, contract: dict, *, require_inputs: bool) -> pd.DataFrame:
    units = unitize(frame, contract)
    if require_inputs:
        missing = [item["name"] for item in contract["inputs"] if item["name"] not in units]
        if missing:
            raise MonitoringContractError(f"Эталонная корзина не содержит входы формулы: {missing}")
    return units


# Эмбеддер GigaChat принимает не более 514 токенов на текст, а единица оценки
# несёт полный ответ агента. Без обрезки один длинный текст роняет весь батч, а
# вместе с ним — узел. Границы те же, что у дрифт-нод в giga_wraper.
EMBEDDING_MAX_CHARS = 1000
EMBEDDING_BATCH_SIZE = 100


class BoundedGigaChatEmbeddings(Embeddings):
    """Чанкует и батчит вход, возвращая один усреднённый вектор на текст."""

    def __init__(
        self,
        embeddings: Embeddings,
        max_chars: int = EMBEDDING_MAX_CHARS,
        batch_size: int = EMBEDDING_BATCH_SIZE,
    ) -> None:
        self.embeddings = embeddings
        self.max_chars = max_chars
        self.batch_size = batch_size

    def _chunks(self, text: str) -> list[str]:
        return [
            text[start : start + self.max_chars]
            for start in range(0, len(text), self.max_chars)
        ] or [""]

    @staticmethod
    def _mean(vectors: list[list[float]]) -> list[float]:
        dimensions = {len(vector) for vector in vectors}
        if len(dimensions) != 1:
            raise MonitoringContractError(
                "Эмбеддер вернул векторы разной размерности"
            )
        return [
            sum(values) / len(vectors)
            for values in zip(*vectors, strict=True)
        ]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        chunks_by_text = [self._chunks(str(text)) for text in texts]
        chunks = [chunk for text_chunks in chunks_by_text for chunk in text_chunks]
        chunk_vectors: list[list[float]] = []
        for start in range(0, len(chunks), self.batch_size):
            chunk_vectors.extend(
                self.embeddings.embed_documents(chunks[start : start + self.batch_size])
            )
        if len(chunk_vectors) != len(chunks):
            raise MonitoringContractError(
                f"Эмбеддер вернул {len(chunk_vectors)} векторов на {len(chunks)} чанков"
            )
        vectors = []
        offset = 0
        for text_chunks in chunks_by_text:
            end = offset + len(text_chunks)
            vectors.append(self._mean(chunk_vectors[offset:end]))
            offset = end
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self._mean([
            self.embeddings.embed_query(chunk)
            for chunk in self._chunks(str(text))
        ])


def _labelled_reference_units(units: pd.DataFrame, source_ids: list[str]) -> pd.DataFrame:
    """Оставляет только размеченные единицы reference-корзины.

    Неразмеченная единица не является few-shot примером и не задаёт шкалу: её
    пустая оценка иначе попадает и в примеры судьи, и в набор допустимых
    значений критерия.
    """
    labelled = units.dropna(subset=source_ids, how="any").reset_index(drop=True)
    if labelled.empty:
        raise MonitoringContractError(
            "В reference-корзине нет ни одной размеченной единицы: "
            "судье нечем задать шкалу оценок"
        )
    return labelled


def _predict(asessor: Asessor, frame: pd.DataFrame, source_ids: list[str], count: int) -> pd.DataFrame:
    if count < 1:
        raise ValueError("num_assessors должен быть положительным")
    processor = AnswersProcessor()
    frames = []
    for index in range(count):
        values = asyncio.run(asessor.run(frame))
        parsed = processor.parse(
            [value.model_dump() if hasattr(value, "model_dump") else value for value in values],
            source_ids,
        )
        if count > 1:
            parsed.columns = [
                column.replace("agent_", f"agent_{index}_", 1)
                for column in parsed.columns
            ]
        frames.append(parsed)
    combined = pd.concat(frames, axis=1)
    if count > 1:
        combined = add_voting_columns(combined, source_ids, mode="scoring")
    missing = [f"agent_{source_id}" for source_id in source_ids if f"agent_{source_id}" not in combined]
    if missing:
        raise MonitoringContractError(f"Assessor не вернул обязательные поля: {missing}")
    return combined


def _broadcast_predictions(
    original: pd.DataFrame,
    units: pd.DataFrame,
    predictions: pd.DataFrame,
    scores: pd.Series,
    plan: JudgePlan,
) -> pd.DataFrame:
    result = broadcast_scores(original, units, scores)
    result = apply_judge_labels(result, units, predictions, plan)
    result["assessor_id"] = "judge"
    for column in predictions.columns:
        result[column] = None
        target = result.columns.get_loc(column)
        for positions, value in zip(units["_row_positions"], predictions[column].tolist()):
            for position in positions:
                result.iat[position, target] = value
    return result


def _build_assessor(
    models,
    rag_units,
    source_ids,
    instruction,
    domain_path,
    instruction_llm_preprocessing,
):
    return Asessor(
        llm=models[0],
        embedding_model=models[1],
        dataset=rag_units,
        context_columns=["assessment_context"],
        answer_columns=source_ids,
        instruction=instruction,
        domain_rag_path=domain_path,
        instruction_structuring=instruction_llm_preprocessing,
        instruction_summarization=instruction_llm_preprocessing,
    )


def _calibrate(
    rag_units: pd.DataFrame,
    source_ids: list[str],
    instruction: str,
    domain_path: str | None,
    models,
    train_fraction: float,
    num_assessors: int,
    instruction_llm_preprocessing: bool,
) -> tuple[dict[str, float], pd.DataFrame, pd.DataFrame]:
    if len(rag_units) < 2:
        raise MonitoringContractError("Для calibration требуется минимум две единицы")
    groups = rag_units["_group_id"] if "_group_id" in rag_units else None
    if groups is not None and groups.notna().any():
        group_ids = list(dict.fromkeys(groups.dropna().tolist()))
        if len(group_ids) < 2:
            raise MonitoringContractError("Для calibration требуется минимум две группы")
        shuffled_groups = pd.Series(group_ids).sample(frac=1, random_state=42384).tolist()
        split = min(max(int(len(shuffled_groups) * train_fraction), 1), len(shuffled_groups) - 1)
        train_groups = set(shuffled_groups[:split])
        train = rag_units[rag_units["_group_id"].isin(train_groups)].reset_index(drop=True)
        test = rag_units[~rag_units["_group_id"].isin(train_groups)].reset_index(drop=True)
    else:
        shuffled = rag_units.sample(frac=1, random_state=42384).reset_index(drop=True)
        split = min(max(int(len(shuffled) * train_fraction), 1), len(shuffled) - 1)
        train, test = shuffled.iloc[:split], shuffled.iloc[split:]
    predictions = _predict(
        _build_assessor(
            models,
            train,
            source_ids,
            instruction,
            domain_path,
            instruction_llm_preprocessing,
        ),
        test,
        source_ids,
        num_assessors,
    )
    # Отказ судьи (квота/сеть) — не неверный ответ: такие строки исключаются
    # из калибровки, иначе каждый допущенный отказ занижает acc_auto как ноль.
    answered = predictions[
        [f"agent_{source_id}" for source_id in source_ids]
    ].notna().any(axis=1)
    if not answered.any():
        raise MonitoringContractError(
            "Calibration невозможна: судья не ответил ни на одну тестовую единицу"
        )
    if not answered.all():
        print(
            f"calibration: судья не ответил на {int((~answered).sum())} из "
            f"{len(answered)} единиц; acc_auto считается по отвеченным"
        )
    labels = test[source_ids].reset_index(drop=True)
    comparison = pd.concat(
        [
            predictions[answered].reset_index(drop=True),
            labels[answered].reset_index(drop=True),
        ],
        axis=1,
    )
    score = ResultsScorer(AnswersProcessor()).score(comparison, source_ids)
    metrics = {
        "acc_auto": float(score["mean_accuracy"]["Mean accuracy"]),
        "baseline_mode_accuracy": float(score["mean_accuracy"]["Mean mode"]),
        "cohen_kappa": score["cohen_kappa"],
        "krippendorff_alpha": score["krippendorff_alpha"],
        "spearman_correlation": float(score["mean_correlation"]),
    }
    print(
        f"calibration: acc_auto={metrics['acc_auto']:.3f}, "
        f"baseline по моде={metrics['baseline_mode_accuracy']:.3f}, "
        f"каппа Коэна={metrics['cohen_kappa']:.3f}, "
        f"альфа Криппендорфа={metrics['krippendorff_alpha']:.3f}, "
        f"корреляция Спирмана={metrics['spearman_correlation']:.3f}"
    )
    return metrics, test, predictions


def _assessment_result(
    contract: dict,
    units: pd.DataFrame,
    *,
    plan: JudgePlan,
    scores: pd.Series | None = None,
    calibration_metrics: dict[str, float] | None = None,
) -> dict[str, object]:
    result: dict[str, object] = {
        "contract_version": "laim-assessment-result.v1",
        "status": "computed",
        "assessment_mode": contract["assessment_mode"],
        "formula": plan.contract["formula"],
        "laim_monitoring_version": laim_monitoring.__version__,
        # contract_formula: судья размечает входы формулы, КМ считает формула
        # контракта (та же, что у адаптера и km-dynamic). judge_final_score:
        # судья ставит готовый score, формула отчёта не воспроизводится.
        "scoring_semantics": plan.semantics,
        "scoring_semantics_reason": plan.reason,
        "judge_fields": list(plan.judge_fields),
        "total_units": len(units),
        "scored_units": len(units) if scores is None else int(scores.notna().sum()),
    }
    if calibration_metrics is not None:
        result["calibration_metrics"] = calibration_metrics
    return result


# Доля единиц без ответа судьи, выше которой разметка мониторинга невалидна:
# КМ по «выжившим» строкам смещена. Единичные отказы остаются NaN и исключаются.
MAX_JUDGE_FAILURE_SHARE = 0.2


def _judge_failure_share(predictions: pd.DataFrame, source_ids: list[str]) -> float:
    columns = [f"agent_{source_id}" for source_id in source_ids]
    return float(predictions[columns].isna().all(axis=1).mean()) if len(predictions) else 0.0


def _unavailable_result(contract: dict) -> dict[str, object]:
    result: dict[str, object] = {
        "contract_version": "laim-assessment-result.v1",
        "status": "not_computable",
        "total_units": None,
        "scored_units": 0,
        "reason": contract.get("reason", "monitoring_metric невычислим"),
    }
    if contract.get("assessment_mode") is not None:
        result["assessment_mode"] = contract["assessment_mode"]
    if contract.get("reason_code") is not None:
        result["reason_code"] = contract["reason_code"]
    return result


def _unavailable_scored_data(frame: pd.DataFrame | None) -> pd.DataFrame | None:
    if frame is None:
        return None
    result = frame.copy()
    result["main_metric"] = pd.Series(float("nan"), index=result.index, dtype="float64")
    result["assessment_unit_id"] = None
    return result


def _build_judge_model(model_id: str, config: ModelsConfig, llm_model: str):
    """Создаёт одного явно выбранного судью на весь запуск."""
    route = str(model_id or "").strip() or "giga"
    if route == "giga":
        configured = getattr(config, "llm_params", {})
        resolved = configured.get("model") or str(llm_model).strip()
        return GigaChat(**config.contour_llm_configs), resolved
    if config.contour != "sds":
        raise ValueError(
            f"Модель {route!r} доступна только через AI Gateway в контуре SDS; "
            "задайте AI_GATEWAY_URL либо выберите GigaChat."
        )
    return (
        SdsChatModel(
            base_url=config.contour_configs.get("base_url"),
            model_id=route,
            temperature=(
                config.llm_params["temperature"]
                if os.environ.get("TEMPERATURE") is not None
                else None
            ),
            top_p=(
                config.llm_params["top_p"]
                if os.environ.get("TOP_P") is not None
                else None
            ),
            timeout=config.llm_params["timeout"],
            verify_ssl_certs=config.verify_ssl_certs,
            # Reasoning-модели гейтвея сжигают бюджет на рассуждения и ловят
            # finish_reason='length' с пустым content; 16384 убирает этот класс.
            max_tokens=16384,
        ),
        route,
    )


def main(
    reference_umr: pd.DataFrame,
    monitoring_metric: dict,
    assessor_instruction: Path | None = None,
    monitoring_umr: pd.DataFrame | None = None,
    scoring_rag_train_size: float = 0.8,
    domain_rag_files_zip: Path | None = None,
    stage: Annotated[str, "scoring", "monitoring", "combined"] = "combined",
    num_assessors: int = 1,
    model_id: str = "giga",
    llm_model: str = "GigaChat-3-Ultra",
    instruction_llm_preprocessing: bool = False,
) -> dict[str, object]:
    contract = validate_monitoring_metric(monitoring_metric, require_computed=False)
    if stage not in {"scoring", "monitoring", "combined"}:
        raise ValueError(f"Неизвестный stage: {stage}")
    monitoring_umr = _load_df(monitoring_umr)
    if stage in {"monitoring", "combined"} and monitoring_umr is None:
        raise MonitoringContractError(f"stage={stage} требует monitoring_umr")
    if contract["status"] != "computed":
        # Plan-less отказ адаптера не несёт assessment_mode — нормализация
        # невозможна и не нужна: наружу уходит машинный not_computable.
        return {
            "scored_data": _unavailable_scored_data(monitoring_umr),
            "acc_auto": None,
            "assessment_result": _unavailable_result(contract),
        }
    if stage in {"monitoring", "combined"}:
        monitoring_umr = normalize_umr(monitoring_umr, contract)

    # Что размечает судья и какой формулой считается score — одно решение на
    # весь запуск: калибровка и мониторинг обязаны измерять одно и то же.
    # Для accuracy prediction (класс агента) в эталоне есть всегда; на
    # мониторинге это свойство конвертера трейсов.
    agent_observed = True
    if stage in {"monitoring", "combined"}:
        agent_observed = all(input_observed(monitoring_umr, item) for item in agent_inputs(contract))
    plan = build_judge_plan(contract, agent_observed=agent_observed)
    assessment_contract = plan.contract
    source_ids = list(plan.judge_fields)
    print(f"assessment: scoring_semantics={plan.semantics} ({plan.reason})")
    # Packed dialogue разворачивается до построения units: broadcast_scores
    # пишет по позициям, поэтому reference_umr обязан совпадать с ними построчно.
    reference_umr = normalize_umr(_load_df(reference_umr), contract)
    rag_assessment = _assessment_frame(reference_umr, contract)
    rag_units = _assessor_units(
        rag_assessment,
        assessment_contract,
        require_inputs=True,
    )
    instruction = _load_instruction(assessor_instruction)
    if not instruction.strip():
        raise MonitoringContractError(
            "LLM-оценка требует непустую инструкцию в assessor_instruction"
        )
    instruction += "\n\n" + judge_instruction(plan)
    domain_path = _domain_path(domain_rag_files_zip)

    judge_units = _labelled_reference_units(rag_units, source_ids)
    config = ModelsConfig(model=llm_model)
    judge_model, _ = _build_judge_model(model_id, config, llm_model)
    models = (
        judge_model,
        BoundedGigaChatEmbeddings(GigaChatEmbeddings(**config.contour_configs)),
    )

    acc_auto = None
    calibration_metrics = None
    scored_output = None
    assessment_result = None
    if stage in {"scoring", "combined"}:
        calibration_metrics, test_units, predictions = _calibrate(
            judge_units,
            source_ids,
            instruction,
            domain_path,
            models,
            scoring_rag_train_size,
            num_assessors,
            instruction_llm_preprocessing,
        )
        acc_auto = calibration_metrics["acc_auto"]
        scores = score_judge_predictions(test_units, predictions, plan)
        scored_output = _broadcast_predictions(
            reference_umr, test_units, predictions, scores, plan,
        )
        assessment_result = _assessment_result(
            contract,
            test_units,
            plan=plan,
            scores=scores,
            calibration_metrics=calibration_metrics,
        )

    if stage in {"monitoring", "combined"} and monitoring_umr is not None:
        # Без наблюдаемого prediction (трейс не несёт класс агента) судья
        # оценивает сам output_answer — это уже учтено в plan (judge_final_score).
        monitoring_assessment = (
            _assessment_frame(monitoring_umr, contract) if agent_observed else monitoring_umr
        )
        monitoring_units = _assessor_units(
            monitoring_assessment,
            assessment_contract,
            require_inputs=False,
        )
        predictions = _predict(
            _build_assessor(
                models,
                judge_units,
                source_ids,
                instruction,
                domain_path,
                instruction_llm_preprocessing,
            ),
            monitoring_units,
            source_ids,
            num_assessors,
        )
        scores = score_judge_predictions(monitoring_units, predictions, plan)
        scored_output = _broadcast_predictions(
            monitoring_umr, monitoring_units, predictions, scores, plan,
        )
        assessment_result = _assessment_result(
            contract,
            monitoring_units,
            plan=plan,
            scores=scores,
            calibration_metrics=calibration_metrics,
        )
        failure_share = _judge_failure_share(predictions, source_ids)
        if failure_share > MAX_JUDGE_FAILURE_SHARE:
            assessment_result["status"] = "not_computable"
            assessment_result["reason"] = (
                f"Судья не ответил на {failure_share:.0%} единиц мониторинга "
                f"(порог {MAX_JUDGE_FAILURE_SHARE:.0%}): КМ по оставшимся смещена"
            )
    return {
        "scored_data": scored_output,
        "acc_auto": acc_auto,
        "assessment_result": assessment_result,
    }
