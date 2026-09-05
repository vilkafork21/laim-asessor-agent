"""Sber DS entrypoint автоассесора для канонического UMR."""

from __future__ import annotations

import asyncio
import io
import json
import logging
import os
from pathlib import Path
from typing import Annotated
from uuid import uuid4

import pandas as pd
from langchain_core.embeddings import Embeddings
from langchain_gigachat.chat_models import GigaChat
from langchain_gigachat.embeddings.gigachat import GigaChatEmbeddings

from agent.asessor_agent import Asessor
from agent.config import ModelsConfig
from agent.sds_chat_model import SdsChatModel
from agent.score_results import score_results
from laim_monitoring import (
    MonitoringContractError,
    broadcast_scores,
    normalize_umr,
    unitize,
    validate_monitoring_metric,
)
from utils import add_voting_columns, extract_zip, remove_directory
from admission import admit, judge_bias

logger = logging.getLogger(__name__)


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


def _source_instruction(contract: dict) -> str:
    return (contract["evaluation"]["rubric"] + "\nПоле JSON assessment_score: итоговая оценка "
            f"по шкале {contract['evaluation']['score_values']}. Не имитируй голоса разметчиков.")


def _assessor_units(frame: pd.DataFrame, contract: dict, *, require_sources: bool) -> pd.DataFrame:
    units = unitize(frame, contract, include_sources=False)
    required = contract["evaluation"]["required_evidence"]
    prediction = next((source["column_name"] for source in contract["scoring"]["sources"]
                       if source["role"] == "prediction"), None)
    for index, row in units.iterrows():
        first = frame.iloc[row["_row_positions"][0]]
        session = first.get("session_id")
        if contract["assessment_mode"] == "qa" and pd.notna(session) and str(session).strip():
            units.at[index, "_group_id"] = str(session)
        context = dict(row["assessment_context"])
        observations = []
        for position in row["_row_positions"]:
            observation = frame.iloc[position]
            evidence = observation.get("evaluation_evidence", {})
            if isinstance(evidence, str):
                evidence = json.loads(evidence)
            if not isinstance(evidence, dict):
                raise MonitoringContractError("evaluation_evidence должен быть JSON object")
            missing = [name for name in required if name not in evidence or evidence[name] is None]
            if missing:
                raise MonitoringContractError(f"Недоступны обязательные свидетельства: {missing}")
            item = {"evidence": {name: evidence[name] for name in required}}
            if prediction:
                item["observed_prediction"] = observation.get(prediction)
            observations.append(item)
        context["observations"] = observations
        units.at[index, "assessment_context"] = context
    if require_sources:
        if "main_metric" not in units:
            raise MonitoringContractError("Reference не содержит итоговый main_metric")
        scores = pd.to_numeric(units["main_metric"], errors="raise")
        if not scores.dropna().isin(contract["evaluation"]["score_values"]).all():
            raise MonitoringContractError("Reference main_metric выходит за утверждённый score_values")
        units["assessment_score"] = scores
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
    frames = []
    for index in range(count):
        values = asyncio.run(asessor.run(frame))
        if len(values) != len(frame):
            raise MonitoringContractError("Assessor: число ответов не совпадает с числом единиц")
        parsed = pd.DataFrame([
            value.model_dump() if hasattr(value, "model_dump") else (value or {})
            for value in values
        ]).reindex(columns=source_ids).add_prefix("agent_")
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


def _score_predictions(units: pd.DataFrame, predictions: pd.DataFrame, contract: dict) -> pd.Series:
    scores = pd.to_numeric(predictions["agent_assessment_score"], errors="raise")
    allowed = contract["evaluation"]["score_values"]
    if not scores.dropna().isin(allowed).all():
        raise MonitoringContractError(f"Судья вернул оценку вне объявленной шкалы: {allowed}")
    return pd.Series(scores.to_numpy(), index=units.index, dtype="float64")


def _broadcast_predictions(
    original: pd.DataFrame,
    units: pd.DataFrame,
    predictions: pd.DataFrame,
    scores: pd.Series,
) -> pd.DataFrame:
    result = broadcast_scores(original, units, scores)
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
    score_values,
):
    return Asessor(
        llm=models[0],
        embedding_model=models[1],
        dataset=rag_units,
        score_values=score_values,
        context_columns=["assessment_context"],
        answer_columns=source_ids,
        instruction=instruction,
        domain_rag_path=domain_path,
        instruction_structuring=instruction_llm_preprocessing,
        instruction_summarization=instruction_llm_preprocessing,
    )


def _split_units(
    rag_units: pd.DataFrame, source_ids: list[str], train_fraction: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Делит эталон на train/holdout, сохраняя долю каждого класса оценок.

    Дефекты в корзинах редки (5-12%): при случайном делении holdout остаётся
    почти без них, и каппа считается по единицам счёта.
    """
    groups = rag_units["_group_id"] if "_group_id" in rag_units else None
    grouped = groups is not None and groups.notna().any()
    if grouped:
        keys = rag_units.groupby("_group_id", sort=False)[source_ids[0]].min()
        if len(keys) < 2:
            raise MonitoringContractError("Для calibration требуется минимум две группы")
    else:
        keys = rag_units[source_ids[0]]

    train_keys: list = []
    for _label, members in keys.groupby(keys.to_numpy(), sort=True):
        shuffled = members.sample(frac=1, random_state=42384)
        split = min(max(int(len(shuffled) * train_fraction), 1), max(len(shuffled) - 1, 1))
        train_keys.extend(shuffled.index[:split].tolist())

    selected = set(train_keys)
    mask = rag_units["_group_id"].isin(selected) if grouped else rag_units.index.isin(selected)
    train = rag_units[mask].reset_index(drop=True)
    test = rag_units[~mask].reset_index(drop=True)
    if train.empty or test.empty:
        raise MonitoringContractError("Для calibration не удалось разделить эталон")
    return train, test


def _calibrate(
    rag_units: pd.DataFrame,
    source_ids: list[str],
    instruction: str,
    domain_path: str | None,
    models,
    train_fraction: float,
    num_assessors: int,
    instruction_llm_preprocessing: bool,
    *,
    assessment_contract: dict,
    admission_settings: dict,
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame, Asessor]:
    if len(rag_units) < 2:
        raise MonitoringContractError("Для calibration требуется минимум две единицы")
    train, test = _split_units(rag_units, source_ids, train_fraction)
    judge = _build_assessor(
        models, train, source_ids, instruction, domain_path,
        instruction_llm_preprocessing, assessment_contract["evaluation"]["score_values"],
    )
    predictions = _predict(
        judge,
        test,
        source_ids,
        num_assessors,
    )
    labels = test[source_ids].reset_index(drop=True)
    comparison = pd.concat([predictions.reset_index(drop=True), labels], axis=1)
    score = score_results(
        comparison, "assessment_score",
        defect_threshold=assessment_contract["evaluation"]["defect_threshold"],
        higher_is_better=assessment_contract["evaluation"]["higher_is_better"],
    )
    if score["invalid_share"]:
        logger.warning("calibration: судья ответил на %s из %s единиц",
                       score["paired_units"], score["holdout_units"])
    # Смещение судьи считается на шкале ключевой метрики: оценка единицы по
    # контракту у судьи против той же оценки по человеческой разметке.
    judge_scores = _score_predictions(test, predictions, assessment_contract)
    human_scores = test["assessment_score"].astype(float)
    paired = judge_scores.notna().to_numpy() & human_scores.notna().to_numpy()
    bias = judge_bias(
        judge_scores[paired].astype(float).tolist(),
        human_scores[paired].astype(float).tolist(),
    )
    metrics: dict[str, object] = {
        **score,
        "bias_mean": None if bias is None else bias["mean"],
        "bias_ci_lower": None if bias is None else bias["ci_lower"],
        "bias_ci_upper": None if bias is None else bias["ci_upper"],
        "bias_units": None if bias is None else bias["units"],
    }
    admission = admit(metrics, **admission_settings)
    metrics["admission_status"] = admission.status
    metrics["admission_reason_code"] = admission.reason_code
    metrics["admission_reason"] = admission.reason
    logger.info(
        "calibration: acc_auto=%s, baseline по моде=%s, каппа Коэна=%s, "
        "альфа Криппендорфа=%s, корреляция Спирмана=%s, полнота на дефектах=%s, "
        "точность на дефектах=%s, смещение судьи=%s, допуск=%s (%s)",
        metrics["acc_auto"], metrics["baseline_mode_accuracy"], metrics["cohen_kappa"],
        metrics["krippendorff_alpha"], metrics["spearman_correlation"],
        metrics["defect_recall"], metrics["defect_precision"], metrics["bias_mean"],
        admission.status, admission.reason,
    )
    return metrics, test, predictions, judge


def _source_by_role(contract: dict, role: str) -> dict:
    return next(source for source in contract["scoring"]["sources"] if source["role"] == role)


def _source_missing(frame: pd.DataFrame, source: dict) -> bool:
    column = source["column_name"]
    if column not in frame:
        return True
    values = frame[column]
    return bool(values.isna().all() or values.astype(str).str.strip().eq("").all())


def _assessment_result(
    contract: dict,
    units: pd.DataFrame,
    *,
    scores: pd.Series | None = None,
    calibration_metrics: dict[str, object] | None = None,
    max_invalid_share: float = 1.0,
) -> dict[str, object]:
    total = len(units)
    scored = total if scores is None else int(scores.notna().sum())
    refused = total - scored
    refused_share = refused / total if total else 0.0
    result: dict[str, object] = {
        "contract_version": "laim-assessment-result.v2",
        "definition_id": contract.get("definition_id"),
        "status": "computed",
        "assessment_mode": contract["assessment_mode"],
        "total_units": total,
        "scored_units": scored,
        "refused_units": refused,
        "refused_share": refused_share,
    }
    if scored == 0:
        result.update(
            status="not_computable", reason_code="no_scored_units",
            reason="Нет ни одной оценённой единицы",
        )
    elif refused_share > max_invalid_share:
        # Переизбыток отказов — отдельный статус, а не падение ноды и не
        # молчаливое сужение выборки.
        result["status"] = "not_computable"
        result["reason_code"] = "judge_refusals"
        result["reason"] = (
            f"Доля отказов судьи {refused_share:.2f} выше допустимой {max_invalid_share:.2f}"
        )
    if calibration_metrics is not None:
        result["calibration_metrics"] = calibration_metrics
    return result


def _unavailable_result(contract: dict) -> dict[str, object]:
    result: dict[str, object] = {
        "contract_version": "laim-assessment-result.v2",
        "definition_id": contract.get("definition_id"),
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
    monitoring_umr: pd.DataFrame | None = None,
    scoring_rag_train_size: float = 0.8,
    domain_rag_files_zip: Path | None = None,
    stage: Annotated[str, "scoring", "monitoring", "combined"] = "combined",
    num_assessors: int = 1,
    model_id: str = "giga",
    llm_model: str = "GigaChat-3-Ultra",
    min_holdout_units: int = 20,
    min_holdout_defect_units: int = 4,
    weak_holdout_defect_units: int = 10,
    min_defect_recall: float = 0.5,
    min_kappa: float = 0.2,
    max_invalid_share: float = 0.2,
) -> dict[str, object]:
    instruction_llm_preprocessing = False
    admission_settings = dict(
        min_holdout_units=min_holdout_units,
        min_holdout_defect_units=min_holdout_defect_units,
        weak_holdout_defect_units=weak_holdout_defect_units,
        min_defect_recall=min_defect_recall,
        min_kappa=min_kappa,
        max_invalid_share=max_invalid_share,
    )
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
        if monitoring_umr.empty:
            reason = "Во входной выборке нет единиц мониторинга"
            logger.warning(reason)
            result = _unavailable_result({
                **contract, "reason_code": "no_monitoring_units", "reason": reason,
            })
            result["total_units"] = 0
            return {
                "scored_data": _unavailable_scored_data(monitoring_umr),
                "acc_auto": None, "assessment_result": result,
            }
        monitoring_umr = normalize_umr(monitoring_umr, contract)
        if contract["scoring"]["method"] == "accuracy":
            column = _source_by_role(contract, "prediction")["column_name"]
            values = monitoring_umr.get(column, pd.Series(None, index=monitoring_umr.index))
            missing = values.isna() | values.astype(str).str.strip().eq("")
            if missing.any():
                reason = (
                    f"UMR: prediction {column!r} недоступен в {int(missing.sum())} "
                    f"из {len(monitoring_umr)} строк; сопоставимая accuracy не вычисляется"
                )
                logger.warning(reason)
                return {
                    "scored_data": _unavailable_scored_data(monitoring_umr),
                    "acc_auto": None,
                    "assessment_result": _unavailable_result({
                        **contract, "reason_code": "missing_prediction", "reason": reason,
                    }),
                }

    reference_umr = _load_df(reference_umr)
    for name, frame in (("reference_umr", reference_umr), ("monitoring_umr", monitoring_umr)):
        if frame is None:
            continue
        expected_role = "reference" if name == "reference_umr" else "monitoring"
        roles = frame.get("dataset_role")
        if roles is None or not roles.eq(expected_role).all():
            raise MonitoringContractError(f"{name}.dataset_role не соответствует назначению данных")
        identifiers = frame.get("definition_id")
        if identifiers is None or not identifiers.eq(contract["definition_id"]).all():
            raise MonitoringContractError(f"{name}.definition_id не соответствует утверждённому определению")
    assessment_contract = contract
    if stage in {"monitoring", "combined"}:
        ready = monitoring_umr.get("evaluation_ready")
        if ready is None or not ready.eq(True).all():
            reason = "Конвертер не подтвердил полноту данных для оценки"
            logger.warning(reason)
            return {"scored_data": _unavailable_scored_data(monitoring_umr), "acc_auto": None,
                    "assessment_result": _unavailable_result({**contract,
                        "reason_code": "evidence_unavailable", "reason": reason})}
        _assessor_units(monitoring_umr, contract, require_sources=False)
    # Packed dialogue разворачивается до построения units: broadcast_scores
    # пишет по позициям, поэтому reference_umr обязан совпадать с ними построчно.
    reference_umr = normalize_umr(_load_df(reference_umr), contract)
    rag_units = _assessor_units(
        reference_umr,
        assessment_contract,
        require_sources=True,
    )
    source_ids = ["assessment_score"]
    instruction = _source_instruction(contract)
    if not instruction.strip():
        raise MonitoringContractError(
            "LLM-оценка требует непустую инструкцию в assessor_instruction"
        )
    domain_path = _domain_path(domain_rag_files_zip)

    judge_units = _labelled_reference_units(rag_units, source_ids)
    config = ModelsConfig(model=llm_model)
    judge_model, _ = _build_judge_model(model_id, config, llm_model)
    models = (
        judge_model,
        BoundedGigaChatEmbeddings(GigaChatEmbeddings(**config.contour_configs)),
    )

    judge = None
    acc_auto = None
    calibration_metrics = None
    scored_output = None
    assessment_result = None
    if stage in {"scoring", "combined"}:
        calibration_metrics, test_units, predictions, judge = _calibrate(
            judge_units,
            source_ids,
            instruction,
            domain_path,
            models,
            scoring_rag_train_size,
            num_assessors,
            instruction_llm_preprocessing,
            assessment_contract=assessment_contract,
            admission_settings=admission_settings,
        )
        acc_auto = calibration_metrics["acc_auto"]
        scores = _score_predictions(
            test_units,
            predictions,
            assessment_contract,
        )
        scored_output = _broadcast_predictions(
            reference_umr,
            test_units,
            predictions,
            scores,
        )
        assessment_result = _assessment_result(
            contract,
            test_units,
            scores=scores,
            calibration_metrics=calibration_metrics,
        )

    if stage in {"monitoring", "combined"} and monitoring_umr is not None:
        monitoring_units = _assessor_units(
            monitoring_umr,
            assessment_contract,
            require_sources=False,
        )
        if judge is None:
            judge = _build_assessor(
                models, judge_units, source_ids, instruction, domain_path,
                instruction_llm_preprocessing, contract["evaluation"]["score_values"],
            )
        predictions = _predict(
            judge,
            monitoring_units,
            source_ids,
            num_assessors,
        )
        scores = _score_predictions(
            monitoring_units,
            predictions,
            assessment_contract,
        )
        scored_output = _broadcast_predictions(
            monitoring_umr,
            monitoring_units,
            predictions,
            scores,
        )
        assessment_result = _assessment_result(
            contract,
            monitoring_units,
            scores=scores,
            calibration_metrics=calibration_metrics,
            max_invalid_share=max_invalid_share,
        )
    assessment_result["run_id"] = uuid4().hex
    if scored_output is not None:
        scored_output["assessment_run_id"] = assessment_result["run_id"]
    assessment_result["purpose"] = "calibration" if stage == "scoring" else "monitoring"
    return {
        "scored_data": scored_output,
        "acc_auto": acc_auto,
        "assessment_result": assessment_result,
    }
