"""Sber DS entrypoint автоассесора для канонического UMR."""

from __future__ import annotations

import asyncio
from copy import deepcopy
import io
import logging
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
from laim_monitoring import (
    MonitoringContractError,
    broadcast_scores,
    normalize_umr,
    score_units,
    unitize,
    validate_monitoring_metric,
)
from utils import add_voting_columns, extract_zip, read_docx, remove_directory
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


def _source_instruction(contract: dict) -> str:
    lines = ["Поля ответа JSON и соответствующие критерии:"]
    for source in contract["scoring"]["sources"]:
        lines.append(
            f"- {source['source_id']}: критерий из колонки {source['column_name']!r}, "
            f"роль {source['role']}, polarity {source['polarity']}"
        )
    return "\n".join(lines)


def _assessment_contract(contract: dict) -> dict:
    """Предсказывать готовый score для accuracy и целого диалога."""
    if (
        contract["scoring"]["method"] != "accuracy"
        and contract["assessment_mode"] != "dialogue"
    ):
        return contract
    result = deepcopy(contract)
    result["scoring"] = {
        "method": "identity",
        "sources": [
            {
                "source_id": "assessment_score",
                "column_name": "main_metric",
                "role": "final_score",
                "normalization": "numeric",
                "polarity": "direct",
            }
        ],
        "missing_policy": contract["scoring"]["missing_policy"],
        "majority_denominator": None,
    }
    return result


def _assessment_frame(frame: pd.DataFrame, contract: dict) -> pd.DataFrame:
    """Выровнять контекст accuracy по prediction эталона и monitoring."""
    if contract["scoring"]["method"] != "accuracy":
        return frame
    prediction = _source_by_role(contract, "prediction")
    column = prediction["column_name"]
    if _source_missing(frame, prediction):
        raise MonitoringContractError(
            f"UMR не содержит наблюдаемое prediction: {column}"
        )
    result = frame.copy()
    result["output_answer"] = result[column]
    return result


def _assessor_units(frame: pd.DataFrame, contract: dict, *, require_sources: bool) -> pd.DataFrame:
    units = unitize(frame, contract)
    source_ids = [source["source_id"] for source in contract["scoring"]["sources"]]
    if require_sources:
        missing = [source_id for source_id in source_ids if source_id not in units]
        if missing:
            raise MonitoringContractError(f"RAG не содержит источники MeasurementPlan: {missing}")
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


def _score_predictions(units: pd.DataFrame, predictions: pd.DataFrame, contract: dict) -> pd.Series:
    source_ids = [source["source_id"] for source in contract["scoring"]["sources"]]
    values = units.copy()
    for source_id in source_ids:
        values[source_id] = predictions[f"agent_{source_id}"].tolist()
    # Строка без единого поля ответа — это отказ судьи, а не пропуск в данных:
    # missing_policy контракта (в том числе fail) к ней не применяется, единица
    # просто исключается из оценки.
    failed = (
        predictions[[f"agent_{source_id}" for source_id in source_ids]]
        .isna()
        .all(axis=1)
        .to_numpy()
    )
    scores = pd.Series(float("nan"), index=values.index, dtype="float64")
    scores.iloc[~failed] = score_units(values.iloc[~failed], contract).to_numpy()
    return scores


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
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    if len(rag_units) < 2:
        raise MonitoringContractError("Для calibration требуется минимум две единицы")
    train, test = _split_units(rag_units, source_ids, train_fraction)
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
        logger.warning(
            "calibration: судья не ответил на %d из %d единиц; acc_auto считается по отвеченным",
            int((~answered).sum()), len(answered),
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
    labels_used = labels[answered].astype(float)
    # Смещение судьи считается на шкале ключевой метрики: оценка единицы по
    # контракту у судьи против той же оценки по человеческой разметке.
    judge_scores = _score_predictions(test, predictions, assessment_contract)
    human_scores = score_units(test, assessment_contract)
    paired = answered.to_numpy() & judge_scores.notna().to_numpy() & human_scores.notna().to_numpy()
    bias = judge_bias(
        judge_scores[paired].astype(float).tolist(),
        human_scores[paired].astype(float).tolist(),
    )
    metrics: dict[str, object] = {
        "acc_auto": float(score["mean_accuracy"]["Mean accuracy"]),
        "holdout_units": int(len(labels_used)),
        "holdout_defect_units": int(
            (labels_used[source_ids[0]] == labels_used[source_ids[0]].min()).sum()
        ),
        "invalid_share": float((~answered).sum() / len(answered)),
        "baseline_mode_accuracy": float(score["mean_accuracy"]["Mean mode"]),
        "cohen_kappa": score["cohen_kappa"],
        "krippendorff_alpha": score["krippendorff_alpha"],
        "spearman_correlation": float(score["mean_correlation"]),
        "defect_recall": float(score["defect_recall"]),
        "defect_precision": float(score["defect_precision"]),
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
        "calibration: acc_auto=%.3f, baseline по моде=%.3f, каппа Коэна=%s, "
        "альфа Криппендорфа=%s, корреляция Спирмана=%.3f, полнота на дефектах=%.3f, "
        "точность на дефектах=%.3f, смещение судьи=%s, допуск=%s (%s)",
        metrics["acc_auto"], metrics["baseline_mode_accuracy"], metrics["cohen_kappa"],
        metrics["krippendorff_alpha"], metrics["spearman_correlation"],
        metrics["defect_recall"], metrics["defect_precision"], metrics["bias_mean"],
        admission.status, admission.reason,
    )
    return metrics, test, predictions


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
        "contract_version": "laim-assessment-result.v1",
        "status": "computed",
        "assessment_mode": contract["assessment_mode"],
        "total_units": total,
        "scored_units": scored,
        "refused_units": refused,
        "refused_share": refused_share,
    }
    if refused_share > max_invalid_share:
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
    min_holdout_units: int = 20,
    min_holdout_defect_units: int = 4,
    weak_holdout_defect_units: int = 10,
    min_defect_recall: float = 0.5,
    min_kappa: float = 0.2,
    max_invalid_share: float = 0.2,
) -> dict[str, object]:
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
        monitoring_umr = normalize_umr(monitoring_umr, contract)

    assessment_contract = _assessment_contract(contract)
    # Packed dialogue разворачивается до построения units: broadcast_scores
    # пишет по позициям, поэтому reference_umr обязан совпадать с ними построчно.
    reference_umr = normalize_umr(_load_df(reference_umr), contract)
    rag_assessment = _assessment_frame(reference_umr, contract)
    rag_units = _assessor_units(
        rag_assessment,
        assessment_contract,
        require_sources=True,
    )
    source_ids = [
        source["source_id"] for source in assessment_contract["scoring"]["sources"]
    ]
    instruction = _load_instruction(assessor_instruction)
    if not instruction.strip():
        raise MonitoringContractError(
            "LLM-оценка требует непустую инструкцию в assessor_instruction"
        )
    instruction += "\n\n" + _source_instruction(assessment_contract)
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
        # Трейсы не несут колонок размеченной корзины: без наблюдаемого
        # prediction судья оценивает сам output_answer, а не отказывается.
        monitoring_assessment = monitoring_umr
        if contract["scoring"]["method"] == "accuracy":
            prediction = _source_by_role(contract, "prediction")
            if _source_missing(monitoring_umr, prediction):
                print(
                    f"monitoring: prediction {prediction['column_name']!r} "
                    "недоступен в UMR, судья оценивает output_answer"
                )
            else:
                monitoring_assessment = _assessment_frame(monitoring_umr, contract)
        monitoring_units = _assessor_units(
            monitoring_assessment,
            assessment_contract,
            require_sources=False,
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
    return {
        "scored_data": scored_output,
        "acc_auto": acc_auto,
        "assessment_result": assessment_result,
    }
