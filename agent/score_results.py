"""
Обработка и оценка результатов асессора.

Основные классы:
- AnswersProcessor: парсинг ответов LLM в DataFrame
- ResultsScorer: вычисление метрик (корреляция, точность)
"""

from typing import Any

import krippendorff
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import cohen_kappa_score


# =====================================================
# ANSWERS PROCESSOR
# =====================================================


class AnswersProcessor:
    def parse(self, dataset: list[dict], answer_columns: list[str]) -> pd.DataFrame:
        """Парсит ответы LLM в DataFrame с префиксом agent_.

        Поддерживает два формата:
        - {answer: {target: ..., cheked: ...}} (старый формат)
        - {target: ..., cheked: ...} (Pydantic output)
        """
        asessor_answers = []
        for val in dataset:
            if val is None:
                asessor_answers.append({column: None for column in answer_columns})
                continue

            # Try old format first: {answer: {...}}
            answer = val.get("answer")
            if answer is not None:
                asessor_answers.append(answer)
            else:
                # Pydantic format: direct fields
                # Filter to only include answer_columns
                answer = {k: v for k, v in val.items() if k in answer_columns}
                if answer:
                    asessor_answers.append(answer)
                else:
                    asessor_answers.append({column: None for column in answer_columns})
        asessor_answers = [
            {
                "agent_" + key: value
                for key, value in answer.items()
                if key in answer_columns
            }
            for answer in asessor_answers
        ]

        return pd.DataFrame(asessor_answers).reset_index(drop=True)

    def merge(
        self,
        agent_asessor_answers_dataframe: pd.DataFrame,
        real_asessor_answers_dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """Объединяет предсказания асессора с реальными ответами."""
        stat_df = agent_asessor_answers_dataframe.merge(
            real_asessor_answers_dataframe.reset_index(drop=True),
            left_index=True,
            right_index=True,
        ).reset_index(drop=True)

        return stat_df.replace([np.inf, -np.inf], np.nan)


# =====================================================
# RESULTS SCORER
# =====================================================


def _pair_cohen_kappa(agent_data: pd.Series, human_data: pd.Series) -> float | None:
    try:
        kappa = cohen_kappa_score(agent_data, human_data)
    except Exception:
        return None
    return None if np.isnan(kappa) else float(kappa)


def _pair_nominal_alpha(agent_data: pd.Series, human_data: pd.Series) -> float | None:
    # Метки могут быть нечисловыми: кодируем их и считаем nominal-альфу —
    # это согласуется с точным совпадением в accuracy.
    codes, _ = pd.factorize(pd.concat([agent_data, human_data], ignore_index=True))
    try:
        alpha = krippendorff.alpha(
            codes.reshape(2, -1).astype(float), level_of_measurement="nominal"
        )
    except Exception:
        return None
    return None if np.isnan(alpha) else float(alpha)


class ResultsScorer:
    def __init__(self, processor: AnswersProcessor):
        self.processor = processor

    def compute_correlation(
        self, full_dataset: pd.DataFrame, answer_columns: list[str]
    ) -> float:
        """Вычисляет среднюю корреляцию Спирмана между агентом и человеком."""
        all_correlations = []
        for answer_column in answer_columns:
            agent_stat_col = f"agent_{answer_column}"
            human_stat_col = answer_column

            if agent_stat_col in full_dataset.columns and human_stat_col in full_dataset.columns:
                mask = full_dataset[[agent_stat_col, human_stat_col]].notna().all(axis=1)
                agent_data = full_dataset.loc[mask, agent_stat_col]
                human_data = full_dataset.loc[mask, human_stat_col]

                if len(agent_data) > 1 and len(human_data) > 1:
                    try:
                        cv_value, _ = stats.spearmanr(agent_data, human_data)
                        if not np.isnan(cv_value):
                            all_correlations.append(cv_value)
                    except Exception:
                        pass

        return np.mean(np.array(all_correlations)) if all_correlations else 0

    def compute_mean_accuracy(
        self, full_dataset: pd.DataFrame, answer_columns: list[str]
    ) -> dict[str, float]:
        """Вычисляет долю точных совпадений и baseline по моде."""

        def _calculate_accuracy_for_columns(
            row: pd.Series, columns: list[str], agent_col_prefix: str
        ) -> float:
            """Сравнивает ответы с разметкой без близости по шкале."""
            accuracies = []
            for col in columns:
                agent_col = f"{agent_col_prefix}{col}"
                agent_val = row.get(agent_col)
                real_val = row[col]
                if pd.notna(agent_val) and pd.notna(real_val):
                    accuracies.append(float(agent_val == real_val))
            return np.mean(accuracies) if accuracies else 0

        accuracies = full_dataset.apply(
            lambda row: _calculate_accuracy_for_columns(row, answer_columns, "agent_"), axis=1
        )

        mode_values = {}
        for col in answer_columns:
            mode_vals = full_dataset[col].dropna().mode()
            mode_values[col] = mode_vals.iloc[0] if not mode_vals.empty else None

        def _calculate_mode_accuracy(row: pd.Series, columns: list[str]) -> float:
            """Вычисляет точность относительно моды (baseline)."""
            accuracies = []
            for col in columns:
                real_val = row[col]
                mode_val = mode_values[col]
                if pd.notna(real_val) and pd.notna(mode_val):
                    accuracies.append(float(real_val == mode_val))
            return np.mean(accuracies) if accuracies else 0

        mode_accuracies = full_dataset.apply(
            lambda row: _calculate_mode_accuracy(row, answer_columns), axis=1
        )

        return {"Mean accuracy": accuracies.mean(), "Mean mode": mode_accuracies.mean()}

    def compute_agreement(
        self, full_dataset: pd.DataFrame, answer_columns: list[str]
    ) -> dict[str, float]:
        """Каппа Коэна и альфа Криппендорфа между агентом и человеком, среднее по критериям."""
        kappas = []
        alphas = []
        for answer_column in answer_columns:
            agent_col = f"agent_{answer_column}"
            if agent_col not in full_dataset.columns or answer_column not in full_dataset.columns:
                continue
            mask = full_dataset[[agent_col, answer_column]].notna().all(axis=1)
            agent_data = full_dataset.loc[mask, agent_col]
            human_data = full_dataset.loc[mask, answer_column]
            if len(agent_data) < 2:
                continue
            kappa = _pair_cohen_kappa(agent_data, human_data)
            if kappa is not None:
                kappas.append(kappa)
            alpha = _pair_nominal_alpha(agent_data, human_data)
            if alpha is not None:
                alphas.append(alpha)

        return {
            "cohen_kappa": float(np.mean(kappas)) if kappas else 0.0,
            "krippendorff_alpha": float(np.mean(alphas)) if alphas else 0.0,
        }

    def compute_defect_quality(
        self, full_dataset: pd.DataFrame, answer_columns: list[str]
    ) -> dict[str, float]:
        """Полнота и точность судьи на дефектах — то, ради чего работает мониторинг.

        Сдвиг судьи в сторону «всё хорошо» поднимает accuracy на перекошенной
        разметке, поэтому деградация обязана быть видна отдельной метрикой.
        """
        caught = flagged = defects = 0
        for answer_column in answer_columns:
            agent_col = f"agent_{answer_column}"
            if agent_col not in full_dataset.columns or answer_column not in full_dataset.columns:
                continue
            mask = full_dataset[[agent_col, answer_column]].notna().all(axis=1)
            human = pd.to_numeric(full_dataset.loc[mask, answer_column], errors="coerce")
            agent = pd.to_numeric(full_dataset.loc[mask, agent_col], errors="coerce")
            if human.isna().all() or agent.isna().all():
                continue
            floor = human.min()
            defects += int((human == floor).sum())
            flagged += int((agent == floor).sum())
            caught += int(((human == floor) & (agent == floor)).sum())

        return {
            "defect_recall": caught / defects if defects else 0.0,
            "defect_precision": caught / flagged if flagged else 0.0,
        }

    def score(
        self,
        full_dataset: pd.DataFrame,
        answer_columns: list[str],
    ) -> dict[str, Any]:
        """Вычисляет все метрики для оценки асессора."""
        mean_correlation = self.compute_correlation(full_dataset, answer_columns)
        mean_accuracy_dict = self.compute_mean_accuracy(full_dataset, answer_columns)
        agreement = self.compute_agreement(full_dataset, answer_columns)
        defects = self.compute_defect_quality(full_dataset, answer_columns)

        return {
            "mean_correlation": mean_correlation,
            "mean_accuracy": mean_accuracy_dict,
            "cohen_kappa": agreement["cohen_kappa"],
            "krippendorff_alpha": agreement["krippendorff_alpha"],
            "defect_recall": defects["defect_recall"],
            "defect_precision": defects["defect_precision"],
        }
