"""Accuracy калибровки — точное совпадение метки, а не близость чисел."""

import pandas as pd

from agent.score_results import AnswersProcessor, ResultsScorer


def test_mean_accuracy_requires_exact_match():
    frame = pd.DataFrame({
        "score": [5, 1],
        "agent_score": [4, 1],
    })

    result = ResultsScorer(AnswersProcessor()).compute_mean_accuracy(frame, ["score"])

    assert result["Mean accuracy"] == 0.5


def test_score_reports_full_agreement_metrics():
    frame = pd.DataFrame({
        "score": [1, 2, 1, 2],
        "agent_score": [1, 2, 1, 2],
    })

    result = ResultsScorer(AnswersProcessor()).score(frame, ["score"], defect_threshold=1, higher_is_better=True)

    assert result["cohen_kappa"] == 1.0
    assert result["krippendorff_alpha"] == 1.0


def test_cohen_kappa_penalizes_chance_agreement():
    frame = pd.DataFrame({
        "score": [1, 2, 1, 2],
        "agent_score": [1, 1, 1, 1],
    })

    result = ResultsScorer(AnswersProcessor()).score(frame, ["score"], defect_threshold=1, higher_is_better=True)

    assert result["mean_accuracy"]["Mean accuracy"] == 0.5
    assert result["cohen_kappa"] == 0.0


def test_cohen_kappa_handles_non_integral_scale():
    # Шкала 0.5/1.0 номинальна: полное совпадение даёт каппу 1.0, а не 0.0 из-за
    # проглоченной ошибки sklearn о «непрерывной» цели.
    frame = pd.DataFrame({
        "score": [0.5, 1.0, 0.5, 1.0],
        "agent_score": [0.5, 1.0, 0.5, 1.0],
    })

    result = ResultsScorer(AnswersProcessor()).score(frame, ["score"], defect_threshold=1, higher_is_better=True)

    assert result["cohen_kappa"] == 1.0


def test_incomputable_agreement_is_none_not_zero():
    frame = pd.DataFrame({"score": [1.0, 0.0], "agent_score": [None, None]})

    result = ResultsScorer(AnswersProcessor()).score(frame, ["score"], defect_threshold=1, higher_is_better=True)

    assert result["cohen_kappa"] is None
    assert result["krippendorff_alpha"] is None


def test_defect_recall_exposes_blind_judge():
    """Судья, зачитывающий всё подряд, обязан быть виден по recall, а не по accuracy."""
    frame = pd.DataFrame({
        "score": [0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
        "agent_score": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    })

    result = ResultsScorer(AnswersProcessor()).score(frame, ["score"], defect_threshold=1, higher_is_better=True)

    assert result["mean_accuracy"]["Mean accuracy"] > 0.6
    assert result["defect_recall"] == 0.0
    assert result["defect_precision"] == 0.0


def test_defect_recall_counts_caught_defects():
    frame = pd.DataFrame({
        "score": [0.0, 0.0, 0.0, 0.0, 1.0, 1.0],
        "agent_score": [0.0, 0.0, 0.0, 1.0, 0.0, 1.0],
    })

    result = ResultsScorer(AnswersProcessor()).score(frame, ["score"], defect_threshold=1, higher_is_better=True)

    assert result["defect_recall"] == 0.75
    assert result["defect_precision"] == 0.75
