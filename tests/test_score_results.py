"""Accuracy калибровки — точное совпадение метки, а не близость чисел."""

import pandas as pd

from agent.score_results import score_results


def test_mean_accuracy_requires_exact_match():
    frame = pd.DataFrame({
        "score": [5, 1],
        "agent_score": [4, 1],
    })

    result = score_results(frame, "score", defect_threshold=1, higher_is_better=True)

    assert result["acc_auto"] == 0.5


def test_score_reports_full_agreement_metrics():
    frame = pd.DataFrame({
        "score": [1, 2, 1, 2],
        "agent_score": [1, 2, 1, 2],
    })

    result = score_results(frame, "score", defect_threshold=1, higher_is_better=True)

    assert result["cohen_kappa"] == 1.0
    assert result["krippendorff_alpha"] == 1.0


def test_cohen_kappa_penalizes_chance_agreement():
    frame = pd.DataFrame({
        "score": [1, 2, 1, 2],
        "agent_score": [1, 1, 1, 1],
    })

    result = score_results(frame, "score", defect_threshold=1, higher_is_better=True)

    assert result["acc_auto"] == 0.5
    assert result["cohen_kappa"] == 0.0


def test_cohen_kappa_handles_non_integral_scale():
    # Шкала 0.5/1.0 номинальна: полное совпадение даёт каппу 1.0, а не 0.0 из-за
    # проглоченной ошибки sklearn о «непрерывной» цели.
    frame = pd.DataFrame({
        "score": [0.5, 1.0, 0.5, 1.0],
        "agent_score": [0.5, 1.0, 0.5, 1.0],
    })

    result = score_results(frame, "score", defect_threshold=1, higher_is_better=True)

    assert result["cohen_kappa"] == 1.0


def test_incomputable_agreement_is_none_not_zero():
    frame = pd.DataFrame({"score": [1.0, 0.0], "agent_score": [None, None]})

    result = score_results(frame, "score", defect_threshold=1, higher_is_better=True)

    assert result["cohen_kappa"] is None
    assert result["krippendorff_alpha"] is None


def test_defect_recall_exposes_blind_judge():
    """Судья, зачитывающий всё подряд, обязан быть виден по recall, а не по accuracy."""
    frame = pd.DataFrame({
        "score": [0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
        "agent_score": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    })

    result = score_results(frame, "score", defect_threshold=1, higher_is_better=True)

    assert result["acc_auto"] > 0.6
    assert result["defect_recall"] == 0.0
    assert result["defect_precision"] is None


def test_defect_recall_counts_caught_defects():
    frame = pd.DataFrame({
        "score": [0.0, 0.0, 0.0, 0.0, 1.0, 1.0],
        "agent_score": [0.0, 0.0, 0.0, 1.0, 0.0, 1.0],
    })

    result = score_results(frame, "score", defect_threshold=1, higher_is_better=True)

    assert result["defect_recall"] == 0.75
    assert result["defect_precision"] == 0.75


def test_agreement_reports_coverage_and_error_magnitude():
    frame = pd.DataFrame({'score': [0., 0., 1., 2.], 'agent_score': [None, 1., 1., 2.]})
    result = score_results(frame, 'score', defect_threshold=1, higher_is_better=True)
    assert result['paired_units'] == 3
    assert result['coverage'] == 0.75
    assert result['mean_absolute_error'] == 1 / 3
    assert result['defect_recall'] == 0
    assert result['defect_precision'] is None
    assert result['defect_confusion'] == {
        'true_positive': 0, 'false_negative': 1, 'false_positive': 0,
        'true_negative': 2, 'abstained_defects': 1, 'abstained_nondefects': 0,
    }


def test_constant_ratings_have_no_chance_agreement_or_correlation():
    frame = pd.DataFrame({'score': [1., 1.], 'agent_score': [1., 1.]})
    result = score_results(frame, 'score', defect_threshold=1, higher_is_better=True)
    assert result['cohen_kappa'] is result['krippendorff_alpha'] is None
    assert result['spearman_correlation'] is None
    assert result['defect_recall'] is None


def test_nominal_alpha_and_kappa_match_independent_coincidence_counts():
    import pytest

    frame = pd.DataFrame({'score': [0, 0, 1, 1], 'agent_score': [0, 1, 1, 1]})
    result = score_results(frame, 'score', defect_threshold=1, higher_is_better=True)
    assert result['cohen_kappa'] == 0.5
    # Do=1/4; De=2*3*5/(8*7), где 3 и 5 — суммы меток двух разметчиков.
    assert result['krippendorff_alpha'] == pytest.approx(8 / 15)


def test_all_refusals_keep_defects_and_unknown_agreement():
    frame = pd.DataFrame({'score': [0., 1.], 'agent_score': [None, None]})
    result = score_results(frame, 'score', defect_threshold=1, higher_is_better=True)
    assert result['paired_units'] == 0 and result['holdout_defect_units'] == 1
    assert result['defect_recall'] == 0
    assert result['acc_auto'] is result['spearman_correlation'] is None
    assert result['defect_confusion']['abstained_defects'] == 1


def test_lower_is_better_uses_the_other_side_of_threshold():
    frame = pd.DataFrame({'score': [2., 0., 3.], 'agent_score': [None, 0., 3.]})
    result = score_results(frame, 'score', defect_threshold=1, higher_is_better=False)
    assert result['defect_recall'] == 0.5
    assert result['defect_precision'] == 1
