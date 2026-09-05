import pandas as pd
import pytest

import main as assessor
from tests.test_monitoring_accuracy import _metric, _reference


@pytest.mark.parametrize("scores", [pd.Series(dtype=float), pd.Series([float("nan")])])
def test_zero_scored_units_are_never_computed(scores):
    result = assessor._assessment_result(
        _metric(), pd.DataFrame(index=scores.index), scores=scores, max_invalid_share=1.0,
    )
    assert result["status"] == "not_computable"
    assert result["scored_units"] == 0
    assert result["reason_code"] == "no_scored_units"


def test_empty_monitoring_does_not_initialize_models(monkeypatch):
    def unexpected_model(**_kwargs):
        pytest.fail("На пустом срезе модель не нужна")

    monkeypatch.setattr(assessor, "ModelsConfig", unexpected_model)
    result = assessor.main(_reference(), _metric(), monitoring_umr=pd.DataFrame())
    assert result["assessment_result"]["status"] == "not_computable"
    assert result["assessment_result"]["reason_code"] == "no_monitoring_units"
    assert result["assessment_result"]["total_units"] == 0
    assert result["scored_data"].empty
