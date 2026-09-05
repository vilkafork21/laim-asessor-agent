import pandas as pd
import pytest

import main as assessor
from tests.test_monitoring_accuracy import _metric, _monitoring, _reference


@pytest.mark.parametrize("value", [None, float("nan"), "", "  ", "absent_column"])
def test_missing_prediction_preserves_population_and_stops_before_judge(monkeypatch, value):
    monitoring = _monitoring()
    if value == "absent_column":
        monitoring = monitoring.drop(columns="class")
    else:
        monitoring.loc[0, "class"] = value

    def unexpected_model(**_kwargs):
        pytest.fail("Недоступное prediction должно быть обнаружено до создания моделей")

    monkeypatch.setattr(assessor, "ModelsConfig", unexpected_model)
    result = assessor.main(
        reference_umr=_reference(), monitoring_metric=_metric(),
        monitoring_umr=monitoring, stage="combined",
    )

    assessment = result["assessment_result"]
    assert assessment["status"] == "not_computable"
    assert assessment["reason_code"] == "missing_prediction"
    assert assessment["scored_units"] == 0
    assert "class" in assessment["reason"]
    assert result["acc_auto"] is None
    assert result["scored_data"]["main_metric"].isna().all()
    pd.testing.assert_frame_equal(result["scored_data"][monitoring.columns], monitoring)
