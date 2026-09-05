from types import SimpleNamespace

import pandas as pd

import main as assessor
from tests.test_monitoring_accuracy import _metric, _monitoring, _reference


def test_combined_uses_calibrated_judge_without_holdout_in_rag(monkeypatch):
    built = []
    calls = []

    def build(_models, train, *_args):
        judge = SimpleNamespace(rag_ids=set(train['_unit_id']))
        built.append(judge)
        return judge

    def predict(judge, units, _sources, _count):
        ids = set(units['_unit_id'])
        calls.append((judge, ids))
        assert judge.rag_ids.isdisjoint(ids)
        values = units.get('assessment_score', pd.Series(1.0, index=units.index))
        return pd.DataFrame({'agent_assessment_score': values.tolist()})

    monkeypatch.setattr(assessor, 'ModelsConfig', lambda **_: SimpleNamespace(contour_configs={}))
    monkeypatch.setattr(assessor, 'GigaChatEmbeddings', lambda **_: object())
    monkeypatch.setattr(assessor, '_build_judge_model', lambda *_: (object(), 'judge'))
    monkeypatch.setattr(assessor, '_build_assessor', build)
    monkeypatch.setattr(assessor, '_predict', predict)

    result = assessor.main(
        reference_umr=_reference(), monitoring_metric=_metric(),
        monitoring_umr=_monitoring(), stage='combined',
    )

    assert len(built) == 1
    assert len(calls) == 2
    assert calls[0][0] is calls[1][0] is built[0]
    assert built[0].rag_ids | calls[0][1] == set(_reference().query_id)
    assert calls[1][1] == set(_monitoring().query_id)
    assert result['assessment_result']['calibration_metrics']['acc_auto'] == 1.0
