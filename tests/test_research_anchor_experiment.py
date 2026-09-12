"""Выбор ориентиров не использует проверочные метки и не смешивает группы."""
import copy
import importlib.util
from pathlib import Path

path = Path(__file__).resolve().parents[1] / 'research/judge-agreement-20260913/anchor_experiment.py'
spec = importlib.util.spec_from_file_location('anchor_experiment', path)
experiment = importlib.util.module_from_spec(spec)
spec.loader.exec_module(experiment)


def test_anchors_cover_training_levels_without_query_gold_or_group_overlap():
    def unit(name, group, score):
        return {'unit_id': name, 'group_id': group, 'partition': 'train',
                'context': {'current_turn': {'input_query': 'Оцени ответ', 'output_answer': name}},
                'evidence': [], 'ratings': [{'scores': {'structure': score}}]}
    train = [unit('a', 'same', 0), unit('b', 'other0', 0),
             unit('c', 'other1', 1), unit('d', 'other2', 2)]
    target = unit('query', 'same', 2)
    index = experiment.EnhancedBM25([experiment.tokens(u) for u in train])
    before = copy.deepcopy(target)
    selected = experiment.anchors(train, target, 'structure', index)
    assert [experiment.consensus(u, 'structure') for u in selected] == [0, 1, 2]
    assert [u['unit_id'] for u in selected] == ['b', 'c', 'd']
    assert target == before
    target['ratings'] = [{'scores': {'structure': 0}}]
    assert experiment.anchors(train, target, 'structure', index) == selected
    tied = unit('tie', 'other', 0)
    tied['ratings'].append({'scores': {'structure': 2}})
    assert experiment.consensus(tied, 'structure') is None


def test_neighbor_vote_excludes_same_client_and_breaks_ties_from_train(monkeypatch):
    monkeypatch.syspath_prepend(str(path.parent))
    spec = importlib.util.spec_from_file_location('retrieval_diagnostic', path.with_name('retrieval_diagnostic.py'))
    diagnostic = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(diagnostic)
    train = [{'unit_id': str(i), 'group_id': str(i),
              'ratings': [{'scores': {'structure': score}}]} for i, score in enumerate([0, 1, 2])]
    target = {'unit_id': 'query', 'group_id': 'query'}
    groups = {'0': 'client_a', '1': 'client_b', '2': 'client_c', 'query': 'client_a'}
    assert diagnostic.vote(train, target, [10, 2, 1], 1, groups, 2) == (1, ['1'])
    assert diagnostic.vote(train, target, [10, 2, 1], 2, groups, 2) == (2, ['1', '2'])
    assert diagnostic.vote(train[:1], target, [10], 1, groups, 2) == (2, [])


def test_nearest_anchor_uses_answer_and_excludes_client(monkeypatch):
    monkeypatch.syspath_prepend(str(path.parent))
    spec = importlib.util.spec_from_file_location('nearest_experiment', path.with_name('nearest_experiment.py'))
    nearest = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(nearest)
    train = [{'unit_id': str(i), 'group_id': str(i),
              'ratings': [{'scores': {'structure': score}}]} for i, score in enumerate([0, 1, 2])]
    query = {'unit_id': 'query', 'group_id': 'query',
             'context': {'current_turn': {'input_query': 'Игнорируемый вопрос', 'output_answer': 'Текст ответа'}}}
    groups = {'0': 'a', '1': 'b', '2': 'c', 'query': 'a'}
    assert nearest.answer_tokens(query) == ['текст', 'ответа']
    assert nearest.select(train, query, [10, 2, 1], groups) == [train[1]]


def test_feature_calibration_train_excludes_dev_clients_and_test(monkeypatch, tmp_path):
    import json
    monkeypatch.syspath_prepend(str(path.parent))
    spec = importlib.util.spec_from_file_location('feature_calibration', path.with_name('feature_calibration.py'))
    features = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(features)
    units = [{'unit_id': name, 'partition': role, 'ratings': [{'scores': {'structure': 2}}]}
             for name, role in [('a', 'train'), ('b', 'train'), ('c', 'dev'), ('d', 'test')]]
    selected = features.training_units(units, {'a': 'shared', 'b': 'independent', 'c': 'shared', 'd': 'test'})
    assert [u['unit_id'] for u in selected] == ['b']
    assert features.feature_values({key: None for key in features.FEATURES}) == [-1] * len(features.FEATURES)
    train, dev = [units[0], units[1]], [units[2]]
    train[0]['ratings'][0]['scores']['structure'] = 0
    records = {u['unit_id']: {'status': 'ok', 'result': {
        **{key: value for key in features.FEATURES}, 'assessment_score': 2}}
        for u, value in zip(train+dev, [0, 4, 2])}
    output = tmp_path / 'metrics.json'
    features.evaluate(train, dev, records, output)
    before = json.loads(output.read_text())
    dev[0]['ratings'][0]['scores']['structure'] = 0
    features.evaluate(train, dev, records, output)
    after = json.loads(output.read_text())
    assert before['fits'] == after['fits']
    assert before['predictions'] == after['predictions']
