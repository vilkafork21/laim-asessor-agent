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
