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
