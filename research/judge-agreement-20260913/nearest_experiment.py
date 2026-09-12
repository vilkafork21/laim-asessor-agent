"""GigaChat с ближайшим train-ответом другого клиента; прежний live-контроль."""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
import shutil
from pathlib import Path
from unittest.mock import patch

import anchor_experiment as experiment
from retrieval_diagnostic import vote


def answer_tokens(unit: dict) -> list[str]:
    return re.findall(r'\w+', str(unit['context']['current_turn'].get('output_answer') or '').lower()) or ['__empty__']


def select(train: list[dict], unit: dict, scores: list[float], groups: dict[str, str]) -> list[dict]:
    _, identifiers = vote(train, unit, scores, 1, groups, 2)
    return [u for u in train if u['unit_id'] in identifiers]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', type=Path, required=True)
    parser.add_argument('--clusters', type=Path, required=True)
    parser.add_argument('--credentials-file', type=Path, required=True)
    parser.add_argument('--baseline-runs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.criterion, args.limit = 'structure', 0
    if args.output.resolve().is_relative_to(experiment.ROOT):
        raise ValueError('Запросы и ответы должны оставаться вне Git')
    mapping = json.loads(args.clusters.read_text())
    groups = mapping.get('evaluation_cluster_by_unit', mapping.get('component_by_unit'))
    case = json.loads(args.case.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    for path in args.baseline_runs.glob('*.json'):
        record = json.loads(path.read_text())
        target = args.output / path.name
        if record['agent'] == case['agent'] and record['arm'] == 'baseline' and not target.exists():
            shutil.copyfile(path, target)
    def closest(train, unit, criterion, index):
        return select(train, unit, index.get_scores(answer_tokens(unit)), groups)
    with patch.object(experiment, 'tokens', answer_tokens), patch.object(experiment, 'anchors', closest):
        asyncio.run(experiment.run(args))


if __name__ == '__main__':
    logging.basicConfig(level=logging.ERROR)
    main()
