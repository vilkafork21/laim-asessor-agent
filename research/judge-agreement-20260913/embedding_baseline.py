"""Диагностический baseline локальной Giga-Embeddings; не production judge."""
from __future__ import annotations

import argparse
import hashlib
import json
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def digest(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def normalized_answer(unit: dict) -> str:
    return ' '.join(unicodedata.normalize('NFKC', unit['context']['current_turn'].get('output_answer') or '').casefold().split())


def pool(vectors: list[list[float]], weights: list[int]) -> np.ndarray:
    result = np.average(vectors, axis=0, weights=weights)
    length = np.linalg.norm(result)
    if not np.isfinite(result).all() or length == 0:
        raise ValueError('Нельзя нормировать embedding: нулевая норма или неконечные значения')
    return result / length


def record_path(output: Path, unit: dict) -> Path:
    return output / (hashlib.sha256(unit['unit_id'].encode()).hexdigest()+'.json')


def embed(args: argparse.Namespace, case: dict) -> None:
    import torch
    from importlib.metadata import version
    from sentence_transformers import SentenceTransformer
    torch.set_num_threads(2)
    model = SentenceTransformer(str(args.model_directory), trust_remote_code=True, local_files_only=True)
    model.max_seq_length = 512
    metadata = {'case_sha256': digest(args.case), 'chunk_tokens': 480, 'max_seq_length': 512,
                'pooling': 'Среднее нормированных chunk-векторов с весом по числу токенов, затем L2',
                'model_files_sha256': {str(p.relative_to(args.model_directory)): digest(p) for p in sorted(args.model_directory.rglob('*'))
                                      if p.is_file() and p.suffix in ['.json', '.py', '.safetensors']},
                'versions': {name: version(name) for name in ['torch', 'transformers', 'sentence-transformers']}}
    units = sorted([u for u in case['units'] if u['partition'] in ['train', 'dev']], key=lambda u: u['unit_id'])
    args.output.mkdir(parents=True, exist_ok=True)
    for number, unit in enumerate(units, 1):
        path = record_path(args.output, unit)
        if path.exists():
            if json.loads(path.read_text())['metadata'] != metadata:
                raise ValueError('Кеш относится к другой версии модели, данных или предобработки')
            continue
        text = unit['context']['current_turn'].get('output_answer') or ''
        ids = model.tokenizer.encode(text, add_special_tokens=False)
        parts = [ids[i:i+480] for i in range(0, len(ids), 480)] or [[]]
        chunks = [model.tokenizer.decode(part, skip_special_tokens=False, clean_up_tokenization_spaces=False) for part in parts] if len(parts) > 1 else [text]
        if any(len(model.tokenizer.encode(chunk)) > 512 for chunk in chunks):
            raise ValueError('Chunk превышает предел tokenizer: молчаливое усечение запрещено')
        vectors = model.encode(chunks, batch_size=4, normalize_embeddings=True, show_progress_bar=False)
        vector = pool(vectors, [max(1, len(part)) for part in parts])
        record = {'unit_id': unit['unit_id'], 'partition': unit['partition'], 'metadata': metadata,
                  'token_count': len(ids), 'chunks': len(chunks), 'embedding': vector.tolist()}
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(record, ensure_ascii=False, allow_nan=False))
        temporary.replace(path)
        print(case['agent'], number, '/', len(units), 'chunks', len(chunks), flush=True)


def evaluate(args: argparse.Namespace, case: dict) -> None:
    from sklearn.linear_model import LogisticRegression
    from feature_calibration import training_units, consensus
    from audit_agreement import audit
    from analyze_anchors import coefficients
    mapping = json.loads(args.clusters.read_text())
    groups = mapping.get('evaluation_cluster_by_unit', mapping.get('component_by_unit'))
    dev = sorted([u for u in case['units'] if u['partition'] == 'dev'], key=lambda u: u['unit_id'])
    train = training_units(case['units'], groups)
    before = len(train)
    exposed_answers = {normalized_answer(u) for u in dev}
    train = [u for u in train if normalized_answer(u) not in exposed_answers]
    case_sha256 = digest(args.case)
    def matrix(units):
        records = [json.loads(record_path(args.output, u).read_text()) for u in units]
        if any(r['metadata']['case_sha256'] != case_sha256 for r in records):
            raise ValueError('Сохранённые векторы относятся к другой корзине')
        return np.array([r['embedding'] for r in records])
    x, query = matrix(train), matrix(dev)
    y = np.array([consensus(u, 'structure') for u in train])
    human = np.array([consensus(u, 'structure') for u in dev], dtype=float)
    prior = Counter(y)
    mode = min(prior, key=lambda value: (-prior[value], value))
    predictions = {'train_mode': [float(mode)]*len(dev)}
    nearest = np.argsort(-(query @ x.T), axis=1, kind='stable')
    for k in [1, 3]:
        values = []
        for indices in nearest[:, :k]:
            counts = Counter(y[indices])
            values.append(float(min(counts, key=lambda value: (-counts[value], value != mode, value))))
        predictions[f'cosine_k{k}'] = values
    for name, weight in [('unweighted', None), ('balanced', 'balanced')]:
        model = LogisticRegression(C=10, class_weight=weight, max_iter=1000, random_state=20260913).fit(x, y)
        predictions[f'logistic_{name}'] = model.predict(query).tolist()
    rows = []
    for name, prediction in predictions.items():
        row = dict(audit(human.tolist(), prediction), profile=name)
        known = np.isfinite(human)
        metrics = coefficients(human[known], np.array(prediction)[known])
        row.update({k: metrics[k] for k in ['defect_recall', 'defect_false_positive_rate', 'defect_confusion']})
        rows.append(row)
        print(case['agent'], name, {k: row[k] for k in ['cohen_kappa', 'krippendorff_alpha_ordinal', 'spearman_correlation', 'defect_recall', 'defect_false_positive_rate']})
    result = {'agent': case['agent'], 'rows': rows, 'train_after_client_exclusion': before,
              'train_after_exact_answer_exclusion': len(train), 'dev_units': len(dev),
              'source_sha256': {p.name: digest(p) for p in [args.case, args.clusters]},
              'unit_ids': [u['unit_id'] for u in dev], 'predictions': predictions,
              'scope': 'Диагностика на локальной Giga-Embeddings; эквивалентность GigaChat API не установлена. Все клиенты dev и совпадающие ответы исключены из train.'}
    (args.output/'metrics.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['embed', 'evaluate'], required=True)
    for name in ['case', 'clusters', 'model-directory', 'output']:
        parser.add_argument('--'+name, type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.output.resolve().is_relative_to(ROOT):
        raise ValueError('Векторы и подробные оценки должны оставаться вне Git')
    case = json.loads(arguments.case.read_text())
    (embed if arguments.stage == 'embed' else evaluate)(arguments, case)
