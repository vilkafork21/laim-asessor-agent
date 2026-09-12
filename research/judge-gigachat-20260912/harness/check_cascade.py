"""Воспроизводимость dev-каскада без сети и без новых ответов моделей."""
import asyncio
import json
from unittest.mock import patch

import run_cascade as cascade


def main() -> None:
    with patch('round4.api',side_effect=KeyboardInterrupt('Запрещена сеть в offline-проверке')) as direct, patch('sdk_round.Asessor.run',side_effect=KeyboardInterrupt('Запрещена сеть в offline-проверке')) as native:
        asyncio.run(cascade.run('dev'))
        assert direct.call_count==native.call_count==0
    expected=json.loads((cascade.OUT/'cascade-dev-predictions.json').read_text())
    for r in expected:
        for uid,score in zip(r['unit_ids'],r['prediction']):
            record=json.loads((cascade.OUT/'cascade-runs'/f'dev-{uid}.json').read_text())
            assert record['scores'][r['criterion']]==score
    print('PASS: весь dev-каскад совпал с replay; новых API-вызовов нет; балл 0 сохраняется')


if __name__=='__main__':
    main()
