#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List


def load_records(path: Path) -> List[Dict]:
    records: List[Dict] = []
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            records.append(json.loads(line))
    return records


def summarize(records: List[Dict], success_thr: float) -> Dict[str, Dict[str, float]]:
    def _avg(key: str, subset: List[Dict]) -> float:
        return sum(r['timing'][key] for r in subset) / len(subset) if subset else 0.0

    success = [r for r in records if r['TE'] < success_thr]
    fail = [r for r in records if r['TE'] >= success_thr]
    total = len(records)
    return {
        'total_frames': total,
        'success_count': len(success),
        'fail_count': len(fail),
        'avg_timing_all': {
            'filter': _avg('filter', records),
            'matching': _avg('matching', records),
            'solver': _avg('solver', records),
            'total': _avg('total', records),
        },
        'avg_timing_success': {
            'filter': _avg('filter', success),
            'matching': _avg('matching', success),
            'solver': _avg('solver', success),
            'total': _avg('total', success),
        },
        'avg_timing_fail': {
            'filter': _avg('filter', fail),
            'matching': _avg('matching', fail),
            'solver': _avg('solver', fail),
            'total': _avg('total', fail),
        },
    }


def main():
    parser = argparse.ArgumentParser(description='Summarize stage timing from matches log')
    parser.add_argument('--matches', required=True)
    parser.add_argument('--success-thr', type=float, default=1.0)
    args = parser.parse_args()

    path = Path(args.matches)
    records = load_records(path)
    report = summarize(records, args.success_thr)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
