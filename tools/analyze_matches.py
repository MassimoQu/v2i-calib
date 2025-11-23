#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List


def load_matches(path: Path) -> List[Dict]:
    records: List[Dict] = []
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            records.append(json.loads(line))
    return records


def compute_metrics(records: List[Dict], thresholds: List[float]) -> Dict[str, Dict[str, float]]:
    summary: Dict[str, Dict[str, float]] = {}
    total = len(records)
    for thr in thresholds:
        key = f"@{thr}m"
        subset = [r for r in records if r['TE'] < thr]
        if subset:
            m_re = sum(r['RE'] for r in subset) / len(subset)
            m_te = sum(r['TE'] for r in subset) / len(subset)
        else:
            m_re = None
            m_te = None
        summary[key] = {
            'success_rate': len(subset) / total if total else 0.0,
            'success_frames': len(subset),
            'mRE': m_re,
            'mTE': m_te,
        }
    return summary


def main():
    parser = argparse.ArgumentParser(description='Analyze matches.jsonl metrics at multiple thresholds')
    parser.add_argument('--matches', required=True, help='Path to matches.jsonl')
    parser.add_argument('--max-threshold', type=int, default=5)
    args = parser.parse_args()

    path = Path(args.matches)
    records = load_matches(path)
    thresholds = [float(i) for i in range(1, args.max_threshold + 1)]
    metrics = compute_metrics(records, thresholds)
    print(json.dumps({'total_frames': len(records), 'threshold_metrics': metrics}, indent=2))


if __name__ == '__main__':
    main()
