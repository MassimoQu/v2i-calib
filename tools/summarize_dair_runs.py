#!/usr/bin/env python3
"""
Summarize metrics.json outputs from the DAIR-V2X pipeline runs.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_dair_pipeline_experiments import EXPERIMENTS  # type: ignore  # pylint: disable=import-error


def load_metrics(tag: str) -> Optional[Dict]:
    path = ROOT / 'outputs' / tag / 'metrics.json'
    if not path.exists():
        return None
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)


def fmt_pct(value: Optional[float]) -> str:
    if value is None:
        return '-'
    return f"{value * 100:.2f}"


def fmt_float(value: Optional[float]) -> str:
    if value is None:
        return '-'
    return f"{value:.3f}"


def summarize() -> List[Tuple[str, str, Dict]]:
    rows: List[Tuple[str, str, Dict]] = []
    for exp in EXPERIMENTS:
        tag = exp['tag']
        metrics = load_metrics(tag)
        if metrics is None:
            continue
        rows.append((tag, exp['description'], metrics))
    return rows


def main() -> None:
    rows = summarize()
    if not rows:
        print('No metrics found for configured experiments.')
        return
    headers = [
        'Tag',
        'Description',
        'Succ@1m(%)',
        'Succ@2m(%)',
        'Succ@3m(%)',
        'mRE@1m',
        'mRE@2m',
        'mRE@3m',
        'mTE@1m',
        'mTE@2m',
        'mTE@3m',
        'AvgTime(s)',
    ]
    col_widths = [max(len(h), 12) for h in headers]
    lines = []
    lines.append(' | '.join(h.ljust(w) for h, w in zip(headers, col_widths)))
    lines.append('-+-'.join('-' * w for w in col_widths))
    for tag, desc, metrics in rows:
        line = [
            tag.ljust(col_widths[0]),
            desc.ljust(col_widths[1]),
            fmt_pct(metrics.get('success_at_1m')).rjust(col_widths[2]),
            fmt_pct(metrics.get('success_at_2m')).rjust(col_widths[3]),
            fmt_pct(metrics.get('success_at_3m')).rjust(col_widths[4]),
            fmt_float(metrics.get('mRE@1m')).rjust(col_widths[5]),
            fmt_float(metrics.get('mRE@2m')).rjust(col_widths[6]),
            fmt_float(metrics.get('mRE@3m')).rjust(col_widths[7]),
            fmt_float(metrics.get('mTE@1m')).rjust(col_widths[8]),
            fmt_float(metrics.get('mTE@2m')).rjust(col_widths[9]),
            fmt_float(metrics.get('mTE@3m')).rjust(col_widths[10]),
            fmt_float(metrics.get('avg_time')).rjust(col_widths[11]),
        ]
        lines.append(' | '.join(line))
    report_path = ROOT / 'outputs' / 'dair_metrics_summary.txt'
    report_path.write_text('\n'.join(lines), encoding='utf-8')
    print(f"Wrote summary to {report_path}")
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
