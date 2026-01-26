#!/usr/bin/env python3
"""
Summarize detection cache statistics (frame counts, per-agent box counts).

Example:
    python tools/summarize_detection_cache.py \\
        --path data/DAIR-V2X/detected/veh_rsu_dual_detection_cache.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Sequence


def _load_cache(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Detection cache not found: {path}")
    with path.open('r', encoding='utf-8') as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Detection cache must be a dict, got {type(data)}")
    return data


def _box_count(entry: Sequence[Any]) -> int:
    if not isinstance(entry, list):
        return 0
    return sum(1 for box in entry if box is not None)


def summarize(cache: Dict[str, Any], field: str) -> Dict[str, Any]:
    stats = {
        'total_records': len(cache),
        'non_null_records': 0,
        'both_agents_present': 0,
        'both_with_boxes': 0,
        'veh_counts': [],
        'infra_counts': [],
        'shared_min_counts': [],
    }
    for key in sorted(cache.keys(), key=lambda x: int(x) if str(x).isdigit() else x):
        entry = cache[key]
        if not isinstance(entry, dict):
            continue
        stats['non_null_records'] += 1
        pred_list = entry.get(field)
        if not isinstance(pred_list, list) or len(pred_list) < 2:
            continue
        stats['both_agents_present'] += 1
        infra_count = _box_count(pred_list[0])
        veh_count = _box_count(pred_list[1])
        stats['infra_counts'].append(infra_count)
        stats['veh_counts'].append(veh_count)
        stats['shared_min_counts'].append(min(infra_count, veh_count))
        if infra_count > 0 and veh_count > 0:
            stats['both_with_boxes'] += 1
    return stats


def _fmt_basic(values: List[int]) -> str:
    if not values:
        return "0 (avg=0.0, min=0, max=0)"
    return f"{len(values)} (avg={mean(values):.2f}, min={min(values)}, max={max(values)})"


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize detection cache stats.")
    parser.add_argument('--path', required=True, help='Path to detection cache JSON file.')
    parser.add_argument('--field', default='pred_corner3d_np_list',
                        help='Field name containing per-agent detections.')
    args = parser.parse_args()

    cache = _load_cache(Path(args.path))
    stats = summarize(cache, args.field)

    print(f"Total records: {stats['total_records']}")
    print(f"Non-null entries: {stats['non_null_records']}")
    print(f"Records with both agents: {stats['both_agents_present']}")
    print(f"Frames where both agents have >=1 boxes: {stats['both_with_boxes']}")
    print(f"Infrastructure box counts: {_fmt_basic(stats['infra_counts'])}")
    print(f"Vehicle box counts: {_fmt_basic(stats['veh_counts'])}")
    print(f"Min shared boxes per frame (infra/veh): {_fmt_basic(stats['shared_min_counts'])}")


if __name__ == '__main__':
    main()
