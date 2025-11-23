#!/usr/bin/env python3
"""Merge per-agent HEAL stage1 box exports into a two-agent record."""

from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping


def load_stage1(path: Path) -> Dict[str, Any]:
    if path.is_dir():
        path = path / 'stage1_boxes.json'
    if not path.exists():
        raise FileNotFoundError(f"Stage1 export not found: {path}")
    with path.open('r', encoding='utf-8') as f:
        raw = json.load(f)
    if not isinstance(raw, Mapping):
        raise ValueError(f"Stage1 export must be JSON object, got {type(raw)}")
    return {str(k): v for k, v in raw.items()}


def _extract_list(entry: MutableMapping[str, Any], key: str) -> list[Any]:
    value = entry.get(key)
    return value if isinstance(value, list) else []


def _extract_first(entry: MutableMapping[str, Any], key: str) -> Any:
    values = _extract_list(entry, key)
    return values[0] if values else []


def merge_records(infra: MutableMapping[str, Any], veh: MutableMapping[str, Any]) -> Dict[str, Any] | None:
    if infra is None or veh is None:
        return None
    merge = OrderedDict()
    infra_id = _extract_first(infra, 'cav_id_list')
    veh_id = _extract_first(veh, 'cav_id_list')
    merge['cav_id_list'] = [infra_id, veh_id]
    merge['pred_corner3d_np_list'] = [
        _extract_first(infra, 'pred_corner3d_np_list'),
        _extract_first(veh, 'pred_corner3d_np_list'),
    ]
    merge['uncertainty_np_list'] = [
        _extract_first(infra, 'uncertainty_np_list'),
        _extract_first(veh, 'uncertainty_np_list'),
    ]
    for pose_key in ('lidar_pose_clean_np', 'lidar_pose_np'):
        merge[pose_key] = [
            _extract_first(infra, pose_key),
            _extract_first(veh, pose_key),
        ]
    return merge


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge vehicle/infra stage1 exports.")
    parser.add_argument('--vehicle', required=True, help='Path to vehicle-only stage1 export directory or JSON file.')
    parser.add_argument('--infrastructure', required=True, help='Path to infrastructure-only stage1 export.')
    parser.add_argument('--output', required=True, help='Destination directory or JSON file for merged stage1 boxes.')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    veh_stage1 = load_stage1(Path(args.vehicle))
    infra_stage1 = load_stage1(Path(args.infrastructure))
    merged: Dict[str, Any] = {}
    for key in sorted(set(veh_stage1.keys()) & set(infra_stage1.keys()), key=lambda x: int(x)):
        record = merge_records(infra_stage1[key], veh_stage1[key])
        if record is None:
            continue
        merged[key] = record
    output_path = Path(args.output)
    if output_path.is_dir() or output_path.suffix == '':
        output_path.mkdir(parents=True, exist_ok=True)
        output_path = output_path / 'stage1_boxes.json'
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', encoding='utf-8') as f:
        json.dump(merged, f, indent=2)
    print(f"Wrote merged stage1 boxes for {len(merged)} samples to {output_path}")


if __name__ == '__main__':
    main()
