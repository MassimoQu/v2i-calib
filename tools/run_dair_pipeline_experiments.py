#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from calib.config import load_config
from calib.pipelines.object_level import ObjectLevelPipeline


EXPERIMENTS: List[Dict] = [
    {
        'tag': 'paper3737_v2xreg_oiou_gt15',
        'description': 'V2X-Reg (oIoU) with GT boxes, top-15',
        'overrides': {
            'filters.top_k': 15,
            'matching.core_components': ['iou'],
            'matching.filter_strategy': 'allRetained',
            'matching.filter_threshold': 0,
        },
    },
    {
        'tag': 'paper3737_v2xregpp_gt_inf',
        'description': 'V2X-Reg++ with GT boxes, keep all boxes',
        'overrides': {
            'filters.top_k': 0,
        },
    },
    {
        'tag': 'paper3737_v2xregpp_gt25',
        'description': 'V2X-Reg++ with GT boxes, top-25',
        'overrides': {
            'filters.top_k': 25,
        },
    },
    {
        'tag': 'paper3737_v2xregpp_gt15',
        'description': 'V2X-Reg++ with GT boxes, top-15',
        'overrides': {
            'filters.top_k': 15,
        },
    },
    {
        'tag': 'paper3737_v2xregpp_gt10',
        'description': 'V2X-Reg++ with GT boxes, top-10',
        'overrides': {
            'filters.top_k': 10,
        },
    },
    {
        'tag': 'paper3737_v2xregpp_gt25_hsvd',
        'description': 'V2X-Reg++ GT top-25 with highest-score SVD (hSVD)',
        'overrides': {
            'filters.top_k': 25,
            'matching.filter_strategy': 'topRetained',
        },
    },
    {
        'tag': 'paper3737_v2xregpp_gt25_msvd',
        'description': 'V2X-Reg++ GT top-25 with mean SVD (mSVD)',
        'overrides': {
            'filters.top_k': 25,
            'matching.matches2extrinsic': 'evenSVD',
        },
    },
    {
        'tag': 'paper3737_pp15_v2xregpp',
        'description': 'V2X-Reg++ with PointPillars detections, top-15',
        'overrides': {
            'filters.top_k': 15,
            'data.use_detection': True,
            'data.detection_cache': 'data/DAIR-V2X/detected/paper3737_pp_dual/stage1_boxes.json',
            'data.detection_field': 'pred_corner3d_np_list',
        },
        'requires_cache': 'data/DAIR-V2X/detected/paper3737_pp_dual/stage1_boxes.json',
    },
    {
        'tag': 'paper3737_sc15_v2xregpp',
        'description': 'V2X-Reg++ with SECOND detections, top-15',
        'overrides': {
            'filters.top_k': 15,
            'data.use_detection': True,
            'data.detection_cache': 'data/DAIR-V2X/detected/paper3737_sc_dual/stage1_boxes.json',
            'data.detection_field': 'pred_corner3d_np_list',
        },
        'requires_cache': 'data/DAIR-V2X/detected/paper3737_sc_dual/stage1_boxes.json',
    },
]


def apply_overrides(cfg, overrides: Dict[str, object]) -> None:
    for path, value in overrides.items():
        parts = path.split('.')
        target = cfg
        for attr in parts[:-1]:
            target = getattr(target, attr)
        setattr(target, parts[-1], value)


def main() -> None:
    parser = argparse.ArgumentParser(description='Run DAIR-V2X pipeline experiments sequentially.')
    parser.add_argument(
        '--config',
        default='configs/pipeline_paper3737.yaml',
        help=(
            'Base pipeline config path (recommended: configs/pipeline_paper3737.yaml for Table III GT sweeps; '
            'use configs/pipeline_top3000.yaml for quick iteration).'
        ),
    )
    parser.add_argument(
        '--tags',
        nargs='*',
        help='Optional subset of experiment tags to run. Defaults to all configured experiments.',
    )
    args = parser.parse_args()

    results = {}
    selected = [exp for exp in EXPERIMENTS if not args.tags or exp['tag'] in args.tags]
    for exp in selected:
        cache_hint = exp.get('requires_cache')
        if cache_hint:
            cache_path = Path(cache_hint)
            if not cache_path.is_absolute():
                cache_path = ROOT / cache_hint
            if not cache_path.exists():
                print(f"[SKIP] {exp['tag']} missing cache: {cache_path}")
                continue
        cfg = load_config(args.config)
        apply_overrides(cfg, {'output.tag': exp['tag']})
        apply_overrides(cfg, exp['overrides'])
        print(f"\n[RUN] {exp['tag']}: {exp['description']}")
        pipeline = ObjectLevelPipeline(cfg)
        summary = pipeline.run()
        results[exp['tag']] = summary
        print(json.dumps(summary, indent=2))

    print('\nAll experiments completed:')
    for tag, summary in results.items():
        succ1 = summary.get('success_at_1m')
        succ2 = summary.get('success_at_2m')
        avg_time = summary.get('avg_time')
        print(f"- {tag}: success@1m={succ1} success@2m={succ2} avg_time={avg_time}")


if __name__ == '__main__':
    main()
