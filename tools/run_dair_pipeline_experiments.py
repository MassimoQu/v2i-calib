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
        'tag': 'dair_v2xreg_oiou_gt15',
        'description': 'V2X-Reg (oIoU) with GT boxes, top-15',
        'overrides': {
            'filters.top_k': 15,
            'matching.core_components': ['iou'],
        },
    },
    {
        'tag': 'dair_v2xregpp_gt_inf',
        'description': 'V2X-Reg++ with GT boxes, keep all boxes',
        'overrides': {
            'filters.top_k': 0,
            'matching.parallel_flag': True,
            'matching.corresponding_parallel': True,
        },
    },
    {
        'tag': 'dair_v2xregpp_gt25',
        'description': 'V2X-Reg++ with GT boxes, top-25',
        'overrides': {
            'filters.top_k': 25,
        },
    },
    {
        'tag': 'dair_v2xregpp_gt15',
        'description': 'V2X-Reg++ with GT boxes, top-15',
        'overrides': {
            'filters.top_k': 15,
        },
    },
    {
        'tag': 'dair_v2xregpp_gt10',
        'description': 'V2X-Reg++ with GT boxes, top-10',
        'overrides': {
            'filters.top_k': 10,
        },
    },
    {
        'tag': 'dair_v2xregpp_pp15',
        'description': 'V2X-Reg++ with PointPillars detections, top-15',
        'overrides': {
            'filters.top_k': 15,
            'data.use_detection': True,
            'data.detection_cache': 'data/DAIR-V2X/detected/detected_boxes_test.json',
        },
    },
    {
        'tag': 'dair_v2xregpp_sc15',
        'description': 'V2X-Reg++ with SECOND detections, top-15',
        'overrides': {
            'filters.top_k': 15,
            'data.use_detection': True,
            'data.detection_cache': 'data/DAIR-V2X/detected/dairv2x-second_uncertainty/test/stage1_boxes.json',
        },
    },
    {
        'tag': 'dair_v2xregpp_gt25_hsvd',
        'description': 'V2X-Reg++ GT top-25 with highest-score SVD (hSVD)',
        'overrides': {
            'filters.top_k': 25,
            'matching.filter_strategy': 'topRetained',
        },
    },
    {
        'tag': 'dair_v2xregpp_gt25_msvd',
        'description': 'V2X-Reg++ GT top-25 with mean SVD (mSVD)',
        'overrides': {
            'filters.top_k': 25,
            'matching.matches2extrinsic': 'evenSVD',
        },
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
    parser.add_argument('--config', default='configs/pipeline.yaml', help='Base pipeline config path.')
    parser.add_argument(
        '--tags',
        nargs='*',
        help='Optional subset of experiment tags to run. Defaults to all configured experiments.',
    )
    args = parser.parse_args()

    results = {}
    selected = [exp for exp in EXPERIMENTS if not args.tags or exp['tag'] in args.tags]
    for exp in selected:
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
        print(f"- {tag}: {summary.get('mRRE@1.0', 0):.3f} deg, {summary.get('mRTE@1.0', 0):.3f} m")


if __name__ == '__main__':
    main()
