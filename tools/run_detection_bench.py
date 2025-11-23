#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import List

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from calib.config import load_config, PipelineConfig
from calib.pipelines.object_level import ObjectLevelPipeline


def slugify(path: Path, root: Path) -> str:
    rel = path.relative_to(root)
    parts = list(rel.with_suffix('').parts)
    return '-'.join(parts)


def run_pipeline(cfg: PipelineConfig) -> dict:
    pipeline = ObjectLevelPipeline(cfg)
    return pipeline.run()


def main():
    parser = argparse.ArgumentParser(description='Run detection benchmarks for all cached files')
    parser.add_argument('--config', default='configs/pipeline_detection.yaml')
    parser.add_argument('--det-root', default='data/DAIR-V2X/detected')
    parser.add_argument('--max-files', type=int, default=None)
    args = parser.parse_args()

    det_root = Path(args.det_root).resolve()
    config_path = Path(args.config)

    json_paths: List[Path] = sorted([p for p in det_root.rglob('*.json') if p.is_file()])
    if args.max_files is not None:
        json_paths = json_paths[: args.max_files]

    summaries = []
    for idx, det_path in enumerate(json_paths, 1):
        slug = slugify(det_path, det_root)
        print(f"[{idx}/{len(json_paths)}] Running detection cache: {det_path}")
        cfg = load_config(config_path)
        cfg.data.detection_cache = str(det_path)
        cfg.data.use_detection = True
        cfg.data.max_samples = 1000  # manageable subset for benchmarking
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        cfg.output.tag = f"det-{slug}-{timestamp}"
        result = run_pipeline(cfg)
        result['detection_source'] = slug
        result['output_tag'] = cfg.output.tag
        summaries.append(result)

    summary_path = Path('outputs/detection_bench_summary.json')
    summary_path.write_text(json.dumps({'results': summaries}, indent=2), encoding='utf-8')
    print(f"Saved summary to {summary_path}")


if __name__ == '__main__':
    main()
