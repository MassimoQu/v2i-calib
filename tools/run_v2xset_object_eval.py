#!/usr/bin/env python3
import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import List

import numpy as np

from calib.config import load_config
from calib.evaluation.metrics import FrameMetrics, aggregate_metrics
from calib.filters.pipeline import FilterPipeline
from calib.matching.engine import MatchingEngine
from calib.solvers.svd import ExtrinsicSolver
from legacy.v2x_calib.reader import V2XSetReader


def ensure_output_dir(tag: str, root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    out_dir = root / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def run_experiment(cfg_path: str, args) -> dict:
    cfg = load_config(cfg_path)
    if args.top_k is not None:
        cfg.filters.top_k = int(args.top_k)
    if args.core_components:
        cfg.matching.core_components = list(args.core_components)
    if args.output_tag:
        cfg.output.tag = args.output_tag
    if args.filter_strategy:
        cfg.matching.filter_strategy = args.filter_strategy
    if args.matches2extrinsic:
        cfg.matching.matches2extrinsic = args.matches2extrinsic

    filters = FilterPipeline(cfg.filters)
    matcher = MatchingEngine(cfg.matching)
    solver = ExtrinsicSolver(cfg.matching, cfg.solver)

    reader = V2XSetReader(
        root_dir=args.root,
        split=args.split,
        frame_stride=args.frame_stride,
        max_cavs=args.max_cavs,
    )
    max_pairs = args.max_pairs if args.max_pairs is None else int(args.max_pairs)
    records: List[FrameMetrics] = []
    total_pairs = 0
    noise_kwargs = {
        "pos_std": args.noise_pos_std,
        "rot_std": np.radians(args.noise_rot_std_deg),
        "pos_mean": args.noise_pos_mean,
        "rot_mean": np.radians(args.noise_rot_mean_deg),
    }
    if args.noise_type is None:
        noise_kwargs = None
    for frame_idx, pair_id, infra_boxes, veh_boxes, T_true in reader.generate_vehicle_vehicle_bboxes_object_list(
        start_idx=args.start_idx,
        end_idx=args.end_idx,
        noise_type=args.noise_type,
        noise=noise_kwargs,
    ):
        if max_pairs is not None and total_pairs >= max_pairs:
            break
        infra_f, veh_f = filters.apply(infra_boxes, veh_boxes)
        matches, stability = matcher.compute(infra_f, veh_f, T_eval=T_true)
        try:
            T6, RE, TE = solver.solve(infra_f, veh_f, matches, T_true)
        except Exception as exc:  # pragma: no cover - diagnostic path
            print(f"[WARN] solver failure on frame {frame_idx} pair {pair_id}: {exc}", file=sys.stderr)
            traceback.print_exc()
            continue
        records.append(
            FrameMetrics(
                infra_id=pair_id,
                veh_id=pair_id,
                RE=RE,
                TE=TE,
                stability=stability,
                time_cost=0.0,
            )
        )
        total_pairs += 1
    summary = aggregate_metrics(records, cfg.evaluation.success_thresholds)
    out_root = Path(cfg.output.root_dir)
    tag = cfg.output.tag or args.output_tag or "v2xset_object_eval"
    out_dir = ensure_output_dir(tag, out_root)
    with (out_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def main():
    parser = argparse.ArgumentParser(description="Run object-level calibration on V2X-Set.")
    parser.add_argument("--config", type=str, default="configs/pipeline.yaml")
    parser.add_argument("--root", type=str, default="/mnt/ssd_gw/cooperative-vehicle-infrastructure/v2xset")
    parser.add_argument("--split", type=str, default="validate")
    parser.add_argument("--frame-stride", type=int, default=10)
    parser.add_argument("--max-cavs", type=int, default=3)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--core-components", nargs="+", default=None)
    parser.add_argument("--filter-strategy", type=str, default=None)
    parser.add_argument("--max-pairs", type=int, default=200)
    parser.add_argument("--start-idx", type=int, default=0)
    parser.add_argument("--end-idx", type=int, default=None)
    parser.add_argument("--output-tag", type=str, default=None)
    parser.add_argument("--matches2extrinsic", type=str, default=None,
                        help="Optional override for matching.matches2extrinsic (e.g., evenSVD).")
    parser.add_argument("--noise-type", type=str, default=None,
                        choices=["gaussian", "laplace", "von_mises"])
    parser.add_argument("--noise-pos-std", type=float, default=0.0)
    parser.add_argument("--noise-rot-std-deg", type=float, default=0.0)
    parser.add_argument("--noise-pos-mean", type=float, default=0.0)
    parser.add_argument("--noise-rot-mean-deg", type=float, default=0.0)
    args = parser.parse_args()
    summary = run_experiment(args.config, args)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
