#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run ICP/PICP sweeps on paper3737 and aggregate paper-style metrics.")
    p.add_argument("--workers", type=int, default=os.cpu_count() or 8)
    p.add_argument("--chunk-size", type=int, default=256)
    p.add_argument("--methods", type=str, default="icp,picp")
    p.add_argument("--noises", type=str, default="0,1,2")
    p.add_argument("--voxel", type=float, default=0.3)
    p.add_argument("--max-corr", type=float, default=1.0)
    p.add_argument("--max-iter", type=int, default=100)
    p.add_argument(
        "--project-config",
        type=str,
        default="configs/hkust_lidar_global_paper3737.yaml",
        help="Project config that defines DAIR paths; values can be overridden via --data-root/--data-info.",
    )
    p.add_argument("--beam-align-infra", action="store_true", help="Enable infra beam alignment before ICP.")
    p.add_argument("--init-source", type=str, default="unadjusted", choices=["gt", "unadjusted"])
    p.add_argument("--noise-mode", type=str, default="left", choices=["left", "right"])
    p.add_argument("--seed", type=int, default=2025)
    p.add_argument("--tag-prefix", type=str, default=None,
                   help="Folder prefix under outputs/baselines/ (default: auto).")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def load_num_pairs() -> int:
    data_info = REPO_ROOT / "data" / "data_info_dair_paper3737.json"
    items = json.loads(data_info.read_text(encoding="utf-8"))
    if not isinstance(items, list):
        raise ValueError(f"Expected list in {data_info}")
    return len(items)


def shard_ranges(total: int, chunk: int) -> List[Tuple[int, int]]:
    ranges: List[Tuple[int, int]] = []
    start = 0
    while start < total:
        end = min(total, start + chunk)
        ranges.append((start, end))
        start = end
    return ranges


def run_cmd(cmd: List[str], *, env: dict, dry_run: bool) -> None:
    print(" ".join(cmd), flush=True)
    if dry_run:
        return
    subprocess.run(cmd, cwd=str(REPO_ROOT), env=env, check=True)


def run_many(cmds: List[List[str]], *, env: dict, workers: int, dry_run: bool) -> None:
    if dry_run:
        for cmd in cmds:
            run_cmd(cmd, env=env, dry_run=True)
        return
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(run_cmd, cmd, env=env, dry_run=False) for cmd in cmds]
        for fut in as_completed(futures):
            fut.result()


def aggregate(pattern: str, output_dir: Path, *, thresholds: str = "1,2,3", merged_name: str = "details.jsonl") -> None:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "tools" / "aggregate_jsonl_metrics.py"),
        "--inputs",
        pattern,
        "--output-dir",
        str(output_dir),
        "--thresholds",
        thresholds,
        "--merged-name",
        merged_name,
    ]
    subprocess.run(cmd, cwd=str(REPO_ROOT), env=os.environ.copy(), check=True)


def _metrics_complete(path: Path, *, expected_frames: int) -> bool:
    if not path.is_file():
        return False
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    try:
        return int(obj.get("num_frames", -1)) == int(expected_frames)
    except Exception:
        return False


def main() -> None:
    args = parse_args()
    num_pairs = load_num_pairs()
    ranges = shard_ranges(num_pairs, int(args.chunk_size))
    methods = [m.strip() for m in str(args.methods).split(",") if m.strip()]
    noises = [int(x.strip()) for x in str(args.noises).split(",") if x.strip()]

    voxel = float(args.voxel)
    max_corr = float(args.max_corr)
    max_iter = int(args.max_iter)

    prefix = args.tag_prefix
    if not prefix:
        prefix = f"paper3737_icp_sweep_vx{voxel:g}_mc{max_corr:g}_it{max_iter}"

    base_env = os.environ.copy()
    base_env.update({
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
    })

    cmds: List[List[str]] = []
    for method in methods:
        for noise in noises:
            for shard_idx, (start, end) in enumerate(ranges):
                out_tag = f"baselines/tmp/{prefix}/{method}_noise{noise}_chunk{shard_idx:03d}"
                out_metrics = REPO_ROOT / "outputs" / out_tag / "metrics.json"
                if (not bool(args.force)) and _metrics_complete(out_metrics, expected_frames=end - start):
                    continue
                cmd = [
                    sys.executable,
                    str(REPO_ROOT / "benchmarks" / "run_dair_lidar_benchmark.py"),
                    "--project-config",
                    str(args.project_config),
                    "--data-info",
                    "data/data_info_dair_paper3737.json",
                    "--data-root",
                    "data/DAIR-V2X/cooperative-vehicle-infrastructure",
                    "--start",
                    str(start),
                    "--end",
                    str(end),
                    "--method",
                    method,
                    "--init-source",
                    str(args.init_source),
                    "--trans-noise",
                    str(noise),
                    "--rot-noise-deg",
                    str(noise),
                    "--noise-mode",
                    str(args.noise_mode),
                    "--voxel",
                    str(voxel),
                    "--max-corr",
                    str(max_corr),
                    "--max-iter",
                    str(max_iter),
                    "--output-root",
                    "outputs",
                    "--output-tag",
                    out_tag,
                    "--seed",
                    str(int(args.seed)),
                ]
                if bool(args.beam_align_infra):
                    cmd += ["--beam-align-infra"]
                cmds.append(cmd)

    print(f"Total pairs: {num_pairs} | shards: {len(ranges)} | commands: {len(cmds)} | prefix: {prefix}")
    run_many(cmds, env=base_env, workers=int(args.workers), dry_run=bool(args.dry_run))
    if args.dry_run:
        return

    for method in methods:
        for noise in noises:
            aggregate(
                f"outputs/baselines/tmp/{prefix}/{method}_noise{noise}_chunk*/details.jsonl",
                REPO_ROOT / "outputs" / "baselines" / f"{prefix}/{method}_noise{noise}",
                thresholds="1,2,3",
                merged_name="details.jsonl",
            )
    print("Done.")


if __name__ == "__main__":
    main()
