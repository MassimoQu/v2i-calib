#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]


def _tag_float(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    s = f"{float(value):.4f}".rstrip("0").rstrip(".")
    return s.replace(".", "p")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run VIPS sweeps on paper3737 and aggregate paper-style metrics.")
    p.add_argument("--workers", type=int, default=os.cpu_count() or 8, help="Max concurrent subprocesses.")
    p.add_argument("--chunk-size", type=int, default=64, help="Pairs per shard.")
    p.add_argument("--noises", type=str, default="0,1,2", help="Comma-separated noise std values (m & deg).")
    p.add_argument(
        "--match-distance-thr",
        type=str,
        default="1.5",
        help="Comma-separated distance gates in meters; use 'none' to disable the gate.",
    )
    p.add_argument("--init-source", type=str, default="gt", choices=["gt", "unadjusted"])
    p.add_argument("--skip-icp", action="store_true", help="Skip point cloud ICP refinement.")
    p.add_argument("--top-k", type=int, default=0, help="Optionally keep only the top-K largest boxes.")
    p.add_argument("--seed", type=int, default=2025)
    p.add_argument("--tag-prefix", type=str, default=None, help="Optional prefix under outputs/vips/tmp/ for shards.")
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
    failures: List[Tuple[List[str], str]] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        future_map = {pool.submit(run_cmd, cmd, env=env, dry_run=False): cmd for cmd in cmds}
        for fut in as_completed(future_map):
            cmd = future_map[fut]
            try:
                fut.result()
            except subprocess.CalledProcessError as exc:
                failures.append((cmd, f"CalledProcessError(rc={exc.returncode})"))
                print(f"[ERROR] subprocess failed (rc={exc.returncode}): {' '.join(cmd)}", file=sys.stderr, flush=True)
            except Exception as exc:  # pragma: no cover
                failures.append((cmd, f"{type(exc).__name__}: {exc}"))
                print(f"[ERROR] subprocess failed: {' '.join(cmd)} | {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
    if failures:
        first_cmd, first_err = failures[0]
        raise RuntimeError(
            f"{len(failures)}/{len(cmds)} VIPS shards failed; "
            f"first error: {first_err} | cmd={' '.join(first_cmd)}"
        )


def aggregate(pattern: str, output_dir: Path, *, thresholds: str = "1,2,3") -> None:
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
        "matches.jsonl",
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


def _parse_int_list(text: str) -> List[int]:
    out: List[int] = []
    for token in str(text).split(","):
        token = token.strip()
        if not token:
            continue
        out.append(int(token))
    return out


def _parse_thr_list(text: str) -> List[Optional[float]]:
    out: List[Optional[float]] = []
    for token in str(text).split(","):
        token = token.strip().lower()
        if not token:
            continue
        if token in {"none", "null", "na", "-"}:
            out.append(None)
        else:
            out.append(float(token))
    return out


def main() -> None:
    args = parse_args()
    num_pairs = load_num_pairs()
    ranges = shard_ranges(num_pairs, int(args.chunk_size))
    noises = _parse_int_list(args.noises)
    thr_list = _parse_thr_list(args.match_distance_thr)

    prefix = str(args.tag_prefix).strip() if args.tag_prefix else "sweep"
    init_tag = "initgt" if str(args.init_source) == "gt" else "initunadj"
    icp_tag = "noicp" if bool(args.skip_icp) else "icp"
    topk = int(args.top_k)
    topk_tag = f"_top{topk}" if topk > 0 else ""

    base_env = os.environ.copy()
    base_env.update({
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
        # VIPS uses numba-parallel kernels; cap threads to avoid oversubscription.
        "NUMBA_NUM_THREADS": "1",
        # Let SciPy cKDTree queries opt into parallel mode explicitly.
        "V2XREG_KDTREE_WORKERS": "1",
    })

    cmds: List[List[str]] = []
    for noise in noises:
        for thr in thr_list:
            thr_tag = "thrnone" if thr is None else f"thr{_tag_float(float(thr))}"
            out_base = f"paper3737_noise{noise}_{thr_tag}_{init_tag}_{icp_tag}{topk_tag}"
            for shard_idx, (start, end) in enumerate(ranges):
                out_tag = f"tmp/{prefix}/{out_base}_chunk{shard_idx:03d}"
                out_metrics = REPO_ROOT / "outputs" / "vips" / out_tag / "metrics.json"
                if (not bool(args.force)) and _metrics_complete(out_metrics, expected_frames=end - start):
                    continue
                cmd = [
                    sys.executable,
                    str(REPO_ROOT / "benchmarks" / "run_vips_benchmark.py"),
                    "--config",
                    "configs/pipeline_paper3737_gt15_seedrefine5.yaml",
                    "--start",
                    str(start),
                    "--max-pairs",
                    str(end - start),
                    "--output-tag",
                    out_tag,
                    "--init-source",
                    str(args.init_source),
                    "--trans-noise",
                    str(noise),
                    "--rot-noise-deg",
                    str(noise),
                    "--seed",
                    str(int(args.seed)),
                ]
                if topk > 0:
                    cmd += ["--top-k", str(topk)]
                if thr is not None:
                    cmd += ["--match-distance-thr", str(float(thr))]
                else:
                    # run_vips_benchmark default is 8.0; explicitly disable the gate by setting a huge threshold.
                    cmd += ["--match-distance-thr", "1e9"]
                if bool(args.skip_icp):
                    cmd += ["--skip-icp"]
                cmds.append(cmd)

    total_jobs = len(noises) * len(thr_list)
    print(f"Total pairs: {num_pairs} | shards: {len(ranges)} | jobs: {total_jobs} | commands: {len(cmds)} | prefix: {prefix}")
    run_many(cmds, env=base_env, workers=int(args.workers), dry_run=bool(args.dry_run))
    if args.dry_run:
        return

    for noise in noises:
        for thr in thr_list:
            thr_tag = "thrnone" if thr is None else f"thr{_tag_float(float(thr))}"
            out_base = f"paper3737_noise{noise}_{thr_tag}_{init_tag}_{icp_tag}{topk_tag}"
            aggregate(
                f"outputs/vips/tmp/{prefix}/{out_base}_chunk*/matches.jsonl",
                REPO_ROOT / "outputs" / "vips" / out_base,
                thresholds="1,2,3",
            )
    print("Done.")


if __name__ == "__main__":
    main()
