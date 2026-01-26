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
    p = argparse.ArgumentParser(description="Run HKUST (Teaser++/FGR/Quatro) baselines on paper3737 in shards.")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 16) // 2))
    p.add_argument("--chunk-size", type=int, default=64)
    p.add_argument(
        "--rotation-algs",
        type=str,
        default="GNC_TLS,FGR,QUATRO",
        help="Comma-separated rotation algorithms: GNC_TLS,FGR,QUATRO.",
    )
    p.add_argument(
        "--config",
        type=str,
        default="configs/hkust_lidar_global_paper3737_table3_ratio0p5.yaml",
        help="HKUST benchmark YAML config.",
    )
    p.add_argument(
        "--variant",
        type=str,
        default="ratio0p5",
        help=(
            "Suffix used in output folder names under outputs/hkust_teaser/ (e.g. 'ratio0p5', 'table3', 'nobeam'). "
            "This does not affect the algorithm; it's purely for bookkeeping."
        ),
    )
    p.add_argument("--tag-prefix", type=str, default=None, help="Prefix under outputs/hkust_teaser/tmp/ for shards.")
    p.add_argument(
        "--log-root",
        type=str,
        default="logs/hkust_teaser_shards_ratio0p5",
        help="Directory to store per-shard stdout/stderr logs (relative to repo root).",
    )
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


def _parse_alg_list(text: str) -> List[str]:
    out: List[str] = []
    for raw in str(text).split(","):
        token = raw.strip()
        if not token:
            continue
        out.append(token.upper())
    return out


def run_cmd(cmd: List[str], log_path: Path, *, env: dict, dry_run: bool) -> None:
    print(" ".join(cmd), flush=True)
    if dry_run:
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as f:
        subprocess.run(cmd, cwd=str(REPO_ROOT), env=env, stdout=f, stderr=subprocess.STDOUT, check=True)


def run_many(cmds: List[Tuple[List[str], Path]], *, env: dict, workers: int, dry_run: bool) -> None:
    if dry_run:
        for cmd, log_path in cmds:
            run_cmd(cmd, log_path, env=env, dry_run=True)
        return
    failures: List[Tuple[List[str], str]] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        future_map = {
            pool.submit(run_cmd, cmd, log_path, env=env, dry_run=False): cmd for cmd, log_path in cmds
        }
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
            f"{len(failures)}/{len(cmds)} HKUST shards failed; "
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


def main() -> None:
    args = parse_args()
    num_pairs = load_num_pairs()
    ranges = shard_ranges(num_pairs, int(args.chunk_size))
    algs = _parse_alg_list(args.rotation_algs)
    variant = str(args.variant).strip() or "variant"

    prefix = str(args.tag_prefix).strip() if args.tag_prefix else "sweep"
    log_root = (REPO_ROOT / str(args.log_root)).resolve()
    log_dir = log_root / prefix

    base_env = os.environ.copy()
    base_env.update({
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
        # cKDTree can use parallel queries; keep deterministic + avoid oversubscription.
        "V2XREG_KDTREE_WORKERS": "1",
    })

    cmds: List[Tuple[List[str], Path]] = []
    out_bases: List[str] = []
    for alg in algs:
        if alg not in {"GNC_TLS", "FGR", "QUATRO"}:
            raise ValueError(f"Unknown rotation alg: {alg}")
        alg_tag = "gnctls" if alg == "GNC_TLS" else alg.lower()
        out_base = f"hkust_teaser_paper3737_{alg_tag}_{variant}"
        out_bases.append(out_base)
        for shard_idx, (start, end) in enumerate(ranges):
            out_tag = f"tmp/{prefix}/{out_base}_chunk{shard_idx:03d}"
            out_metrics = REPO_ROOT / "outputs" / "hkust_teaser" / out_tag / "metrics.json"
            if (not bool(args.force)) and _metrics_complete(out_metrics, expected_frames=end - start):
                continue
            cmd = [
                sys.executable,
                str(REPO_ROOT / "benchmarks" / "hkust_lidar_global_registration_benchmark.py"),
                "--config",
                str(args.config),
                "--rotation-alg",
                alg,
                "--start",
                str(start),
                "--max-pairs",
                str(end - start),
                "--output-tag",
                out_tag,
            ]
            cmds.append((cmd, log_dir / f"{out_base}_chunk{shard_idx:03d}.log"))

    print(
        f"Total pairs: {num_pairs} | shards: {len(ranges)} | algs: {len(algs)} | "
        f"commands: {len(cmds)} | workers: {int(args.workers)} | prefix: {prefix}"
    )
    run_many(cmds, env=base_env, workers=int(args.workers), dry_run=bool(args.dry_run))
    if bool(args.dry_run):
        return

    for out_base in out_bases:
        aggregate(
            f"outputs/hkust_teaser/tmp/{prefix}/{out_base}_chunk*/matches.jsonl",
            REPO_ROOT / "outputs" / "hkust_teaser" / f"{out_base}_full",
            thresholds="1,2,3",
        )
    print("Done.")


if __name__ == "__main__":
    main()
