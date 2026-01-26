#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Iterable, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Measure single-process runtimes for Table III methods and patch canonical metrics.json avg_time.\n"
            "This avoids inflated runtimes caused by running many shards concurrently."
        )
    )
    p.add_argument("--num-frames", type=int, default=500, help="How many frames to time per method (default: 500).")
    p.add_argument("--start", type=int, default=0, help="Start index within paper3737 (default: 0).")
    p.add_argument("--threads", type=int, default=8, help="Thread count for BLAS/numba/OMP (default: 8).")
    p.add_argument("--dry-run", action="store_true", help="Print commands without executing.")
    p.add_argument("--force", action="store_true", help="Rerun timing even if timing metrics already exist.")
    return p.parse_args()


def run_cmd(cmd: List[str], *, env: dict, dry_run: bool) -> None:
    print(" ".join(cmd), flush=True)
    if dry_run:
        return
    subprocess.run(cmd, cwd=str(REPO_ROOT), env=env, check=True)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _metrics_exists(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > 0
    except FileNotFoundError:
        return False


def _patch_time(target_metrics: Path, timing_metrics: Path, *, dry_run: bool) -> None:
    cmd = [
        "python",
        str(REPO_ROOT / "tools" / "override_metrics_time.py"),
        "--metrics",
        str(target_metrics),
        "--time-metrics",
        str(timing_metrics),
    ]
    run_cmd(cmd, env=os.environ.copy(), dry_run=dry_run)


def main() -> None:
    args = parse_args()
    start = int(args.start)
    num_frames = int(args.num_frames)
    end = start + num_frames

    timing_env = os.environ.copy()
    threads = max(1, int(args.threads))
    timing_env.update({
        "OMP_NUM_THREADS": str(threads),
        "OPENBLAS_NUM_THREADS": str(threads),
        "MKL_NUM_THREADS": str(threads),
        "NUMEXPR_NUM_THREADS": str(threads),
        "VECLIB_MAXIMUM_THREADS": str(threads),
        "NUMBA_NUM_THREADS": str(threads),
        # Keep CBM on CPU by default to avoid requiring a GPU setup for timing.
        "V2XREG_CBM_DEVICE": "cpu",
    })

    # (name, target_metrics, timing_tag, command)
    tasks: List[Tuple[str, Path, Path, List[str]]] = []

    # --- ICP / PICP ---
    for method in ["icp", "picp"]:
        for noise in [0, 1, 2]:
            target = REPO_ROOT / "outputs_paper_3737" / "baselines" / f"{method}_paper_noise{noise}" / "metrics.json"
            if not _metrics_exists(target):
                continue
            timing_dir = REPO_ROOT / "outputs_tmp" / "table3_timing" / f"{method}_noise{noise}"
            timing_metrics = timing_dir / "metrics.json"
            if (not bool(args.force)) and _metrics_exists(timing_metrics):
                tasks.append((f"{method}-noise{noise}", target, timing_metrics, []))
                continue
            out_tag = str(Path("table3_timing") / f"{method}_noise{noise}")
            cmd = [
                "python",
                str(REPO_ROOT / "benchmarks" / "run_dair_lidar_benchmark.py"),
                "--project-config",
                "configs/hkust_lidar_global_paper3737_table3.yaml",
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
                "unadjusted",
                "--trans-noise",
                str(noise),
                "--rot-noise-deg",
                str(noise),
                "--noise-seed-mode",
                "per_index",
                "--voxel",
                "0.3",
                "--max-corr",
                "1.0",
                "--max-iter",
                "100",
                "--output-root",
                "outputs_tmp",
                "--output-tag",
                out_tag,
                "--seed",
                "2025",
            ] + (["--picp-loss", "huber", "--huber-k", "0.5"] if method == "picp" else [])
            tasks.append((f"{method}-noise{noise}", target, timing_metrics, cmd))

    # --- VIPS / CBM ---
    for noise in [0, 1, 2]:
        vips_target = REPO_ROOT / "outputs" / "vips" / f"paper3737_noise{noise}_thr1p5_initgt_icp" / "metrics.json"
        if _metrics_exists(vips_target):
            vips_timing = REPO_ROOT / "outputs" / "vips" / "timing" / f"paper3737_noise{noise}_thr1p5_initgt_icp_{num_frames}" / "metrics.json"
            if (not bool(args.force)) and _metrics_exists(vips_timing):
                tasks.append((f"vips-noise{noise}", vips_target, vips_timing, []))
            else:
                out_tag = str(Path("timing") / f"paper3737_noise{noise}_thr1p5_initgt_icp_{num_frames}")
                tasks.append((
                    f"vips-noise{noise}",
                    vips_target,
                    vips_timing,
                    [
                        "python",
                        str(REPO_ROOT / "benchmarks" / "run_vips_benchmark.py"),
                        "--config",
                        "configs/pipeline_paper3737_gt15_seedrefine5.yaml",
                        "--start",
                        str(start),
                        "--max-pairs",
                        str(num_frames),
                        "--output-tag",
                        out_tag,
                        "--init-source",
                        "gt",
                        "--trans-noise",
                        str(noise),
                        "--rot-noise-deg",
                        str(noise),
                        "--match-distance-thr",
                        "1.5",
                        "--seed",
                        "2025",
                    ],
                ))

        cbm_target = REPO_ROOT / "outputs" / "cbm" / f"paper3737_noise{noise}_min20_thr4p0_initgt_noicp" / "metrics.json"
        if _metrics_exists(cbm_target):
            cbm_timing = REPO_ROOT / "outputs" / "cbm" / "timing" / f"paper3737_noise{noise}_min20_thr4p0_initgt_noicp_{num_frames}" / "metrics.json"
            if (not bool(args.force)) and _metrics_exists(cbm_timing):
                tasks.append((f"cbm-noise{noise}", cbm_target, cbm_timing, []))
            else:
                out_tag = str(Path("timing") / f"paper3737_noise{noise}_min20_thr4p0_initgt_noicp_{num_frames}")
                tasks.append((
                    f"cbm-noise{noise}",
                    cbm_target,
                    cbm_timing,
                    [
                        "python",
                        str(REPO_ROOT / "benchmarks" / "run_cbm_benchmark.py"),
                        "--config",
                        "configs/pipeline_paper3737_gt15_seedrefine5.yaml",
                        "--start",
                        str(start),
                        "--max-pairs",
                        str(num_frames),
                        "--output-tag",
                        out_tag,
                        "--init-source",
                        "gt",
                        "--trans-noise",
                        str(noise),
                        "--rot-noise-deg",
                        str(noise),
                        "--min-matches",
                        "20",
                        "--match-distance-thr",
                        "4.0",
                        "--skip-icp",
                        "--seed",
                        "2025",
                    ],
                ))

    # --- No-init point cloud baselines ---
    hkust_map = {
        "fgr": "FGR",
        "quatro": "QUATRO",
        "gnctls": "GNC_TLS",
    }
    for tag, alg in hkust_map.items():
        target = REPO_ROOT / "outputs" / "hkust_teaser" / f"hkust_teaser_paper3737_{tag}_full" / "metrics.json"
        if not _metrics_exists(target):
            continue
        timing_metrics = REPO_ROOT / "outputs" / "hkust_teaser" / "timing" / f"paper3737_{tag}_{num_frames}" / "metrics.json"
        if (not bool(args.force)) and _metrics_exists(timing_metrics):
            tasks.append((f"hkust-{tag}", target, timing_metrics, []))
            continue
        out_tag = str(Path("timing") / f"paper3737_{tag}_{num_frames}")
        tasks.append((
            f"hkust-{tag}",
            target,
            timing_metrics,
            [
                "python",
                str(REPO_ROOT / "benchmarks" / "hkust_lidar_global_registration_benchmark.py"),
                "--config",
                "configs/hkust_lidar_global_paper3737_table3.yaml",
                "--start",
                str(start),
                "--end",
                str(end),
                "--rotation-alg",
                alg,
                "--output-tag",
                out_tag,
            ],
        ))

    if not tasks:
        raise SystemExit("No canonical metrics.json found; run tools/reproduce_table3_baselines.py first.")

    print(f"Timing tasks: {len(tasks)} | frames per task: {num_frames} | threads: {threads}")

    for name, target, timing_metrics, cmd in tasks:
        if cmd:
            print(f"[timing] {name}")
            run_cmd(cmd, env=timing_env, dry_run=bool(args.dry_run))
        else:
            print(f"[timing] {name} (reuse existing timing metrics)")

        if args.dry_run:
            continue
        if not _metrics_exists(timing_metrics):
            raise SystemExit(f"Timing metrics missing: {timing_metrics}")
        _patch_time(target, timing_metrics, dry_run=False)

    print("Done.")


if __name__ == "__main__":
    main()

