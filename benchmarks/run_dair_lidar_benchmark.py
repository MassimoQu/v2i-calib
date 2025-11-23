#!/usr/bin/env python3
"""
Run LiDAR-Registration-Benchmark methods on the DAIR-V2X dataset that is bundled
with this repository. Results (per-sample and aggregate) are written to outputs/.
"""
import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import List, Dict, Any

import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R

SCRIPT_DIR = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from configs.legacy_api import cfg as project_cfg  # noqa: E402  pylint: disable=wrong-import-position
from configs.legacy_api import cfg_from_yaml_file as project_cfg_from_yaml  # noqa: E402
from configs.legacy_api import Logger  # noqa: E402
from v2x_calib.reader.CooperativeBatchingReader import CooperativeBatchingReader  # noqa: E402
from v2x_calib.utils import get_RE_TE_by_compare_T_6DOF_result_true, convert_T_to_6DOF  # noqa: E402


BENCHMARK_ROOT = PROJECT_ROOT / 'benchmarks' / 'third_party' / 'LiDAR-Registration-Benchmark'

if not BENCHMARK_ROOT.exists():
    raise FileNotFoundError(
        f"LiDAR-Registration-Benchmark submodule not found at {BENCHMARK_ROOT}. "
        "Please run `git submodule update --init --recursive` first.")

# Import benchmark utilities after ensuring the path is available.
sys.path.insert(0, str(BENCHMARK_ROOT))
from misc import config as benchmark_cfg_module  # type: ignore

try:
    from misc.registration import fpfh_teaser  # type: ignore
except ImportError as exc:  # pragma: no cover - surface missing dependency clearly
    raise ImportError(
        "Failed to import LiDAR-Registration-Benchmark.misc.registration. "
        "Please ensure `teaserpp_python` and `open3d` are installed.") from exc


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate LiDAR-Registration-Benchmark methods on DAIR-V2X data.")
    parser.add_argument('--project-config', type=str,
                        default='configs/hkust_lidar_global_config.yaml',
                        help='Path to the project config that includes DAIR-V2X paths.')
    parser.add_argument('--benchmark-config', type=str,
                        default=str(BENCHMARK_ROOT / 'configs/dataset.yaml'),
                        help='Config inside LiDAR-Registration-Benchmark that defines registration params.')
    parser.add_argument('--start', type=int, default=0,
                        help='Start index within the data_info list.')
    parser.add_argument('--end', type=int, default=-1,
                        help='End index (exclusive) within the data_info list. -1 means all.')
    parser.add_argument('--max-pairs', type=int, default=None,
                        help='Optional hard limit on how many pairs to evaluate.')
    parser.add_argument('--visualize', action='store_true',
                        help='Visualize the registration result via Open3D.')
    parser.add_argument('--method', type=str, default='teaser',
                        choices=['teaser', 'icp', 'picp'],
                        help='Registration method to run.')
    parser.add_argument('--trans-noise', type=float, default=0.0,
                        help='Std of translation noise applied to GT transform (meters).')
    parser.add_argument('--rot-noise-deg', type=float, default=0.0,
                        help='Std of rotation noise applied to GT transform (degrees).')
    parser.add_argument('--voxel', type=float, default=0.3,
                        help='Voxel size for ICP down-sampling (meters).')
    parser.add_argument('--max-corr', type=float, default=1.5,
                        help='Max correspondence distance for ICP (meters).')
    parser.add_argument('--seed', type=int, default=2025,
                        help='Random seed for noise injection.')
    return parser.parse_args()


def resolve_path(base_dir: Path, maybe_relative: str) -> str:
    path = Path(maybe_relative)
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return str(path)


def format_result(inf_id: str, veh_id: str, re: float, te: float,
                  runtime: float, success: bool) -> Dict[str, Any]:
    return {
        "infra_id": inf_id,
        "vehicle_id": veh_id,
        "rotation_error_deg": re,
        "translation_error_m": te,
        "runtime_s": runtime,
        "success": bool(success)
    }


def summarize_results(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not records:
        return {
            "total_pairs": 0,
            "success_pairs": 0,
            "success_rate": 0.0,
            "avg_rotation_error_deg": None,
            "avg_translation_error_m": None,
            "avg_runtime_s": None
        }
    rotation_errors = np.array([entry["rotation_error_deg"] for entry in records])
    translation_errors = np.array([entry["translation_error_m"] for entry in records])
    runtimes = np.array([entry["runtime_s"] for entry in records])
    successes = np.array([entry["success"] for entry in records])
    return {
        "total_pairs": int(len(records)),
        "success_pairs": int(successes.sum()),
        "success_rate": float(successes.mean()),
        "avg_rotation_error_deg": float(rotation_errors.mean()),
        "avg_translation_error_m": float(translation_errors.mean()),
        "avg_runtime_s": float(runtimes.mean())
    }


def add_transform_noise(T: np.ndarray, trans_std: float, rot_std_deg: float,
                        rng: np.random.Generator) -> np.ndarray:
    if trans_std <= 0 and rot_std_deg <= 0:
        return T.copy()
    delta_t = rng.normal(scale=trans_std, size=3)
    delta_euler = rng.normal(scale=math.radians(rot_std_deg), size=3)
    delta_R = R.from_euler('xyz', delta_euler).as_matrix()
    noise_T = np.eye(4)
    noise_T[:3, :3] = delta_R
    noise_T[:3, 3] = delta_t
    return noise_T @ T


def numpy_to_pcd(points: np.ndarray) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    return pcd


def run_icp_solver(infra_pc: np.ndarray,
                   veh_pc: np.ndarray,
                   T_init: np.ndarray,
                   *,
                   voxel: float,
                   max_corr: float,
                   point_to_plane: bool) -> np.ndarray:
    src = numpy_to_pcd(infra_pc)
    tgt = numpy_to_pcd(veh_pc)
    if voxel > 0:
        src = src.voxel_down_sample(voxel)
        tgt = tgt.voxel_down_sample(voxel)
    if point_to_plane:
        tgt.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=max(voxel * 2.0, 0.3), max_nn=30)
        )
        estimation = o3d.pipelines.registration.TransformationEstimationPointToPlane()
    else:
        estimation = o3d.pipelines.registration.TransformationEstimationPointToPoint()
    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100)
    reg = o3d.pipelines.registration.registration_icp(
        src,
        tgt,
        max_corr,
        T_init,
        estimation,
        criteria,
    )
    return reg.transformation


def main():
    args = parse_args()

    project_cfg_from_yaml(resolve_path(PROJECT_ROOT, args.project_config), project_cfg)
    benchmark_cfg_module.cfg_from_yaml_file(
        resolve_path(BENCHMARK_ROOT, args.benchmark_config), benchmark_cfg_module.cfg)

    project_cfg.data.data_root_path = resolve_path(PROJECT_ROOT, project_cfg.data.data_root_path)
    project_cfg.data.data_info_path = resolve_path(PROJECT_ROOT, project_cfg.data.data_info_path)

    reader = CooperativeBatchingReader(path_data_info=project_cfg.data.data_info_path,
                                       path_data_folder=project_cfg.data.data_root_path)
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    logger = Logger(f"dair_lidar_benchmark_{args.method}_{timestamp}")

    output_root = PROJECT_ROOT / 'outputs'
    output_root.mkdir(parents=True, exist_ok=True)
    tag_name = f"dair_lidar_benchmark_{args.method}_{timestamp}"
    output_dir = output_root / tag_name
    output_dir.mkdir(parents=True, exist_ok=True)
    detail_path = output_dir / 'details.jsonl'
    rng = np.random.default_rng(args.seed)

    records: List[Dict[str, Any]] = []
    processed = 0
    success_threshold_rot = benchmark_cfg_module.cfg.evaluation.rot_thd
    success_threshold_trans = benchmark_cfg_module.cfg.evaluation.trans_thd

    with open(detail_path, 'w') as detail_file:
        for (inf_id, veh_id, inf_pc, veh_pc, T_true) in \
                reader.generate_infra_vehicle_pointcloud(start_idx=args.start, end_idx=args.end):
            if args.max_pairs is not None and processed >= args.max_pairs:
                break

            t_start = perf_counter()
            if args.method == 'teaser':
                T_pred = fpfh_teaser(inf_pc, veh_pc, args.visualize)
            elif args.method in {'icp', 'picp'}:
                T_init = add_transform_noise(
                    T_true, args.trans_noise, args.rot_noise_deg, rng
                )
                T_pred = run_icp_solver(
                    inf_pc,
                    veh_pc,
                    T_init,
                    voxel=args.voxel,
                    max_corr=args.max_corr,
                    point_to_plane=(args.method == 'picp'),
                )
            else:  # pragma: no cover
                raise ValueError(f"Unsupported method: {args.method}")
            t_end = perf_counter()

            RE, TE = get_RE_TE_by_compare_T_6DOF_result_true(
                convert_T_to_6DOF(T_pred), convert_T_to_6DOF(T_true))
            success = (RE <= success_threshold_rot) and (TE <= success_threshold_trans)
            runtime = t_end - t_start

            record = format_result(inf_id, veh_id, RE, TE, runtime, success)
            detail_file.write(json.dumps(record) + '\n')
            records.append(record)
            processed += 1

            logger.info(
                f"[{processed}] inf_id={inf_id} veh_id={veh_id} "
                f"RE={RE:.2f}deg TE={TE:.2f}m time={runtime:.3f}s success={success}")

    summary = summarize_results(records)
    metrics_path = output_dir / 'metrics.json'
    with open(metrics_path, 'w') as metrics_file:
        json.dump(summary, metrics_file, indent=2)

    logger.info("==== Summary ====")
    logger.info(json.dumps(summary, indent=2))
    logger.close()
    print(f"Detailed per-pair metrics: {detail_path}")
    print(f"Aggregated metrics: {metrics_path}")


if __name__ == '__main__':
    main()
