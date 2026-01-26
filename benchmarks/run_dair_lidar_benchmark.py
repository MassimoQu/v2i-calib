#!/usr/bin/env python3
"""
Run LiDAR-Registration-Benchmark methods on the DAIR-V2X dataset that is bundled
with this repository. Results (per-sample and aggregate) are written to outputs/.
"""
import argparse
import json
import math
import sys
import zlib
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
from calib.evaluation.metrics import FrameMetrics, aggregate_metrics  # noqa: E402
from v2x_calib.reader.CooperativeBatchingReader import CooperativeBatchingReader  # noqa: E402
from v2x_calib.reader.CooperativeReader import CooperativeReader  # noqa: E402
from v2x_calib.utils import (  # noqa: E402
    convert_T_to_6DOF,
    get_RE_TE_by_compare_T_6DOF_result_true,
)


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
except ImportError:  # pragma: no cover
    # Only required for the "teaser" method. ICP/PICP runs do not need teaserpp_python.
    fpfh_teaser = None


def _cfg_value(block, key, default=None):
    if block is None:
        return default
    if isinstance(block, dict):
        return block.get(key, default)
    return getattr(block, key, default)


def _stable_seed_token(value: object) -> int:
    payload = str(value).encode("utf-8", errors="ignore")
    return int(zlib.crc32(payload) & 0xFFFFFFFF)


def compute_vertical_angles(points: np.ndarray) -> np.ndarray:
    if points.size == 0:
        return np.empty(0)
    xy_norm = np.linalg.norm(points[:, :2], axis=1)
    return np.degrees(np.arctan2(points[:, 2], np.maximum(xy_norm, 1e-6)))


def _filter_points_by_vehicle_angle(points: np.ndarray, veh_points: np.ndarray, beam_cfg) -> np.ndarray:
    if veh_points is None or veh_points.size == 0:
        return points

    percentiles = _cfg_value(beam_cfg, 'angle_percentiles', (1.0, 99.0))
    if not isinstance(percentiles, (list, tuple)) or len(percentiles) != 2:
        percentiles = (1.0, 99.0)
    theta_low, theta_high = percentiles

    veh_angles = compute_vertical_angles(veh_points)
    if veh_angles.size == 0:
        return points

    veh_theta_min = float(np.percentile(veh_angles, theta_low))
    veh_theta_max = float(np.percentile(veh_angles, theta_high))

    infra_angles = compute_vertical_angles(points)
    mask = (infra_angles >= veh_theta_min) & (infra_angles <= veh_theta_max)
    filtered = points[mask]
    min_after_angle = int(_cfg_value(beam_cfg, 'min_points_after_angle', 3000))
    if filtered.shape[0] < max(1, min_after_angle):
        return points
    return filtered


def apply_infra_beam_alignment(
    infra_points: np.ndarray,
    veh_points: np.ndarray,
    *,
    beam_cfg,
    seed_components: tuple[object, ...],
) -> np.ndarray:
    if beam_cfg is None or not _cfg_value(beam_cfg, 'enabled', False):
        return infra_points

    points_aligned = infra_points
    if _cfg_value(beam_cfg, 'match_vehicle_angle', False):
        points_aligned = _filter_points_by_vehicle_angle(points_aligned, veh_points, beam_cfg)

    ratio = _cfg_value(beam_cfg, 'subsample_ratio', None)
    if ratio is None:
        source_lines = _cfg_value(beam_cfg, 'source_lines', None)
        target_lines = _cfg_value(beam_cfg, 'target_lines', None)
        if source_lines and target_lines:
            ratio = float(target_lines) / float(source_lines)
    if ratio is None:
        return points_aligned
    ratio = float(np.clip(float(ratio), 0.0, 1.0))
    if ratio <= 0.0 or ratio >= 0.999:
        return points_aligned

    num_points = points_aligned.shape[0]
    min_points = int(_cfg_value(beam_cfg, 'min_points', 0))
    target_points = max(int(num_points * ratio), min_points)
    target_points = min(target_points, num_points)
    if target_points <= 0 or target_points == num_points:
        return points_aligned

    base_seed = _cfg_value(beam_cfg, "random_seed", None)
    if base_seed is None:
        rng = np.random.default_rng()
    else:
        base_seed_int = int(base_seed)
        tokens = [_stable_seed_token(c) for c in seed_components]
        rng = np.random.default_rng(np.random.SeedSequence([base_seed_int, *tokens]))
    sampled_indices = rng.choice(num_points, size=target_points, replace=False)
    return points_aligned[sampled_indices]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate LiDAR-Registration-Benchmark methods on DAIR-V2X data.")
    parser.add_argument('--project-config', type=str,
                        default='configs/hkust_lidar_global_config.yaml',
                        help='Path to the project config that includes DAIR-V2X paths.')
    parser.add_argument('--data-info', type=str, default=None,
                        help='Optional data_info.json override (recommended for paper subsets).')
    parser.add_argument('--data-root', type=str, default=None,
                        help='Optional dataset root override (folder containing infrastructure-side/vehicle-side).')
    parser.add_argument('--benchmark-config', type=str,
                        default=str(BENCHMARK_ROOT / 'configs/dataset.yaml'),
                        help='Config inside LiDAR-Registration-Benchmark that defines registration params.')
    parser.add_argument('--start', type=int, default=0,
                        help='Start index within the data_info list.')
    parser.add_argument('--end', type=int, default=-1,
                        help='End index (exclusive) within the data_info list. -1 means all.')
    parser.add_argument('--max-pairs', type=int, default=None,
                        help='Optional hard limit on how many pairs to evaluate.')
    parser.add_argument('--output-root', type=str, default='outputs',
                        help='Output root directory (default: outputs).')
    parser.add_argument('--output-tag', type=str, default=None,
                        help='Optional stable output folder name. '
                             'Defaults to a timestamp-based name.')
    parser.add_argument('--visualize', action='store_true',
                        help='Visualize the registration result via Open3D.')
    parser.add_argument('--method', type=str, default='teaser',
                        choices=['teaser', 'icp', 'picp'],
                        help='Registration method to run.')
    parser.add_argument(
        '--init-source',
        type=str,
        default='unadjusted',
        choices=['gt', 'unadjusted'],
        help="Base initial transform before adding noise. "
        "'gt' uses the cooperative ground truth; "
        "'unadjusted' uses the infrastructure pose without applying its per-frame relative_error.",
    )
    parser.add_argument('--trans-noise', type=float, default=0.0,
                        help='Std of translation noise applied to GT transform (meters).')
    parser.add_argument('--rot-noise-deg', type=float, default=0.0,
                        help='Std of rotation noise applied to GT transform (degrees).')
    parser.add_argument(
        '--noise-mode',
        type=str,
        default='left',
        choices=['left', 'right'],
        help="How to compose the SE(3) noise with T_true. "
             "'left' applies T_init = noise @ T_true; "
             "'right' applies T_init = T_true @ noise.",
    )
    parser.add_argument('--voxel', type=float, default=0.3,
                        help='Voxel size for ICP down-sampling (meters).')
    parser.add_argument('--max-corr', type=float, default=1.5,
                        help='Max correspondence distance for ICP (meters).')
    parser.add_argument('--max-iter', type=int, default=100,
                        help='Max ICP iterations (default: 100).')
    parser.add_argument('--max-delta-trans', type=float, default=float('inf'),
                        help='Optional sanity gate: if ICP changes translation more than this (meters) '
                             'relative to the provided initial transform, fall back to the initial transform.')
    parser.add_argument('--max-delta-rot-deg', type=float, default=float('inf'),
                        help='Optional sanity gate: if ICP changes rotation more than this (degrees) '
                             'relative to the provided initial transform, fall back to the initial transform.')
    parser.add_argument('--log-every', type=int, default=50,
                        help='Log a detailed line every N frames (default: 50).')
    parser.add_argument('--seed', type=int, default=2025,
                        help='Random seed for noise injection.')
    parser.add_argument(
        '--beam-align-infra',
        action='store_true',
        help='Apply DAIR beam-alignment subsampling to the infrastructure point cloud '
             '(uses cfg.infra.beam_alignment from the provided project config).',
    )
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
                        rng: np.random.Generator, mode: str = 'left') -> np.ndarray:
    if trans_std <= 0 and rot_std_deg <= 0:
        return T.copy()
    delta_t = rng.normal(scale=trans_std, size=3)
    delta_euler_deg = rng.normal(scale=rot_std_deg, size=3)
    delta_R = R.from_euler("xyz", delta_euler_deg, degrees=True).as_matrix()
    noise_T = np.eye(4)
    noise_T[:3, :3] = delta_R
    noise_T[:3, 3] = delta_t
    if mode == 'right':
        return T @ noise_T
    return noise_T @ T


def numpy_to_pcd(points: np.ndarray) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    return pcd


def run_icp_solver(
    infra_pc: np.ndarray,
    veh_pc: np.ndarray,
    T_init: np.ndarray,
    *,
    voxel: float,
    max_corr: float,
    point_to_plane: bool,
    max_iter: int,
) -> tuple[np.ndarray, float, float]:
    src = numpy_to_pcd(infra_pc)
    tgt = numpy_to_pcd(veh_pc)
    if voxel > 0:
        src = src.voxel_down_sample(voxel)
        tgt = tgt.voxel_down_sample(voxel)
    if point_to_plane:
        # Match LiDAR-Registration-Benchmark defaults: point-to-plane ICP with a robust Huber kernel.
        radius = max(voxel * 2.0, 0.3)
        src.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30))
        tgt.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30))
        loss = o3d.pipelines.registration.HuberLoss(k=0.5)
        estimation = o3d.pipelines.registration.TransformationEstimationPointToPlane(loss)
    else:
        estimation = o3d.pipelines.registration.TransformationEstimationPointToPoint()
    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iter)
    reg = o3d.pipelines.registration.registration_icp(
        src,
        tgt,
        max_corr,
        T_init,
        estimation,
        criteria,
    )
    return reg.transformation, float(reg.fitness), float(reg.inlier_rmse)


def compute_se3_delta(T_new: np.ndarray, T_ref: np.ndarray) -> tuple[float, float]:
    """Return (delta_rot_deg, delta_trans_m) for T_new relative to T_ref."""
    T_delta = T_new @ np.linalg.inv(T_ref)
    rot = R.from_matrix(T_delta[:3, :3])
    delta_rot_deg = float(np.degrees(rot.magnitude()))
    delta_trans_m = float(np.linalg.norm(T_delta[:3, 3]))
    return delta_rot_deg, delta_trans_m


def main():
    args = parse_args()

    project_cfg_from_yaml(resolve_path(PROJECT_ROOT, args.project_config), project_cfg)
    benchmark_cfg_module.cfg_from_yaml_file(
        resolve_path(BENCHMARK_ROOT, args.benchmark_config), benchmark_cfg_module.cfg)

    if args.data_root:
        project_cfg.data.data_root_path = resolve_path(PROJECT_ROOT, args.data_root)
    else:
        project_cfg.data.data_root_path = resolve_path(PROJECT_ROOT, project_cfg.data.data_root_path)

    if args.data_info:
        project_cfg.data.data_info_path = resolve_path(PROJECT_ROOT, args.data_info)
    else:
        project_cfg.data.data_info_path = resolve_path(PROJECT_ROOT, project_cfg.data.data_info_path)

    reader = CooperativeBatchingReader(path_data_info=project_cfg.data.data_info_path,
                                       path_data_folder=project_cfg.data.data_root_path)
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    logger = Logger(f"dair_lidar_benchmark_{args.method}_{timestamp}")

    output_root = Path(resolve_path(PROJECT_ROOT, args.output_root))
    output_root.mkdir(parents=True, exist_ok=True)
    tag_name = args.output_tag or f"dair_lidar_benchmark_{args.method}_{timestamp}"
    output_dir = output_root / tag_name
    output_dir.mkdir(parents=True, exist_ok=True)
    detail_path = output_dir / 'details.jsonl'
    records: List[FrameMetrics] = []
    processed = 0

    with open(detail_path, 'w') as detail_file:
        infra_names = reader.infra_file_names
        veh_names = reader.vehicle_file_names
        end_idx = len(infra_names) if args.end == -1 else args.end
        if args.start < 0 or args.start >= end_idx:
            raise ValueError('start should be in [0, end)')
        if end_idx > len(infra_names):
            raise ValueError('end should be in [start, len(data_info)]')

        for local_idx, (inf_id, veh_id) in enumerate(
            zip(infra_names[args.start:end_idx], veh_names[args.start:end_idx])
        ):
            if args.max_pairs is not None and processed >= args.max_pairs:
                break

            t_start = perf_counter()
            delta_rot_deg = None
            delta_trans_m = None
            used_fallback = False
            icp_fitness = None
            icp_inlier_rmse = None
            coop_reader = CooperativeReader(inf_id, veh_id, project_cfg.data.data_root_path)
            inf_pc, veh_pc = coop_reader.get_cooperative_infra_vehicle_pointcloud()
            if args.beam_align_infra:
                beam_cfg = _cfg_value(_cfg_value(project_cfg, 'infra', None), 'beam_alignment', None)
                inf_pc = apply_infra_beam_alignment(
                    inf_pc,
                    veh_pc,
                    beam_cfg=beam_cfg,
                    seed_components=(inf_id, veh_id),
                )
            T_true = coop_reader.get_cooperative_T_i2v()
            if args.method == 'teaser':
                if fpfh_teaser is None:
                    raise ImportError(
                        "Failed to import LiDAR-Registration-Benchmark.misc.registration.fpfh_teaser. "
                        "Please ensure `teaserpp_python` and `open3d` are installed if you want to run '--method teaser'."
                    )
                T_pred = fpfh_teaser(inf_pc, veh_pc, args.visualize)
            elif args.method in {'icp', 'picp'}:
                T_base = (
                    coop_reader.get_cooperative_T_i2v_unadjusted()
                    if args.init_source == 'unadjusted'
                    else T_true
                )
                global_idx = int(args.start) + int(local_idx)
                rng = np.random.default_rng(np.random.SeedSequence([int(args.seed), global_idx]))
                T_init = add_transform_noise(
                    T_base, args.trans_noise, args.rot_noise_deg, rng, mode=args.noise_mode
                )
                if args.max_iter <= 0:
                    T_pred = T_init
                    used_fallback = True
                else:
                    T_icp, icp_fitness, icp_inlier_rmse = run_icp_solver(
                        inf_pc,
                        veh_pc,
                        T_init,
                        voxel=args.voxel,
                        max_corr=args.max_corr,
                        point_to_plane=(args.method == 'picp'),
                        max_iter=args.max_iter,
                    )
                    delta_rot_deg, delta_trans_m = compute_se3_delta(T_icp, T_init)
                    if delta_trans_m > args.max_delta_trans or delta_rot_deg > args.max_delta_rot_deg:
                        T_pred = T_init
                        used_fallback = True
                    else:
                        T_pred = T_icp
            else:  # pragma: no cover
                raise ValueError(f"Unsupported method: {args.method}")
            t_end = perf_counter()

            RE, TE = get_RE_TE_by_compare_T_6DOF_result_true(
                convert_T_to_6DOF(T_pred), convert_T_to_6DOF(T_true))
            runtime = t_end - t_start

            detail_file.write(json.dumps({
                "infra_id": inf_id,
                "veh_id": veh_id,
                "RE": float(RE),
                "TE": float(TE),
                "time": float(runtime),
                "delta_rot_deg": delta_rot_deg,
                "delta_trans_m": delta_trans_m,
                "used_fallback": used_fallback,
                "icp_fitness": icp_fitness,
                "icp_inlier_rmse": icp_inlier_rmse,
            }) + '\n')
            records.append(FrameMetrics(
                infra_id=str(inf_id),
                veh_id=str(veh_id),
                RE=float(RE),
                TE=float(TE),
                stability=0.0,
                time_cost=float(runtime),
                matches_count=0,
            ))
            processed += 1

            if args.log_every > 0 and (processed == 1 or processed % args.log_every == 0):
                logger.info(
                    f"[{processed}] inf_id={inf_id} veh_id={veh_id} "
                    f"RE={RE:.2f}deg TE={TE:.2f}m time={runtime:.3f}s")

    summary = aggregate_metrics(records, thresholds=[1.0, 2.0, 3.0])
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
