#!/usr/bin/env python3
"""
Evaluate the CBM (Cooperative Bounding-box Matching) baseline on DAIR-V2X pairs.
The pipeline follows the description in the CBM paper:
1. Use detected/GT 3D boxes from infrastructure & vehicle LiDARs.
2. Run CBM matching to obtain correspondences.
3. Construct pseudo point clouds from matched box corners and estimate pose via SVD.
4. Refine the pose using point-to-plane ICP on the raw point clouds.
"""
import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import List, Tuple
from time import perf_counter

import numpy as np
import open3d as o3d
import torch
from scipy.spatial.transform import Rotation as R

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from configs.legacy_api import Logger, cfg, cfg_from_yaml_file
from calib.evaluation.metrics import FrameMetrics, aggregate_metrics
from v2x_calib.reader.CooperativeReader import CooperativeReader
from v2x_calib.reader.CooperativeBatchingReader import CooperativeBatchingReader
from v2x_calib.utils.bbox_utils import (
    get_bbox3d_8_3_from_xyz_lwh_yaw,
    get_lwh_from_bbox3d_8_3,
    get_xyz_from_bbox3d_8_3,
)
from v2x_calib.utils import (
    convert_T_to_6DOF,
    get_RE_TE_by_compare_T_6DOF_result_true,
)

from benchmarks.third_party.CBM.CBM_torch import CBM as CBMMatcher


def parse_args():
    parser = argparse.ArgumentParser(description="Run CBM baseline on DAIR-V2X.")
    parser.add_argument("--config", type=str, default="configs/pipeline.yaml")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--max-pairs", type=int, default=30)
    parser.add_argument("--output-tag", type=str, default=None)
    parser.add_argument("--trans-noise", type=float, default=2.0,
                        help="Std of translation noise (meters) applied to initial transform.")
    parser.add_argument("--rot-noise-deg", type=float, default=10.0,
                        help="Std of rotation noise (degrees) for XYZ Euler angles.")
    parser.add_argument("--voxel", type=float, default=0.3,
                        help="ICP voxel down-sample size.")
    parser.add_argument("--max-corr", type=float, default=1.5,
                        help="ICP max correspondence distance.")
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--use-prediction", action="store_true",
                        help="Use detection results (if available) instead of GT boxes.")
    parser.add_argument("--identity-init", action="store_true",
                        help="Ignore GT init and feed identity transform into CBM.")
    parser.add_argument("--skip-icp", action="store_true",
                        help="Skip point cloud ICP refinement; use SVD pose directly.")
    return parser.parse_args()


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def add_transform_noise(T: np.ndarray, trans_std: float, rot_std_deg: float, rng: np.random.Generator) -> np.ndarray:
    if trans_std <= 0 and rot_std_deg <= 0:
        return T.copy()
    delta_t = rng.normal(scale=trans_std, size=3)
    rot_std_rad = math.radians(rot_std_deg)
    delta_euler = rng.normal(scale=rot_std_rad, size=3)
    delta_R = R.from_euler("xyz", delta_euler).as_matrix()
    noise_T = np.eye(4)
    noise_T[:3, :3] = delta_R
    noise_T[:3, 3] = delta_t
    return noise_T @ T


def bbox_object_to_state(bbox_obj) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert BBox3d object to (center, dims, yaw, corners).
    Returns:
        center (3,), dims (l,w,h), yaw rad, corners (8,3)
    """
    corners = bbox_obj.get_bbox3d_8_3()
    center = get_xyz_from_bbox3d_8_3(corners)
    l, w, h = get_lwh_from_bbox3d_8_3(corners)
    # yaw estimation: vector between corners 0 and 3 follows the length axis
    vec = corners[0] - corners[3]
    yaw = math.atan2(vec[1], vec[0])
    return center, (l, w, h), yaw, corners


def boxes_to_array(bbox_list) -> np.ndarray:
    states = []
    for bbox in bbox_list:
        center, (l, w, h), yaw, _ = bbox_object_to_state(bbox)
        states.append([center[0], center[1], center[2], h, w, l, yaw])
    if not states:
        return np.zeros((0, 7), dtype=np.float32)
    return np.asarray(states, dtype=np.float32)


def estimate_pose_from_matches(infra_boxes, veh_boxes, pairs: np.ndarray) -> Tuple[np.ndarray, int]:
    """Estimate pose using SVD on box corners."""
    if pairs.size == 0:
        return None, 0
    src_pts: List[np.ndarray] = []
    dst_pts: List[np.ndarray] = []
    for ego_idx, cav_idx in pairs:
        _, _, _, infra_corners = bbox_object_to_state(infra_boxes[int(cav_idx)])
        _, _, _, veh_corners = bbox_object_to_state(veh_boxes[int(ego_idx)])
        src_pts.append(infra_corners)
        dst_pts.append(veh_corners)
    src = np.vstack(src_pts)
    dst = np.vstack(dst_pts)
    if src.shape[0] < 3:
        return None, src.shape[0]
    src_cent = src.mean(axis=0)
    dst_cent = dst.mean(axis=0)
    src_centered = src - src_cent
    dst_centered = dst - dst_cent
    H = src_centered.T @ dst_centered
    U, _, Vt = np.linalg.svd(H)
    R_est = Vt.T @ U.T
    if np.linalg.det(R_est) < 0:
        Vt[-1, :] *= -1
        R_est = Vt.T @ U.T
    t_est = dst_cent - R_est @ src_cent
    T = np.eye(4)
    T[:3, :3] = R_est
    T[:3, 3] = t_est
    return T, src.shape[0]


def refine_with_icp(T_init: np.ndarray, infra_points: np.ndarray, veh_points: np.ndarray,
                    voxel: float, max_corr: float) -> np.ndarray:
    src = o3d.geometry.PointCloud()
    src.points = o3d.utility.Vector3dVector(infra_points)
    tgt = o3d.geometry.PointCloud()
    tgt.points = o3d.utility.Vector3dVector(veh_points)
    if voxel > 0:
        src = src.voxel_down_sample(voxel)
        tgt = tgt.voxel_down_sample(voxel)
    tgt.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 2.0, max_nn=30))
    result = o3d.pipelines.registration.registration_icp(
        src, tgt, max_corr, T_init,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50))
    return result.transformation


def main():
    args = parse_args()
    cfg_from_yaml_file(os.path.join(cfg.ROOT_DIR, args.config), cfg)
    data_root = getattr(cfg.data, "data_root_path", None) or getattr(cfg.data, "data_root", None)
    data_info = getattr(cfg.data, "data_info_path", None) or getattr(cfg.data, "data_info", None)
    if data_root is None or data_info is None:
        raise ValueError("Config must specify data_root(_path) and data_info_path.")
    if not hasattr(cfg.data, "data_root_path"):
        cfg.data.data_root_path = data_root
    if not hasattr(cfg.data, "data_info_path"):
        cfg.data.data_info_path = data_info

    tag = args.output_tag or f"cbm_{Path(args.config).stem}"
    output_dir = Path(cfg.ROOT_DIR) / "outputs" / "cbm" / tag
    ensure_dir(output_dir)
    logger = Logger(tag)
    matches_path = output_dir / "matches.jsonl"

    reader = CooperativeBatchingReader(
        path_data_info=data_info,
        path_data_folder=data_root,
    )
    cbm_default_args = argparse.Namespace(
        sigma1=10 * math.pi / 180,
        sigma2=3.0,
        sigma3=1.0,
        absolute_dis_lim=20.0,
    )
    cbm_matcher = CBMMatcher(args=cbm_default_args)
    rng = np.random.default_rng(args.seed)

    records: List[FrameMetrics] = []
    processed = 0
    start_idx = args.start
    end_idx = min(start_idx + args.max_pairs, len(reader.infra_file_names))

    with matches_path.open("w", encoding="utf-8") as f_match:
        for idx in range(start_idx, end_idx):
            infra_id = reader.infra_file_names[idx]
            veh_id = reader.vehicle_file_names[idx]
            coop = CooperativeReader(infra_id, veh_id, data_root)
            inf_boxes, veh_boxes = (
                coop.get_cooperative_infra_vehicle_boxes_object_list_predicted()
                if args.use_prediction else
                coop.get_cooperative_infra_vehicle_boxes_object_list()
            )
            if len(inf_boxes) == 0 or len(veh_boxes) == 0:
                logger.info(f"[{idx}] Skip {infra_id}-{veh_id}: empty boxes.")
                continue
            inf_pc, veh_pc = coop.get_cooperative_infra_vehicle_pointcloud()
            T_true = coop.get_cooperative_T_i2v()
            if args.identity_init:
                T_init = np.eye(4)
            else:
                T_init = add_transform_noise(
                    T_true, args.trans_noise, args.rot_noise_deg, rng)
            init_RE, init_TE = get_RE_TE_by_compare_T_6DOF_result_true(
                convert_T_to_6DOF(T_init), convert_T_to_6DOF(T_true))

            cav_array = boxes_to_array(inf_boxes)
            ego_array = boxes_to_array(veh_boxes)
            t_process_start = perf_counter()
            matching = cbm_matcher(ego_array, cav_array, T_init)
            if torch.is_tensor(matching):
                matching = matching.cpu().numpy()
            if matching is None or matching.size == 0:
                logger.info(f"[{idx}] CBM returned no matches for {infra_id}-{veh_id}.")
                continue

            T_svd, num_pts = estimate_pose_from_matches(inf_boxes, veh_boxes, matching)
            if T_svd is None:
                logger.info(f"[{idx}] Not enough correspondences after CBM for {infra_id}-{veh_id}.")
                continue
            if args.skip_icp:
                T_refined = T_svd
            else:
                T_refined = refine_with_icp(T_svd, inf_pc, veh_pc, args.voxel, args.max_corr)
            elapsed = perf_counter() - t_process_start

            RE, TE = get_RE_TE_by_compare_T_6DOF_result_true(
                convert_T_to_6DOF(T_refined), convert_T_to_6DOF(T_true))
            logger.info(
                f"[{idx}] {infra_id}-{veh_id} matches={len(matching)} points={num_pts} "
                f"RE={RE:.2f} TE={TE:.2f} time={elapsed:.2f}s | init RE={init_RE:.2f} TE={init_TE:.2f}"
            )
            f_match.write(json.dumps({
                "infra_id": infra_id,
                "veh_id": veh_id,
                "RE": RE,
                "TE": TE,
                "init_RE": init_RE,
                "init_TE": init_TE,
                "num_matches": int(len(matching)),
                "num_points": int(num_pts),
                "time": elapsed,
                "best_mode": None,
            }) + "\n")

            records.append(FrameMetrics(
                infra_id=str(infra_id),
                veh_id=str(veh_id),
                RE=float(RE),
                TE=float(TE),
                stability=0.0,
                time_cost=elapsed,
            ))
            processed += 1
            if processed >= args.max_pairs:
                break

    thresholds = [1.0, 2.0, 3.0, 4.0, 5.0]
    summary = aggregate_metrics(records, thresholds)
    summary_path = output_dir / "metrics.json"
    with summary_path.open("w", encoding="utf-8") as f_summary:
        json.dump(summary, f_summary, indent=2)
    logger.info(f"Summary: {json.dumps(summary, indent=2)}")


if __name__ == "__main__":
    main()
