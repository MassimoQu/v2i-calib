#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

from time import perf_counter

import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R
from scipy.sparse.linalg import eigsh

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.legacy_api import Logger, cfg, cfg_from_yaml_file
from calib.evaluation.metrics import FrameMetrics, aggregate_metrics
from v2x_calib.reader.CooperativeReader import CooperativeReader
from v2x_calib.reader.CooperativeBatchingReader import CooperativeBatchingReader
from v2x_calib.utils.bbox_utils import (
    get_lwh_from_bbox3d_8_3,
    get_xyz_from_bbox3d_8_3,
)
from v2x_calib.utils import (
    convert_T_to_6DOF,
    get_RE_TE_by_compare_T_6DOF_result_true,
)

VIPS_DIR = ROOT / "benchmarks" / "third_party" / "VIPS_co_visible_object_matching"
if str(VIPS_DIR) not in sys.path:
    sys.path.append(str(VIPS_DIR))

from matrix import create_affinity_matrix  # type: ignore  # pylint: disable=import-error
from matching import find_optimal_matching  # type: ignore  # pylint: disable=import-error


def parse_args():
    parser = argparse.ArgumentParser(description="Run VIPS graph matching baseline on DAIR-V2X.")
    parser.add_argument("--config", type=str, default="configs/pipeline_hkust.yaml")
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
                        help="Ignore GT init and feed identity transform into VIPS.")
    parser.add_argument("--skip-icp", action="store_true",
                        help="Skip point cloud ICP refinement; use SVD pose directly.")
    parser.add_argument("--match-distance-thr", type=float, default=8.0,
                        help="Reject VIPS matches whose centers differ more than this threshold (in meters).")
    return parser.parse_args()


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def add_transform_noise(T: np.ndarray, trans_std: float, rot_std_deg: float,
                        rng: np.random.Generator) -> np.ndarray:
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


def bbox_object_to_state(bbox) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    corners = bbox.get_bbox3d_8_3()
    center = get_xyz_from_bbox3d_8_3(corners)
    l, w, h = get_lwh_from_bbox3d_8_3(corners)
    vec = corners[0] - corners[3]
    yaw = math.atan2(vec[1], vec[0])
    return center, np.array([l, w, h], dtype=np.float32), yaw, corners


def transform_points(T: np.ndarray, points: np.ndarray) -> np.ndarray:
    pts_h = np.hstack([points, np.ones((points.shape[0], 1))])
    transformed = (T @ pts_h.T).T
    return transformed[:, :3]


def transform_yaw(yaw: float, rot_mat: np.ndarray) -> float:
    dir_vec = np.array([math.cos(yaw), math.sin(yaw), 0.0])
    rotated = rot_mat @ dir_vec
    return math.atan2(rotated[1], rotated[0])


class CategoryEncoder:
    def __init__(self) -> None:
        self._mapping: Dict[str, int] = {}

    def encode(self, name: str) -> int:
        key = name.lower()
        if key not in self._mapping:
            self._mapping[key] = len(self._mapping)
        return self._mapping[key]


def build_car_graph(bboxes, encoder: CategoryEncoder, world_T: np.ndarray) -> Dict[str, List]:
    categories: List[int] = []
    positions: List[List[float]] = []
    bbox_dims: List[List[float]] = []
    world_positions: List[List[float]] = []
    headings: List[List[float]] = []
    rot = world_T[:3, :3]
    for bbox in bboxes:
        center, dims, yaw, _ = bbox_object_to_state(bbox)
        categories.append(encoder.encode(bbox.get_bbox_type()))
        positions.append(center.astype(float).tolist())
        bbox_dims.append(dims.astype(float).tolist())
        transformed_center = transform_points(world_T, center.reshape(1, 3))[0]
        world_positions.append(transformed_center.tolist())
        world_yaw = transform_yaw(yaw, rot)
        headings.append([world_yaw])
    return {
        "category": categories,
        "position": positions,
        "bounding_box": bbox_dims,
        "world_position": world_positions,
        "heading": headings,
    }


def run_vips_matching(car1: Dict[str, List], car2: Dict[str, List], threshold: float = 0.5) -> List[List[int]]:
    L1 = len(car1["category"])
    L2 = len(car2["category"])
    if L1 == 0 or L2 == 0:
        return []
    G1 = np.zeros((L1, 11), dtype=np.float32)
    G2 = np.zeros((L2, 11), dtype=np.float32)
    for i in range(L1):
        G1[i, 0] = car1["category"][i]
        G1[i, 1:4] = np.asarray(car1["position"][i])
        G1[i, 4:7] = np.asarray(car1["bounding_box"][i])
        G1[i, 7:10] = np.asarray(car1["world_position"][i])
        G1[i, 10] = car1["heading"][i][0]
    for i in range(L2):
        G2[i, 0] = car2["category"][i]
        G2[i, 1:4] = np.asarray(car2["position"][i])
        G2[i, 4:7] = np.asarray(car2["bounding_box"][i])
        G2[i, 7:10] = np.asarray(car2["world_position"][i])
        G2[i, 10] = car2["heading"][i][0]
    M = create_affinity_matrix(G1, G2, L1, L2)
    dim = M.shape[0]
    if dim > 256:
        _, eigvecs = eigsh(M, k=1, which="LA")
        w = eigvecs[:, 0]
    else:
        _, eigvecs = np.linalg.eigh(M)
        w = eigvecs[:, -1]
    if np.max(w) > np.min(w):
        w = (w - np.min(w)) / (np.max(w) - np.min(w))
    return find_optimal_matching(w, L1, L2, threshold=threshold)


def filter_matches_by_distance(matches: np.ndarray, infra_boxes, veh_boxes,
                               T_init: np.ndarray, thr: float) -> np.ndarray:
    if matches.size == 0:
        return matches
    filtered: List[List[int]] = []
    transformed_cache: Dict[int, np.ndarray] = {}
    for veh_idx, infra_idx in matches:
        veh_center, _, _, _ = bbox_object_to_state(veh_boxes[int(veh_idx)])
        if infra_idx not in transformed_cache:
            infra_center, _, _, _ = bbox_object_to_state(infra_boxes[int(infra_idx)])
            transformed = transform_points(T_init, infra_center.reshape(1, 3))[0]
            transformed_cache[int(infra_idx)] = transformed
        else:
            transformed = transformed_cache[int(infra_idx)]
        if np.linalg.norm(transformed - veh_center) <= thr:
            filtered.append([int(veh_idx), int(infra_idx)])
    return np.asarray(filtered, dtype=np.int32)


def estimate_pose_from_matches(infra_boxes, veh_boxes, pairs: np.ndarray) -> Tuple[np.ndarray | None, int]:
    if pairs.size == 0:
        return None, 0
    src_pts: List[np.ndarray] = []
    dst_pts: List[np.ndarray] = []
    for veh_idx, infra_idx in pairs:
        _, _, _, infra_corners = bbox_object_to_state(infra_boxes[int(infra_idx)])
        _, _, _, veh_corners = bbox_object_to_state(veh_boxes[int(veh_idx)])
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

    tag = args.output_tag or f"vips_{Path(args.config).stem}"
    output_dir = Path(cfg.ROOT_DIR) / "outputs" / "vips" / tag
    ensure_dir(output_dir)
    logger = Logger(tag)
    matches_path = output_dir / "matches.jsonl"

    reader = CooperativeBatchingReader(
        path_data_info=data_info,
        path_data_folder=data_root,
    )
    rng = np.random.default_rng(args.seed)
    encoder = CategoryEncoder()

    records: List[FrameMetrics] = []
    processed = 0
    start_idx = args.start
    end_idx = min(start_idx + args.max_pairs, len(reader.infra_file_names))

    def append_failure(reason: str, elapsed: float, infra_id: str, veh_id: str):
        logger.info(f"[{infra_id}-{veh_id}] VIPS failure counted: {reason}")
        records.append(FrameMetrics(
            infra_id=str(infra_id),
            veh_id=str(veh_id),
            RE=180.0,
            TE=1e6,
            stability=0.0,
            time_cost=elapsed,
        ))

    with matches_path.open("w", encoding="utf-8") as f_match:
        for idx in range(start_idx, end_idx):
            infra_id = reader.infra_file_names[idx]
            veh_id = reader.vehicle_file_names[idx]
            coop = CooperativeReader(infra_id, veh_id, data_root)
            start_time = perf_counter()
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
                T_init = add_transform_noise(T_true, args.trans_noise, args.rot_noise_deg, rng)
            init_RE, init_TE = get_RE_TE_by_compare_T_6DOF_result_true(
                convert_T_to_6DOF(T_init), convert_T_to_6DOF(T_true))

            veh_graph = build_car_graph(veh_boxes, encoder, np.eye(4))
            infra_graph = build_car_graph(inf_boxes, encoder, T_init)

            try:
                matches = run_vips_matching(veh_graph, infra_graph)
            except Exception as exc:  # pragma: no cover - defensive logging
                elapsed = perf_counter() - start_time
                logger.error(f"[{idx}] VIPS failed on {infra_id}-{veh_id}: {exc}")
                append_failure("solver exception", elapsed, infra_id, veh_id)
                processed += 1
                if processed >= args.max_pairs:
                    break
                continue
            matches = np.asarray(matches, dtype=np.int32)
            raw_count = matches.shape[0]
            matches = filter_matches_by_distance(matches, inf_boxes, veh_boxes, T_init, args.match_distance_thr)
            if matches.size == 0:
                elapsed = perf_counter() - start_time
                logger.info(f"[{idx}] VIPS returned no matches for {infra_id}-{veh_id}.")
                append_failure("distance gate removed all matches", elapsed, infra_id, veh_id)
                processed += 1
                if processed >= args.max_pairs:
                    break
                continue
            pose_T, num_pts = estimate_pose_from_matches(inf_boxes, veh_boxes, matches)
            if pose_T is None:
                elapsed = perf_counter() - start_time
                logger.info(f"[{idx}] Not enough correspondences after VIPS for {infra_id}-{veh_id}.")
                append_failure("SVD received < 3 matches", elapsed, infra_id, veh_id)
                processed += 1
                if processed >= args.max_pairs:
                    break
                continue
            if args.skip_icp:
                T_refined = pose_T
            else:
                T_refined = refine_with_icp(pose_T, inf_pc, veh_pc, args.voxel, args.max_corr)

            RE, TE = get_RE_TE_by_compare_T_6DOF_result_true(
                convert_T_to_6DOF(T_refined), convert_T_to_6DOF(T_true))
            elapsed = perf_counter() - start_time
            logger.info(
                f"[{idx}] {infra_id}-{veh_id} matches={len(matches)} (raw {raw_count}) points={num_pts} "
                f"RE={RE:.2f} TE={TE:.2f} time={elapsed:.2f}s | init RE={init_RE:.2f} TE={init_TE:.2f}"
            )
            f_match.write(json.dumps({
                "infra_id": infra_id,
                "veh_id": veh_id,
                "RE": RE,
                "TE": TE,
                "init_RE": init_RE,
                "init_TE": init_TE,
                "num_matches": int(len(matches)),
                "num_points": int(num_pts),
                "time": elapsed,
            }) + "\n")

            records.append(FrameMetrics(
                infra_id=str(infra_id),
                veh_id=str(veh_id),
                RE=float(RE),
                TE=float(TE),
                stability=0.0,
                time_cost=elapsed,
                matches_count=int(len(matches)),
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
