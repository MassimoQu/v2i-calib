#!/usr/bin/env python3
"""
Convert HEAL stage1 box export (stage1_boxes.json) into the detection cache
format consumed by calib.data.detection_adapter.DetectionAdapter.

Typical usage:
    python tools/heal_stage1_to_detection_cache.py \\
        --stage1 HEAL/opencood/logs/xxx/test/stage1_boxes.json \\
        --output data/DAIR-V2X/detected/my_stage1_cache.json \\
        --swap-order
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Sequence, List

import math
import numpy as np

CAV_ORDER_KEYS = {
    'pred_corner3d_np_list',
    'pred_box3d_np_list',
    'pred_score_np_list',
    'uncertainty_np_list',
    'lidar_pose_clean_np',
    'lidar_pose_np',
    'cav_id_list',
    'occ_map_level0',
    'feature_corner3d_np_list',
}


def pose_to_matrix(pose: Sequence[float]) -> np.ndarray:
    """Convert [x, y, z, roll, yaw, pitch] (degree) to 4x4 transform."""
    x, y, z, roll, yaw, pitch = pose
    r = math.radians(roll)
    y_rad = math.radians(yaw)
    p = math.radians(pitch)
    c_y = math.cos(y_rad)
    s_y = math.sin(y_rad)
    c_r = math.cos(r)
    s_r = math.sin(r)
    c_p = math.cos(p)
    s_p = math.sin(p)
    matrix = np.identity(4, dtype=np.float64)
    matrix[0, 3] = x
    matrix[1, 3] = y
    matrix[2, 3] = z
    matrix[0, 0] = c_p * c_y
    matrix[0, 1] = c_y * s_p * s_r - s_y * c_r
    matrix[0, 2] = -c_y * s_p * c_r - s_y * s_r
    matrix[1, 0] = s_y * c_p
    matrix[1, 1] = s_y * s_p * s_r + c_y * c_r
    matrix[1, 2] = -s_y * s_p * c_r + c_y * s_r
    matrix[2, 0] = s_p
    matrix[2, 1] = -c_p * s_r
    matrix[2, 2] = c_p * c_r
    return matrix


def transform_boxes_from_ego_to_local(boxes: Sequence, T_cav_ref: np.ndarray) -> Sequence:
    arr = np.asarray(boxes, dtype=np.float64)
    if arr.size == 0:
        return boxes
    ones = np.ones(arr.shape[:-1] + (1,), dtype=np.float64)
    hom = np.concatenate([arr, ones], axis=-1)
    transformed = hom @ T_cav_ref.T
    return transformed[..., :3].tolist()


def load_stage1_file(path: Path) -> Dict[str, Any]:
    if path.is_dir():
        path = path / 'stage1_boxes.json'
    if not path.exists():
        raise FileNotFoundError(f"Stage1 file not found: {path}")
    with path.open('r', encoding='utf-8') as f:
        data = json.load(f)
    if not isinstance(data, Mapping):
        raise ValueError(f"Stage1 file must contain a JSON object, got {type(data)}")
    return {str(k): v for k, v in data.items()}


def maybe_swap_cav_order(value: Any) -> Any:
    if not isinstance(value, list):
        return value
    if len(value) < 2:
        return value
    swapped = list(value)
    swapped[0], swapped[1] = swapped[1], swapped[0]
    return swapped


def normalize_entry(
    entry: MutableMapping[str, Any],
    swap_order: bool,
    ensure_two_cavs: bool,
    deproject_to_local: bool,
    ego_index: int,
    required_cav_substrings: Sequence[str] | None,
) -> Dict[str, Any]:
    if entry is None:
        return {}
    pred_list = entry.get('pred_corner3d_np_list')
    if ensure_two_cavs and (not isinstance(pred_list, list) or len(pred_list) < 2):
        return {}
    if required_cav_substrings:
        cav_ids = entry.get('cav_id_list')
        if not isinstance(cav_ids, list):
            return {}
        normalized_ids = [str(cav_id).lower() for cav_id in cav_ids]
        required = [str(tag).lower() for tag in required_cav_substrings if tag]
        for tag in required:
            if not any(tag in cav_id for cav_id in normalized_ids):
                return {}
    if deproject_to_local and isinstance(pred_list, list):
        poses = entry.get('lidar_pose_clean_np') or entry.get('lidar_pose_np')
        if isinstance(poses, list) and len(poses) >= 1:
            ego_idx = ego_index if ego_index < len(poses) else 0
            T_world_ref = pose_to_matrix(poses[ego_idx])
            transforms = []
            for pose in poses:
                T_world_cav = pose_to_matrix(pose)
                transforms.append(np.linalg.inv(T_world_cav) @ T_world_ref)
            for cav_idx, boxes in enumerate(pred_list):
                if not isinstance(boxes, list):
                    continue
                if cav_idx >= len(transforms):
                    continue
                converted_boxes = []
                for box in boxes:
                    if isinstance(box, dict):
                        corners = box.get('corners')
                        if corners is None:
                            converted_boxes.append(box)
                            continue
                        transformed = transform_boxes_from_ego_to_local(
                            corners, transforms[cav_idx]
                        )
                        updated = dict(box)
                        updated['corners'] = transformed
                        converted_boxes.append(updated)
                    else:
                        converted_boxes.append(
                            transform_boxes_from_ego_to_local(box, transforms[cav_idx])
                        )
                pred_list[cav_idx] = converted_boxes
            box3d_list = entry.get('pred_box3d_np_list')
            if isinstance(box3d_list, list):
                for cav_idx, boxes in enumerate(box3d_list):
                    if not isinstance(boxes, list) or cav_idx >= len(transforms):
                        continue
                    T_cav_ref = transforms[cav_idx]
                    R_cav_ref = T_cav_ref[:3, :3]
                    converted = []
                    for box in boxes:
                        payload = box
                        is_dict = isinstance(box, dict)
                        if is_dict:
                            payload = box.get('box7d') or box.get('box3d') or box.get('pred_box3d')
                        if payload is None:
                            converted.append(box)
                            continue
                        arr = np.asarray(payload, dtype=np.float64).reshape(-1)
                        if arr.size < 7:
                            converted.append(box)
                            continue
                        center = np.array(arr[:3], dtype=np.float64)
                        dims = np.array(arr[3:6], dtype=np.float64)
                        yaw = float(arr[6])
                        center_h = np.concatenate([center, [1.0]])
                        center_new = (center_h @ T_cav_ref.T)[:3]
                        forward = np.array([math.cos(yaw), math.sin(yaw), 0.0], dtype=np.float64)
                        forward_new = R_cav_ref @ forward
                        yaw_new = math.atan2(float(forward_new[1]), float(forward_new[0]))
                        new_box = [
                            float(center_new[0]),
                            float(center_new[1]),
                            float(center_new[2]),
                            float(dims[0]),
                            float(dims[1]),
                            float(dims[2]),
                            float(yaw_new),
                        ]
                        if is_dict:
                            updated = dict(box)
                            if 'box7d' in updated:
                                updated['box7d'] = new_box
                            elif 'box3d' in updated:
                                updated['box3d'] = new_box
                            else:
                                updated['box3d'] = new_box
                            converted.append(updated)
                        else:
                            converted.append(new_box)
                    box3d_list[cav_idx] = converted
            feature_list = entry.get('feature_corner3d_np_list')
            if isinstance(feature_list, list):
                for cav_idx, features in enumerate(feature_list):
                    if cav_idx >= len(transforms) or not isinstance(features, list):
                        continue
                    converted: List[Any] = []
                    for feat in features:
                        if isinstance(feat, dict):
                            corners = feat.get('corners')
                            if corners is None:
                                converted.append(feat)
                                continue
                            transformed_corners = transform_boxes_from_ego_to_local(
                                corners, transforms[cav_idx]
                            )
                            updated = dict(feat)
                            updated['corners'] = transformed_corners
                            converted.append(updated)
                        else:
                            converted.append(
                                transform_boxes_from_ego_to_local(
                                    feat, transforms[cav_idx]
                                )
                            )
                    feature_list[cav_idx] = converted
    normalized: Dict[str, Any] = {}
    for key, value in entry.items():
        if swap_order and key in CAV_ORDER_KEYS:
            normalized[key] = maybe_swap_cav_order(value)
        else:
            normalized[key] = value
    return normalized


def _extract_frame_id(path: str | None) -> str:
    if not path:
        return ''
    parts = str(path).replace('\\', '/').split('/')
    if not parts:
        return ''
    stem = parts[-1]
    if '.' in stem:
        stem = stem.split('.')[0]
    return stem


def stage1_to_detection(
    stage1: Mapping[str, Any],
    swap_order: bool,
    ensure_two_cavs: bool,
    max_samples: int | None,
    keep_null: bool,
    deproject_to_local: bool,
    ego_index: int,
    required_cav_substrings: Sequence[str] | None,
    data_info: Sequence[Mapping[str, Any]] | None = None,
) -> Dict[str, Any]:
    output: Dict[str, Any] = {}
    processed = 0
    if data_info:
        key_iter: Iterable[int] = range(len(data_info))
    else:
        key_iter = sorted(int(k) for k in stage1.keys())
    for key_int in key_iter:
        if max_samples is not None and processed >= max_samples:
            break
        entry = stage1.get(str(key_int))
        if entry is None:
            if keep_null:
                output[str(key_int)] = None
            continue
        normalized = normalize_entry(
            entry,
            swap_order,
            ensure_two_cavs,
            deproject_to_local,
            ego_index,
            required_cav_substrings,
        )
        if not normalized:
            if keep_null:
                output[str(key_int)] = None
            continue
        if data_info and 0 <= key_int < len(data_info):
            info = data_info[key_int]
            infra_path = info.get('infrastructure_pointcloud_path')
            veh_path = info.get('vehicle_pointcloud_path')
            normalized['infra_frame_id'] = _extract_frame_id(infra_path)
            normalized['veh_frame_id'] = _extract_frame_id(veh_path)
        output[str(key_int)] = normalized
        processed += 1
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert HEAL stage1_boxes.json to calibration detection cache format."
    )
    parser.add_argument('--stage1', required=True,
                        help='Path to HEAL stage1_boxes.json (file or directory).')
    parser.add_argument('--output', required=True,
                        help='Destination JSON path for the detection cache.')
    parser.add_argument('--swap-order', action='store_true',
                        help='Swap the first two CAV entries (useful when HEAL exports vehicle-first).')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Optional limit on number of samples to export.')
    parser.add_argument('--keep-null', action='store_true',
                        help='Keep entries with missing predictions as null in the output.')
    parser.add_argument('--require-two-cavs', action='store_true',
                        help='Skip entries that do not contain at least two CAV predictions.')
    parser.add_argument('--deproject-to-local', action='store_true',
                        help='Convert boxes from ego frame back to each CAV local frame using lidar_pose_clean_np.')
    parser.add_argument('--ego-index', type=int, default=0,
                        help='Index of the ego/reference pose inside lidar_pose_clean_np when deprojecting (default: 0).')
    parser.add_argument('--data-info', type=str, default=None,
                        help='Optional data_info.json used to attach infra/vehicle frame IDs to each record.')
    parser.add_argument(
        '--require-cav-substrings',
        nargs='+',
        default=None,
        help='Only keep entries whose cav_id_list contains all provided substrings (case-insensitive).',
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stage1_path = Path(args.stage1)
    output_path = Path(args.output)
    data_info = None
    if args.data_info:
        info_path = Path(args.data_info)
        if not info_path.exists():
            raise FileNotFoundError(f"data_info file not found: {info_path}")
        with info_path.open('r', encoding='utf-8') as f:
            loaded = json.load(f)
        if not isinstance(loaded, list):
            raise ValueError(f"data_info file must be a list, got {type(loaded)}")
        data_info = loaded
    stage1_data = load_stage1_file(stage1_path)
    detection = stage1_to_detection(
        stage1_data,
        swap_order=args.swap_order,
        ensure_two_cavs=args.require_two_cavs,
        max_samples=args.max_samples,
        keep_null=args.keep_null,
        deproject_to_local=args.deproject_to_local,
        ego_index=args.ego_index,
        required_cav_substrings=args.require_cav_substrings,
        data_info=data_info,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', encoding='utf-8') as f:
        json.dump(detection, f, indent=2)
    print(f"Wrote detection cache with {len(detection)} entries to {output_path}")


if __name__ == '__main__':
    main()
