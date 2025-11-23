import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import yaml

from .BBox3d import BBox3d
from . import noise_utils
from ..utils import (
    convert_6DOF_to_T,
    implement_T_3dbox_object_list,
    implement_T_to_3dbox_with_own_center,
)


def _pose_to_matrix(pose_vec: Sequence[float]) -> np.ndarray:
    """
    Converts [x, y, z, roll, yaw, pitch] (angles in degrees)
    into a 4x4 homogeneous matrix that maps the local frame to the world frame.
    """
    if len(pose_vec) != 6:
        raise ValueError(f"Expected pose with 6 DoF, got {pose_vec}")
    x, y, z, roll, yaw, pitch = pose_vec
    r = np.radians([roll, yaw, pitch])
    cr, cy, cp = np.cos(r)
    sr, sy, sp = np.sin(r)

    matrix = np.eye(4, dtype=np.float64)
    matrix[0, 3], matrix[1, 3], matrix[2, 3] = x, y, z

    # Rotation follows Carla's intrinsic xyz convention (roll, yaw, pitch).
    matrix[0, 0] = cp * cy
    matrix[0, 1] = cy * sp * sr - sy * cr
    matrix[0, 2] = -cy * sp * cr - sy * sr
    matrix[1, 0] = sy * cp
    matrix[1, 1] = sy * sp * sr + cy * cr
    matrix[1, 2] = -sy * sp * cr + cy * sr
    matrix[2, 0] = sp
    matrix[2, 1] = -cp * sr
    matrix[2, 2] = cp * cr
    return matrix


def _create_bbox(extent: Sequence[float]) -> np.ndarray:
    """
    Build an 8x3 array of bbox corners for Carla-style extents (half lengths).
    """
    if len(extent) != 3:
        raise ValueError(f"Extent must have 3 entries, got {extent}")
    l, w, h = extent
    return np.array(
        [
            [l, -w, -h],
            [l, w, -h],
            [-l, w, -h],
            [-l, -w, -h],
            [l, -w, h],
            [l, w, h],
            [-l, w, h],
            [-l, -w, h],
        ],
        dtype=np.float64,
    )


def _load_yaml(path: Path) -> Dict:
    if path.suffix == ".json":
        return json.loads(path.read_text())
    with path.open("r") as f:
        return yaml.safe_load(f)


def _sortable_id(value: str):
    try:
        return int(value)
    except (ValueError, TypeError):
        return value


@dataclass(frozen=True)
class _FrameMeta:
    scene: Path
    timestamp: str
    cav_dirs: Tuple[Path, ...]


class V2XSetReader:
    """
    Lightweight reader that mirrors the V2X-Sim API but consumes the
    OpenCOOD V2X-Set split stored under cooperative-vehicle-infrastructure/v2xset.
    """

    def __init__(
        self,
        root_dir: str = "/mnt/ssd_gw/cooperative-vehicle-infrastructure/v2xset",
        split: str = "validate",
        max_cavs: int = 4,
        frame_stride: int = 1,
        range_limits: Optional[Sequence[float]] = (-45.0, -45.0, -3.0, 45.0, 45.0, 2.0),
    ) -> None:
        self.root_dir = Path(root_dir)
        self.split = split
        self.max_cavs = max_cavs
        self.frame_stride = max(1, frame_stride)
        self.range_limits = range_limits
        split_dir = self.root_dir / split
        if not split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {split_dir}")
        self._frames: List[_FrameMeta] = self._index_frames(split_dir)

    def _index_frames(self, split_dir: Path) -> List[_FrameMeta]:
        frames: List[_FrameMeta] = []
        scene_dirs = sorted([p for p in split_dir.iterdir() if p.is_dir()])
        for scene in scene_dirs:
            cav_dirs = []
            for sub in sorted(scene.iterdir()):
                if not sub.is_dir():
                    continue
                try:
                    _ = int(sub.name)
                except ValueError:
                    continue
                cav_dirs.append(sub)
            if len(cav_dirs) < 2:
                continue
            cav_dirs = cav_dirs[: self.max_cavs]
            timestamps = self._collect_timestamps(cav_dirs[0])
            for idx, ts in enumerate(timestamps):
                if idx % self.frame_stride != 0:
                    continue
                frames.append(_FrameMeta(scene=scene, timestamp=ts, cav_dirs=tuple(cav_dirs)))
        return frames

    @staticmethod
    def _collect_timestamps(cav_dir: Path) -> List[str]:
        timestamps = sorted(
            f.stem for f in cav_dir.glob("*.yaml") if "additional" not in f.stem
        )
        return timestamps

    def __len__(self) -> int:
        return len(self._frames)

    def _load_frame(self, meta: _FrameMeta) -> Dict[str, Dict]:
        frame: Dict[str, Dict] = {}
        for cav_dir in meta.cav_dirs:
            yaml_path = cav_dir / f"{meta.timestamp}.yaml"
            if not yaml_path.exists():
                continue
            params = _load_yaml(yaml_path)
            lidar_pose = params.get("lidar_pose")
            if lidar_pose is None:
                continue
            frame[cav_dir.name] = {
                "pose_vec": lidar_pose,
                "pose_mat": _pose_to_matrix(lidar_pose),
                "vehicles": params.get("vehicles", {}),
                "pcd_path": cav_dir / f"{meta.timestamp}.pcd",
            }
        return frame

    def _merge_world_bboxes(self, frame: Dict[str, Dict]) -> List[BBox3d]:
        objects: Dict[str, Dict] = {}
        for cav in frame.values():
            objects.update(cav.get("vehicles", {}))

        bbox_world_list: List[BBox3d] = []
        for obj_id in sorted(objects.keys(), key=_sortable_id):
            info = objects[obj_id]
            location = info.get("location")
            extent = info.get("extent")
            angle = info.get("angle", [0.0, 0.0, 0.0])
            center_offset = info.get("center", [0.0, 0.0, 0.0])
            if location is None or extent is None:
                continue
            center = [location[0] + center_offset[0],
                      location[1] + center_offset[1],
                      location[2] + center_offset[2]]
            pose = [
                center[0],
                center[1],
                center[2],
                float(angle[0]),
                float(angle[1]),
                float(angle[2]),
            ]
            corners = _create_bbox(extent)
            corners_h = np.concatenate([corners, np.ones((8, 1))], axis=1)
            corners_world = ( _pose_to_matrix(pose) @ corners_h.T ).T[:, :3]
            bbox_world_list.append(BBox3d('car', corners_world))
        return bbox_world_list

    def _filter_boxes(self, boxes: List[BBox3d]) -> List[BBox3d]:
        if self.range_limits is None:
            return boxes
        x_min, y_min, z_min, x_max, y_max, z_max = self.range_limits
        filtered: List[BBox3d] = []
        for box in boxes:
            center = np.mean(box.get_bbox3d_8_3(), axis=0)
            if (
                x_min <= center[0] <= x_max
                and y_min <= center[1] <= y_max
                and z_min <= center[2] <= z_max
            ):
                filtered.append(box)
        return filtered

    def _apply_noise(
        self,
        boxes: List[BBox3d],
        noise_type: Optional[str],
        noise: Dict[str, float],
    ) -> List[BBox3d]:
        if noise_type is None:
            return boxes
        noised: List[BBox3d] = []
        for box in boxes:
            if noise_type == "gaussian":
                noise_6d = noise_utils.generate_noise(
                    noise.get("pos_std", 0.0),
                    noise.get("rot_std", 0.0),
                    noise.get("pos_mean", 0.0),
                    noise.get("rot_mean", 0.0),
                )
            elif noise_type == "laplace":
                pos_b = noise.get("pos_std", 0.0) / np.sqrt(2.0)
                rot_b = noise.get("rot_std", 0.0) / np.sqrt(2.0)
                noise_6d = noise_utils.generate_noise_laplace(
                    pos_b,
                    rot_b,
                    noise.get("pos_mean", 0.0),
                    noise.get("rot_mean", 0.0),
                )
            elif noise_type == "von_mises":
                noise_6d = noise_utils.generate_noise_von_mises(
                    noise.get("pos_std", 0.0),
                    noise.get("rot_std", 0.0),
                    noise.get("pos_mean", 0.0),
                    noise.get("rot_mean", 0.0),
                )
            else:
                raise ValueError(f"Unsupported noise type: {noise_type}")

            noised_bbox = box.copy()
            transformed = implement_T_to_3dbox_with_own_center(
                convert_6DOF_to_T(noise_6d), noised_bbox.get_bbox3d_8_3()
            )
            noised_bbox.bbox3d_8_3 = transformed
            noised.append(noised_bbox)
        return noised

    def _iter_pairs(
        self,
        start_idx: int = 0,
        end_idx: Optional[int] = None,
    ) -> Iterator[Tuple[int, Dict[str, Dict], List[BBox3d]]]:
        for frame_idx, meta in enumerate(self._frames):
            if frame_idx < start_idx:
                continue
            if end_idx is not None and frame_idx >= end_idx:
                break
            frame = self._load_frame(meta)
            if len(frame) < 2:
                continue
            bbox_world = self._merge_world_bboxes(frame)
            if not bbox_world:
                continue
            yield frame_idx, frame, bbox_world

    def generate_vehicle_vehicle_bboxes_object_list(
        self,
        start_idx: int = 0,
        end_idx: Optional[int] = None,
        noise_type: Optional[str] = None,
        noise: Optional[Dict[str, float]] = None,
    ):
        noise = noise or {"pos_std": 0.0, "rot_std": 0.0, "pos_mean": 0.0, "rot_mean": 0.0}
        for frame_idx, frame, bbox_world in self._iter_pairs(start_idx, end_idx):
            cav_ids = sorted(frame.keys(), key=_sortable_id)
            for idx_a in range(len(cav_ids) - 1):
                cav_a = frame[cav_ids[idx_a]]
                pose_a = cav_a["pose_mat"]
                world_to_a = np.linalg.inv(pose_a)
                boxes_a = implement_T_3dbox_object_list(world_to_a, bbox_world)
                boxes_a = self._filter_boxes(boxes_a)
                boxes_a = self._apply_noise(boxes_a, noise_type, noise)
                if not boxes_a:
                    continue
                for idx_b in range(idx_a + 1, len(cav_ids)):
                    cav_b = frame[cav_ids[idx_b]]
                    pose_b = cav_b["pose_mat"]
                    world_to_b = np.linalg.inv(pose_b)
                    boxes_b = implement_T_3dbox_object_list(world_to_b, bbox_world)
                    boxes_b = self._filter_boxes(boxes_b)
                    boxes_b = self._apply_noise(boxes_b, noise_type, noise)
                    if not boxes_b:
                        continue
                    T_b_a = np.dot(world_to_b, pose_a)
                    yield (
                        frame_idx,
                        f"{cav_ids[idx_a]}->{cav_ids[idx_b]}",
                        boxes_a,
                        boxes_b,
                        T_b_a,
                    )

    def generate_vehicle_vehicle_pointcloud(
        self,
        start_idx: int = 0,
        end_idx: Optional[int] = None,
    ):
        import open3d as o3d

        def _load_pcd(path: Path) -> np.ndarray:
            if not path.exists():
                return np.zeros((0, 3), dtype=np.float32)
            pcd = o3d.io.read_point_cloud(str(path))
            return np.asarray(pcd.points, dtype=np.float32)

        for frame_idx, frame, _ in self._iter_pairs(start_idx, end_idx):
            cav_ids = sorted(frame.keys(), key=int)
            for idx_a in range(len(cav_ids) - 1):
                cav_a = frame[cav_ids[idx_a]]
                for idx_b in range(idx_a + 1, len(cav_ids)):
                    cav_b = frame[cav_ids[idx_b]]
                    T_b_a = np.dot(
                        np.linalg.inv(cav_b["pose_mat"]), cav_a["pose_mat"]
                    )
                    pc_a = _load_pcd(cav_a["pcd_path"])
                    pc_b = _load_pcd(cav_b["pcd_path"])
                    yield (
                        f"{frame_idx}_{cav_ids[idx_a]}",
                        f"{frame_idx}_{cav_ids[idx_b]}",
                        pc_a,
                        pc_b,
                        T_b_a,
                    )
