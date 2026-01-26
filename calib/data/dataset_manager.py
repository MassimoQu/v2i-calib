from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Generator, List, Optional, Tuple

import numpy as np
try:
    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None  # type: ignore[assignment]

from v2x_calib.reader import CooperativeBatchingReader, CooperativeReader
from v2x_calib.utils import (
    convert_6DOF_to_T,
    implement_T_3dbox_object_list,
    get_volume_from_bbox3d_8_3,
)
from legacy.v2x_calib.reader import noise_utils

from calib.config import DataConfig
from .interfaces import CalibrationSample
from .detection_adapter import DetectionAdapter


class DatasetManager:
    def __init__(self, config: DataConfig) -> None:
        self.config = config
        self.reader = CooperativeBatchingReader(
            path_data_info=config.data_info_path,
            path_data_folder=config.data_root,
        )
        self.detection_adapter = DetectionAdapter(config.detection_cache)
        self.feature_adapter = DetectionAdapter(config.feature_cache)
        self._detection_vehicle_flip_y: Optional[bool] = None
        self._sensor_frame = str(getattr(config, 'sensor_frame', 'lidar')).lower().strip()
        if self._sensor_frame not in {'lidar', 'camera'}:
            self._sensor_frame = 'lidar'
        self._use_image_descriptors = bool(getattr(config, 'use_image_descriptors', False))
        self._image_descriptor_cfg = dict(getattr(config, 'image_descriptor', {}) or {})
        self._canonicalize_detection_corners = bool(
            getattr(config, 'canonicalize_detection_corners', False)
        )
        self._data_infos = self._load_data_infos(config.data_info_path)
        self._start_index = max(0, int(getattr(config, 'start_index', 0) or 0))
        self._shuffle_flags = {
            key.lower(): bool(value) for key, value in (config.shuffle_box_vertices or {}).items()
        }
        self._vertex_perm = np.array([7, 6, 5, 4, 3, 2, 1, 0], dtype=int)
        self._noise_cfg = dict(config.noise or {})

    @staticmethod
    def _load_data_infos(path: str) -> Optional[list]:
        try:
            with Path(path).open('r', encoding='utf-8') as f:
                payload = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return None
        if isinstance(payload, list):
            return payload
        return None

    def _should_shuffle(self, agent: str) -> bool:
        if not self._shuffle_flags:
            return False
        agent_key = agent.lower()
        if agent_key in self._shuffle_flags:
            return self._shuffle_flags[agent_key]
        return self._shuffle_flags.get('both', False)

    def _shuffle_boxes(self, boxes, agent: str):
        if not boxes or not self._should_shuffle(agent):
            return boxes
        shuffled = []
        for box in boxes:
            copied = box.copy()
            corners = np.asarray(copied.get_bbox3d_8_3())
            copied.bbox3d_8_3 = np.asarray(corners[self._vertex_perm])
            shuffled.append(copied)
        return shuffled

    def _apply_noise(self, boxes, agent: str):
        if not boxes or not self._noise_cfg:
            return boxes
        target = str(self._noise_cfg.get('target', 'vehicle')).lower()
        if target not in {'infra', 'vehicle', 'both'}:
            target = 'vehicle'
        if target != 'both' and agent.lower() != target:
            return boxes
        pos_std = float(self._noise_cfg.get('pos_std', 0.0))
        rot_std = float(self._noise_cfg.get('rot_std', 0.0))
        pos_mean = float(self._noise_cfg.get('pos_mean', 0.0))
        rot_mean = float(self._noise_cfg.get('rot_mean', 0.0))
        offset = self._noise_cfg.get('offset')
        has_gaussian = any(val != 0.0 for val in (pos_std, rot_std, pos_mean, rot_mean))
        if not has_gaussian and offset is None:
            return boxes
        if has_gaussian:
            noise_vec = noise_utils.generate_noise(pos_std, rot_std, pos_mean, rot_mean)
        else:
            noise_vec = np.zeros(6, dtype=float)
        if offset is not None:
            extra = np.zeros(6, dtype=float)
            values = list(offset) if isinstance(offset, (list, tuple, np.ndarray)) else [offset]
            for idx, val in enumerate(values[:6]):
                extra[idx] = float(val)
            noise_vec = noise_vec + extra
        if np.allclose(noise_vec, 0.0):
            return boxes
        delta_T = convert_6DOF_to_T(noise_vec)
        return implement_T_3dbox_object_list(delta_T, boxes)

    def _get_cooperative_reader(self, infra_id: str, veh_id: str) -> CooperativeReader:
        candidate = getattr(self.reader, 'cooperative_reader', None)
        if candidate is not None:
            try:
                cand_infra = getattr(candidate.infra_reader, 'infra_file_name', None)
                cand_veh = getattr(candidate.vehicle_reader, 'vehicle_file_name', None)
                if cand_infra == infra_id and cand_veh == veh_id:
                    return candidate
            except Exception:
                pass
        return CooperativeReader(infra_id, veh_id, self.config.data_root)

    def _apply_camera_frame(self, boxes, lidar2cam_T):
        if boxes is None:
            return None
        if not boxes:
            return []
        if lidar2cam_T is None:
            return boxes
        return implement_T_3dbox_object_list(lidar2cam_T, boxes)

    def _resolve_image_paths(self, idx: int, infra_id: str, veh_id: str) -> Tuple[Optional[Path], Optional[Path]]:
        root = Path(self.config.data_root)
        infra_path = None
        veh_path = None
        if self._data_infos and 0 <= idx < len(self._data_infos):
            record = self._data_infos[idx]
            if isinstance(record, dict):
                infra_rel = record.get('infrastructure_image_path')
                veh_rel = record.get('vehicle_image_path')
                if infra_rel:
                    infra_path = root / str(infra_rel)
                if veh_rel:
                    veh_path = root / str(veh_rel)
        if infra_path is None:
            infra_path = root / 'infrastructure-side' / 'image' / f'{infra_id}.jpg'
        if veh_path is None:
            veh_path = root / 'vehicle-side' / 'image' / f'{veh_id}.jpg'
        return infra_path, veh_path

    def _load_image(self, path: Optional[Path]) -> Optional[Image.Image]:
        if Image is None:
            return None
        if path is None:
            return None
        try:
            img = Image.open(path)
        except (FileNotFoundError, OSError):
            return None
        try:
            return img.convert('RGB')
        except OSError:
            return None

    def _attach_image_descriptors(self, boxes, image: Optional[Image.Image]) -> None:
        if Image is None:
            return
        if not self._use_image_descriptors or not boxes or image is None:
            return
        cfg = self._image_descriptor_cfg
        merge_mode = str(
            cfg.get('merge_mode') or cfg.get('merge_strategy') or cfg.get('merge') or ''
        ).lower()
        allow_merge = merge_mode in {'concat', 'avg', 'mean', 'sum'}
        max_boxes = cfg.get('max_boxes') or cfg.get('top_k')
        try:
            max_boxes = int(max_boxes)
        except Exception:
            max_boxes = 0
        selected_indices = None
        if max_boxes and len(boxes) > max_boxes:
            raw_categories = cfg.get('priority_categories') or []
            priority_categories = [str(cat).lower() for cat in raw_categories]
            priority_map = {cat: idx for idx, cat in enumerate(priority_categories)}

            def _priority_key(item):
                idx, box = item
                try:
                    cat = str(box.get_bbox_type()).lower()
                except Exception:
                    cat = 'detected'
                priority = priority_map.get(cat, len(priority_map))
                try:
                    volume = float(get_volume_from_bbox3d_8_3(box.get_bbox3d_8_3()))
                except Exception:
                    volume = 0.0
                confidence = box.get_confidence() if hasattr(box, 'get_confidence') else 1.0
                return (priority, -volume, -confidence)

            ranked = sorted(enumerate(boxes), key=_priority_key)
            selected_indices = {idx for idx, _ in ranked[:max_boxes]}
        method = str(cfg.get('method') or cfg.get('descriptor_type') or cfg.get('type') or 'pixel').lower()
        patch_size = cfg.get('patch_size') or cfg.get('size') or (16, 16)
        try:
            patch_w, patch_h = int(patch_size[0]), int(patch_size[1])
        except Exception:
            patch_w, patch_h = 16, 16
        patch_w = max(4, patch_w)
        patch_h = max(4, patch_h)
        grayscale = bool(cfg.get('grayscale', True))
        min_area = float(cfg.get('min_area', 16.0))
        hog_cells = cfg.get('hog_cells') or cfg.get('cells') or (2, 2)
        try:
            hog_cells_w, hog_cells_h = int(hog_cells[0]), int(hog_cells[1])
        except Exception:
            hog_cells_w, hog_cells_h = 2, 2
        hog_bins = int(cfg.get('hog_bins', 8) or 8)
        hog_bins = max(4, hog_bins)
        model_name = str(cfg.get('model', 'resnet18') or 'resnet18')
        model_input_size = cfg.get('model_input_size') or cfg.get('input_size')
        try:
            if model_input_size is not None:
                if isinstance(model_input_size, (list, tuple)):
                    model_input_w, model_input_h = int(model_input_size[0]), int(model_input_size[1])
                else:
                    side = int(model_input_size)
                    model_input_w = model_input_h = side
            else:
                model_input_w = model_input_h = None
        except Exception:
            model_input_w = model_input_h = None

        img_w, img_h = image.size

        for idx, box in enumerate(boxes):
            if selected_indices is not None and idx not in selected_indices:
                continue
            existing_descriptor = getattr(box, 'descriptor', None)
            if existing_descriptor is not None and not allow_merge:
                continue
            if not hasattr(box, 'get_bbox2d_4'):
                continue
            bbox = box.get_bbox2d_4()
            if not bbox or len(bbox) != 4:
                continue
            try:
                x1, y1, x2, y2 = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
            except Exception:
                continue
            if x2 <= x1 or y2 <= y1:
                continue
            area = (x2 - x1) * (y2 - y1)
            if area < min_area:
                continue

            left = int(max(0, min(img_w - 1, round(x1))))
            upper = int(max(0, min(img_h - 1, round(y1))))
            right = int(max(0, min(img_w, round(x2))))
            lower = int(max(0, min(img_h, round(y2))))
            if right <= left or lower <= upper:
                continue
            try:
                patch = image.crop((left, upper, right, lower)).resize(
                    (patch_w, patch_h), resample=Image.BILINEAR
                )
            except Exception:
                continue
            if model_input_w and model_input_h and (model_input_w, model_input_h) != (patch_w, patch_h):
                try:
                    patch = patch.resize((model_input_w, model_input_h), resample=Image.BILINEAR)
                except Exception:
                    pass
            arr = np.asarray(patch, dtype=np.float32)
            if arr.size == 0:
                continue
            if arr.ndim == 3 and grayscale:
                arr = arr.mean(axis=2)
            if method in {'resnet', 'cnn', 'timm'}:
                try:
                    import torch
                    import torch.nn as nn
                except Exception:
                    method = 'pixel'
                else:
                    backend = getattr(self, '_image_descriptor_backend', None)
                    backend_name = backend.get('name') if isinstance(backend, dict) else None
                    if backend is None or backend_name != model_name:
                        device_name = str(cfg.get('device') or '').strip().lower()
                        if not device_name:
                            device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
                        if model_name.startswith('timm:'):
                            try:
                                import timm
                                core_name = model_name.split(':', 1)[1]
                                model = timm.create_model(core_name, pretrained=True, num_classes=0, global_pool='avg')
                            except Exception:
                                model = None
                        else:
                            try:
                                import torchvision
                                if hasattr(torchvision.models, model_name):
                                    base = getattr(torchvision.models, model_name)(pretrained=True)
                                    model = nn.Sequential(*list(base.children())[:-1])
                                else:
                                    model = None
                            except Exception:
                                model = None
                        if model is None:
                            method = 'pixel'
                        else:
                            model.eval()
                            model.to(device_name)
                            backend = {
                                'name': model_name,
                                'model': model,
                                'device': device_name,
                                'mean': torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32),
                                'std': torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32),
                            }
                            self._image_descriptor_backend = backend
                    if method in {'resnet', 'cnn', 'timm'} and backend:
                        if arr.ndim == 2:
                            arr = np.repeat(arr[:, :, None], 3, axis=2)
                        arr = arr / 255.0
                        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).float()
                        mean = backend['mean'][:, None, None]
                        std = backend['std'][:, None, None]
                        tensor = (tensor - mean) / std
                        model = backend['model']
                        device = backend['device']
                        with torch.no_grad():
                            feat = model(tensor.to(device))
                        feat = feat.reshape(-1).detach().cpu().numpy()
                        vec = feat.astype(np.float32, copy=False)
                    else:
                        vec = None
            if method in {'hog', 'grad', 'gradient'}:
                if arr.ndim != 2:
                    arr = arr.mean(axis=2)
                arr = arr / 255.0
                gx = np.zeros_like(arr)
                gy = np.zeros_like(arr)
                gx[:, 1:-1] = arr[:, 2:] - arr[:, :-2]
                gy[1:-1, :] = arr[2:, :] - arr[:-2, :]
                mag = np.hypot(gx, gy)
                ang = (np.arctan2(gy, gx) + np.pi) % (2 * np.pi)
                bin_idx = np.floor(ang / (2 * np.pi) * hog_bins).astype(np.int32)
                bin_idx[bin_idx >= hog_bins] = hog_bins - 1
                cell_w = max(1, int(arr.shape[1] // max(1, hog_cells_w)))
                cell_h = max(1, int(arr.shape[0] // max(1, hog_cells_h)))
                hist = np.zeros((hog_cells_h, hog_cells_w, hog_bins), dtype=np.float32)
                for cy in range(hog_cells_h):
                    y0 = cy * cell_h
                    y1 = arr.shape[0] if cy == hog_cells_h - 1 else (cy + 1) * cell_h
                    for cx in range(hog_cells_w):
                        x0 = cx * cell_w
                        x1 = arr.shape[1] if cx == hog_cells_w - 1 else (cx + 1) * cell_w
                        cell_bins = bin_idx[y0:y1, x0:x1].reshape(-1)
                        cell_mag = mag[y0:y1, x0:x1].reshape(-1)
                        if cell_bins.size == 0:
                            continue
                        hist[cy, cx] = np.bincount(cell_bins, weights=cell_mag, minlength=hog_bins)
                vec = hist.reshape(-1)
            elif method not in {'resnet', 'cnn', 'timm'}:
                vec = arr.reshape(-1) / 255.0
            if vec is None:
                continue
            if existing_descriptor is not None and allow_merge:
                try:
                    prev = np.asarray(existing_descriptor, dtype=np.float32).reshape(-1)
                except Exception:
                    prev = None
                if prev is not None and prev.size > 0:
                    if merge_mode == 'concat':
                        vec = np.concatenate([prev, vec.astype(np.float32, copy=False)], axis=0)
                    else:
                        dim = min(int(prev.size), int(vec.size))
                        if dim > 0:
                            merged = prev[:dim] + vec[:dim]
                            if merge_mode in {'avg', 'mean'}:
                                merged = merged / 2.0
                            vec = merged
            norm = float(np.linalg.norm(vec))
            if norm > 1e-6:
                vec = vec / norm
            try:
                box.descriptor = vec.astype(np.float32, copy=False)
            except Exception:
                continue

    def _maybe_get_detections(
        self, idx: int, infra_id: str, veh_id: str
    ) -> Tuple[Optional[list], Optional[list], Optional[Any], Optional[List[float]]]:
        use_boxes = bool(self.config.use_detection)
        load_hints = bool(getattr(self.config, 'load_detection_hints', False))
        if not use_boxes and not load_hints:
            return None, None, None, None
        record = self.detection_adapter.get_record(idx=idx, infra_id=infra_id, veh_id=veh_id)
        if record is None:
            return ([] if use_boxes else None), ([] if use_boxes else None), None, None
        field = getattr(self.config, 'detection_field', 'pred_corner3d_np_list')
        infra_boxes = veh_boxes = None
        if use_boxes:
            infra_boxes, veh_boxes = self.detection_adapter.convert_record(
                record, field=field, default_type='detected'
            )
            if self._canonicalize_detection_corners:
                infra_boxes = self._canonicalize_boxes(infra_boxes)
                veh_boxes = self._canonicalize_boxes(veh_boxes)
        occ_map = None
        if load_hints:
            occ_map = record.get('occ_map_level0')
            if occ_map is None:
                occ_paths = record.get('occ_map_level0_path')
                if occ_paths:
                    def _load_occ(path):
                        try:
                            data = np.load(path)
                        except Exception:
                            return None
                        try:
                            if hasattr(data, 'files'):
                                if 'occ_map_level0' in data.files:
                                    return data['occ_map_level0']
                                if data.files:
                                    return data[data.files[0]]
                            return data
                        finally:
                            try:
                                data.close()
                            except Exception:
                                pass
                    if isinstance(occ_paths, (list, tuple)):
                        occ_map = []
                        for p in occ_paths:
                            occ_map.append(_load_occ(p) if p else None)
                    else:
                        occ_map = _load_occ(occ_paths)
        bev_range = record.get('bev_range') if load_hints else None
        return infra_boxes, veh_boxes, occ_map, bev_range

    def _maybe_configure_detection_vehicle_flip(self, gt_vehicle_boxes, detections_vehicle) -> None:
        if self._detection_vehicle_flip_y is not None:
            return
        # Skip early frames without usable evidence; we'll decide once we have
        # both GT + detection boxes available.
        if not gt_vehicle_boxes or not detections_vehicle:
            return

        def _centers_xy(boxes):
            centers = []
            for box in boxes:
                pts = np.asarray(box.get_bbox3d_8_3(), dtype=np.float32)
                if pts.size == 0:
                    continue
                centers.append(pts.mean(axis=0)[:2])
            if not centers:
                return np.zeros((0, 2), dtype=np.float32)
            return np.stack(centers, axis=0)

        def _median_nn(a: np.ndarray, b: np.ndarray) -> float:
            if a.size == 0 or b.size == 0:
                return float('inf')
            diff = a[:, None, :] - b[None, :, :]
            dists = np.linalg.norm(diff, axis=2)
            return float(np.median(dists.min(axis=1)))

        gt_xy = _centers_xy(gt_vehicle_boxes)
        det_xy = _centers_xy(detections_vehicle)
        if gt_xy.size == 0 or det_xy.size == 0:
            self._detection_vehicle_flip_y = False
            return
        det_xy_flip = det_xy.copy()
        det_xy_flip[:, 1] *= -1
        med_raw = _median_nn(gt_xy, det_xy)
        med_flip = _median_nn(gt_xy, det_xy_flip)
        # Enable flip only when the improvement is obvious and stable.
        self._detection_vehicle_flip_y = bool(med_flip + 1e-6 < med_raw * 0.6 and med_flip + 1e-6 < med_raw - 1.0)

    def _apply_flip_y(self, boxes):
        flipped = []
        for box in boxes:
            copied = box.copy()
            pts = np.asarray(copied.get_bbox3d_8_3(), dtype=np.float32).copy()
            pts[:, 1] *= -1
            copied.bbox3d_8_3 = pts
            flipped.append(copied)
        return flipped

    @staticmethod
    def _canonicalize_box_corners(box):
        try:
            pts = np.asarray(box.get_bbox3d_8_3(), dtype=np.float64).reshape(-1, 3)
        except Exception:
            return box
        if pts.shape[0] != 8:
            return box

        z_min = float(np.min(pts[:, 2]))
        z_max = float(np.max(pts[:, 2]))
        h = float(max(z_max - z_min, 1e-3))
        center_xy = np.mean(pts[:, :2], axis=0)
        center_z = 0.5 * (z_min + z_max)
        center = np.array([float(center_xy[0]), float(center_xy[1]), float(center_z)], dtype=np.float64)

        bottom_idx = np.argsort(pts[:, 2])[:4]
        bottom_xy = pts[bottom_idx, :2]
        bottom_rel = bottom_xy - center_xy[None, :]
        cov = (bottom_rel.T @ bottom_rel) / max(1.0, float(bottom_rel.shape[0]))
        if not np.all(np.isfinite(cov)):
            return box

        yaw = 0.0
        try:
            eigvals, eigvecs = np.linalg.eigh(cov)
            principal = eigvecs[:, int(np.argmax(eigvals))]
            yaw = float(math.atan2(float(principal[1]), float(principal[0])))
        except Exception:
            yaw = 0.0

        cos_y = math.cos(yaw)
        sin_y = math.sin(yaw)
        rot = np.array([[cos_y, sin_y], [-sin_y, cos_y]], dtype=np.float64)  # -yaw
        bottom_rot = bottom_rel @ rot.T
        l = float(np.ptp(bottom_rot[:, 0]))
        w = float(np.ptp(bottom_rot[:, 1]))
        if not math.isfinite(l) or not math.isfinite(w):
            return box
        l = max(l, 1e-3)
        w = max(w, 1e-3)
        if l < w:
            l, w = w, l
            yaw += math.pi / 2.0

        # Canonicalize yaw modulo pi (eliminate 180° direction ambiguity).
        yaw = ((yaw + math.pi / 2.0) % math.pi) - math.pi / 2.0

        try:
            from v2x_calib.utils import get_bbox3d_8_3_from_xyz_lwh_yaw
        except Exception:
            return box

        canonical = np.asarray(
            get_bbox3d_8_3_from_xyz_lwh_yaw(center, [l, w, h], yaw), dtype=np.float32
        )
        copied = box.copy()
        copied.bbox3d_8_3 = canonical
        return copied

    def _canonicalize_boxes(self, boxes):
        if not boxes:
            return boxes
        return [self._canonicalize_box_corners(box) for box in boxes]

    def _maybe_get_features(
        self, idx: int, infra_id: str, veh_id: str
    ) -> Tuple[Optional[list], Optional[list]]:
        if not self.config.use_features:
            return None, None
        return self.feature_adapter.get(
            idx=idx,
            infra_id=infra_id,
            veh_id=veh_id,
            field=self.config.feature_field,
            default_type='feature',
        )

    def samples(self) -> Generator[CalibrationSample, None, None]:
        wrapper = self.reader.generate_infra_vehicle_bboxes_object_list(start_idx=self._start_index)
        produced = 0
        for idx, (inf_id, veh_id, infra_boxes, veh_boxes, T_true) in enumerate(wrapper, start=self._start_index):
            if self.config.max_samples is not None and produced >= self.config.max_samples:
                break
            produced += 1
            detections_infra, detections_vehicle, occ_map_level0, bev_range = self._maybe_get_detections(
                idx, inf_id, veh_id
            )
            feature_infra, feature_vehicle = self._maybe_get_features(
                idx, inf_id, veh_id
            )
            if detections_vehicle:
                self._maybe_configure_detection_vehicle_flip(veh_boxes, detections_vehicle)
            elif feature_vehicle:
                self._maybe_configure_detection_vehicle_flip(veh_boxes, feature_vehicle)
            if self._detection_vehicle_flip_y:
                if detections_vehicle:
                    detections_vehicle = self._apply_flip_y(detections_vehicle)
                if feature_vehicle:
                    feature_vehicle = self._apply_flip_y(feature_vehicle)
            if self._sensor_frame == 'camera':
                coop = self._get_cooperative_reader(inf_id, veh_id)
                try:
                    T_true = coop.get_cooperative_camera_T_i2v()
                except (FileNotFoundError, OSError):
                    continue
                try:
                    infra_lidar2cam, veh_lidar2cam = coop.get_infra_vehicle_lidar2camera()
                except Exception:
                    infra_lidar2cam = veh_lidar2cam = None
                infra_boxes = self._apply_camera_frame(infra_boxes, infra_lidar2cam)
                veh_boxes = self._apply_camera_frame(veh_boxes, veh_lidar2cam)
                detections_infra = self._apply_camera_frame(detections_infra, infra_lidar2cam)
                detections_vehicle = self._apply_camera_frame(detections_vehicle, veh_lidar2cam)
                feature_infra = self._apply_camera_frame(feature_infra, infra_lidar2cam)
                feature_vehicle = self._apply_camera_frame(feature_vehicle, veh_lidar2cam)

                if self._use_image_descriptors:
                    infra_img_path, veh_img_path = self._resolve_image_paths(idx, inf_id, veh_id)
                    infra_img = self._load_image(infra_img_path)
                    veh_img = self._load_image(veh_img_path)
                    for boxes in (infra_boxes, detections_infra, feature_infra):
                        self._attach_image_descriptors(boxes, infra_img)
                    for boxes in (veh_boxes, detections_vehicle, feature_vehicle):
                        self._attach_image_descriptors(boxes, veh_img)
            infra_boxes = self._shuffle_boxes(infra_boxes, 'infra')
            veh_boxes = self._shuffle_boxes(veh_boxes, 'vehicle')
            infra_boxes = self._apply_noise(infra_boxes, 'infra')
            veh_boxes = self._apply_noise(veh_boxes, 'vehicle')
            if detections_infra:
                detections_infra = self._shuffle_boxes(detections_infra, 'infra')
            if detections_vehicle:
                detections_vehicle = self._shuffle_boxes(detections_vehicle, 'vehicle')
            if feature_infra:
                feature_infra = self._shuffle_boxes(feature_infra, 'infra')
            if feature_vehicle:
                feature_vehicle = self._shuffle_boxes(feature_vehicle, 'vehicle')
            yield CalibrationSample(
                index=idx,
                infra_id=inf_id,
                veh_id=veh_id,
                infra_boxes=infra_boxes,
                veh_boxes=veh_boxes,
                T_true=T_true,
                detections_infra=detections_infra,
                detections_vehicle=detections_vehicle,
                features_infra=feature_infra,
                features_vehicle=feature_vehicle,
                occ_maps=occ_map_level0,
                bev_range=bev_range,
            )


__all__ = ['DatasetManager']
