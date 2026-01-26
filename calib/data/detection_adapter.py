from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from v2x_calib.reader.BBox3d import BBox3d
from v2x_calib.utils import get_bbox3d_8_3_from_xyz_lwh_yaw


class DetectionAdapter:
    """Loads cached detection results and converts them into BBox3d objects."""

    def __init__(self, cache_path: Optional[str] = None) -> None:
        self.cache_path = Path(cache_path) if cache_path else None
        self._indexed: List[Dict[str, Any]] = []
        self._map: Dict[str, Any] = {}
        if self.cache_path and self.cache_path.exists():
            with self.cache_path.open('r', encoding='utf-8') as f:
                raw = json.load(f)
            if isinstance(raw, dict):
                self._map = {str(k): v for k, v in raw.items()}
                digit_keys = [k for k in self._map.keys() if str(k).isdigit()]
                if len(digit_keys) == len(self._map):
                    numeric_keys = sorted(int(k) for k in digit_keys)
                    if numeric_keys == list(range(len(numeric_keys))):
                        self._indexed = [self._map[str(idx)] for idx in numeric_keys]

    def _convert_bbox(
        self,
        entry: Any,
        default_type: str = 'detected',
    ) -> BBox3d:
        bbox_type = default_type
        confidence = 1.0
        corners = entry
        box7d = None
        descriptor = None
        if isinstance(entry, dict):
            corners = entry.get('corners') or entry.get('points') or entry.get('bbox')
            bbox_type = entry.get('type', default_type)
            confidence = entry.get('score', entry.get('confidence', confidence))
            descriptor = entry.get('descriptor')
            box7d = entry.get('box7d') or entry.get('box3d') or entry.get('pred_box3d')
            if descriptor is not None:
                descriptor = np.asarray(descriptor, dtype=np.float32)
        if corners is None and box7d is not None:
            corners = box7d
        arr = np.asarray(corners, dtype=np.float32)
        if arr.ndim == 1 and arr.size == 24:
            arr = arr.reshape(8, 3)
        if arr.shape == (7,):
            xyz = arr[:3]
            lwh = arr[3:6]
            yaw = float(arr[6])
            arr = np.asarray(get_bbox3d_8_3_from_xyz_lwh_yaw(xyz, lwh, yaw), dtype=np.float32)
        return BBox3d(bbox_type, arr, confidence=confidence, descriptor=descriptor)

    def _record_can_validate_ids(self, record: Optional[Dict[str, Any]]) -> bool:
        if not isinstance(record, dict):
            return False
        rec_infra = str(record.get('infra_frame_id') or record.get('infra_id') or '').strip()
        rec_veh = str(record.get('veh_frame_id') or record.get('veh_id') or '').strip()
        return bool(rec_infra) and bool(rec_veh)

    def _record_matches_ids(
        self,
        record: Optional[Dict[str, Any]],
        infra_id: Optional[str],
        veh_id: Optional[str],
    ) -> bool:
        if not isinstance(record, dict) or not infra_id or not veh_id:
            return False
        if not self._record_can_validate_ids(record):
            return False
        rec_infra = str(record.get('infra_frame_id') or record.get('infra_id') or '').lower()
        rec_veh = str(record.get('veh_frame_id') or record.get('veh_id') or '').lower()
        return rec_infra == str(infra_id).lower() and rec_veh == str(veh_id).lower()

    def _resolve_record(
        self,
        idx: Optional[int] = None,
        infra_id: Optional[str] = None,
        veh_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        record: Optional[Dict[str, Any]] = None
        if idx is not None:
            key = str(idx)
            if key in self._map:
                candidate = self._map[key]
                if self._record_matches_ids(candidate, infra_id, veh_id) or not infra_id or not veh_id:
                    record = candidate
                elif not self._record_can_validate_ids(candidate):
                    record = candidate
            elif 0 <= idx < len(self._indexed):
                candidate = self._indexed[idx]
                if self._record_matches_ids(candidate, infra_id, veh_id) or not infra_id or not veh_id:
                    record = candidate
                elif not self._record_can_validate_ids(candidate):
                    record = candidate
        if record is None and infra_id and veh_id:
            record = self._find_by_ids(infra_id, veh_id)
        return record

    def _convert_record(
        self,
        record: Dict[str, Any],
        field: str,
        default_type: str = 'detected',
    ) -> Tuple[List[BBox3d], List[BBox3d]]:
        infra_boxes: List[BBox3d] = []
        veh_boxes: List[BBox3d] = []
        pred_list = record.get(field, [])
        if not isinstance(pred_list, list):
            return infra_boxes, veh_boxes

        def _infer_agent_indices() -> Tuple[int, int]:
            cav_ids = record.get('cav_id_list')
            if not isinstance(cav_ids, list):
                return 0, 1
            infra_idx = None
            veh_idx = None
            for cav_idx, cav_id in enumerate(cav_ids):
                cav_norm = str(cav_id).lower()
                if infra_idx is None and ('infra' in cav_norm or 'infrastructure' in cav_norm or 'rsu' in cav_norm):
                    infra_idx = cav_idx
                if veh_idx is None and ('veh' in cav_norm or 'vehicle' in cav_norm):
                    veh_idx = cav_idx
            if infra_idx is None:
                infra_idx = 0
            if veh_idx is None:
                veh_idx = 1 if infra_idx == 0 else 0
            return infra_idx, veh_idx

        infra_idx, veh_idx = _infer_agent_indices()
        score_all = None
        for key in ('pred_score_np_list', 'score_np_list', 'scores_np_list'):
            if key in record:
                score_all = record.get(key)
                break

        def _convert_agent(cav_idx: int) -> List[BBox3d]:
            if cav_idx < 0 or cav_idx >= len(pred_list):
                return []
            cav_boxes = pred_list[cav_idx]
            if not isinstance(cav_boxes, list):
                return []
            cav_scores = None
            if isinstance(score_all, list) and cav_idx < len(score_all) and isinstance(score_all[cav_idx], list):
                cav_scores = score_all[cav_idx]
            converted: List[BBox3d] = []
            for box_idx, box in enumerate(cav_boxes):
                if cav_scores is not None and box_idx < len(cav_scores) and not isinstance(box, dict):
                    box = {'corners': box, 'score': cav_scores[box_idx]}
                converted.append(self._convert_bbox(box, default_type=default_type))
            return converted

        infra_boxes = _convert_agent(infra_idx)
        veh_boxes = _convert_agent(veh_idx)
        return infra_boxes, veh_boxes

    def _find_by_ids(self, infra_id: str, veh_id: str) -> Optional[Dict[str, Any]]:
        candidate_keys = (
            f"{infra_id}_{veh_id}",
            f"{veh_id}_{infra_id}",
            infra_id,
            veh_id,
        )
        for key in candidate_keys:
            if key in self._map:
                return self._map[key]
        if infra_id and veh_id:
            infra_norm = str(infra_id).lower()
            veh_norm = str(veh_id).lower()
            for record in self._map.values():
                if not isinstance(record, dict):
                    continue
                rec_infra = str(record.get('infra_frame_id') or record.get('infra_id') or '').lower()
                rec_veh = str(record.get('veh_frame_id') or record.get('veh_id') or '').lower()
                if rec_infra == infra_norm and rec_veh == veh_norm:
                    return record
        return None

    def get(
        self,
        idx: Optional[int] = None,
        infra_id: Optional[str] = None,
        veh_id: Optional[str] = None,
        field: str = 'pred_corner3d_np_list',
        default_type: str = 'detected',
    ) -> Tuple[Optional[List[BBox3d]], Optional[List[BBox3d]]]:
        record = self._resolve_record(idx=idx, infra_id=infra_id, veh_id=veh_id)
        if record is None:
            return None, None
        boxes = self._convert_record(record, field=field, default_type=default_type)
        return boxes

    def get_record(
        self,
        idx: Optional[int] = None,
        infra_id: Optional[str] = None,
        veh_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        return self._resolve_record(idx=idx, infra_id=infra_id, veh_id=veh_id)

    def convert_record(
        self,
        record: Dict[str, Any],
        field: str = 'pred_corner3d_np_list',
        default_type: str = 'detected',
    ) -> Tuple[List[BBox3d], List[BBox3d]]:
        return self._convert_record(record, field=field, default_type=default_type)


__all__ = ['DetectionAdapter']
