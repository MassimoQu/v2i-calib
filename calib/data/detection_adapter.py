from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from v2x_calib.reader.BBox3d import BBox3d


class DetectionAdapter:
    """Loads cached detection results and converts them into BBox3d objects."""

    def __init__(self, cache_path: Optional[str] = None) -> None:
        self.cache_path = Path(cache_path) if cache_path else None
        self._indexed: List[Dict[str, Any]] = []
        self._map: Dict[str, Dict[str, Any]] = {}
        if self.cache_path and self.cache_path.exists():
            with self.cache_path.open('r', encoding='utf-8') as f:
                raw = json.load(f)
            if isinstance(raw, dict) and all(k.isdigit() for k in raw.keys()):
                for idx in sorted(map(int, raw.keys())):
                    self._indexed.append(raw[str(idx)])
            elif isinstance(raw, dict):
                self._map = raw

    def _convert_bbox(
        self,
        entry: Any,
        default_type: str = 'detected',
    ) -> BBox3d:
        bbox_type = default_type
        confidence = 1.0
        corners = entry
        descriptor = None
        if isinstance(entry, dict):
            corners = entry.get('corners') or entry.get('points') or entry.get('bbox')
            bbox_type = entry.get('type', default_type)
            confidence = entry.get('score', entry.get('confidence', confidence))
            descriptor = entry.get('descriptor')
            if descriptor is not None:
                descriptor = np.asarray(descriptor, dtype=np.float32)
        arr = np.asarray(corners, dtype=np.float32)
        return BBox3d(bbox_type, arr, confidence=confidence, descriptor=descriptor)

    def _resolve_record(
        self,
        idx: Optional[int] = None,
        infra_id: Optional[str] = None,
        veh_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        record: Optional[Dict[str, Any]] = None
        if idx is not None and 0 <= idx < len(self._indexed):
            record = self._indexed[idx]
        elif infra_id and veh_id and self._map:
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
        for idx, cav_boxes in enumerate(pred_list):
            converted = []
            if isinstance(cav_boxes, list):
                for box in cav_boxes:
                    converted.append(self._convert_bbox(box, default_type=default_type))
            if idx == 0:
                infra_boxes = converted
            elif idx == 1:
                veh_boxes = converted
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
