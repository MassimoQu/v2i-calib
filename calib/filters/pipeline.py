from __future__ import annotations

from typing import Dict, List, Tuple

from calib.config import FilterConfig
from v2x_calib.preprocess import Filter3dBoxes
from v2x_calib.utils import get_volume_from_bbox3d_8_3, get_lwh_from_bbox3d_8_3


class FilterPipeline:
    def __init__(self, config: FilterConfig) -> None:
        self.config = config
        self._priority_map = {
            cat.lower(): idx for idx, cat in enumerate(config.priority_categories or [])
        }
        self._per_category_limits: Dict[str, int] = {
            cat.lower(): limit
            for cat, limit in (config.per_category_top_k or {}).items()
            if limit is not None
        }
        self._size_bounds: Dict[str, Dict[str, float]] = {}
        for cat, bounds in (config.size_bounds or {}).items():
            key = cat.lower() if cat.lower() != '__default__' else '__default__'
            self._size_bounds[key] = bounds or {}

    def _distance_filter(self, boxes):
        if self.config.distance_m <= 0:
            return boxes
        return Filter3dBoxes(boxes).filter_according_to_distance(self.config.distance_m)

    def _confidence_filter(self, boxes):
        thr = self.config.min_confidence or 0.0
        if thr <= 0.0:
            return boxes
        filtered = []
        for box in boxes:
            confidence = box.get_confidence() if hasattr(box, 'get_confidence') else 1.0
            if confidence >= thr:
                filtered.append(box)
        return filtered

    def _size_filter(self, boxes):
        if not self._size_bounds:
            return boxes
        filtered = []
        for box in boxes:
            if self._within_size_bounds(box):
                filtered.append(box)
        return filtered

    def _within_size_bounds(self, box) -> bool:
        cat = box.get_bbox_type().lower()
        bounds = self._size_bounds.get(cat) or self._size_bounds.get('__default__')
        if not bounds:
            return True
        l, w, h = get_lwh_from_bbox3d_8_3(box.get_bbox3d_8_3())

        def _check(value: float, min_key: str, max_key: str) -> bool:
            min_val = bounds.get(min_key)
            max_val = bounds.get(max_key)
            if min_val is not None and value < float(min_val):
                return False
            if max_val is not None and value > float(max_val):
                return False
            return True

        return (
            _check(l, 'min_l', 'max_l')
            and _check(w, 'min_w', 'max_w')
            and _check(h, 'min_h', 'max_h')
        )

    def _priority_key(self, box):
        cat = box.get_bbox_type().lower()
        priority = self._priority_map.get(cat, len(self._priority_map))
        volume = get_volume_from_bbox3d_8_3(box.get_bbox3d_8_3())
        confidence = box.get_confidence()
        return (priority, -volume, -confidence)

    def _apply_per_category_limits(self, boxes):
        if not self._per_category_limits:
            return boxes
        grouped: Dict[str, List] = {}
        for box in boxes:
            grouped.setdefault(box.get_bbox_type().lower(), []).append(box)
        selected: List = []
        leftovers: List = []
        for cat, cat_boxes in grouped.items():
            limit = self._per_category_limits.get(cat)
            if limit is None:
                selected.extend(cat_boxes)
                continue
            if limit <= 0:
                leftovers.extend(cat_boxes)
                continue
            sorted_cat = sorted(cat_boxes, key=self._priority_key)
            selected.extend(sorted_cat[:limit])
            leftovers.extend(sorted_cat[limit:])
        selected.extend(leftovers)
        return selected

    def _apply(self, boxes):
        boxes = self._distance_filter(boxes)
        boxes = self._confidence_filter(boxes)
        boxes = self._size_filter(boxes)
        boxes = self._apply_per_category_limits(boxes)
        sorted_boxes = sorted(boxes, key=self._priority_key)
        if self.config.top_k > 0:
            sorted_boxes = sorted_boxes[: self.config.top_k]
        return sorted_boxes

    def apply(self, infra_boxes, veh_boxes) -> Tuple[List, List]:
        filtered_infra = self._apply(infra_boxes)
        filtered_vehicle = self._apply(veh_boxes)
        return filtered_infra, filtered_vehicle


__all__ = ['FilterPipeline']
