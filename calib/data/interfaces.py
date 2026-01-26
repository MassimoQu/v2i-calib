from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class DetectionRecord:
    sample_id: str
    boxes: List[Dict[str, Any]]


@dataclass
class CalibrationSample:
    index: int
    infra_id: str
    veh_id: str
    infra_boxes: Any  # List[BBox3d]
    veh_boxes: Any
    T_true: Any
    detections_infra: Optional[List[Dict[str, Any]]] = None
    detections_vehicle: Optional[List[Dict[str, Any]]] = None
    features_infra: Optional[List[Any]] = None
    features_vehicle: Optional[List[Any]] = None
    occ_maps: Optional[List[Any]] = None
    bev_range: Optional[List[float]] = None


__all__ = ['CalibrationSample', 'DetectionRecord']
