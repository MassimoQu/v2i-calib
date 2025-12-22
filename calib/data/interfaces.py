from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class CalibrationSample:
    index: int
    infra_id: str
    veh_id: str
    infra_boxes: Any  # List[BBox3d]
    veh_boxes: Any
    T_true: Any


__all__ = ['CalibrationSample']
