from __future__ import annotations

from typing import Any, Generator, List, Optional, Tuple

import numpy as np

from v2x_calib.reader import CooperativeBatchingReader

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
        self._shuffle_flags = {
            key.lower(): bool(value) for key, value in (config.shuffle_box_vertices or {}).items()
        }
        self._vertex_perm = np.array([7, 6, 5, 4, 3, 2, 1, 0], dtype=int)

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

    def _maybe_get_detections(
        self, idx: int, infra_id: str, veh_id: str
    ) -> Tuple[Optional[list], Optional[list], Optional[Any], Optional[List[float]]]:
        if not self.config.use_detection:
            return None, None, None, None
        record = self.detection_adapter.get_record(idx=idx, infra_id=infra_id, veh_id=veh_id)
        if record is None:
            return None, None, None, None
        infra_boxes, veh_boxes = self.detection_adapter.convert_record(
            record, field='pred_corner3d_np_list', default_type='detected'
        )
        occ_map = record.get('occ_map_level0')
        bev_range = record.get('bev_range')
        return infra_boxes, veh_boxes, occ_map, bev_range

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
        wrapper = self.reader.generate_infra_vehicle_bboxes_object_list()
        for idx, (inf_id, veh_id, infra_boxes, veh_boxes, T_true) in enumerate(wrapper):
            if self.config.max_samples is not None and idx >= self.config.max_samples:
                break
            detections_infra, detections_vehicle, occ_map_level0, bev_range = self._maybe_get_detections(
                idx, inf_id, veh_id
            )
            feature_infra, feature_vehicle = self._maybe_get_features(
                idx, inf_id, veh_id
            )
            infra_boxes = self._shuffle_boxes(infra_boxes, 'infra')
            veh_boxes = self._shuffle_boxes(veh_boxes, 'vehicle')
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
