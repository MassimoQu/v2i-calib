from __future__ import annotations

from typing import Generator

import numpy as np

from v2x_calib.reader import CooperativeBatchingReader
from v2x_calib.utils import convert_6DOF_to_T, implement_T_3dbox_object_list
from legacy.v2x_calib.reader import noise_utils

from calib.config import DataConfig
from .interfaces import CalibrationSample


class DatasetManager:
    def __init__(self, config: DataConfig) -> None:
        self.config = config
        self.reader = CooperativeBatchingReader(
            path_data_info=config.data_info_path,
            path_data_folder=config.data_root,
        )
        self._shuffle_flags = {
            key.lower(): bool(value) for key, value in (config.shuffle_box_vertices or {}).items()
        }
        self._vertex_perm = np.array([7, 6, 5, 4, 3, 2, 1, 0], dtype=int)
        self._noise_cfg = dict(config.noise or {})

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

    def samples(self) -> Generator[CalibrationSample, None, None]:
        wrapper = self.reader.generate_infra_vehicle_bboxes_object_list()
        for idx, (inf_id, veh_id, infra_boxes, veh_boxes, T_true) in enumerate(wrapper):
            if self.config.max_samples is not None and idx >= self.config.max_samples:
                break
            infra_boxes = self._shuffle_boxes(infra_boxes, 'infra')
            veh_boxes = self._shuffle_boxes(veh_boxes, 'vehicle')
            infra_boxes = self._apply_noise(infra_boxes, 'infra')
            veh_boxes = self._apply_noise(veh_boxes, 'vehicle')
            yield CalibrationSample(
                index=idx,
                infra_id=inf_id,
                veh_id=veh_id,
                infra_boxes=infra_boxes,
                veh_boxes=veh_boxes,
                T_true=T_true,
            )


__all__ = ['DatasetManager']
