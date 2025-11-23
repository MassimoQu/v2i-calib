from __future__ import annotations

from typing import List, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from calib.config import MatchingConfig
from v2x_calib.corresponding import BoxesMatch, CorrespondingDetector
from v2x_calib.utils import implement_T_3dbox_object_list


class MatchingEngine:
    def __init__(self, config: MatchingConfig) -> None:
        self.config = config

    def _available_matches(self, T, infra_boxes, veh_boxes):
        if T is None:
            return []
        converted = implement_T_3dbox_object_list(T, infra_boxes)
        detector = CorrespondingDetector(
            converted,
            veh_boxes,
            distance_threshold=self.config.distance_thresholds,
            parallel=self.config.corresponding_parallel,
        )
        return list(detector.get_matches())

    def _extract_descriptors(self, boxes):
        descs = []
        for box in boxes:
            descriptor = getattr(box, 'descriptor', None)
            if descriptor is None and hasattr(box, 'get_descriptor'):
                descriptor = box.get_descriptor()
            if descriptor is None:
                descs.append(None)
            else:
                arr = np.asarray(descriptor, dtype=np.float32)
                if arr.ndim != 1:
                    arr = arr.reshape(-1)
                descs.append(arr)
        return descs

    def _match_descriptor_only(self, infra_boxes, veh_boxes):
        desc_infra = self._extract_descriptors(infra_boxes)
        desc_vehicle = self._extract_descriptors(veh_boxes)
        infra_idx = [i for i, d in enumerate(desc_infra) if d is not None]
        veh_idx = [j for j, d in enumerate(desc_vehicle) if d is not None]
        if not infra_idx or not veh_idx:
            return [], 0.0
        infra_vecs = []
        for i in infra_idx:
            vec = desc_infra[i].copy()
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            infra_vecs.append(vec)
        veh_vecs = []
        for j in veh_idx:
            vec = desc_vehicle[j].copy()
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            veh_vecs.append(vec)
        infra_mat = np.stack(infra_vecs)
        veh_mat = np.stack(veh_vecs)
        sim_matrix = infra_mat @ veh_mat.T
        min_sim = max(0.0, float(self.config.descriptor_min_similarity))
        cost_matrix = 1.0 - sim_matrix
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        matches = []
        for r, c in zip(row_ind, col_ind):
            sim = sim_matrix[r, c]
            if sim < min_sim:
                continue
            score = sim * max(1.0, self.config.descriptor_weight or 1.0)
            matches.append(((infra_idx[r], veh_idx[c]), float(score)))
        if not matches:
            return [], 0.0
        matches.sort(key=lambda x: x[1], reverse=True)
        max_pairs = max(1, int(self.config.descriptor_max_pairs or len(matches)))
        matches = matches[:max_pairs]
        stability = matches[0][1] if matches else 0.0
        return matches, stability

    def descriptor_matches(self, infra_boxes, veh_boxes):
        matches, stability = self._match_descriptor_only(infra_boxes, veh_boxes)
        return matches, stability

    def compute(
        self,
        infra_boxes,
        veh_boxes,
        *,
        T_hint=None,
        T_eval=None,
        sensor_combo: str = 'lidar-lidar',
    ) -> Tuple[List[Tuple[Tuple[int, int], float]], float]:
        """
        Args:
            T_hint: prior extrinsic estimation (e.g., previous frame)
            T_eval: ground-truth or GT-like extrinsic (used only for filtering diagnostics)
            sensor_combo: reserved for camera/lidar combinations
        """
        if 'descriptor_only' in self.config.strategy:
            matches_score, stability = self._match_descriptor_only(infra_boxes, veh_boxes)
            return matches_score, stability
        available_matches = self._available_matches(T_hint, infra_boxes, veh_boxes)
        if not available_matches and T_eval is not None:
            # fall back to GT matches for evaluation purpose
            available_matches = self._available_matches(T_eval, infra_boxes, veh_boxes)
        matcher = BoxesMatch(
            infra_boxes,
            veh_boxes,
            similarity_strategy=self.config.strategy,
            core_similarity_component=self.config.core_components,
            matches_filter_strategy=self.config.filter_strategy,
            filter_threshold=self.config.filter_threshold,
            true_matches=available_matches,
            distance_threshold=self.config.distance_thresholds,
            svd_starategy=self.config.svd_strategy,
            parallel_flag=int(self.config.parallel_flag),
            corresponding_parallel=self.config.corresponding_parallel,
            descriptor_weight=self.config.descriptor_weight,
            descriptor_metric=self.config.descriptor_metric,
        )
        matches_score = matcher.get_matches_with_score()
        stability = matcher.get_stability() if matches_score else 0.0
        return matches_score, stability


__all__ = ['MatchingEngine']
