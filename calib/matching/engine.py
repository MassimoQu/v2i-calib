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

    def _hint_core_component(self) -> str:
        raw = getattr(self.config, 'hint_core_component', None)
        comp = str(raw or 'centerpoint_distance').strip().lower()
        allowed = {
            'iou': 'iou',
            'centerpoint_distance': 'centerpoint_distance',
            'vertex_distance': 'vertex_distance',
            'overall_distance': 'overall_distance',
        }
        return allowed.get(comp, 'centerpoint_distance')

    def _available_matches(self, T, infra_boxes, veh_boxes):
        if T is None:
            return []
        converted = implement_T_3dbox_object_list(T, infra_boxes)
        detector = CorrespondingDetector(
            converted,
            veh_boxes,
            core_similarity_component=self._hint_core_component(),
            distance_threshold=self.config.distance_thresholds,
            parallel=self.config.corresponding_parallel,
            resolve_180_ambiguity=getattr(self.config, 'resolve_180_ambiguity', False),
        )
        return list(detector.get_matches())

    def _hint_matches(self, T_hint, infra_boxes, veh_boxes):
        if T_hint is None:
            return [], 0.0
        prior_weight = float(getattr(self.config, 'prior_weight', 0.0) or 0.0)
        if prior_weight <= 0.0:
            return [], 0.0
        converted = implement_T_3dbox_object_list(T_hint, infra_boxes)
        thresholds = {str(k).lower(): float(v) for k, v in (self.config.distance_thresholds or {}).items()}
        fallback_thr = thresholds.get('detected', max(thresholds.values()) if thresholds else 1.0)

        metric = str(getattr(self.config, 'descriptor_metric', 'cosine')).lower()
        descriptor_weight = float(getattr(self.config, 'descriptor_weight', 0.0) or 0.0)
        use_descriptor = descriptor_weight > 0.0 and 'descriptor' in (self.config.strategy or [])
        assignment_mode = str(getattr(self.config, 'hint_assignment', 'greedy') or 'greedy').lower()
        hint_component = self._hint_core_component()

        def _descriptor_sim(box_a, box_b):
            desc_a = getattr(box_a, 'descriptor', None)
            if desc_a is None and hasattr(box_a, 'get_descriptor'):
                desc_a = box_a.get_descriptor()
            desc_b = getattr(box_b, 'descriptor', None)
            if desc_b is None and hasattr(box_b, 'get_descriptor'):
                desc_b = box_b.get_descriptor()
            if desc_a is None or desc_b is None:
                return None
            vec_a = np.asarray(desc_a, dtype=np.float32).reshape(-1)
            vec_b = np.asarray(desc_b, dtype=np.float32).reshape(-1)
            dim = min(int(vec_a.size), int(vec_b.size))
            if dim <= 0:
                return None
            vec_a = vec_a[:dim]
            vec_b = vec_b[:dim]
            if metric in {'l2', 'euclidean'}:
                dist = float(np.linalg.norm(vec_a - vec_b))
                return float(np.exp(-dist))
            denom = float(np.linalg.norm(vec_a) * np.linalg.norm(vec_b) + 1e-6)
            if denom <= 0:
                return None
            return float(max(0.0, np.dot(vec_a, vec_b) / denom))

        matches = []
        if assignment_mode == 'hungarian':
            num_infra = len(converted)
            num_veh = len(veh_boxes)
            if num_infra == 0 or num_veh == 0:
                return [], 0.0

            infra_vertices = np.stack(
                [np.asarray(box.get_bbox3d_8_3(), dtype=np.float32) for box in converted], axis=0
            )
            veh_vertices = np.stack(
                [np.asarray(box.get_bbox3d_8_3(), dtype=np.float32) for box in veh_boxes], axis=0
            )
            infra_centers = infra_vertices.mean(axis=1)
            veh_centers = veh_vertices.mean(axis=1)
            center_dist = np.linalg.norm(
                infra_centers[:, None, :] - veh_centers[None, :, :], axis=2
            ).astype(np.float32, copy=False)

            if hint_component == 'vertex_distance':
                infra_flat = infra_vertices.reshape(num_infra, -1)
                veh_flat = veh_vertices.reshape(num_veh, -1)
                dist = (
                    np.linalg.norm(infra_flat[:, None, :] - veh_flat[None, :, :], axis=2) / 8.0
                ).astype(np.float32, copy=False)
            elif hint_component == 'overall_distance':
                infra_flat = infra_vertices.reshape(num_infra, -1)
                veh_flat = veh_vertices.reshape(num_veh, -1)
                vertex_dist = (
                    np.linalg.norm(infra_flat[:, None, :] - veh_flat[None, :, :], axis=2) / 8.0
                ).astype(np.float32, copy=False)
                dist = ((center_dist + vertex_dist) / 2.0).astype(np.float32, copy=False)
            else:
                dist = center_dist

            box_types = [
                str(box.get_bbox_type()).lower() if hasattr(box, 'get_bbox_type') else 'detected'
                for box in infra_boxes
            ]
            thr_row = np.array([float(thresholds.get(t, fallback_thr)) for t in box_types], dtype=np.float32)
            allowed = dist <= thr_row[:, None]
            large_cost = 1e6
            cost = dist.astype(np.float32, copy=True)
            cost[~allowed] = large_cost
            row_ind, col_ind = linear_sum_assignment(cost)
            for r, c in zip(row_ind.tolist(), col_ind.tolist()):
                if r >= num_infra or c >= num_veh:
                    continue
                if not allowed[r, c]:
                    continue
                thr = float(thr_row[r])
                if thr <= 0:
                    continue
                dist_rc = float(dist[r, c])
                geom_sim = max(0.0, (thr - dist_rc) / max(thr, 1e-6))
                score = prior_weight * geom_sim
                if use_descriptor:
                    sim = _descriptor_sim(infra_boxes[r], veh_boxes[c])
                    if sim is not None:
                        score += descriptor_weight * sim
                if score > 0.0:
                    matches.append(((int(r), int(c)), float(score)))
        else:
            detector = CorrespondingDetector(
                converted,
                veh_boxes,
                core_similarity_component=hint_component,
                distance_threshold=self.config.distance_thresholds,
                parallel=self.config.corresponding_parallel,
                resolve_180_ambiguity=getattr(self.config, 'resolve_180_ambiguity', False),
            )
            score_dict = detector.get_matches_with_score()
            if not score_dict:
                return [], 0.0
            for (i, j), neg_dist in score_dict.items():
                if i >= len(infra_boxes) or j >= len(veh_boxes):
                    continue
                box_type = 'detected'
                if hasattr(infra_boxes[i], 'get_bbox_type'):
                    box_type = str(infra_boxes[i].get_bbox_type()).lower()
                thr = float(thresholds.get(box_type, fallback_thr))
                if thr <= 0:
                    continue
                dist_ij = float(-neg_dist)
                geom_sim = max(0.0, (thr - dist_ij) / max(thr, 1e-6))
                score = prior_weight * geom_sim
                if use_descriptor:
                    sim = _descriptor_sim(infra_boxes[i], veh_boxes[j])
                    if sim is not None:
                        score += descriptor_weight * sim
                if score > 0.0:
                    matches.append(((int(i), int(j)), float(score)))
        if not matches:
            return [], 0.0
        matches.sort(key=lambda x: x[1], reverse=True)
        stability = float(matches[0][1]) if matches else 0.0
        return matches, stability

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

    def _extract_types(self, boxes):
        types = []
        for box in boxes:
            if hasattr(box, 'get_bbox_type'):
                types.append(str(box.get_bbox_type()).lower())
            else:
                types.append(str(getattr(box, 'bbox_type', 'unknown')).lower())
        return types

    def _match_descriptor_only(self, infra_boxes, veh_boxes):
        desc_infra = self._extract_descriptors(infra_boxes)
        desc_vehicle = self._extract_descriptors(veh_boxes)
        type_infra = self._extract_types(infra_boxes)
        type_vehicle = self._extract_types(veh_boxes)
        metric = str(getattr(self.config, 'descriptor_metric', 'cosine')).lower()
        min_sim = float(getattr(self.config, 'descriptor_min_similarity', 0.0) or 0.0)
        weight = float(getattr(self.config, 'descriptor_weight', 1.0) or 1.0)

        matches = []
        for cat in sorted(set(type_infra) & set(type_vehicle)):
            infra_idx = [i for i, d in enumerate(desc_infra) if d is not None and type_infra[i] == cat]
            veh_idx = [j for j, d in enumerate(desc_vehicle) if d is not None and type_vehicle[j] == cat]
            if not infra_idx or not veh_idx:
                continue

            dims = []
            for i in infra_idx:
                dims.append(int(desc_infra[i].shape[0]))
            for j in veh_idx:
                dims.append(int(desc_vehicle[j].shape[0]))
            dim = min(dims) if dims else 0
            if dim <= 0:
                continue

            infra_mat = np.stack([desc_infra[i][:dim].astype(np.float32, copy=False) for i in infra_idx])
            veh_mat = np.stack([desc_vehicle[j][:dim].astype(np.float32, copy=False) for j in veh_idx])

            if metric in {'l2', 'euclidean'}:
                a2 = np.sum(infra_mat * infra_mat, axis=1, keepdims=True)
                b2 = np.sum(veh_mat * veh_mat, axis=1, keepdims=True).T
                dist2 = np.maximum(a2 + b2 - 2.0 * (infra_mat @ veh_mat.T), 0.0)
                dist = np.sqrt(dist2, dtype=np.float32)
                sim_matrix = np.exp(-dist)
                cost_matrix = dist
            else:
                infra_norm = np.linalg.norm(infra_mat, axis=1, keepdims=True)
                veh_norm = np.linalg.norm(veh_mat, axis=1, keepdims=True)
                infra_mat_n = infra_mat / np.maximum(infra_norm, 1e-6)
                veh_mat_n = veh_mat / np.maximum(veh_norm, 1e-6)
                sim_matrix = infra_mat_n @ veh_mat_n.T
                cost_matrix = 1.0 - sim_matrix

            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            for r, c in zip(row_ind.tolist(), col_ind.tolist()):
                sim = float(sim_matrix[r, c])
                if sim < min_sim:
                    continue
                score = sim * weight
                matches.append(((infra_idx[r], veh_idx[c]), float(score)))

        if not matches:
            return [], 0.0
        matches.sort(key=lambda x: x[1], reverse=True)
        max_pairs = int(getattr(self.config, 'descriptor_max_pairs', 0) or 0)
        if max_pairs > 0 and len(matches) > max_pairs:
            matches = matches[:max_pairs]
        stability = float(matches[0][1]) if matches else 0.0
        return matches, stability

    def descriptor_matches(self, infra_boxes, veh_boxes):
        matches, stability = self._match_descriptor_only(infra_boxes, veh_boxes)
        return matches, stability

    def hint_matches(self, T_hint, infra_boxes, veh_boxes):
        return self._hint_matches(T_hint, infra_boxes, veh_boxes)

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
        hint_matches, hint_stability = self._hint_matches(T_hint, infra_boxes, veh_boxes)
        if hint_matches:
            return hint_matches, hint_stability
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
            seed_top_k=self.config.seed_top_k,
            resolve_180_ambiguity=getattr(self.config, 'resolve_180_ambiguity', False),
            confidence_weight_exponent=getattr(self.config, 'confidence_weight_exponent', 0.0),
            confidence_weight_min=getattr(self.config, 'confidence_weight_min', 0.0),
            size_similarity_weight=getattr(self.config, 'size_similarity_weight', 0.0),
            size_similarity_min=getattr(self.config, 'size_similarity_min', 0.0),
            confidence_boost_weight=getattr(self.config, 'confidence_boost_weight', 0.0),
            size_similarity_boost_weight=getattr(self.config, 'size_similarity_boost_weight', 0.0),
        )
        matches_score = matcher.get_matches_with_score()
        max_matches = getattr(self.config, 'max_retained_matches', None)
        if max_matches is not None:
            max_matches = int(max_matches)
            if max_matches > 0 and len(matches_score) > max_matches:
                matches_score = matches_score[:max_matches]
        stability = matcher.get_stability() if matches_score else 0.0
        return matches_score, stability


__all__ = ['MatchingEngine']
