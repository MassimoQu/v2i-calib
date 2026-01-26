from __future__ import annotations

from typing import List, Tuple

import numpy as np

from calib.config import MatchingConfig, SolverConfig
from v2x_calib.search import Matches2Extrinsics
from v2x_calib.utils import (
    convert_T_to_6DOF,
    get_RE_TE_by_compare_T_6DOF_result_true,
    get_xyz_from_bbox3d_8_3,
)


class ExtrinsicSolver:
    def __init__(self, matching_cfg: MatchingConfig, solver_cfg: SolverConfig) -> None:
        self.matching_cfg = matching_cfg
        self.solver_cfg = solver_cfg

    def solve(self, infra_boxes, veh_boxes, matches_score, T_true):
        if not matches_score:
            return [0, 0, 0, 0, 0, 0], float('inf'), float('inf')

        thr_cons = float(getattr(self.solver_cfg, 'consistency_threshold_m', 0.0) or 0.0)
        min_support = int(getattr(self.solver_cfg, 'consistency_min_support', 0) or 0)
        if thr_cons > 0.0 and min_support > 0 and len(matches_score) >= max(2, min_support):
            filtered = self._filter_consistent_matches(
                infra_boxes, veh_boxes, matches_score, thr_cons, min_support
            )
            if filtered:
                matches_score = filtered
        conf_exp = float(getattr(self.solver_cfg, 'confidence_weight_exponent', 0.0) or 0.0)
        conf_min = float(getattr(self.solver_cfg, 'confidence_weight_min', 0.0) or 0.0)
        if conf_exp > 0.0:
            adjusted = []

            def _conf(box) -> float:
                if box is None:
                    return 1.0
                if hasattr(box, 'get_confidence'):
                    try:
                        val = box.get_confidence()
                    except Exception:
                        val = None
                    if val is not None:
                        try:
                            return float(val)
                        except Exception:
                            return 1.0
                try:
                    return float(getattr(box, 'confidence', 1.0))
                except Exception:
                    return 1.0

            for (i, j), score in matches_score:
                try:
                    base = float(score)
                except Exception:
                    base = 0.0
                if base <= 0.0:
                    adjusted.append(((int(i), int(j)), base))
                    continue
                try:
                    ci = max(0.0, _conf(infra_boxes[int(i)]))
                    cj = max(0.0, _conf(veh_boxes[int(j)]))
                except Exception:
                    ci = cj = 1.0
                prod = max(float(ci) * float(cj), conf_min)
                weight = base * (prod ** conf_exp)
                adjusted.append(((int(i), int(j)), float(weight)))
            matches_score = adjusted
        solver = Matches2Extrinsics(
            infra_boxes,
            veh_boxes,
            matches_score_list=matches_score,
            svd_strategy=self.matching_cfg.svd_strategy,
            # Prefer solver-level control so we can optionally enable 180° corner-permutation
            # handling without perturbing the correspondence stage.
            resolve_180_ambiguity=bool(
                getattr(self.solver_cfg, 'resolve_180_ambiguity', False)
                or getattr(self.matching_cfg, 'resolve_180_ambiguity', False)
            ),
            max_iterations=getattr(self.solver_cfg, 'max_iterations', 1),
            inlier_threshold_m=getattr(self.solver_cfg, 'inlier_threshold_m', 0.0),
            mad_scale=getattr(self.solver_cfg, 'mad_scale', 2.5),
            min_inliers=getattr(self.solver_cfg, 'min_inliers', 1),
        )
        T6 = solver.get_combined_extrinsic(
            matches2extrinsic_strategies=self.matching_cfg.matches2extrinsic
        )
        RE, TE = get_RE_TE_by_compare_T_6DOF_result_true(T6, convert_T_to_6DOF(T_true))
        return T6, RE, TE

    @staticmethod
    def _filter_consistent_matches(
        infra_boxes,
        veh_boxes,
        matches_score,
        threshold_m: float,
        min_support: int,
    ):
        centers_infra = []
        centers_veh = []
        weights = []
        for (i, j), score in matches_score:
            try:
                infra_box = infra_boxes[int(i)]
                veh_box = veh_boxes[int(j)]
                ci = get_xyz_from_bbox3d_8_3(infra_box.get_bbox3d_8_3())
                cj = get_xyz_from_bbox3d_8_3(veh_box.get_bbox3d_8_3())
            except Exception:
                return matches_score
            centers_infra.append(np.asarray(ci, dtype=np.float64))
            centers_veh.append(np.asarray(cj, dtype=np.float64))
            try:
                weights.append(float(score))
            except Exception:
                weights.append(0.0)

        n = len(centers_infra)
        best_mask = None
        best_support = -1
        best_weight = float("-inf")
        for i in range(n):
            support = 1
            weight_sum = weights[i]
            mask = [True] * n
            for j in range(n):
                if i == j:
                    continue
                dist_infra = float(np.linalg.norm(centers_infra[i] - centers_infra[j]))
                dist_veh = float(np.linalg.norm(centers_veh[i] - centers_veh[j]))
                if abs(dist_infra - dist_veh) <= threshold_m:
                    support += 1
                    weight_sum += weights[j]
                else:
                    mask[j] = False
            if support > best_support or (support == best_support and weight_sum > best_weight):
                best_support = support
                best_weight = weight_sum
                best_mask = mask

        if best_mask is None or best_support < min_support:
            return matches_score
        filtered = [matches_score[idx] for idx in range(n) if best_mask[idx]]
        if len(filtered) < min_support:
            return matches_score
        return filtered


__all__ = ['ExtrinsicSolver']
