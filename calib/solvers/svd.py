from __future__ import annotations

from typing import List, Tuple

from calib.config import MatchingConfig, SolverConfig
from v2x_calib.search import Matches2Extrinsics
from v2x_calib.utils import convert_T_to_6DOF, get_RE_TE_by_compare_T_6DOF_result_true


class ExtrinsicSolver:
    def __init__(self, matching_cfg: MatchingConfig, solver_cfg: SolverConfig) -> None:
        self.matching_cfg = matching_cfg
        self.solver_cfg = solver_cfg

    def solve(self, infra_boxes, veh_boxes, matches_score, T_true):
        if not matches_score:
            return [0, 0, 0, 0, 0, 0], float('inf'), float('inf')
        solver = Matches2Extrinsics(
            infra_boxes,
            veh_boxes,
            matches_score_list=matches_score,
            svd_strategy=self.matching_cfg.svd_strategy,
        )
        T6 = solver.get_combined_extrinsic(
            matches2extrinsic_strategies=self.matching_cfg.matches2extrinsic
        )
        RE, TE = get_RE_TE_by_compare_T_6DOF_result_true(T6, convert_T_to_6DOF(T_true))
        return T6, RE, TE


__all__ = ['ExtrinsicSolver']
