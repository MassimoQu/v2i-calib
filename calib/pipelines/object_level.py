from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from calib.config import PipelineConfig, load_config
from calib.data.dataset_manager import DatasetManager
from calib.evaluation.metrics import FrameMetrics, aggregate_metrics
from calib.filters.pipeline import FilterPipeline
from calib.matching.engine import MatchingEngine
from calib.solvers.svd import ExtrinsicSolver
from v2x_calib.utils import convert_6DOF_to_T, convert_T_to_6DOF, get_RE_TE_by_compare_T_6DOF_result_true


class ObjectLevelPipeline:
    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.dataset = DatasetManager(config.data)
        self.filters = FilterPipeline(config.filters)
        self.matching = MatchingEngine(config.matching)
        self.solver = ExtrinsicSolver(config.matching, config.solver)
        self._prior_T = None

    def _prepare_output_dir(self) -> Path:
        tag = self.config.output.tag or datetime.now().strftime('%Y%m%d-%H%M%S')
        root = Path(self.config.output.root_dir).expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        out_dir = root / tag
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir

    def _should_use_prior(self) -> bool:
        if self.config.solver.stability_gate <= 0:
            return False
        return True

    def _update_prior(self, stability: float, T6, TE: float) -> None:
        gate = self.config.solver.stability_gate
        max_thr = max(self.config.evaluation.success_thresholds or [float('inf')])
        if gate <= 0:
            self._prior_T = None
            return
        if stability < gate or TE > max_thr or T6 is None:
            self._prior_T = None
            return
        self._prior_T = convert_6DOF_to_T(T6)

    def run(self) -> Dict[str, float]:
        output_dir = self._prepare_output_dir()
        frame_records: List[FrameMetrics] = []
        matches_log = output_dir / 'matches.jsonl'
        use_prior = self._should_use_prior()
        with matches_log.open('w', encoding='utf-8') as match_f:
            for sample in self.dataset.samples():
                if sample.index % 50 == 0:
                    print(f'[calibration] processing sample #{sample.index}')
                start = time.perf_counter()
                infra_boxes = sample.infra_boxes
                veh_boxes = sample.veh_boxes

                t_filter_start = time.perf_counter()
                filtered_infra, filtered_vehicle = self.filters.apply(infra_boxes, veh_boxes)
                filter_time = time.perf_counter() - t_filter_start

                t_match_start = time.perf_counter()
                T_hint = self._prior_T if (use_prior and self._prior_T is not None) else None
                matches_with_score, stability = self.matching.compute(
                    filtered_infra,
                    filtered_vehicle,
                    T_hint=T_hint,
                    T_eval=sample.T_true,
                    sensor_combo='lidar-lidar',
                )
                match_time = time.perf_counter() - t_match_start

                fallback_used = False
                if matches_with_score:
                    t_solve_start = time.perf_counter()
                    T6, RE, TE = self.solver.solve(
                        filtered_infra, filtered_vehicle, matches_with_score, sample.T_true
                    )
                    solver_time = time.perf_counter() - t_solve_start
                elif use_prior and self._prior_T is not None:
                    fallback_used = True
                    T6 = convert_T_to_6DOF(self._prior_T)
                    solver_time = 0.0
                    TE_compare = convert_T_to_6DOF(sample.T_true)
                    RE, TE = get_RE_TE_by_compare_T_6DOF_result_true(T6, TE_compare)
                else:
                    t_solve_start = time.perf_counter()
                    T6, RE, TE = self.solver.solve(
                        filtered_infra, filtered_vehicle, matches_with_score, sample.T_true
                    )
                    solver_time = time.perf_counter() - t_solve_start
                elapsed = time.perf_counter() - start
                frame_records.append(
                    FrameMetrics(
                        infra_id=sample.infra_id,
                        veh_id=sample.veh_id,
                        RE=RE,
                        TE=TE,
                        stability=stability,
                        time_cost=elapsed,
                        matches_count=len(matches_with_score),
                    )
                )
                self._update_prior(stability, T6, TE)
                json_record = {
                    'index': sample.index,
                    'infra_id': sample.infra_id,
                    'veh_id': sample.veh_id,
                    'RE': float(RE),
                    'TE': float(TE),
                    'stability': float(stability),
                    'time': float(elapsed),
                    'matches': [
                        {'infra_idx': int(m[0][0]), 'veh_idx': int(m[0][1]), 'score': float(m[1])}
                        for m in matches_with_score
                    ],
                    'timing': {
                        'filter': filter_time,
                        'matching': match_time,
                        'solver': solver_time,
                        'total': elapsed,
                    },
                    'fallback_used': fallback_used,
                }
                match_f.write(json.dumps(json_record) + '\n')
        summary = aggregate_metrics(frame_records, self.config.evaluation.success_thresholds)
        summary_path = output_dir / 'metrics.json'
        with summary_path.open('w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        return summary

def run_from_file(config_path: str) -> Dict[str, float]:
    cfg = load_config(config_path)
    pipeline = ObjectLevelPipeline(cfg)
    return pipeline.run()


__all__ = ['ObjectLevelPipeline', 'run_from_file']
