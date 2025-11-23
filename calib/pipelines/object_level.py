from __future__ import annotations

import json
import time
import numpy as np
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

    def _estimate_occ_hint(self, sample: CalibrationSample):
        if not sample.occ_maps:
            return None
        occ_maps = sample.occ_maps
        if len(occ_maps) < 2:
            return None

        def _squeeze_map(raw):
            arr = np.asarray(raw, dtype=np.float32)
            if arr.ndim >= 3:
                arr = arr.squeeze()
            return arr

        occ_infra = _squeeze_map(occ_maps[0])
        occ_veh = _squeeze_map(occ_maps[1])
        if occ_infra.size == 0 or occ_veh.size == 0:
            return None
        H, W = occ_infra.shape[-2], occ_infra.shape[-1]
        Fa = np.fft.fft2(occ_infra)
        Fb = np.fft.fft2(occ_veh)
        corr = np.fft.ifft2(Fa * np.conj(Fb))
        corr = np.abs(np.fft.fftshift(corr))
        peak = np.unravel_index(np.argmax(corr), corr.shape)
        shift_y = peak[0] - H // 2
        shift_x = peak[1] - W // 2
        resolution = 0.5  # meters per pixel (approx)
        offset = np.array([shift_x * resolution, shift_y * resolution, 0.0, 0.0, 0.0, 0.0])
        return convert_6DOF_to_T(offset)

    def run(self) -> Dict[str, float]:
        output_dir = self._prepare_output_dir()
        frame_records: List[FrameMetrics] = []
        matches_log = output_dir / 'matches.jsonl'
        use_prior = self._should_use_prior()
        use_detection = self.config.data.use_detection
        use_features = getattr(self.config.data, 'use_features', False)
        with matches_log.open('w', encoding='utf-8') as match_f:
            for sample in self.dataset.samples():
                if sample.index % 50 == 0:
                    print(f'[calibration] processing sample #{sample.index}')
                start = time.perf_counter()
                infra_source = 'groundtruth'
                veh_source = 'groundtruth'
                if use_features and sample.features_infra:
                    infra_boxes = list(sample.features_infra)
                    infra_source = 'feature'
                    if use_detection and sample.detections_infra:
                        infra_boxes = infra_boxes + list(sample.detections_infra)
                        infra_source = 'feature+detection'
                elif use_detection and sample.detections_infra:
                    infra_boxes = sample.detections_infra
                    infra_source = 'detection'
                else:
                    infra_boxes = sample.infra_boxes
                if use_features and sample.features_vehicle:
                    veh_boxes = list(sample.features_vehicle)
                    veh_source = 'feature'
                    if use_detection and sample.detections_vehicle:
                        veh_boxes = veh_boxes + list(sample.detections_vehicle)
                        veh_source = 'feature+detection'
                elif use_detection and sample.detections_vehicle:
                    veh_boxes = sample.detections_vehicle
                    veh_source = 'detection'
                else:
                    veh_boxes = sample.veh_boxes
                    veh_source = 'groundtruth'

                descriptor_T = None
                occ_T = self._estimate_occ_hint(sample)
                descriptor_seed_enabled = getattr(self.config.matching, 'descriptor_seed', False)
                descriptor_matches = None
                if descriptor_seed_enabled:
                    desc_matches, desc_stability = self.matching.descriptor_matches(infra_boxes, veh_boxes)
                    descriptor_matches = desc_matches
                    if desc_matches:
                        try:
                            T6_desc, _, _ = self.solver.solve(infra_boxes, veh_boxes, desc_matches, sample.T_true)
                            if T6_desc is not None:
                                descriptor_T = convert_6DOF_to_T(T6_desc)
                        except Exception:
                            descriptor_T = None

                t_filter_start = time.perf_counter()
                filtered_infra, filtered_vehicle = self.filters.apply(infra_boxes, veh_boxes)
                filter_time = time.perf_counter() - t_filter_start

                t_match_start = time.perf_counter()
                T_hint = None
                if use_prior and self._prior_T is not None:
                    T_hint = self._prior_T
                elif descriptor_T is not None:
                    T_hint = descriptor_T
                elif occ_T is not None:
                    T_hint = occ_T
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
                    'detections': {
                        'infra_count': int(len(sample.detections_infra or [])),
                        'vehicle_count': int(len(sample.detections_vehicle or [])),
                    },
                    'bbox_source': {
                        'infra': infra_source,
                        'vehicle': veh_source,
                    },
                    'timing': {
                        'filter': filter_time,
                        'matching': match_time,
                        'solver': solver_time,
                        'total': elapsed,
                    },
                    'fallback_used': fallback_used,
                    'descriptor_seed_used': bool(descriptor_T is not None),
                    'descriptor_seed_matches': int(len(descriptor_matches or [])),
                }
                match_f.write(json.dumps(json_record) + '\n')
        summary = aggregate_metrics(frame_records, self.config.evaluation.success_thresholds)
        summary_path = output_dir / 'metrics.json'
        with summary_path.open('w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        return summary

    def _estimate_occ_hint(self, sample: CalibrationSample):
        if not sample.occ_maps:
            return None
        occ_maps = sample.occ_maps
        if len(occ_maps) < 2:
            return None

        def _squeeze_map(raw):
            arr = np.asarray(raw, dtype=np.float32)
            if arr.ndim >= 3:
                arr = arr.squeeze()
            return arr

        occ_infra = _squeeze_map(occ_maps[0])
        occ_veh = _squeeze_map(occ_maps[1])
        if occ_infra.size == 0 or occ_veh.size == 0:
            return None
        H, W = occ_infra.shape[-2], occ_infra.shape[-1]
        Fa = np.fft.fft2(occ_infra)
        Fb = np.fft.fft2(occ_veh)
        corr = np.fft.ifft2(Fa * np.conj(Fb))
        idx = np.unravel_index(np.argmax(np.abs(corr)), corr.shape)
        shift_row, shift_col = idx
        if shift_row > H // 2:
            shift_row -= H
        if shift_col > W // 2:
            shift_col -= W
        bev_range = sample.bev_range or [-102.4, -51.2, -3.5, 102.4, 51.2, 1.5]
        extent_x = bev_range[3] - bev_range[0]
        extent_y = bev_range[4] - bev_range[1]
        resolution_x = extent_x / W if W else 1.0
        resolution_y = extent_y / H if H else 1.0
        offset = np.array([shift_col * resolution_x, shift_row * resolution_y, 0.0, 0.0, 0.0, 0.0])
        return convert_6DOF_to_T(offset)


def run_from_file(config_path: str) -> Dict[str, float]:
    cfg = load_config(config_path)
    pipeline = ObjectLevelPipeline(cfg)
    return pipeline.run()


__all__ = ['ObjectLevelPipeline', 'run_from_file']
