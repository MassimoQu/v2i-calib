from __future__ import annotations

import json
import time
import numpy as np
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from calib.config import PipelineConfig, load_config
from calib.data.dataset_manager import DatasetManager
from calib.evaluation.metrics import FrameMetrics, aggregate_metrics_with_gate
from calib.filters.pipeline import FilterPipeline
from calib.matching.engine import MatchingEngine
from calib.solvers.svd import ExtrinsicSolver
from v2x_calib.corresponding import CorrespondingDetector
from v2x_calib.utils import (
    convert_6DOF_to_T,
    convert_T_to_6DOF,
    get_RE_TE_by_compare_T_6DOF_result_true,
    implement_T_3dbox_object_list,
    implement_T_points_n_3,
    get_extrinsic_from_two_points,
)


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
        if gate <= 0:
            self._prior_T = None
            return
        keep_on_failure = bool(getattr(self.config.solver, 'keep_prior_on_failure', False))
        if stability < gate or T6 is None:
            if not keep_on_failure:
                self._prior_T = None
            return
        self._prior_T = convert_6DOF_to_T(T6)

    def _refine_icp(self, T_init, infra_boxes, veh_boxes):
        cfg = self.config.solver
        max_iters = int(getattr(cfg, 'icp_refine_iters', 0) or 0)
        if max_iters <= 0 or T_init is None:
            return None
        if not infra_boxes or not veh_boxes:
            return None

        def _centers(boxes):
            centers = []
            for box in boxes:
                try:
                    pts = np.asarray(box.get_bbox3d_8_3(), dtype=np.float32)
                except Exception:
                    continue
                if pts.size == 0:
                    continue
                centers.append(pts.mean(axis=0))
            if not centers:
                return None
            return np.stack(centers, axis=0)

        pts_infra = _centers(infra_boxes)
        pts_veh = _centers(veh_boxes)
        if pts_infra is None or pts_veh is None:
            return None

        dist_thr = float(getattr(cfg, 'icp_distance_threshold_m', 0.0) or 0.0)
        trim_ratio = float(getattr(cfg, 'icp_trim_ratio', 0.0) or 0.0)
        min_matches = int(getattr(cfg, 'icp_min_matches', 0) or 0)
        min_matches = max(1, min_matches)

        T = T_init
        for _ in range(max_iters):
            try:
                pts_trans = implement_T_points_n_3(T, pts_infra)
            except Exception:
                return None
            if pts_trans.size == 0:
                return None
            diff = pts_trans[:, None, :] - pts_veh[None, :, :]
            dists = np.linalg.norm(diff, axis=2)
            nn_idx = np.argmin(dists, axis=1)
            nn_dist = dists[np.arange(dists.shape[0]), nn_idx]
            mask = np.isfinite(nn_dist)
            if dist_thr > 0.0:
                mask &= nn_dist <= dist_thr
            if not np.any(mask):
                break

            sel = np.where(mask)[0]
            order = np.argsort(nn_dist[sel])
            if trim_ratio > 0.0 and trim_ratio < 1.0:
                keep = int(max(min_matches, round(trim_ratio * sel.size)))
            else:
                keep = sel.size
            keep = max(min_matches, min(keep, sel.size))
            sel_keep = sel[order[:keep]]
            if sel_keep.size < min_matches:
                # Fall back to closest matches overall.
                order_all = np.argsort(nn_dist)
                sel_keep = order_all[: min(min_matches, order_all.size)]
            if sel_keep.size == 0:
                break

            try:
                T_new = get_extrinsic_from_two_points(pts_infra[sel_keep], pts_veh[nn_idx[sel_keep]])
            except Exception:
                break
            if np.allclose(T_new, T):
                T = T_new
                break
            T = T_new
        return T

    def run(self) -> Dict[str, float]:
        output_dir = self._prepare_output_dir()
        frame_records: List[FrameMetrics] = []
        matches_log = output_dir / 'matches.jsonl'
        use_prior = self._should_use_prior()
        use_detection = self.config.data.use_detection
        use_features = getattr(self.config.data, 'use_features', False)
        sensor_frame = str(getattr(self.config.data, 'sensor_frame', 'lidar')).lower().strip()
        sensor_combo = 'camera-camera' if sensor_frame == 'camera' else 'lidar-lidar'
        filter_topk_candidates = list(getattr(self.config.filters, 'top_k_candidates', []) or [])
        filter_topk_candidates = [int(k) for k in filter_topk_candidates if k is not None]
        if filter_topk_candidates:
            topk_values = sorted(set(filter_topk_candidates + [int(self.config.filters.top_k)]))
            candidate_filters = {
                k: FilterPipeline(replace(self.config.filters, top_k=int(k), top_k_candidates=[]))
                for k in topk_values
            }
        else:
            topk_values = [int(self.config.filters.top_k)]
            candidate_filters = {topk_values[0]: self.filters}
        if not topk_values:
            topk_values = [int(getattr(self.config.filters, 'top_k', 0) or 0)]
            candidate_filters = {topk_values[0]: self.filters}
        with matches_log.open('w', encoding='utf-8') as match_f:
            for sample in self.dataset.samples():
                if sample.index % 50 == 0:
                    print(f'[calibration] processing sample #{sample.index}')
                start = time.perf_counter()
                infra_source = 'groundtruth'
                veh_source = 'groundtruth'
                if use_features and sample.features_infra is not None:
                    infra_boxes = list(sample.features_infra)
                    infra_source = 'feature'
                    if use_detection and sample.detections_infra is not None:
                        infra_boxes = infra_boxes + list(sample.detections_infra)
                        infra_source = 'feature+detection'
                elif use_detection:
                    infra_boxes = sample.detections_infra or []
                    infra_source = 'detection'
                else:
                    infra_boxes = sample.infra_boxes
                if use_features and sample.features_vehicle is not None:
                    veh_boxes = list(sample.features_vehicle)
                    veh_source = 'feature'
                    if use_detection and sample.detections_vehicle is not None:
                        veh_boxes = veh_boxes + list(sample.detections_vehicle)
                        veh_source = 'feature+detection'
                elif use_detection:
                    veh_boxes = sample.detections_vehicle or []
                    veh_source = 'detection'
                else:
                    veh_boxes = sample.veh_boxes
                    veh_source = 'groundtruth'

                descriptor_T = None
                descriptor_seed_enabled = getattr(self.config.matching, 'descriptor_seed', False)
                descriptor_matches = None

                hint_candidates = []
                hint_source = None
                self._last_occ_hint_peak = None
                self._last_occ_hint_ratio = None
                if use_prior and self._prior_T is not None:
                    hint_candidates.append({'source': 'prior', 'T': self._prior_T})
                else:
                    if descriptor_seed_enabled:
                        desc_matches, _ = self.matching.descriptor_matches(infra_boxes, veh_boxes)
                        descriptor_matches = desc_matches
                        if desc_matches:
                            try:
                                T6_desc, _, _ = self.solver.solve(
                                    infra_boxes, veh_boxes, desc_matches, sample.T_true
                                )
                                if T6_desc is not None:
                                    descriptor_T = convert_6DOF_to_T(T6_desc)
                            except Exception:
                                descriptor_T = None
                        if descriptor_T is not None:
                            hint_candidates.append({'source': 'descriptor', 'T': descriptor_T})
                    if sensor_combo == 'lidar-lidar':
                        occ_T = self._estimate_occ_hint(sample)
                        if occ_T is not None:
                            hint_candidates.append({'source': 'occ', 'T': occ_T})

                def _candidate_quality(T6, infra, veh):
                    if T6 is None:
                        return float('-inf'), 0
                    try:
                        T = convert_6DOF_to_T(T6)
                    except Exception:
                        return float('-inf'), 0
                    try:
                        converted = implement_T_3dbox_object_list(T, infra)
                    except Exception:
                        return float('-inf'), 0
                    detector = CorrespondingDetector(
                        converted,
                        veh,
                        distance_threshold=self.config.matching.distance_thresholds,
                        parallel=self.config.matching.corresponding_parallel,
                    )
                    matched = int(detector.get_matched_num())
                    precision = float(detector.get_distance_corresponding_precision())
                    return precision, matched

                best = None
                total_filter_time = 0.0
                total_match_time = 0.0
                total_solver_time = 0.0
                for top_k in topk_values:
                    t_filter_start = time.perf_counter()
                    filtered_infra, filtered_vehicle = candidate_filters[top_k].apply(
                        infra_boxes, veh_boxes, sensor_frame=sensor_frame
                    )
                    filter_time = time.perf_counter() - t_filter_start
                    total_filter_time += filter_time

                    t_match_start = time.perf_counter()
                    # Always compute a late-fusion baseline match set. When a hint is available,
                    # also compute a hint-guided correspondence set and pick the better solution
                    # using internal correspondence quality (no GT).
                    matches_base, stability_base = self.matching.compute(
                        filtered_infra,
                        filtered_vehicle,
                        T_hint=None,
                        T_eval=sample.T_true,
                        sensor_combo=sensor_combo,
                    )
                    match_time = time.perf_counter() - t_match_start

                    solver_time = 0.0

                    def _maybe_solve(source: str, matches, stability, hint: Optional[str] = None):
                        nonlocal solver_time
                        if not matches:
                            return None
                        t_solve_start = time.perf_counter()
                        T6_cand, RE_cand, TE_cand = self.solver.solve(
                            filtered_infra, filtered_vehicle, matches, sample.T_true
                        )
                        solver_time += time.perf_counter() - t_solve_start
                        return {
                            'source': source,
                            'matches': matches,
                            'stability': float(stability),
                            'T6': T6_cand,
                            'RE': float(RE_cand),
                            'TE': float(TE_cand),
                            'quality': _candidate_quality(T6_cand, filtered_infra, filtered_vehicle),
                            'hint_source': hint,
                        }

                    def _add_icp_candidate(source: str, cand):
                        if not cand:
                            return
                        if not bool(getattr(self.config.solver, 'icp_refine_on_solution', False)):
                            return
                        if cand.get('T6') is None:
                            return
                        icp_T = self._refine_icp(convert_6DOF_to_T(cand['T6']), filtered_infra, filtered_vehicle)
                        if icp_T is None:
                            return
                        try:
                            icp_T6 = convert_T_to_6DOF(icp_T)
                            TE_compare = convert_T_to_6DOF(sample.T_true)
                            RE_icp, TE_icp = get_RE_TE_by_compare_T_6DOF_result_true(icp_T6, TE_compare)
                            candidates.append(
                                {
                                    'source': f'{source}_icp',
                                    'matches': cand.get('matches', []),
                                    'stability': float(cand.get('stability', 0.0)),
                                    'T6': icp_T6,
                                    'RE': float(RE_icp),
                                    'TE': float(TE_icp),
                                    'quality': _candidate_quality(icp_T6, filtered_infra, filtered_vehicle),
                                }
                            )
                        except Exception:
                            return

                    candidates = []
                    cand_base = _maybe_solve('late', matches_base, stability_base)
                    if cand_base is not None:
                        candidates.append(cand_base)
                        _add_icp_candidate('late', cand_base)
                    for hint in hint_candidates:
                        hint_T = hint['T']
                        hint_label = str(hint.get('source') or 'hint')
                        t_match_hint = time.perf_counter()
                        hint_matches, stability_hint = self.matching.compute(
                            filtered_infra,
                            filtered_vehicle,
                            T_hint=hint_T,
                            T_eval=sample.T_true,
                            sensor_combo=sensor_combo,
                        )
                        match_time += time.perf_counter() - t_match_hint
                        cand_hint = _maybe_solve('hint', hint_matches, stability_hint, hint=hint_label)
                        if cand_hint is not None:
                            candidates.append(cand_hint)
                            _add_icp_candidate('hint', cand_hint)
                        try:
                            aligned_infra = implement_T_3dbox_object_list(hint_T, filtered_infra)
                        except Exception:
                            aligned_infra = None
                        if aligned_infra is not None:
                            t_match_aligned = time.perf_counter()
                            matches_aligned, stability_aligned = self.matching.compute(
                                aligned_infra,
                                filtered_vehicle,
                                T_hint=None,
                                T_eval=sample.T_true,
                                sensor_combo=sensor_combo,
                            )
                            match_time += time.perf_counter() - t_match_aligned
                            cand_aligned = _maybe_solve('aligned', matches_aligned, stability_aligned, hint=hint_label)
                            if cand_aligned is not None:
                                candidates.append(cand_aligned)
                                _add_icp_candidate('aligned', cand_aligned)
                        if (
                            hint_label == 'occ'
                            and bool(getattr(self.config.matching, 'occ_hint_raw_candidate', False))
                        ):
                            try:
                                raw_T6 = convert_T_to_6DOF(hint_T)
                                TE_compare = convert_T_to_6DOF(sample.T_true)
                                RE_raw, TE_raw = get_RE_TE_by_compare_T_6DOF_result_true(raw_T6, TE_compare)
                                candidates.append(
                                    {
                                        'source': 'occ_raw',
                                        'matches': [],
                                        'stability': float(stability_hint),
                                        'T6': raw_T6,
                                        'RE': float(RE_raw),
                                        'TE': float(TE_raw),
                                        'quality': _candidate_quality(raw_T6, filtered_infra, filtered_vehicle),
                                        'hint_source': hint_label,
                                    }
                                )
                            except Exception:
                                pass
                        icp_T = self._refine_icp(hint_T, filtered_infra, filtered_vehicle)
                        if icp_T is not None:
                            try:
                                icp_T6 = convert_T_to_6DOF(icp_T)
                                TE_compare = convert_T_to_6DOF(sample.T_true)
                                RE_icp, TE_icp = get_RE_TE_by_compare_T_6DOF_result_true(icp_T6, TE_compare)
                                candidates.append(
                                    {
                                        'source': 'icp',
                                        'matches': [],
                                        'stability': float(stability_hint),
                                        'T6': icp_T6,
                                        'RE': float(RE_icp),
                                        'TE': float(TE_icp),
                                        'quality': _candidate_quality(icp_T6, filtered_infra, filtered_vehicle),
                                        'hint_source': hint_label,
                                    }
                                )
                            except Exception:
                                pass

                    total_match_time += match_time

                    if (
                        use_prior
                        and getattr(self.config.solver, 'consider_prior_candidate', False)
                        and self._prior_T is not None
                    ):
                        try:
                            prior_T6 = convert_T_to_6DOF(self._prior_T)
                            TE_compare = convert_T_to_6DOF(sample.T_true)
                            RE_prior, TE_prior = get_RE_TE_by_compare_T_6DOF_result_true(prior_T6, TE_compare)
                            candidates.append(
                                {
                                    'source': 'prior',
                                    'matches': [],
                                    'stability': float(stability_base),
                                    'T6': prior_T6,
                                    'RE': float(RE_prior),
                                    'TE': float(TE_prior),
                                    'quality': _candidate_quality(prior_T6, filtered_infra, filtered_vehicle),
                                }
                            )
                        except Exception:
                            pass

                    def _better(a, b):
                        if b is None:
                            return True
                        if a is None:
                            return False
                        if a['quality'][0] > b['quality'][0] + 1e-9:
                            return True
                        if abs(a['quality'][0] - b['quality'][0]) <= 1e-9:
                            if a['quality'][1] > b['quality'][1]:
                                return True
                            if a['quality'][1] == b['quality'][1] and a['stability'] > b['stability']:
                                return True
                        return False

                    best_candidate = None
                    occ_ratio = getattr(self, '_last_occ_hint_ratio', None)
                    force_ratio = float(
                        getattr(self.config.matching, 'occ_hint_raw_force_ratio', 0.0) or 0.0
                    )
                    if force_ratio > 0.0 and occ_ratio is not None:
                        try:
                            if float(occ_ratio) >= force_ratio:
                                for cand in candidates:
                                    if cand.get('source') == 'occ_raw':
                                        best_candidate = cand
                                        break
                        except Exception:
                            pass
                    if best_candidate is None:
                        for cand in candidates:
                            if _better(cand, best_candidate):
                                best_candidate = cand

                    fallback_used = False
                    matching_source = 'late'
                    hint_source = None
                    matches_with_score = matches_base
                    stability = float(stability_base)
                    if best_candidate is not None:
                        matching_source = str(best_candidate['source'])
                        hint_source = best_candidate.get('hint_source')
                        matches_with_score = best_candidate['matches']
                        stability = float(best_candidate['stability'])
                        T6 = best_candidate['T6']
                        RE = float(best_candidate['RE'])
                        TE = float(best_candidate['TE'])
                        fallback_used = matching_source == 'prior'
                    elif use_prior and self._prior_T is not None:
                        fallback_used = True
                        matching_source = 'prior'
                        T6 = convert_T_to_6DOF(self._prior_T)
                        TE_compare = convert_T_to_6DOF(sample.T_true)
                        RE, TE = get_RE_TE_by_compare_T_6DOF_result_true(T6, TE_compare)
                    else:
                        # keep baseline outputs (may be empty / unsolved)
                        T6 = None
                        RE = 180.0
                        TE = float('inf')
                    total_solver_time += solver_time

                    quality = _candidate_quality(T6, filtered_infra, filtered_vehicle)
                    record = {
                        'top_k': int(top_k),
                        'filtered_infra': filtered_infra,
                        'filtered_vehicle': filtered_vehicle,
                        'matches_with_score': matches_with_score,
                        'stability': stability,
                        'T6': T6,
                        'RE': RE,
                        'TE': TE,
                        'filter_time': filter_time,
                        'match_time': match_time,
                        'solver_time': solver_time,
                        'fallback_used': fallback_used,
                        'quality': quality,
                        'matching_source': matching_source,
                        'hint_source': hint_source,
                    }
                    if best is None:
                        best = record
                        continue
                    # Prefer higher correspondence precision (closer to 0), then more matches, then stability.
                    if record['quality'][0] > best['quality'][0] + 1e-9:
                        best = record
                    elif abs(record['quality'][0] - best['quality'][0]) <= 1e-9:
                        if record['quality'][1] > best['quality'][1]:
                            best = record
                        elif record['quality'][1] == best['quality'][1] and record['stability'] > best['stability']:
                            best = record

                if best is None:
                    best = {
                        'top_k': int(getattr(self.config.filters, 'top_k', 0) or 0),
                        'filtered_infra': [],
                        'filtered_vehicle': [],
                        'matches_with_score': [],
                        'stability': 0.0,
                        'T6': None,
                        'RE': 180.0,
                        'TE': float('inf'),
                        'filter_time': 0.0,
                        'match_time': 0.0,
                        'solver_time': 0.0,
                        'fallback_used': False,
                        'quality': (float('-inf'), 0),
                        'matching_source': 'none',
                    }
                assert best is not None
                filtered_infra = best['filtered_infra']
                filtered_vehicle = best['filtered_vehicle']
                matches_with_score = best['matches_with_score']
                stability = best['stability']
                T6 = best['T6']
                RE = best['RE']
                TE = best['TE']
                filter_time = best['filter_time']
                match_time = best['match_time']
                solver_time = best['solver_time']
                fallback_used = best['fallback_used']
                matching_source = best.get('matching_source', 'late')
                hint_source = best.get('hint_source')
                chosen_top_k = best['top_k']
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
                    'sensor_combo': sensor_combo,
                    'filter_top_k': int(chosen_top_k),
                    'filter_top_k_candidates': [int(k) for k in topk_values],
                    'matching_source': matching_source,
                    'T6': None if T6 is None else [float(x) for x in T6],
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
                        'filter_total': float(total_filter_time),
                        'matching_total': float(total_match_time),
                        'solver_total': float(total_solver_time),
                        'total': elapsed,
                    },
                    'fallback_used': fallback_used,
                    'descriptor_seed_used': bool(descriptor_T is not None),
                    'descriptor_seed_matches': int(len(descriptor_matches or [])),
                    'hint_source': hint_source,
                    'occ_hint_peak': getattr(self, '_last_occ_hint_peak', None),
                    'occ_hint_ratio': getattr(self, '_last_occ_hint_ratio', None),
                }
                match_f.write(json.dumps(json_record) + '\n')
        summary = aggregate_metrics_with_gate(
            frame_records,
            self.config.evaluation.success_thresholds,
            success_gate=self.config.evaluation.success_gate,
        )
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
        if occ_infra.ndim != 2 or occ_veh.ndim != 2:
            return None
        if occ_infra.shape != occ_veh.shape:
            return None

        H, W = int(occ_infra.shape[-2]), int(occ_infra.shape[-1])

        # Downsample to keep rotation search cheap (phase correlation is robust to mild aliasing).
        max_dim = 128
        stride = int(np.ceil(max(H, W) / float(max_dim))) if max(H, W) > max_dim else 1
        if stride > 1:
            occ_infra = occ_infra[::stride, ::stride]
            occ_veh = occ_veh[::stride, ::stride]
            H, W = int(occ_infra.shape[-2]), int(occ_infra.shape[-1])

        occ_veh_zm = occ_veh - float(np.mean(occ_veh))
        Fb_conj = np.conj(np.fft.fft2(occ_veh_zm))

        def _phase_corr(a, Fb_conj_local):
            a = a - float(np.mean(a))
            Fa = np.fft.fft2(a)
            R = Fa * Fb_conj_local
            R /= (np.abs(R) + 1e-6)
            corr = np.fft.ifft2(R)
            corr_abs = np.abs(corr)
            idx = np.unravel_index(int(np.argmax(corr_abs)), corr_abs.shape)
            peak = float(corr_abs[idx])

            def _parabola_offset(v_m1: float, v_0: float, v_p1: float) -> float:
                denom = (v_m1 - 2.0 * v_0 + v_p1)
                if abs(denom) < 1e-9:
                    return 0.0
                delta = 0.5 * (v_m1 - v_p1) / denom
                if not np.isfinite(delta):
                    return 0.0
                return float(np.clip(delta, -0.5, 0.5))

            row, col = int(idx[0]), int(idx[1])
            row_m1 = (row - 1) % H if H else row
            row_p1 = (row + 1) % H if H else row
            col_m1 = (col - 1) % W if W else col
            col_p1 = (col + 1) % W if W else col

            delta_row = _parabola_offset(
                float(corr_abs[row_m1, col]),
                float(corr_abs[row, col]),
                float(corr_abs[row_p1, col]),
            )
            delta_col = _parabola_offset(
                float(corr_abs[row, col_m1]),
                float(corr_abs[row, col]),
                float(corr_abs[row, col_p1]),
            )

            shift_row = float(row) + float(delta_row)
            shift_col = float(col) + float(delta_col)
            if shift_row > H / 2.0:
                shift_row -= float(H)
            if shift_col > W / 2.0:
                shift_col -= float(W)
            return peak, float(shift_row), float(shift_col)

        max_rot = float(getattr(self.config.matching, 'occ_hint_rotation_max_deg', 0.0) or 0.0)
        step_rot = float(getattr(self.config.matching, 'occ_hint_rotation_step_deg', 0.0) or 0.0)
        best_peak = -1.0
        best_shift_row = 0
        best_shift_col = 0
        best_yaw = 0.0
        coarse_second_peak = 0.0

        if max_rot > 0.0 and step_rot > 0.0:
            from scipy.ndimage import rotate as _rotate

            angles = np.arange(-max_rot, max_rot + 1e-3, step_rot, dtype=np.float32)
            coarse_candidates = []
            for angle in angles.tolist():
                rotated = _rotate(
                    occ_infra,
                    angle=float(angle),
                    reshape=False,
                    order=1,
                    mode='constant',
                    cval=0.0,
                    prefilter=False,
                )
                peak, shift_row, shift_col = _phase_corr(rotated, Fb_conj)
                coarse_candidates.append((float(peak), float(angle), float(shift_row), float(shift_col)))
            coarse_candidates.sort(key=lambda x: x[0], reverse=True)
            if len(coarse_candidates) > 1:
                coarse_second_peak = float(coarse_candidates[1][0])
            top_n = min(5, len(coarse_candidates))
            refine_step = max(1.0, float(step_rot) / 3.0)
            refine_radius = float(step_rot)
            for peak0, angle0, _, _ in coarse_candidates[:top_n]:
                angle_min = max(-max_rot, angle0 - refine_radius)
                angle_max = min(max_rot, angle0 + refine_radius)
                refine_angles = np.arange(angle_min, angle_max + 1e-3, refine_step, dtype=np.float32)
                for angle in refine_angles.tolist():
                    rotated = _rotate(
                        occ_infra,
                        angle=float(angle),
                        reshape=False,
                        order=1,
                        mode='constant',
                        cval=0.0,
                        prefilter=False,
                    )
                    peak, shift_row, shift_col = _phase_corr(rotated, Fb_conj)
                    if peak > best_peak:
                        best_peak = peak
                        best_shift_row = shift_row
                        best_shift_col = shift_col
                        best_yaw = float(angle)
        else:
            best_peak, best_shift_row, best_shift_col = _phase_corr(occ_infra, Fb_conj)

        min_peak = float(getattr(self.config.matching, 'occ_hint_min_peak', 0.0) or 0.0)
        min_ratio = float(getattr(self.config.matching, 'occ_hint_min_peak_ratio', 0.0) or 0.0)
        try:
            self._last_occ_hint_peak = float(best_peak)
            self._last_occ_hint_ratio = float(best_peak) / float(coarse_second_peak + 1e-6)
        except Exception:
            self._last_occ_hint_peak = None
            self._last_occ_hint_ratio = None
        if min_peak > 0.0 and best_peak < min_peak:
            return None
        if min_ratio > 0.0:
            ratio = float(best_peak) / float(coarse_second_peak + 1e-6)
            if ratio < min_ratio:
                return None

        bev_range = sample.bev_range or [-102.4, -51.2, -3.5, 102.4, 51.2, 1.5]
        extent_x = bev_range[3] - bev_range[0]
        extent_y = bev_range[4] - bev_range[1]
        resolution_x = extent_x / W if W else 1.0
        resolution_y = extent_y / H if H else 1.0
        # Phase correlation estimates the relative shift between two maps (vehicle w.r.t infra);
        # we need an i->v hint, so invert the direction and account for image coordinate sign.
        offset = np.array(
            [
                -best_shift_col * resolution_x,
                -best_shift_row * resolution_y,
                0.0,
                0.0,
                0.0,
                -best_yaw,
            ],
            dtype=np.float32,
        )
        return convert_6DOF_to_T(offset)


def run_from_file(config_path: str) -> Dict[str, float]:
    cfg = load_config(config_path)
    pipeline = ObjectLevelPipeline(cfg)
    return pipeline.run()


__all__ = ['ObjectLevelPipeline', 'run_from_file']
