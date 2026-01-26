from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml


@dataclass
class DataConfig:
    dataset: str
    split: str
    data_info_path: str
    data_root: str
    sensor_frame: str = 'lidar'
    detection_cache: Optional[str] = None
    use_detection: bool = False
    load_detection_hints: bool = False
    detection_field: str = 'pred_corner3d_np_list'
    canonicalize_detection_corners: bool = False
    feature_cache: Optional[str] = None
    use_features: bool = False
    feature_field: str = 'feature_corner3d_np_list'
    use_image_descriptors: bool = False
    image_descriptor: Dict[str, Any] = field(default_factory=dict)
    noise: Dict[str, Any] = field(default_factory=dict)
    max_samples: Optional[int] = None
    start_index: int = 0
    shuffle_box_vertices: Dict[str, bool] = field(default_factory=dict)


@dataclass
class FilterConfig:
    top_k: int
    distance_m: float
    priority_categories: List[str] = field(default_factory=list)
    min_confidence: float = 0.0
    per_category_top_k: Dict[str, int] = field(default_factory=dict)
    size_bounds: Dict[str, Dict[str, float]] = field(default_factory=dict)
    top_k_candidates: List[int] = field(default_factory=list)
    confidence_first_for_detected: bool = False


@dataclass
class MatchingConfig:
    strategy: List[str]
    core_components: List[str]
    filter_strategy: str
    filter_threshold: float
    matches2extrinsic: str
    svd_strategy: str
    distance_thresholds: Dict[str, float]
    hint_core_component: str = 'centerpoint_distance'
    hint_assignment: str = 'greedy'
    prior_weight: float = 0.0
    parallel_flag: bool = False
    corresponding_parallel: bool = False
    descriptor_weight: float = 0.0
    descriptor_metric: str = 'cosine'
    descriptor_min_similarity: float = 0.0
    descriptor_max_pairs: int = 50
    descriptor_seed: bool = False
    seed_top_k: int = 0
    max_retained_matches: Optional[int] = None
    resolve_180_ambiguity: bool = False
    # Optional weighting for core correspondence scoring.
    confidence_weight_exponent: float = 0.0
    confidence_weight_min: float = 0.0
    size_similarity_weight: float = 0.0
    size_similarity_min: float = 0.0
    confidence_boost_weight: float = 0.0
    size_similarity_boost_weight: float = 0.0
    occ_hint_rotation_max_deg: float = 0.0
    occ_hint_rotation_step_deg: float = 0.0
    occ_hint_min_peak: float = 0.0
    occ_hint_min_peak_ratio: float = 0.0
    occ_hint_raw_candidate: bool = False
    occ_hint_raw_force_ratio: float = 0.0


@dataclass
class SolverConfig:
    stability_gate: float
    max_iterations: int = 1
    inlier_threshold_m: float = 0.0
    mad_scale: float = 2.5
    min_inliers: int = 1
    confidence_weight_exponent: float = 0.0
    confidence_weight_min: float = 0.0
    consistency_threshold_m: float = 0.0
    consistency_min_support: int = 0
    # Solve-time only: allow 180-degree (corner ordering) ambiguity resolution without
    # changing the correspondence stage (BoxesMatch/CorrespondingDetector).
    resolve_180_ambiguity: bool = False
    keep_prior_on_failure: bool = False
    consider_prior_candidate: bool = False
    # Optional: seed refinement (config may be present in recovered experiment YAMLs).
    seed_refine_top_n: int = 0
    seed_refine_min_matches: int = 0
    # Optional: ICP refinement on feature centers when a hint is available.
    icp_refine_iters: int = 0
    icp_distance_threshold_m: float = 0.0
    icp_trim_ratio: float = 0.0
    icp_min_matches: int = 0
    icp_refine_on_solution: bool = False


@dataclass
class EvalConfig:
    success_thresholds: List[float]
    # Table-III default: a frame is successful at threshold λ
    # iff (TE < λ meters AND RE < λ degrees). Set to "te" to gate by TE only.
    success_gate: str = "te_re"
    time_verbose: bool = False


@dataclass
class OutputConfig:
    root_dir: str
    tag: Optional[str] = None


@dataclass
class PipelineConfig:
    data: DataConfig
    filters: FilterConfig
    matching: MatchingConfig
    solver: SolverConfig
    evaluation: EvalConfig
    output: OutputConfig


def _dict_to_dataclass(dc_cls, payload: Dict) -> object:
    """Convert a raw dict to a dataclass instance.

    This helper is intentionally tolerant: recovered configs may contain extra fields
    from newer experiment branches. Unknown keys are ignored so the pipeline can run
    in best-effort mode.
    """
    allowed = {f.name for f in fields(dc_cls)}
    filtered = {k: v for k, v in (payload or {}).items() if k in allowed}
    return dc_cls(**filtered)


def load_config(path: str | Path) -> PipelineConfig:
    cfg_path = Path(path)
    with cfg_path.open('r', encoding='utf-8') as f:
        payload = yaml.safe_load(f)
    return PipelineConfig(
        data=_dict_to_dataclass(DataConfig, payload['data']),
        filters=_dict_to_dataclass(FilterConfig, payload['filters']),
        matching=_dict_to_dataclass(MatchingConfig, payload['matching']),
        solver=_dict_to_dataclass(SolverConfig, payload['solver']),
        evaluation=_dict_to_dataclass(EvalConfig, payload['evaluation']),
        output=_dict_to_dataclass(OutputConfig, payload['output']),
    )


__all__ = ['PipelineConfig', 'load_config']
