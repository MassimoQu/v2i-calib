from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional
import yaml


@dataclass
class DataConfig:
    dataset: str
    split: str
    data_info_path: str
    data_root: str
    detection_cache: Optional[str] = None
    use_detection: bool = False
    feature_cache: Optional[str] = None
    use_features: bool = False
    feature_field: str = 'feature_corner3d_np_list'
    noise: Dict[str, float] = field(default_factory=dict)
    max_samples: Optional[int] = None
    shuffle_box_vertices: Dict[str, bool] = field(default_factory=dict)


@dataclass
class FilterConfig:
    top_k: int
    distance_m: float
    priority_categories: List[str] = field(default_factory=list)
    min_confidence: float = 0.0
    per_category_top_k: Dict[str, int] = field(default_factory=dict)
    size_bounds: Dict[str, Dict[str, float]] = field(default_factory=dict)


@dataclass
class MatchingConfig:
    strategy: List[str]
    core_components: List[str]
    filter_strategy: str
    filter_threshold: float
    matches2extrinsic: str
    svd_strategy: str
    distance_thresholds: Dict[str, float]
    prior_weight: float = 0.0
    parallel_flag: bool = False
    corresponding_parallel: bool = False
    descriptor_weight: float = 0.0
    descriptor_metric: str = 'cosine'
    descriptor_min_similarity: float = 0.0
    descriptor_max_pairs: int = 50
    descriptor_seed: bool = False


@dataclass
class SolverConfig:
    stability_gate: float
    max_iterations: int = 1


@dataclass
class EvalConfig:
    success_thresholds: List[float]
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
    return dc_cls(**payload)


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
