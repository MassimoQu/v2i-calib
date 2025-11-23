#!/usr/bin/env python3
import argparse
import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np

from calib.config import load_config
from calib.filters.pipeline import FilterPipeline
from calib.matching.engine import MatchingEngine
from legacy.v2x_calib.reader import V2XSetReader
from legacy.v2x_calib.utils import convert_6DOF_to_T, implement_T_3dbox_object_list


@dataclass
class Sample:
    infra_boxes: List
    veh_boxes: List
    T_true: np.ndarray


def clone_boxes(boxes: List) -> List:
    return [box.copy() for box in boxes]


def collect_samples(reader: V2XSetReader, limit: int) -> List[Sample]:
    samples: List[Sample] = []
    for _, _, infra_boxes, veh_boxes, T_true in reader.generate_vehicle_vehicle_bboxes_object_list():
        samples.append(Sample(clone_boxes(infra_boxes), clone_boxes(veh_boxes), T_true))
        if len(samples) >= limit:
            break
    return samples


def evaluate_stability(samples: List[Sample], filters: FilterPipeline,
                       matcher: MatchingEngine, bias_T: np.ndarray) -> float:
    stabilities: List[float] = []
    for sample in samples:
        infra = clone_boxes(sample.infra_boxes)
        veh = clone_boxes(sample.veh_boxes)
        if bias_T is not None:
            veh = implement_T_3dbox_object_list(bias_T, veh)
        infra_f, veh_f = filters.apply(infra, veh)
        if not infra_f or not veh_f:
            continue
        _, stability = matcher.compute(infra_f, veh_f, T_eval=sample.T_true)
        stabilities.append(float(stability))
    return float(np.mean(stabilities)) if stabilities else float("nan")


def evaluate_matches(samples: List[Sample], filters: FilterPipeline,
                     matcher: MatchingEngine) -> List[int]:
    counts: List[int] = []
    for sample in samples:
        infra = clone_boxes(sample.infra_boxes)
        veh = clone_boxes(sample.veh_boxes)
        infra_f, veh_f = filters.apply(infra, veh)
        if not infra_f or not veh_f:
            continue
        matches, _ = matcher.compute(infra_f, veh_f, T_eval=sample.T_true)
        counts.append(len(matches))
    return counts


def build_matcher(cfg, core_components=None, strategy=None):
    cfg_match = deepcopy(cfg.matching)
    if core_components is not None:
        cfg_match.core_components = list(core_components)
    if strategy is not None:
        cfg_match.strategy = list(strategy)
    return MatchingEngine(cfg_match)


def main():
    parser = argparse.ArgumentParser(description="Generate V2X-Set indicator curves and ablation data.")
    parser.add_argument("--config", type=str, default="configs/pipeline.yaml")
    parser.add_argument("--root", type=str, default="/mnt/ssd_gw/cooperative-vehicle-infrastructure/v2xset")
    parser.add_argument("--split", type=str, default="validate")
    parser.add_argument("--frame-stride", type=int, default=40)
    parser.add_argument("--max-cavs", type=int, default=3)
    parser.add_argument("--max-samples", type=int, default=60)
    parser.add_argument("--output-dir", type=str, default="outputs")
    args = parser.parse_args()

    cfg = load_config(args.config)
    filters = FilterPipeline(cfg.filters)
    reader = V2XSetReader(
        root_dir=args.root,
        split=args.split,
        frame_stride=args.frame_stride,
        max_cavs=args.max_cavs,
    )
    samples = collect_samples(reader, args.max_samples)
    if not samples:
        raise RuntimeError("No samples collected from V2X-Set.")

    odist_matcher = build_matcher(cfg, core_components=['centerpoint_distance', 'vertex_distance'])
    oiou_matcher = build_matcher(cfg, core_components=['iou'])

    trans_biases = np.linspace(0.0, 2.0, 9)  # meters
    rot_biases = np.linspace(0.0, 20.0, 9)   # degrees

    indicator = {
        "translation": {"bias_m": [], "oDist": [], "oIoU": []},
        "rotation": {"bias_deg": [], "oDist": [], "oIoU": []},
    }
    for bias in trans_biases:
        T_bias = convert_6DOF_to_T([bias, 0.0, 0.0, 0.0, 0.0, 0.0])
        indicator["translation"]["bias_m"].append(bias)
        indicator["translation"]["oDist"].append(evaluate_stability(samples, filters, odist_matcher, T_bias))
        indicator["translation"]["oIoU"].append(evaluate_stability(samples, filters, oiou_matcher, T_bias))
    for bias in rot_biases:
        T_bias = convert_6DOF_to_T([0.0, 0.0, 0.0, 0.0, bias, 0.0])
        indicator["rotation"]["bias_deg"].append(bias)
        indicator["rotation"]["oDist"].append(evaluate_stability(samples, filters, odist_matcher, T_bias))
        indicator["rotation"]["oIoU"].append(evaluate_stability(samples, filters, oiou_matcher, T_bias))

    strategies = {
        "strategy1_oDist": {"core_components": ['centerpoint_distance', 'vertex_distance'], "strategy": ['category', 'core']},
        "strategy2_oIoU": {"core_components": ['iou'], "strategy": ['category', 'core']},
        "strategy3_angle": {"core_components": None, "strategy": ['category', 'angle']},
        "strategy4_length": {"core_components": None, "strategy": ['category', 'length']},
    }
    ablation = {}
    for name, params in strategies.items():
        matcher = build_matcher(cfg, core_components=params["core_components"], strategy=params["strategy"])
        ablation[name] = evaluate_matches(samples, filters, matcher)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "v2xset_indicator_curves.json").write_text(json.dumps(indicator, indent=2))
    (out_dir / "v2xset_association_ablation.json").write_text(json.dumps(ablation, indent=2))
    print(f"Indicator curves saved to {out_dir / 'v2xset_indicator_curves.json'}")
    print(f"Ablation data saved to {out_dir / 'v2xset_association_ablation.json'}")


if __name__ == "__main__":
    main()
