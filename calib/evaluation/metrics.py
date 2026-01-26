from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class FrameMetrics:
    infra_id: str
    veh_id: str
    RE: float
    TE: float
    stability: float
    time_cost: float
    matches_count: int = 0


def aggregate_metrics(records: List[FrameMetrics], thresholds: List[float]) -> Dict[str, float]:
    return aggregate_metrics_with_gate(records, thresholds, success_gate="te_re")


def aggregate_metrics_with_gate(
    records: List[FrameMetrics],
    thresholds: List[float],
    *,
    success_gate: str = "te_re",
) -> Dict[str, float]:
    summary: Dict[str, float] = {}
    if not records:
        return summary

    def _format_thr(value: float) -> str:
        if float(value).is_integer():
            return f"{int(value)}"
        return f"{value:.2f}".rstrip('0').rstrip('.')

    gate = str(success_gate).strip().lower()
    if gate not in {"te", "te_re"}:
        raise ValueError(f"Unknown success_gate: {success_gate}")

    summary["success_gate"] = gate
    summary["success_definition"] = (
        "TE < thr (meters)" if gate == "te" else "TE < thr (meters) AND RE < thr (degrees)"
    )

    summary['avg_time'] = sum(r.time_cost for r in records) / len(records)
    summary['num_frames'] = len(records)
    match_counts = [r.matches_count for r in records]
    summary['frames_with_matches'] = sum(1 for count in match_counts if count > 0)
    summary['avg_matches'] = sum(match_counts) / len(match_counts)

    if not thresholds:
        summary['success_frames'] = len(records)
        return summary

    for idx, thr in enumerate(thresholds):
        if gate == "te":
            filtered = [r for r in records if r.TE < thr]
        else:
            filtered = [r for r in records if r.TE < thr and r.RE < thr]
        filtered_with_matches = [r for r in filtered if r.matches_count > 0]
        label = _format_thr(thr)
        summary[f"success_at_{label}m"] = len(filtered) / len(records)
        summary[f"success_with_matches_at_{label}m"] = (
            len(filtered_with_matches) / len(records) if records else 0.0
        )
        if filtered:
            summary[f"mRE@{label}m"] = sum(r.RE for r in filtered) / len(filtered)
            summary[f"mTE@{label}m"] = sum(r.TE for r in filtered) / len(filtered)
        else:
            summary[f"mRE@{label}m"] = None  # type: ignore[assignment]
            summary[f"mTE@{label}m"] = None  # type: ignore[assignment]
        if idx == 0:
            summary['success_frames'] = len(filtered)
            summary['success_with_matches_frames'] = len(filtered_with_matches)

    return summary


__all__ = ['FrameMetrics', 'aggregate_metrics', 'aggregate_metrics_with_gate']
