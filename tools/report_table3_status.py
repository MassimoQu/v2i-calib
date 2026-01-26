#!/usr/bin/env python3
"""
Report how current local outputs match the paper's Table III (DAIR-V2X, 3737 pairs).

Run (recommended, Python>=3.8 with deps already in v2x env):
  MAMBA_ROOT_PREFIX=$PWD/.micromamba ./bin/micromamba run -n v2x python tools/report_table3_status.py
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except Exception:
        return None


def _load_metrics(path: Path) -> Optional[Dict[str, Any]]:
    """Load a metrics dict from either a `metrics.json` or a single-run `matches.jsonl`."""
    if path.suffix.lower() == ".jsonl":
        try:
            from calib.evaluation.metrics import FrameMetrics, aggregate_metrics
        except Exception:
            return None
        records: List[FrameMetrics] = []
        try:
            with path.open("r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    re_deg = obj.get("RE")
                    te_m = obj.get("TE")
                    if re_deg is None or te_m is None:
                        continue
                    time_s = obj.get("time") or 0.0
                    records.append(FrameMetrics(
                        infra_id=str(obj.get("infra_id", "")),
                        veh_id=str(obj.get("veh_id", "")),
                        RE=float(re_deg),
                        TE=float(te_m),
                        stability=0.0,
                        time_cost=float(time_s),
                        matches_count=int(obj.get("num_matches") or 0),
                    ))
        except FileNotFoundError:
            return None
        except Exception:
            return None
        return aggregate_metrics(records, thresholds=[1.0, 2.0, 3.0])
    return _load_json(path)


def _pct(value: Optional[float]) -> str:
    if value is None:
        return "-"
    return f"{value * 100:.2f}"


def _f3(value: Any) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.3f}"
    except Exception:
        return "-"


def _row_key(method: str, noise: Optional[int]) -> str:
    return f"{method}-noise{noise}" if noise is not None else method


def _metric_get(metrics: Optional[Dict[str, Any]], key: str) -> Optional[float]:
    if not metrics:
        return None
    v = metrics.get(key)
    if v is None:
        return None
    try:
        return float(v)
    except Exception:
        return None


def _print_table(rows: List[List[str]]) -> None:
    widths = [0] * max(1, len(rows[0]))
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))
    for idx, row in enumerate(rows):
        line = " | ".join(str(cell).ljust(widths[i]) for i, cell in enumerate(row))
        print(line)
        if idx == 0:
            print("-+-".join("-" * w for w in widths))


def main() -> None:
    # Paper Table III numbers (DAIR-V2X, |S|=3737).
    paper: Dict[str, Dict[str, Any]] = {}

    def add_paper(method: str, noise: Optional[int], *, mre: Tuple[float, float, float], mte: Tuple[float, float, float],
                  succ: Tuple[float, float, float], time_s: Optional[float]) -> None:
        paper[_row_key(method, noise)] = {
            "mRE@1m": mre[0], "mRE@2m": mre[1], "mRE@3m": mre[2],
            "mTE@1m": mte[0], "mTE@2m": mte[1], "mTE@3m": mte[2],
            "success_at_1m": succ[0] / 100.0, "success_at_2m": succ[1] / 100.0, "success_at_3m": succ[2] / 100.0,
            "avg_time": time_s,
        }

    # Init ✓ baselines.
    add_paper("ICP", 0, mre=(0.65, 0.98, 1.07), mte=(0.42, 0.54, 0.58), succ=(47.52, 89.55, 96.01), time_s=2.91)
    add_paper("ICP", 1, mre=(0.80, 1.36, 1.72), mte=(0.66, 1.31, 1.62), succ=(0.86, 37.93, 80.50), time_s=2.92)
    add_paper("ICP", 2, mre=(0.00, 1.48, 2.11), mte=(0.00, 1.33, 2.03), succ=(0.00, 3.66, 19.94), time_s=2.86)

    add_paper("PICP", 0, mre=(0.52, 0.80, 0.88), mte=(0.42, 0.54, 0.57), succ=(59.59, 90.41, 96.12), time_s=1.35)
    add_paper("PICP", 1, mre=(0.74, 1.31, 1.67), mte=(0.75, 1.32, 1.63), succ=(2.91, 42.78, 87.93), time_s=1.76)
    add_paper("PICP", 2, mre=(0.80, 1.40, 2.11), mte=(0.53, 1.45, 2.10), succ=(0.22, 2.69, 21.12), time_s=1.70)

    add_paper("VIPS", 0, mre=(0.63, 0.89, 0.99), mte=(0.54, 0.78, 0.89), succ=(54.20, 88.69, 97.63), time_s=0.46)
    add_paper("VIPS", 1, mre=(0.66, 1.04, 1.24), mte=(0.54, 0.82, 1.02), succ=(18.53, 39.01, 47.74), time_s=0.44)
    add_paper("VIPS", 2, mre=(0.58, 1.17, 1.56), mte=(0.48, 0.96, 1.39), succ=(2.37, 7.87, 13.15), time_s=0.47)

    add_paper("CBM", 0, mre=(0.61, 0.97, 1.21), mte=(0.53, 0.80, 1.06), succ=(17.11, 23.04, 26.49), time_s=0.35)
    add_paper("CBM", 1, mre=(0.71, 0.94, 1.14), mte=(0.61, 0.74, 1.00), succ=(9.91, 15.63, 16.49), time_s=0.36)
    add_paper("CBM", 2, mre=(0.69, 1.09, 1.38), mte=(0.58, 0.76, 1.06), succ=(6.03, 12.28, 16.81), time_s=0.35)

    # Init ✕ baselines.
    add_paper("FGR", None, mre=(0.71, 1.15, 1.47), mte=(0.70, 1.13, 1.45), succ=(14.76, 31.57, 35.34), time_s=22.73)
    add_paper("Quatro", None, mre=(0.62, 1.22, 1.46), mte=(0.65, 1.19, 1.51), succ=(12.07, 30.50, 45.04), time_s=21.58)
    add_paper("Teaser++", None, mre=(0.69, 1.13, 1.47), mte=(0.66, 1.09, 1.44), succ=(14.33, 29.74, 34.81), time_s=22.43)

    # Ours / object-level methods.
    add_paper("V2X-Reg", None, mre=(0.66, 1.03, 1.25), mte=(0.54, 0.91, 1.18), succ=(25.54, 55.93, 72.31), time_s=0.21)
    add_paper("V2X-Reg++ GT∞", None, mre=(0.62, 1.01, 1.26), mte=(0.49, 0.83, 1.07), succ=(22.88, 48.03, 61.49), time_s=0.46)
    add_paper("V2X-Reg++ GT25", None, mre=(0.63, 1.01, 1.23), mte=(0.52, 0.85, 1.05), succ=(32.27, 67.59, 82.93), time_s=0.12)
    add_paper("V2X-Reg++ GT15", None, mre=(0.65, 1.05, 1.30), mte=(0.54, 0.87, 1.10), succ=(26.79, 61.17, 78.75), time_s=0.09)
    add_paper("V2X-Reg++ GT10", None, mre=(0.66, 1.11, 1.36), mte=(0.57, 0.92, 1.15), succ=(20.02, 54.86, 71.98), time_s=0.04)
    add_paper("V2X-Reg++ PP15", None, mre=(0.66, 1.06, 1.29), mte=(0.55, 0.86, 1.07), succ=(24.91, 56.62, 70.94), time_s=None)
    add_paper("V2X-Reg++ SC15", None, mre=(0.65, 1.05, 1.29), mte=(0.54, 0.86, 1.06), succ=(25.15, 56.89, 71.23), time_s=None)

    ours_paths: Dict[str, Path] = {
        _row_key("ICP", 0): REPO_ROOT / "outputs" / "paper3737_icp_noise0" / "metrics.json",
        _row_key("ICP", 1): REPO_ROOT / "outputs" / "paper3737_icp_noise1" / "metrics.json",
        _row_key("ICP", 2): REPO_ROOT / "outputs" / "paper3737_icp_noise2" / "metrics.json",
        _row_key("PICP", 0): REPO_ROOT / "outputs" / "paper3737_picp_noise0" / "metrics.json",
        _row_key("PICP", 1): REPO_ROOT / "outputs" / "paper3737_picp_noise1" / "metrics.json",
        _row_key("PICP", 2): REPO_ROOT / "outputs" / "paper3737_picp_noise2" / "metrics.json",
        # NOTE: use single-run (non-merged) outputs for VIPS/CBM.
        _row_key("VIPS", 0): REPO_ROOT / "outputs" / "vips" / "paper3737_vips_noise0_noicp" / "metrics.json",
        _row_key("VIPS", 1): REPO_ROOT / "outputs" / "vips" / "paper3737_vips_noise1_noicp" / "metrics.json",
        _row_key("VIPS", 2): REPO_ROOT / "outputs" / "vips" / "paper3737_vips_noise2_noicp" / "metrics.json",
        _row_key("CBM", 0): REPO_ROOT / "outputs" / "cbm" / "paper3737_cbm_noise0_noicp" / "metrics.json",
        _row_key("CBM", 1): REPO_ROOT / "outputs" / "cbm" / "paper3737_cbm_noise1_noicp" / "metrics.json",
        _row_key("CBM", 2): REPO_ROOT / "outputs" / "cbm" / "paper3737_cbm_noise2_noicp" / "metrics.json",
        # HKUST baselines: compute paper-style metrics from the single full-run matches.jsonl.
        _row_key("FGR", None): REPO_ROOT / "outputs" / "hkust_teaser" / "hkust_teaser_paper3737_fgr_full" / "matches.jsonl",
        _row_key("Quatro", None): REPO_ROOT / "outputs" / "hkust_teaser" / "hkust_teaser_paper3737_quatro_full" / "matches.jsonl",
        _row_key("Teaser++", None): REPO_ROOT / "outputs" / "hkust_teaser" / "hkust_teaser_paper3737_gnctls_full" / "matches.jsonl",
        _row_key("V2X-Reg", None): REPO_ROOT / "outputs" / "paper3737_dair_v2xreg_oiou_gt15" / "metrics.json",
        _row_key("V2X-Reg++ GT∞", None): REPO_ROOT / "outputs" / "paper3737_dair_v2xregpp_gt_inf" / "metrics.json",
        _row_key("V2X-Reg++ GT25", None): REPO_ROOT / "outputs" / "paper3737_dair_v2xregpp_gt25" / "metrics.json",
        _row_key("V2X-Reg++ GT15", None): REPO_ROOT / "outputs" / "paper3737_dair_v2xregpp_gt15" / "metrics.json",
        _row_key("V2X-Reg++ GT10", None): REPO_ROOT / "outputs" / "paper3737_dair_v2xregpp_gt10" / "metrics.json",
        # PP/SC rows: use the best paper3737 reproduction runs (HEAL dual stage1, score>=0.3, iter=2).
        _row_key("V2X-Reg++ PP15", None): REPO_ROOT / "outputs" / "paper3737_pp15_heal_pp_corners_conf0p3_iter2" / "metrics.json",
        _row_key("V2X-Reg++ SC15", None): REPO_ROOT / "outputs" / "paper3737_sc15_heal_sc_corners_conf0p3_iter2" / "metrics.json",
    }

    header = [
        "Row",
        "Our Succ@1/2/3(%)",
        "Paper Succ@1/2/3(%)",
        "ΔSucc@2m(%)",
        "Our mRE@1/2/3",
        "Paper mRE@1/2/3",
        "Our mTE@1/2/3",
        "Paper mTE@1/2/3",
        "Our time(s)",
        "Paper time(s)",
        "Metrics path",
    ]
    table: List[List[str]] = [header]

    order: List[Tuple[str, Optional[int]]] = [
        ("ICP", 0), ("ICP", 1), ("ICP", 2),
        ("PICP", 0), ("PICP", 1), ("PICP", 2),
        ("VIPS", 0), ("VIPS", 1), ("VIPS", 2),
        ("CBM", 0), ("CBM", 1), ("CBM", 2),
        ("FGR", None), ("Quatro", None), ("Teaser++", None),
        ("V2X-Reg", None),
        ("V2X-Reg++ GT∞", None),
        ("V2X-Reg++ GT25", None),
        ("V2X-Reg++ GT15", None),
        ("V2X-Reg++ GT10", None),
        ("V2X-Reg++ PP15", None),
        ("V2X-Reg++ SC15", None),
    ]

    for method, noise in order:
        key = _row_key(method, noise)
        paper_metrics = paper.get(key)
        if paper_metrics is None:
            continue
        metrics_path = ours_paths.get(key)
        ours = _load_metrics(metrics_path) if metrics_path else None
        our_s1 = _metric_get(ours, "success_at_1m")
        our_s2 = _metric_get(ours, "success_at_2m")
        our_s3 = _metric_get(ours, "success_at_3m")
        pap_s1 = _metric_get(paper_metrics, "success_at_1m")
        pap_s2 = _metric_get(paper_metrics, "success_at_2m")
        pap_s3 = _metric_get(paper_metrics, "success_at_3m")

        delta_s2 = None
        if our_s2 is not None and pap_s2 is not None:
            delta_s2 = (our_s2 - pap_s2) * 100.0

        table.append([
            key,
            f"{_pct(our_s1)}/{_pct(our_s2)}/{_pct(our_s3)}",
            f"{_pct(pap_s1)}/{_pct(pap_s2)}/{_pct(pap_s3)}",
            "-" if delta_s2 is None else f"{delta_s2:+.2f}",
            f"{_f3(_metric_get(ours, 'mRE@1m'))}/{_f3(_metric_get(ours, 'mRE@2m'))}/{_f3(_metric_get(ours, 'mRE@3m'))}",
            f"{_f3(_metric_get(paper_metrics, 'mRE@1m'))}/{_f3(_metric_get(paper_metrics, 'mRE@2m'))}/{_f3(_metric_get(paper_metrics, 'mRE@3m'))}",
            f"{_f3(_metric_get(ours, 'mTE@1m'))}/{_f3(_metric_get(ours, 'mTE@2m'))}/{_f3(_metric_get(ours, 'mTE@3m'))}",
            f"{_f3(_metric_get(paper_metrics, 'mTE@1m'))}/{_f3(_metric_get(paper_metrics, 'mTE@2m'))}/{_f3(_metric_get(paper_metrics, 'mTE@3m'))}",
            _f3(_metric_get(ours, "avg_time")),
            _f3(_metric_get(paper_metrics, "avg_time")),
            str(metrics_path) if metrics_path else "-",
        ])

    _print_table(table)


if __name__ == "__main__":
    main()
