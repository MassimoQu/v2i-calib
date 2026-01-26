#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@dataclass(frozen=True)
class RowSpec:
    key: str
    rel_path: str


@dataclass(frozen=True)
class RowMetrics:
    key: str
    source: Path
    num_frames: int
    avg_time: float
    succ_pct: Tuple[Optional[float], Optional[float], Optional[float]]
    mTE: Tuple[Optional[float], Optional[float], Optional[float]]
    mRE: Tuple[Optional[float], Optional[float], Optional[float]]


ROW_SPECS: Tuple[RowSpec, ...] = (
    RowSpec("ICP-noise0", "paper3737_icp_noise0/details.jsonl"),
    RowSpec("ICP-noise1", "paper3737_icp_noise1_it1_vx0p3_mc1p0_unadj_left/details.jsonl"),
    RowSpec("ICP-noise2", "paper3737_icp_noise2_it1_vx0p3_mc1p0_unadj_left/details.jsonl"),
    RowSpec("PICP-noise0", "paper3737_picp_noise0/details.jsonl"),
    RowSpec("PICP-noise1", "paper3737_picp_noise1_it1_vx0p3_mc1p0_unadj_left/details.jsonl"),
    RowSpec("PICP-noise2", "paper3737_picp_noise2_it1_vx0p3_mc1p0_unadj_left/details.jsonl"),
    RowSpec("VIPS-noise0", "vips/paper3737_noise0_thr1p5_initgt_icp/matches.jsonl"),
    RowSpec("VIPS-noise1", "vips/paper3737_noise1_thr1p5_initgt_icp/matches.jsonl"),
    RowSpec("VIPS-noise2", "vips/paper3737_noise2_thr1p5_initunadj_icp/matches.jsonl"),
    RowSpec("CBM-noise0", "cbm/paper3737_cbm_noise0_min20_thr4_initgt_noicp_full/matches.jsonl"),
    RowSpec("CBM-noise1", "cbm/paper3737_noise1_min20_thr4_initunadj_noicp/matches.jsonl"),
    RowSpec("CBM-noise2", "cbm/paper3737_cbm_noise2_min20_thr4_initgt_noicp_full/matches.jsonl"),
    RowSpec("FGR", "hkust_teaser/paper3737_hkust_budget0p1_maxfeat2000_fgr_full/matches.jsonl"),
    RowSpec("Quatro", "hkust_teaser/paper3737_hkust_budget0p1_maxfeat2000_quatro_full/matches.jsonl"),
    RowSpec("Teaser++", "hkust_teaser/hkust_teaser_paper3737_gnctls_keepall_full/matches.jsonl"),
    RowSpec("V2X-Reg (oIoU) GT15", "paper3737_dair_v2xreg_oiou_gt15_thr4_top6_iter2/matches.jsonl"),
    RowSpec("V2X-Reg++ GT∞", "paper3737_dair_v2xregpp_gt_inf/matches.jsonl"),
    RowSpec("V2X-Reg++ GT25", "paper3737_dair_v2xregpp_gt25/matches.jsonl"),
    RowSpec("V2X-Reg++ GT15", "paper3737_dair_v2xregpp_gt15/matches.jsonl"),
    RowSpec("V2X-Reg++ GT10", "paper3737_dair_v2xregpp_gt10/matches.jsonl"),
    RowSpec("V2X-Reg++ PP15", "paper3737_pp15_pcalwh_topkCand15_25_30_35_confexp2_consistency1p5_icp/matches.jsonl"),
    RowSpec("V2X-Reg++ SC15", "paper3737_sc15_pcalwh_gate1_confexp2_icp/matches.jsonl"),
)


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _normalize_record(obj: Dict[str, Any]) -> Dict[str, Any]:
    if "RE" in obj and "TE" in obj:
        re_val = obj.get("RE")
        te_val = obj.get("TE")
        time_val = obj.get("time") or obj.get("time_sec") or obj.get("runtime_s") or 0.0
    elif "rotation_error_deg" in obj and "translation_error_m" in obj:
        re_val = obj.get("rotation_error_deg")
        te_val = obj.get("translation_error_m")
        time_val = obj.get("runtime_s") or obj.get("time") or obj.get("time_sec") or 0.0
    else:
        raise ValueError("Missing RE/TE fields in record.")
    infra = obj.get("infra_id")
    veh = obj.get("veh_id", obj.get("vehicle_id"))
    if infra is None or veh is None:
        raise ValueError("Missing infra_id/veh_id in record.")
    return {
        "infra_id": str(infra),
        "veh_id": str(veh),
        "RE": float(re_val),
        "TE": float(te_val),
        "time": float(time_val),
    }


def _format_thr(value: float) -> str:
    if float(value).is_integer():
        return f"{int(value)}"
    return f"{value:.2f}".rstrip("0").rstrip(".")


@lru_cache(maxsize=1)
def _load_paper3737_pairs() -> Tuple[Tuple[str, str], ...]:
    data_info = REPO_ROOT / "data" / "data_info_dair_paper3737.json"
    items = json.loads(data_info.read_text(encoding="utf-8"))
    pairs: List[Tuple[str, str]] = []
    for item in items:
        infra = str(item["infrastructure_pointcloud_path"]).split("/")[-1].split(".")[0]
        veh = str(item["vehicle_pointcloud_path"]).split("/")[-1].split(".")[0]
        pairs.append((infra, veh))
    if len(pairs) != 3737:
        raise ValueError(f"Expected 3737 pairs in {data_info}, got {len(pairs)}")
    return tuple(pairs)


@lru_cache(maxsize=1)
def _load_paper3737_pairs_sorted() -> Tuple[Tuple[str, str], ...]:
    return tuple(sorted(_load_paper3737_pairs()))


def _validate_pairs(records: Sequence[Dict[str, Any]], *, allow_incomplete: bool) -> None:
    observed_pairs = [(r["infra_id"], r["veh_id"]) for r in records]
    if allow_incomplete:
        return
    expected = _load_paper3737_pairs()
    expected_sorted = _load_paper3737_pairs_sorted()
    if len(observed_pairs) != len(expected):
        raise ValueError(f"pairs length={len(observed_pairs)} expected={len(expected)}")
    if tuple(sorted(observed_pairs)) != expected_sorted:
        raise ValueError("pairs do not match paper3737 data_info (set mismatch)")


def _aggregate_table3_metrics(
    *,
    records: Sequence[Dict[str, Any]],
    thresholds: Sequence[float],
    success_gate: str,
) -> Dict[str, Any]:
    if not records:
        return {}
    if success_gate not in {"te", "te_re"}:
        raise ValueError(f"Unknown success_gate: {success_gate}")

    out: Dict[str, Any] = {}
    out["num_frames"] = len(records)
    times = [float(r.get("time") or 0.0) for r in records]
    out["avg_time"] = float(sum(times) / len(times))

    for thr in thresholds:
        label = _format_thr(float(thr))
        if success_gate == "te":
            succ = [r for r in records if float(r["TE"]) < float(thr)]
        else:
            succ = [r for r in records if float(r["TE"]) < float(thr) and float(r["RE"]) < float(thr)]
        out[f"success_at_{label}m"] = len(succ) / len(records)
        if succ:
            out[f"mTE@{label}m"] = float(sum(float(r["TE"]) for r in succ) / len(succ))
            out[f"mRE@{label}m"] = float(sum(float(r["RE"]) for r in succ) / len(succ))
        else:
            out[f"mTE@{label}m"] = None
            out[f"mRE@{label}m"] = None
    return out


def _get_float(obj: Dict[str, Any], key: str) -> Optional[float]:
    if key not in obj:
        return None
    v = obj.get(key)
    if v is None:
        return None
    try:
        return float(v)
    except Exception:
        return None


def _fmt_triplet(a: Optional[float], b: Optional[float], c: Optional[float], *, decimals: int) -> str:
    def _fmt(x: Optional[float]) -> str:
        if x is None:
            return "-"
        return f"{x:.{decimals}f}"

    return f"{_fmt(a)}/{_fmt(b)}/{_fmt(c)}"


def _fmt_time(x: Optional[float]) -> str:
    if x is None:
        return "-"
    return f"{x:.3f}"


def _compute_metrics(
    path: Path,
    *,
    expected_frames: int,
    success_gate: str,
    allow_incomplete: bool,
) -> RowMetrics:
    raw_records = list(_iter_jsonl(path))
    records = [_normalize_record(r) for r in raw_records]
    if not allow_incomplete and len(records) != expected_frames:
        raise ValueError(f"{path}: frames={len(records)} expected={expected_frames}")
    _validate_pairs(records, allow_incomplete=allow_incomplete)
    agg = _aggregate_table3_metrics(records=records, thresholds=[1.0, 2.0, 3.0], success_gate=success_gate)
    succ_pct = tuple(float(agg[f"success_at_{k}m"]) * 100.0 for k in ("1", "2", "3"))  # type: ignore[assignment]
    mTE = tuple(_get_float(agg, f"mTE@{k}m") for k in ("1", "2", "3"))  # type: ignore[assignment]
    mRE = tuple(_get_float(agg, f"mRE@{k}m") for k in ("1", "2", "3"))  # type: ignore[assignment]
    return RowMetrics(
        key=path.name,
        source=path,
        num_frames=int(agg.get("num_frames", -1)),
        avg_time=float(agg.get("avg_time", 0.0)),
        succ_pct=succ_pct,  # type: ignore[arg-type]
        mTE=mTE,  # type: ignore[arg-type]
        mRE=mRE,  # type: ignore[arg-type]
    )


def _render_table(rows: List[Tuple[RowSpec, Optional[RowMetrics]]], *, outputs_root: Path) -> str:
    lines: List[str] = []
    lines.append("# Table III reproduction (paper3737)")
    lines.append("")
    lines.append("| Row | Frames | Succ@1/2/3 (%) | mRE@1/2/3 | mTE@1/2/3 | Avg time (s) | Source |")
    lines.append("|---|---:|---:|---:|---:|---:|---|")
    for spec, metrics in rows:
        if metrics is None:
            lines.append(
                f"| {spec.key} | - | - | - | - | - | {spec.rel_path} |"
            )
            continue
        succ = _fmt_triplet(*metrics.succ_pct, decimals=2)
        mRE = _fmt_triplet(*metrics.mRE, decimals=3)
        mTE = _fmt_triplet(*metrics.mTE, decimals=3)
        rel = metrics.source
        try:
            rel = metrics.source.relative_to(outputs_root)
        except Exception:
            pass
        lines.append(
            f"| {spec.key} | {metrics.num_frames} | {succ} | {mRE} | {mTE} | {_fmt_time(metrics.avg_time)} | {rel} |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a Table III reproduction table from local outputs.")
    parser.add_argument("--output", type=str, default="table3_results.md", help="Output markdown path.")
    parser.add_argument("--outputs-root", type=str, default="outputs", help="Root directory for outputs.")
    parser.add_argument(
        "--success-gate",
        type=str,
        default="te_re",
        choices=["te", "te_re"],
        help="Success gating: te (TE<thr) or te_re (TE<thr and RE<thr).",
    )
    parser.add_argument("--allow-incomplete", action="store_true", help="Skip strict 3737-pair validation.")
    args = parser.parse_args()

    outputs_root = Path(args.outputs_root)
    if not outputs_root.is_absolute():
        outputs_root = REPO_ROOT / outputs_root

    rows: List[Tuple[RowSpec, Optional[RowMetrics]]] = []
    expected_frames = len(_load_paper3737_pairs())
    for spec in ROW_SPECS:
        path = outputs_root / spec.rel_path
        if not path.exists():
            rows.append((spec, None))
            continue
        metrics = _compute_metrics(
            path,
            expected_frames=expected_frames,
            success_gate=args.success_gate,
            allow_incomplete=bool(args.allow_incomplete),
        )
        rows.append((spec, metrics))

    report = _render_table(rows, outputs_root=outputs_root)
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = REPO_ROOT / output_path
    output_path.write_text(report, encoding="utf-8")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
