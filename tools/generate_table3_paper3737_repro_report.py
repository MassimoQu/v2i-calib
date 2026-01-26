#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@dataclass(frozen=True)
class PaperRow:
    key: str
    mRE: Tuple[float, float, float]
    mTE: Tuple[float, float, float]
    succ_pct: Tuple[float, float, float]
    time_s: Optional[float]


@dataclass(frozen=True)
class RunMetrics:
    key: str
    source: Path
    num_frames: int
    avg_time: float
    succ_pct: Tuple[float, float, float]
    mTE: Tuple[Optional[float], Optional[float], Optional[float]]
    mRE: Tuple[Optional[float], Optional[float], Optional[float]]


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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


def _fmt_delta_triplet(
    a: Optional[float],
    b: Optional[float],
    c: Optional[float],
    *,
    decimals: int,
    unit: str = "",
) -> str:
    def _fmt(x: Optional[float]) -> str:
        if x is None:
            return "-"
        sign = "+" if x >= 0 else ""
        return f"{sign}{x:.{decimals}f}{unit}"

    return f"{_fmt(a)}/{_fmt(b)}/{_fmt(c)}"


def _fmt_delta_time(x: Optional[float]) -> str:
    if x is None:
        return "-"
    sign = "+" if x >= 0 else ""
    return f"{sign}{x:.3f}"


def _delta(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None:
        return None
    return float(a) - float(b)


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


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _extract_pair(obj: Dict[str, Any]) -> Tuple[str, str]:
    infra = obj.get("infra_id")
    veh = obj.get("veh_id", obj.get("vehicle_id"))
    if infra is None or veh is None:
        raise ValueError("Missing infra_id/veh_id in record")
    return str(infra), str(veh)


def _validate_pairs_match_paper3737(observed_pairs: Sequence[Tuple[str, str]]) -> None:
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
    times = [float(r.get("time") or r.get("time_sec") or 0.0) for r in records]
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


def _compute_run_metrics(row_key: str, jsonl_path: Path, *, expected_frames: int, success_gate: str) -> RunMetrics:
    records = list(_iter_jsonl(jsonl_path))
    if len(records) != expected_frames:
        raise ValueError(f"{jsonl_path}: frames={len(records)} expected={expected_frames}")
    observed_pairs = [_extract_pair(r) for r in records]
    _validate_pairs_match_paper3737(observed_pairs)
    agg = _aggregate_table3_metrics(records=records, thresholds=[1.0, 2.0, 3.0], success_gate=success_gate)
    succ_pct = tuple(float(agg[f"success_at_{k}m"]) * 100.0 for k in ("1", "2", "3"))  # type: ignore[assignment]
    mTE = tuple(_get_float(agg, f"mTE@{k}m") for k in ("1", "2", "3"))  # type: ignore[assignment]
    mRE = tuple(_get_float(agg, f"mRE@{k}m") for k in ("1", "2", "3"))  # type: ignore[assignment]
    return RunMetrics(
        key=row_key,
        source=jsonl_path,
        num_frames=int(agg.get("num_frames", -1)),
        avg_time=float(agg.get("avg_time", 0.0)),
        succ_pct=succ_pct,  # type: ignore[arg-type]
        mTE=mTE,  # type: ignore[arg-type]
        mRE=mRE,  # type: ignore[arg-type]
    )


def build_paper_rows() -> Dict[str, PaperRow]:
    paper: Dict[str, PaperRow] = {}

    def add(
        key: str,
        *,
        mRE: Tuple[float, float, float],
        mTE: Tuple[float, float, float],
        succ_pct: Tuple[float, float, float],
        time_s: Optional[float],
    ) -> None:
        paper[key] = PaperRow(key=key, mRE=mRE, mTE=mTE, succ_pct=succ_pct, time_s=time_s)

    # Table III (DAIR-V2X, |S|=3737).
    add("ICP-noise0", mRE=(0.65, 0.98, 1.07), mTE=(0.42, 0.54, 0.58), succ_pct=(47.52, 89.55, 96.01), time_s=2.91)
    add("ICP-noise1", mRE=(0.80, 1.36, 1.72), mTE=(0.66, 1.31, 1.62), succ_pct=(0.86, 37.93, 80.50), time_s=2.92)
    add("ICP-noise2", mRE=(0.00, 1.48, 2.11), mTE=(0.00, 1.33, 2.03), succ_pct=(0.00, 3.66, 19.94), time_s=2.86)

    add("PICP-noise0", mRE=(0.52, 0.80, 0.88), mTE=(0.42, 0.54, 0.57), succ_pct=(59.59, 90.41, 96.12), time_s=1.35)
    add("PICP-noise1", mRE=(0.74, 1.31, 1.67), mTE=(0.75, 1.32, 1.63), succ_pct=(2.91, 42.78, 87.93), time_s=1.76)
    add("PICP-noise2", mRE=(0.80, 1.40, 2.11), mTE=(0.53, 1.45, 2.10), succ_pct=(0.22, 2.69, 21.12), time_s=1.70)

    add("VIPS-noise0", mRE=(0.63, 0.89, 0.99), mTE=(0.54, 0.78, 0.89), succ_pct=(54.20, 88.69, 97.63), time_s=0.46)
    add("VIPS-noise1", mRE=(0.66, 1.04, 1.24), mTE=(0.54, 0.82, 1.02), succ_pct=(18.53, 39.01, 47.74), time_s=0.44)
    add("VIPS-noise2", mRE=(0.58, 1.17, 1.56), mTE=(0.48, 0.96, 1.39), succ_pct=(2.37, 7.87, 13.15), time_s=0.47)

    add("CBM-noise0", mRE=(0.61, 0.97, 1.21), mTE=(0.53, 0.80, 1.06), succ_pct=(17.11, 23.04, 26.49), time_s=0.35)
    add("CBM-noise1", mRE=(0.71, 0.94, 1.14), mTE=(0.61, 0.74, 1.00), succ_pct=(9.91, 15.63, 16.49), time_s=0.36)
    add("CBM-noise2", mRE=(0.69, 1.09, 1.38), mTE=(0.58, 0.76, 1.06), succ_pct=(6.03, 12.28, 16.81), time_s=0.35)

    add("FGR", mRE=(0.71, 1.15, 1.47), mTE=(0.70, 1.13, 1.45), succ_pct=(14.76, 31.57, 35.34), time_s=22.73)
    add("Quatro", mRE=(0.62, 1.22, 1.46), mTE=(0.65, 1.19, 1.51), succ_pct=(12.07, 30.50, 45.04), time_s=21.58)
    add("Teaser++", mRE=(0.69, 1.13, 1.47), mTE=(0.66, 1.09, 1.44), succ_pct=(14.33, 29.74, 34.81), time_s=22.43)

    add("V2X-Reg", mRE=(0.66, 1.03, 1.25), mTE=(0.54, 0.91, 1.18), succ_pct=(25.54, 55.93, 72.31), time_s=0.21)
    add("V2X-Reg++ GT∞", mRE=(0.62, 1.01, 1.26), mTE=(0.49, 0.83, 1.07), succ_pct=(22.88, 48.03, 61.49), time_s=0.46)
    add("V2X-Reg++ GT25", mRE=(0.63, 1.01, 1.23), mTE=(0.52, 0.85, 1.05), succ_pct=(32.27, 67.59, 82.93), time_s=0.12)
    add("V2X-Reg++ GT15", mRE=(0.65, 1.05, 1.30), mTE=(0.54, 0.87, 1.10), succ_pct=(26.79, 61.17, 78.75), time_s=0.09)
    add("V2X-Reg++ GT10", mRE=(0.66, 1.11, 1.36), mTE=(0.57, 0.92, 1.15), succ_pct=(20.02, 54.86, 71.98), time_s=0.04)
    add("V2X-Reg++ PP15", mRE=(0.66, 1.06, 1.29), mTE=(0.55, 0.86, 1.07), succ_pct=(24.91, 56.62, 70.94), time_s=None)
    add("V2X-Reg++ SC15", mRE=(0.65, 1.05, 1.29), mTE=(0.54, 0.86, 1.06), succ_pct=(25.15, 56.89, 71.23), time_s=None)

    return paper


def _glob_candidates(*, root: Path, patterns: Sequence[str]) -> List[Path]:
    out: List[Path] = []
    for pat in patterns:
        out.extend(sorted(root.glob(pat)))
    # Dedup while preserving order.
    seen = set()
    uniq: List[Path] = []
    for p in out:
        if p in seen:
            continue
        seen.add(p)
        uniq.append(p)
    return uniq


def discover_candidates(row_key: str) -> List[Path]:
    """Return jsonl paths that plausibly correspond to the given Table-III row."""
    outputs = REPO_ROOT / "outputs"
    vips_root = outputs / "vips"
    cbm_root = outputs / "cbm"
    hkust_root = outputs / "hkust_teaser"

    def _noise_patterns(filename: str, tokens: Sequence[str]) -> List[str]:
        # Only scan within one folder depth to keep discovery cheap.
        pats = []
        for t in tokens:
            pats.append(f"*{t}*/{filename}")
        return pats

    # --- ICP / PICP ---
    if row_key.startswith("ICP-noise"):
        noise = row_key.split("ICP-")[-1]
        # NOTE: Avoid matching PICP runs (folder name contains "picp", which includes "icp" as a substring).
        return _glob_candidates(root=outputs, patterns=[f"paper3737_icp*{noise}*/details.jsonl"])
    if row_key.startswith("PICP-noise"):
        noise = row_key.split("PICP-")[-1]
        return _glob_candidates(root=outputs, patterns=[f"paper3737_picp*{noise}*/details.jsonl"])

    # --- VIPS / CBM ---
    if row_key.startswith("VIPS-noise"):
        noise = row_key.split("VIPS-")[-1]
        return _glob_candidates(root=vips_root, patterns=_noise_patterns("matches.jsonl", [f"paper3737*{noise}"]))
    if row_key.startswith("CBM-noise"):
        noise = row_key.split("CBM-")[-1]
        return _glob_candidates(root=cbm_root, patterns=_noise_patterns("matches.jsonl", [f"paper3737*{noise}"]))

    # --- HKUST baselines ---
    if row_key == "FGR":
        return _glob_candidates(root=hkust_root, patterns=["*paper3737*fgr*/matches.jsonl"])
    if row_key == "Quatro":
        return _glob_candidates(root=hkust_root, patterns=["*paper3737*quatro*/matches.jsonl"])
    if row_key == "Teaser++":
        # Our HKUST implementation uses GNC_TLS as the Teaser++ row in Table III.
        return _glob_candidates(root=hkust_root, patterns=["*paper3737*gnctls*/matches.jsonl", "*paper3737*gnc_tls*/matches.jsonl"])

    # --- Object-level methods ---
    if row_key == "V2X-Reg":
        return _glob_candidates(root=outputs, patterns=["paper3737_*v2xreg_oiou*/matches.jsonl"])
    if row_key.startswith("V2X-Reg++ "):
        suffix = row_key.split("V2X-Reg++ ", 1)[1].strip().lower()
        if suffix.startswith("gt"):
            tag = suffix.replace("∞", "inf").replace(" ", "")
            # GT∞ becomes gtinf (folder uses gt_inf); normalize.
            tag = tag.replace("gt∞", "gt_inf").replace("gtinf", "gt_inf")
            return _glob_candidates(root=outputs, patterns=[f"paper3737_*v2xregpp*{tag}*/matches.jsonl"])
        if suffix.startswith("pp"):
            return _glob_candidates(root=outputs, patterns=["paper3737_pp15*/matches.jsonl"])
        if suffix.startswith("sc"):
            return _glob_candidates(root=outputs, patterns=["paper3737_sc15*/matches.jsonl"])

    return []


def _merge_metadata(jsonl_path: Path) -> Dict[str, Any]:
    """Best-effort metadata about whether this run was merged from shards.

    Note: We *do not* reject merged runs outright, because long-running baselines
    (e.g. HKUST FGR/QUATRO/GNC_TLS) are often executed as multiple disjoint shards
    and concatenated. Instead, we rely on strict validation that the final jsonl:
      - contains exactly `expected_frames` lines
      - covers the exact paper3737 pair set (set equality)
    Metrics are always recomputed from the jsonl, never taken from metrics.json.
    """

    metrics_path = jsonl_path.parent / "metrics.json"
    if not metrics_path.exists():
        return {}
    try:
        metrics = _load_json(metrics_path)
    except Exception:
        return {}
    meta: Dict[str, Any] = {}
    merged_inputs = metrics.get("merged_inputs")
    merged_records = metrics.get("merged_records")
    if merged_inputs is not None:
        meta["merged_inputs"] = merged_inputs
        if isinstance(merged_inputs, list):
            meta["merged_inputs_count"] = len(merged_inputs)
    if merged_records is not None:
        meta["merged_records"] = merged_records
    return meta


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate a Table-III (paper3737) repro status markdown report.")
    p.add_argument("--output", type=str, default="-", help="Write markdown to this path, or '-' for stdout.")
    p.add_argument(
        "--success-gate",
        type=str,
        default="te_re",
        choices=["te", "te_re"],
        help=(
            "Success gating for Table III. "
            "'te' uses TE < thr. "
            "'te_re' uses (TE < thr AND RE < thr). "
            "Default follows Table III numbers for V2X-Reg++ rows."
        ),
    )
    return p.parse_args()


def _select_closest(candidates: Sequence[RunMetrics], paper_row: PaperRow) -> Optional[RunMetrics]:
    if not candidates:
        return None

    def key(m: RunMetrics) -> tuple:
        # "Closest" means overall closest to the paper row across all reported thresholds:
        #   1) success@{1,2,3} (percentage points)
        #   2) mTE@{1,2,3} (meters) among successful frames
        #   3) mRE@{1,2,3} (degrees) among successful frames
        # Time is used only as a final tie-breaker (and only if the paper provides it).
        #
        # Note: mTE/mRE can be None when success@thr == 0; treat as a large mismatch.
        missing_penalty = 1e3

        s1, s2, s3 = m.succ_pct
        p1, p2, p3 = paper_row.succ_pct
        succ_diff = abs(s1 - p1) + abs(s2 - p2) + abs(s3 - p3)

        t1, t2, t3 = m.mTE
        pt1, pt2, pt3 = paper_row.mTE
        te_diff = (
            abs((t1 if t1 is not None else missing_penalty) - pt1)
            + abs((t2 if t2 is not None else missing_penalty) - pt2)
            + abs((t3 if t3 is not None else missing_penalty) - pt3)
        )

        r1, r2, r3 = m.mRE
        pr1, pr2, pr3 = paper_row.mRE
        re_diff = (
            abs((r1 if r1 is not None else missing_penalty) - pr1)
            + abs((r2 if r2 is not None else missing_penalty) - pr2)
            + abs((r3 if r3 is not None else missing_penalty) - pr3)
        )

        time_diff = 0.0
        if paper_row.time_s is not None:
            time_diff = abs(float(m.avg_time) - float(paper_row.time_s))

        # Keep a mild preference on the 2m threshold as a tie-breaker since it's most commonly cited.
        succ2_diff = abs(s2 - p2)
        return (succ_diff, succ2_diff, te_diff, re_diff, time_diff)

    return sorted(candidates, key=key)[0]


def _select_best(candidates: Sequence[RunMetrics]) -> Optional[RunMetrics]:
    if not candidates:
        return None

    def key(m: RunMetrics) -> tuple:
        s1, s2, s3 = m.succ_pct
        t2 = m.mTE[1] if m.mTE else None
        r2 = m.mRE[1] if m.mRE else None
        # Maximize success@2m, then success@1m/@3m; break ties by lower mTE/mRE.
        return (
            s2,
            s1,
            s3,
            -(t2 if t2 is not None else 1e9),
            -(r2 if r2 is not None else 1e9),
        )

    return sorted(candidates, key=key, reverse=True)[0]


def main() -> None:
    args = parse_args()

    paper = build_paper_rows()
    gate = str(args.success_gate)

    rows_order = [
        "ICP-noise0", "ICP-noise1", "ICP-noise2",
        "PICP-noise0", "PICP-noise1", "PICP-noise2",
        "VIPS-noise0", "VIPS-noise1", "VIPS-noise2",
        "CBM-noise0", "CBM-noise1", "CBM-noise2",
        "FGR", "Quatro", "Teaser++",
        "V2X-Reg",
        "V2X-Reg++ GT∞",
        "V2X-Reg++ GT25",
        "V2X-Reg++ GT15",
        "V2X-Reg++ GT10",
        "V2X-Reg++ PP15",
        "V2X-Reg++ SC15",
    ]

    lines = []
    lines.append("# Table III (DAIR-V2X, paper3737=3737 pairs) 复现差距汇总")
    lines.append("")
    lines.append(f"- 更新时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("- 论文来源：V2X-Reg++ 论文 Table III（DAIR-V2X）")
    lines.append("- 约束：本报告 **只** 使用每个 run 的全量 per-frame 输出（`matches.jsonl` / `details.jsonl`，`num_frames=3737`），并对照 `data/data_info_dair_paper3737.json` 校验 pair 集合一致。")
    lines.append(
        "- 约束：允许 **同一配置的 disjoint shards 合并**（常见于耗时较长的 HKUST baselines），但必须满足："
        "`matches.jsonl/details.jsonl` 覆盖 3737/3737 且 pair 集合与 `data/data_info_dair_paper3737.json` 完全一致；"
        "并且所有指标 **只从 jsonl 重新计算**（不读取 `metrics.json` 的聚合值）。"
    )
    lines.append(f"- Success 口径：`{gate}`（`te`=仅 TE<thr；`te_re`=TE<thr 且 RE<thr；Table III 的 V2X-Reg++ 行更符合 `te_re`）。")
    lines.append("")
    lines.append("关键说明：")
    lines.append("- Table III 的 `mRRE` 子列标注为 `@1°/@2°/@3°`，且我们观察到 `V2X-Reg++` 的若干 GT 行更符合 `Success = (TE<thr AND RE<thr)` 的口径；因此本报告默认用 `te_re` 来对齐表格数值（可用 `--success-gate te` 切回仅 TE）。")
    lines.append("- 之前出现“PP/SC 比 GT 更好”的现象，根因是不同 run 的 `metrics.json` 生成时采用了不同 Success 口径（有的用 `te_re`，有的用 `te`）。本报告统一从同一 run 的 jsonl 重新计算，避免口径混用。")
    lines.append("")
    lines.append("表格说明：每个指标单元格三行依次为 `paper / closest / best`；并在 closest/best 行给出 `Δ(ours-paper)`（Success 的 Δ 单位为百分点(pp)）。")
    lines.append("- `closest` 选取：在所有候选 runs 中最小化 `Σ|ΔSuccess@{1,2,3}|`，再以 `|ΔSuccess@2| → Σ|ΔmRTE@{1,2,3}| → Σ|ΔmRRE@{1,2,3}|` 作为 tie-break（若论文提供 Time，则最后再比 `|ΔTime|`）。")
    lines.append("")
    lines.append("| Row | Success@1/2/3 (%) | mRTE@1/2/3 (m) | mRRE@1/2/3 (°) | Time (s) | Closest source | Best source | Frames |")
    lines.append("|---|---|---|---|---|---|---|---:|")

    for key in rows_order:
        paper_row = paper.get(key)
        if paper_row is None:
            continue
        candidate_paths = discover_candidates(key)
        candidates: List[RunMetrics] = []
        for p in candidate_paths:
            try:
                _ = _merge_metadata(p)
                candidates.append(_compute_run_metrics(key, p, expected_frames=3737, success_gate=gate))
            except Exception:
                # Skip invalid/mismatched runs silently to keep the report resilient.
                continue

        closest = _select_closest(candidates, paper_row)
        best = _select_best(candidates)

        pap_s1, pap_s2, pap_s3 = paper_row.succ_pct
        pap_t1, pap_t2, pap_t3 = paper_row.mTE
        pap_r1, pap_r2, pap_r3 = paper_row.mRE
        pap_time = paper_row.time_s

        def _cell_triplet(
            *,
            paper_vals: Tuple[float, float, float],
            closest_vals: Optional[Tuple[Optional[float], Optional[float], Optional[float]]],
            best_vals: Optional[Tuple[Optional[float], Optional[float], Optional[float]]],
            decimals: int,
            delta_decimals: int,
            delta_unit: str = "",
        ) -> str:
            c1, c2, c3 = closest_vals if closest_vals is not None else (None, None, None)
            b1, b2, b3 = best_vals if best_vals is not None else (None, None, None)
            p1, p2, p3 = paper_vals
            return "<br>".join([
                f"paper: {_fmt_triplet(p1, p2, p3, decimals=decimals)}",
                (
                    "closest: -"
                    if closest_vals is None
                    else "closest: "
                    + _fmt_triplet(c1, c2, c3, decimals=decimals)
                    + "<br>Δ: "
                    + _fmt_delta_triplet(_delta(c1, p1), _delta(c2, p2), _delta(c3, p3), decimals=delta_decimals, unit=delta_unit)
                ),
                (
                    "best: -"
                    if best_vals is None
                    else "best: "
                    + _fmt_triplet(b1, b2, b3, decimals=decimals)
                    + "<br>Δ: "
                    + _fmt_delta_triplet(_delta(b1, p1), _delta(b2, p2), _delta(b3, p3), decimals=delta_decimals, unit=delta_unit)
                ),
            ])

        succ_cell = _cell_triplet(
            paper_vals=(pap_s1, pap_s2, pap_s3),
            closest_vals=(closest.succ_pct if closest is not None else None),  # type: ignore[arg-type]
            best_vals=(best.succ_pct if best is not None else None),  # type: ignore[arg-type]
            decimals=2,
            delta_decimals=2,
            delta_unit="",
        ).replace("Δ: ", "Δpp: ")

        te_cell = _cell_triplet(
            paper_vals=(pap_t1, pap_t2, pap_t3),
            closest_vals=(closest.mTE if closest is not None else None),
            best_vals=(best.mTE if best is not None else None),
            decimals=3,
            delta_decimals=3,
        )
        re_cell = _cell_triplet(
            paper_vals=(pap_r1, pap_r2, pap_r3),
            closest_vals=(closest.mRE if closest is not None else None),
            best_vals=(best.mRE if best is not None else None),
            decimals=3,
            delta_decimals=3,
        )

        def _time_block(label: str, value: Optional[float]) -> str:
            return f"{label}: {_fmt_time(value)}"

        closest_time = None if closest is None else float(closest.avg_time)
        best_time = None if best is None else float(best.avg_time)
        time_cell = "<br>".join([
            _time_block("paper", pap_time),
            _time_block("closest", closest_time) + ("" if closest is None or pap_time is None else f"<br>Δ: {_fmt_delta_time(_delta(closest_time, pap_time))}"),
            _time_block("best", best_time) + ("" if best is None or pap_time is None else f"<br>Δ: {_fmt_delta_time(_delta(best_time, pap_time))}"),
        ])

        def _rel(p: Optional[Path]) -> str:
            if p is None:
                return "-"
            try:
                return str(p.relative_to(REPO_ROOT))
            except ValueError:
                return str(p)

        frames = 3737 if (closest is not None or best is not None) else 0
        lines.append(
            "| {row} | {succ} | {te} | {re} | {time} | `{csrc}` | `{bsrc}` | {frames} |".format(
                row=key,
                succ=succ_cell,
                te=te_cell,
                re=re_cell,
                time=time_cell,
                csrc=_rel(closest.source if closest is not None else None),
                bsrc=_rel(best.source if best is not None else None),
                frames=frames,
            )
        )

    output = "\n".join(lines) + "\n"
    if str(args.output).strip() == "-":
        print(output, end="")
        return
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(output, encoding="utf-8")
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
