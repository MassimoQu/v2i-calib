#!/usr/bin/env python3

import json
from pathlib import Path
from typing import List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt_pct(value: Optional[float]) -> str:
    if value is None:
        return "N/A"
    return f"{100.0 * float(value):.2f}"


def _fmt_num(value: Optional[float]) -> str:
    if value is None:
        return "N/A"
    return f"{float(value):.3f}"


def _read_success_and_mte(metrics: dict) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float], Optional[float], Optional[float]]:
    s1 = metrics.get("success_at_1m")
    s2 = metrics.get("success_at_2m")
    s3 = metrics.get("success_at_3m")
    t1 = metrics.get("mTE@1m")
    t2 = metrics.get("mTE@2m")
    t3 = metrics.get("mTE@3m")
    return s1, s2, s3, t1, t2, t3


def _paper_pct_to_ratio(pct: float) -> float:
    return float(pct) / 100.0


def build_rows():
    out = []

    # Paper Table III (DAIR-V2X, |S|=3737).
    out += [
        {
            "key": "ICP_noise0",
            "paper_success_pct": (47.52, 89.55, 96.01),
            "paper_mte": (0.42, 0.54, 0.58),
            "metrics_path": REPO_ROOT / "outputs" / "paper3737_icp_noise0" / "metrics.json",
        },
        {
            "key": "ICP_noise1",
            "paper_success_pct": (0.86, 37.93, 80.50),
            "paper_mte": (0.66, 1.31, 1.62),
            "metrics_path": REPO_ROOT / "outputs" / "paper3737_icp_noise1" / "metrics.json",
        },
        {
            "key": "ICP_noise2",
            "paper_success_pct": (0.00, 3.66, 19.94),
            "paper_mte": (0.00, 1.33, 2.03),
            "metrics_path": REPO_ROOT / "outputs" / "paper3737_icp_noise2" / "metrics.json",
        },
        {
            "key": "PICP_noise0",
            "paper_success_pct": (59.59, 90.41, 96.12),
            "paper_mte": (0.42, 0.54, 0.57),
            "metrics_path": REPO_ROOT / "outputs" / "paper3737_picp_noise0" / "metrics.json",
        },
        {
            "key": "PICP_noise1",
            "paper_success_pct": (2.91, 42.78, 87.93),
            "paper_mte": (0.75, 1.32, 1.63),
            "metrics_path": REPO_ROOT / "outputs" / "paper3737_picp_noise1" / "metrics.json",
        },
        {
            "key": "PICP_noise2",
            "paper_success_pct": (0.22, 2.69, 21.12),
            "paper_mte": (0.53, 1.45, 2.10),
            "metrics_path": REPO_ROOT / "outputs" / "paper3737_picp_noise2" / "metrics.json",
        },
        {
            "key": "VIPS_noise0",
            "paper_success_pct": (54.20, 88.69, 97.63),
            "paper_mte": (0.54, 0.78, 0.89),
            "metrics_path": REPO_ROOT / "outputs" / "vips" / "paper3737_noise0_thr1p5_initgt_icp" / "metrics.json",
        },
        {
            "key": "VIPS_noise1",
            "paper_success_pct": (18.53, 39.01, 47.74),
            "paper_mte": (0.54, 0.82, 1.02),
            "metrics_path": REPO_ROOT / "outputs" / "vips" / "paper3737_noise1_thr1p5_initgt_icp" / "metrics.json",
        },
        {
            "key": "VIPS_noise2",
            "paper_success_pct": (2.37, 7.87, 13.15),
            "paper_mte": (0.48, 0.96, 1.39),
            "metrics_path": REPO_ROOT / "outputs" / "vips" / "paper3737_noise2_thr1p5_initgt_icp" / "metrics.json",
        },
        {
            "key": "CBM_noise0",
            "paper_success_pct": (17.11, 23.04, 26.49),
            "paper_mte": (0.53, 0.80, 1.06),
            "metrics_path": REPO_ROOT / "outputs" / "cbm" / "paper3737_noise0_min20_thr4_initgt_noicp" / "metrics.json",
        },
        {
            "key": "CBM_noise1",
            "paper_success_pct": (9.91, 15.63, 16.49),
            "paper_mte": (0.61, 0.74, 1.00),
            "metrics_path": REPO_ROOT / "outputs" / "cbm" / "paper3737_noise1_min20_thr4_initgt_noicp" / "metrics.json",
        },
        {
            "key": "CBM_noise2",
            "paper_success_pct": (6.03, 12.28, 16.81),
            "paper_mte": (0.58, 0.76, 1.06),
            "metrics_path": REPO_ROOT / "outputs" / "cbm" / "paper3737_noise2_min20_thr4_initgt_noicp" / "metrics.json",
        },
        {
            "key": "FGR",
            "paper_success_pct": (14.76, 31.57, 35.34),
            "paper_mte": (0.70, 1.13, 1.45),
            "metrics_path": REPO_ROOT / "outputs" / "hkust_teaser" / "hkust_teaser_paper3737_fgr_full" / "metrics.json",
        },
        {
            "key": "Quatro",
            "paper_success_pct": (12.07, 30.50, 45.04),
            "paper_mte": (0.65, 1.19, 1.51),
            "metrics_path": REPO_ROOT / "outputs" / "hkust_teaser" / "hkust_teaser_paper3737_quatro_full" / "metrics.json",
        },
        {
            "key": "Teaser++",
            "paper_success_pct": (14.33, 29.74, 34.81),
            "paper_mte": (0.66, 1.09, 1.44),
            "metrics_path": REPO_ROOT / "outputs" / "hkust_teaser" / "hkust_teaser_paper3737_gnctls_full" / "metrics.json",
        },
        {
            "key": "V2X-Reg (oIoU) GT15",
            "paper_success_pct": (25.54, 55.93, 72.31),
            "paper_mte": (0.54, 0.91, 1.18),
            "metrics_path": REPO_ROOT / "outputs" / "paper3737_dair_v2xreg_oiou_gt15" / "metrics.json",
        },
        {
            "key": "V2X-Reg++GT_inf",
            "paper_success_pct": (22.88, 48.03, 61.49),
            "paper_mte": (0.49, 0.83, 1.07),
            "metrics_path": REPO_ROOT / "outputs" / "paper3737_dair_v2xregpp_gt_inf" / "metrics.json",
        },
        {
            "key": "V2X-Reg++GT25",
            "paper_success_pct": (32.27, 67.59, 82.93),
            "paper_mte": (0.52, 0.85, 1.05),
            "metrics_path": REPO_ROOT / "outputs" / "paper3737_dair_v2xregpp_gt25" / "metrics.json",
        },
        {
            "key": "V2X-Reg++GT15",
            "paper_success_pct": (26.79, 61.17, 78.75),
            "paper_mte": (0.54, 0.87, 1.10),
            "metrics_path": REPO_ROOT / "outputs" / "paper3737_dair_v2xregpp_gt15" / "metrics.json",
        },
        {
            "key": "V2X-Reg++GT10",
            "paper_success_pct": (20.02, 54.86, 71.98),
            "paper_mte": (0.57, 0.92, 1.15),
            "metrics_path": REPO_ROOT / "outputs" / "paper3737_dair_v2xregpp_gt10" / "metrics.json",
        },
        {
            "key": "V2X-Reg++PP15",
            "paper_success_pct": (24.91, 56.62, 70.94),
            "paper_mte": (0.55, 0.86, 1.07),
            "metrics_path": REPO_ROOT / "outputs" / "paper3737_pp15_heal_pp_corners_conf0p3_iter2" / "metrics.json",
        },
        {
            "key": "V2X-Reg++SC15",
            "paper_success_pct": (25.15, 56.89, 71.23),
            "paper_mte": (0.54, 0.86, 1.06),
            "metrics_path": REPO_ROOT / "outputs" / "paper3737_sc15_heal_sc_corners_conf0p3_iter2" / "metrics.json",
        },
    ]
    return out


def main() -> None:
    rows = build_rows()
    print("# Table III status (paper3737)\n")
    print("| row | paper succ@1/2/3 (%) | ours succ@1/2/3 (%) | paper mTE@1/2/3 | ours mTE@1/2/3 | frames | ok? | metrics |")
    print("|---|---:|---:|---:|---:|---:|:---:|---|")

    for row in rows:
        paper_s1, paper_s2, paper_s3 = row["paper_success_pct"]
        paper_t1, paper_t2, paper_t3 = row.get("paper_mte") or (None, None, None)

        ours_s1 = ours_s2 = ours_s3 = None
        ours_t1 = ours_t2 = ours_t3 = None
        frames = None
        ok = False
        metrics_path_str = "N/A"

        metrics_path = row.get("metrics_path")
        if metrics_path is not None:
            metrics_path_str = str(metrics_path)
            if metrics_path.is_file():
                obj = _load_json(metrics_path)
                ours_s1, ours_s2, ours_s3, ours_t1, ours_t2, ours_t3 = _read_success_and_mte(obj)
                frames = obj.get("num_frames")

        ok = (
            ours_s1 is not None
            and ours_s2 is not None
            and ours_s3 is not None
            and float(ours_s1) >= _paper_pct_to_ratio(paper_s1)
            and float(ours_s2) >= _paper_pct_to_ratio(paper_s2)
            and float(ours_s3) >= _paper_pct_to_ratio(paper_s3)
        )

        paper_s = f"{paper_s1:.2f}/{paper_s2:.2f}/{paper_s3:.2f}"
        ours_s = f"{_fmt_pct(ours_s1)}/{_fmt_pct(ours_s2)}/{_fmt_pct(ours_s3)}"
        paper_t = f"{_fmt_num(paper_t1)}/{_fmt_num(paper_t2)}/{_fmt_num(paper_t3)}"
        ours_t = f"{_fmt_num(ours_t1)}/{_fmt_num(ours_t2)}/{_fmt_num(ours_t3)}"
        frames_s = "N/A" if frames is None else str(frames)
        ok_s = "YES" if ok else "NO"

        print(f"| {row['key']} | {paper_s} | {ours_s} | {paper_t} | {ours_t} | {frames_s} | {ok_s} | {metrics_path_str} |")


if __name__ == "__main__":
    main()
