#!/usr/bin/env python3
"""
Create visualization figures for V2X-Set experiments.
Generates PNGs under outputs/v2xset_plots/.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "v2xset_plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_metrics(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def plot_hkust_runtime():
    configs = {
        "Teaser++ (GNC-TLS)": ROOT / "outputs" / "hkust_teaser" / "v2xset_teaser_gnctls" / "metrics.json",
        "FGR": ROOT / "outputs" / "hkust_teaser" / "v2xset_fgr" / "metrics.json",
        "QUATRO": ROOT / "outputs" / "hkust_teaser" / "v2xset_quatro" / "metrics.json",
    }
    labels, times = [], []
    for name, path in configs.items():
        metrics = load_metrics(path)
        labels.append(name)
        times.append(metrics.get("avg_time", 0.0))

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, times, color=["#6baed6", "#fd8d3c", "#74c476"])
    ax.set_ylabel("Avg runtime per pair (s)")
    ax.set_title("HKUST LiDAR baselines on V2X-Set")
    ax.set_ylim(0, max(times) * 1.2 if times else 1)
    for idx, t in enumerate(times):
        ax.text(idx, t + 0.1, f"{t:.2f}s", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "hkust_runtime.png", dpi=200)
    plt.close(fig)


def plot_success_vs_topk():
    regpp_paths = {
        25: ROOT / "outputs" / "v2xset_regpp_gt25" / "metrics.json",
        15: ROOT / "outputs" / "v2xset_regpp_gt15" / "metrics.json",
        10: ROOT / "outputs" / "v2xset_regpp_gt10" / "metrics.json",
    }
    reg_paths = {
        25: ROOT / "outputs" / "v2xset_reg_gt25" / "metrics.json",
        15: ROOT / "outputs" / "v2xset_reg_gt15" / "metrics.json",
        10: ROOT / "outputs" / "v2xset_reg_gt10" / "metrics.json",
    }

    def extract_success(paths: Dict[int, Path]) -> Tuple[List[int], List[float]]:
        xs, ys = [], []
        for topk, path in sorted(paths.items(), reverse=True):
            xs.append(topk)
            metrics = load_metrics(path)
            ys.append(metrics.get("success_at_1m", 0.0))
        return xs, ys

    xs_pp, ys_pp = extract_success(regpp_paths)
    xs_reg, ys_reg = extract_success(reg_paths)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(xs_pp, ys_pp, marker="o", label="V2X-Reg++ (oDist)")
    ax.plot(xs_reg, ys_reg, marker="s", label="V2X-Reg (oIoU, evenSVD)")
    ax.set_xlabel("Top-K boxes kept")
    ax.set_ylabel("Success@1 m")
    ax.set_title("V2X-Set object-level calibration")
    ax.set_xticks(xs_pp)
    ax.set_ylim(0, 1.05)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "success_vs_topk.png", dpi=200)
    plt.close(fig)


def plot_noise_heatmap():
    grid_path = ROOT / "outputs" / "v2xset_noise_grid.json"
    if not grid_path.exists():
        return
    with grid_path.open("r", encoding="utf-8") as f:
        grid = json.load(f)
    trans_vals = sorted(float(k) for k in grid.keys())
    rot_vals = sorted(float(k) for k in next(iter(grid.values())).keys())
    matrix = np.zeros((len(trans_vals), len(rot_vals)))
    for i, t in enumerate(trans_vals):
        for j, r in enumerate(rot_vals):
            matrix[i, j] = grid[str(t)][str(r)]["success_at_1m"]

    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(matrix, origin="lower", cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(rot_vals)))
    ax.set_xticklabels([f"{r:.2f}" for r in rot_vals], rotation=45, ha="right")
    ax.set_yticks(range(len(trans_vals)))
    ax.set_yticklabels([f"{t:.2f}" for t in trans_vals])
    ax.set_xlabel("Rotation noise σ (deg)")
    ax.set_ylabel("Translation noise σ (m)")
    ax.set_title("Success@1 m under synthetic noise (V2X-Set)")
    fig.colorbar(im, ax=ax, label="Success@1 m")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "noise_heatmap.png", dpi=200)
    plt.close(fig)


def plot_indicator_curves():
    path = ROOT / "outputs" / "v2xset_indicator_curves.json"
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    trans = data["translation"]
    rot = data["rotation"]

    axes[0].plot(trans["bias_m"], trans["oDist"], marker="o", label="oDist")
    axes[0].plot(trans["bias_m"], trans["oIoU"], marker="s", label="oIoU")
    axes[0].set_xlabel("Translation bias (m)")
    axes[0].set_ylabel("Stability score")
    axes[0].set_title("Translation sweep")
    axes[0].grid(True, linestyle="--", alpha=0.4)

    axes[1].plot(rot["bias_deg"], rot["oDist"], marker="o", label="oDist")
    axes[1].plot(rot["bias_deg"], rot["oIoU"], marker="s", label="oIoU")
    axes[1].set_xlabel("Rotation bias (deg)")
    axes[1].set_title("Rotation sweep")
    axes[1].grid(True, linestyle="--", alpha=0.4)

    axes[1].legend(loc="best")
    fig.suptitle("oDist vs oIoU stability on V2X-Set")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "indicator_curves.png", dpi=200)
    plt.close(fig)


def plot_association_violin():
    path = ROOT / "outputs" / "v2xset_association_ablation.json"
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    fig, ax = plt.subplots(figsize=(7, 4))
    names = list(data.keys())
    values = [data[name] for name in names]
    vp = ax.violinplot(values, showmeans=True, showextrema=False)
    for body in vp['bodies']:
        body.set_alpha(0.6)
    ax.set_xticks(range(1, len(names) + 1))
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylabel("Matches per frame")
    ax.set_title("Association strategies on V2X-Set")
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "association_violin.png", dpi=200)
    plt.close(fig)


def main():
    plot_hkust_runtime()
    plot_success_vs_topk()
    plot_noise_heatmap()
    plot_indicator_curves()
    plot_association_violin()
    print(f"Saved figures to {OUT_DIR}")


if __name__ == "__main__":
    main()
