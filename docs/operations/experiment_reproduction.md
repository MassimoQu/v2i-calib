# Experiment Reproduction Guide

This repository provides the **calibration solver** and the scripts/configs used for the paper’s experiments (object-level, using 3D bounding boxes).

For the reported quantitative results, please refer to the papers. For a minimal entrypoint, see `docs/operations/experiment_progress_public.md`.

## 1. Environment

```bash
conda create -n v2xreg python=3.10 -y
conda activate v2xreg
pip install -r requirements.txt
```

Optional (only needed for `benchmarks/` baselines):
```bash
pip install open3d==0.17.* torch==2.3.*
git submodule update --init --recursive
```

## 2. Data (DAIR-V2X)

Place the official DAIR-V2X cooperative split under:
- `data/DAIR-V2X/cooperative-vehicle-infrastructure/`

If you keep the dataset elsewhere, symlink it:
```bash
ln -s /path/to/cooperative-vehicle-infrastructure data/DAIR-V2X/cooperative-vehicle-infrastructure
```

## 3. Run the core pipeline (DAIR-V2X)

- Full test split (GT boxes):
  ```bash
  python tools/run_calibration.py --config configs/pipeline.yaml --print
  ```

- Paper-aligned Table III (DAIR-V2X, GT rows) on the provided 3737-frame subset:
  ```bash
  python tools/run_dair_pipeline_experiments.py --config configs/pipeline_paper3737.yaml
  ```

- Fast sweeps (Top-3000 subset, for debugging / quick iteration):
  ```bash
  python tools/run_dair_pipeline_experiments.py --config configs/pipeline_top3000.yaml
  ```

Outputs are written to `outputs/<tag>/`:
- `metrics.json`: aggregated metrics + avg runtime
- `matches.jsonl`: per-frame timing + match details

Notes:
- `success_at_{λ}m` uses the paper-aligned criterion: `RTE < λ (m)` **and** `RRE < λ (deg)`.
- `success_te_only_at_{λ}m` is kept as a TE-only debug indicator.

## 4. Baselines (optional)

Baseline scripts live in `benchmarks/` and require extra dependencies/build steps (Open3D / TEASER++ / PyTorch, etc.). They are provided for convenience and for sanity-checking against common registration baselines.
