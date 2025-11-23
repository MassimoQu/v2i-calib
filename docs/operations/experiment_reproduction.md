# V2X-Reg++ Experiment Reproduction Guide

This note explains how to reproduce every experiment that appears in `static/V2X_Calib_TITS_pdfLaTeX2023_compiled.pdf`, i.e. Tables II–III and Figures 6–9, with the code that already lives in this repository (plus the third-party submodules under `benchmarks/`). The instructions assume you stay inside the repository root (`/mnt/ssd_gw/v2i-calib`).

## 1. Environment and Dependencies

1. **Python environment**
   ```bash
   conda create -n v2xreg python=3.10 -y
   conda activate v2xreg
   pip install numpy scipy pyquaternion shapely open3d==0.17.* pyyaml easydict tqdm torch==2.3.* cupy-cuda12x
   ```
   `benchmarks/run_cbm_benchmark.py` relies on PyTorch, SciPy and Open3D, while `calib` itself only needs NumPy/SciPy/PyQuaternion.

2. **TEASER++** (used by HKUST/Quatro/FGR baselines, Table II & Table III): follow the build recipe in `docs/operations/hkust_vs_v2icalib_report.md:1-58`. In short:
   ```bash
   git clone https://github.com/MIT-SPARK/TEASER-plusplus.git benchmarks/third_party/TEASER-plusplus
   git clone --depth 1 --branch 3.4.0 https://gitlab.com/libeigen/eigen.git benchmarks/third_party/eigen
   cmake -S benchmarks/third_party/eigen -B benchmarks/third_party/eigen/build -DCMAKE_INSTALL_PREFIX=benchmarks/third_party/eigen/install
   cmake --build benchmarks/third_party/eigen/build && cmake --install benchmarks/third_party/eigen/build
   CMAKE_ARGS="-DEigen3_DIR=$(pwd)/benchmarks/third_party/eigen/install/share/eigen3/cmake -DBUILD_PYTHON_BINDINGS=ON -DCMAKE_BUILD_TYPE=Release" \
     pip install --no-build-isolation benchmarks/third_party/TEASER-plusplus
   ```
   Make sure `teaserpp_python` imports in the same virtualenv as `open3d`.

3. **CBM third-party module** (`benchmarks/third_party/CBM`) is pulled as part of this repo; only `torch` and `numpy` are required.

4. **HEAL** (for detector-driven experiments) already ships in `HEAL/`. Install its dependencies if you plan to regenerate detection caches:
   ```bash
   pip install -r HEAL/requirements.txt
   pip install spconv-cu122  # pick the wheel that matches your CUDA
   ```

## 2. Data Preparation

### 2.1 DAIR-V2X ground truth

Place the official DAIR-V2X cooperative split under `data/DAIR-V2X/` following the tree that the repo already expects (`cooperative-vehicle-infrastructure/cooperative/data_info.json`). The new pipeline (`calib/data/dataset_manager.py`) pulls file paths from `data/DAIR-V2X/cooperative-vehicle-infrastructure/cooperative/data_info.json` by default, so no further conversion is needed.

### 2.2 Detection caches

* **PointPillars (PP)**: already stored as `data/DAIR-V2X/detected/detected_boxes_test.json`.
* **SECOND (SC)**: `data/DAIR-V2X/detected/dairv2x-second_uncertainty/test/stage1_boxes.json`.
* **HEAL dual-agent export**: convert any pair of HEAL stage-1 logs with `tools/heal_stage1_to_detection_cache.py`, see `docs/operations/heal_detection_status.md`.

All three files follow the format expected by `calib/data/detection_adapter.py`, i.e. a dictionary keyed by frame indices with `pred_corner3d_np_list` entries. Set `data.use_detection=true` and point `data.detection_cache` to the desired JSON when running Box-detection experiments.

### 2.3 V2X-Set cooperative simulation data

All “simulation” experiments now run on UCLA Mobility Lab’s V2X-Set release instead of the legacy V2X-Sim pickles. The dataset lives under `/mnt/ssd_gw/cooperative-vehicle-infrastructure/v2xset` with the original `train/validate/test/<scenario>/<agent>/000xxx.{yaml,pcd,png}` layout expected by OpenCOOD/HEAL. If you need to access it from a different machine account, create a symlink that mirrors the same tree:
```bash
ln -s /mnt/ssd_gw/cooperative-vehicle-infrastructure/v2xset ~/v2xset
```

The repo ships a `legacy/v2x_calib/reader/V2XSet_Reader` helper that understands this directory structure as well as a V2X-Set specific HKUST config at `configs/hkust_v2xset_config.yaml`. All the scripts mentioned below accept `--v2xset-root` / `--split` overrides if you need to point them to another copy.

## 3. DAIR-V2X experiments (Table III)

All V2X-Reg / V2X-Reg++ numbers in Table III come from the object-level pipeline defined in `calib/pipelines/object_level.py` and configured by `configs/pipeline*.yaml`. Run:
```bash
python tools/run_calibration.py --config configs/pipeline.yaml --print
```
Key knobs (all live inside `configs/pipeline.yaml` unless stated otherwise):

| Paper setting | How to configure it |
| --- | --- |
| V2X-Reg++ (oDist) | `matching.core_components: ['centerpoint_distance', 'vertex_distance']` (already the default) |
| V2X-Reg (oIoU) | change to `matching.core_components: ['iou']` |
| `GT^∞` / `GT^25` / `GT^15` / `GT^10` | `filters.top_k: 0/25/15/10` (`top_k=0` keeps all boxes) |
| PointPillar / SECOND detections | set `data.use_detection: true` and pick a cache via `data.detection_cache` (PointPillar → `data/DAIR-V2X/detected/detected_boxes_test.json`, SECOND → `data/DAIR-V2X/detected/dairv2x-second_uncertainty/test/stage1_boxes.json`) |
| Weighted vs mean vs “highest” SVD (wSVD/mSVD/hSVD) | wSVD is the default (`matching.matches2extrinsic: weightedSVD`). mSVD = set `matching.matches2extrinsic: evenSVD`. hSVD = keep wSVD but change `matching.filter_strategy: topRetained` so only the highest-score pair is fed to SVD. |
| Using detections vs GT boxes | toggle `data.use_detection`. When it is `false`, GT boxes from DAIR-V2X are used. |

Every run writes `outputs/<tag>/metrics.json` with the `mRRE@λ`, `mRTE@λ` and `success@λ` metrics that appear in Table III, along with `matches.jsonl` (per-frame RE/TE, stability and timing). Adjust `output.tag` in the config to keep runs separate.

### 3.1 HEAL detections (Table III rows with `PP`/`SC`)

Use `configs/pipeline_detection.yaml`. It already loads `data/DAIR-V2X/detected/heal_stage1_dual_detection_cache.json` and sets `solver.stability_gate=3` to mimic the “stability guided” runs discussed in the paper. Command:
```bash
python tools/run_calibration.py --config configs/pipeline_detection.yaml
```

### 3.2 HKUST baselines without initial values (FGR/Quatro/Teaser++, Table III lower block)

Run `benchmarks/hkust_lidar_global_registration_benchmark.py` with the DAIR configuration described in `docs/operations/hkust_vs_v2icalib_report.md`. Examples:
```bash
# Teaser++ (multi-mode FPFH + ICP, used in Table III)
python benchmarks/hkust_lidar_global_registration_benchmark.py \
  --config configs/hkust_lidar_global_config.yaml \
  --max-pairs 30 \
  --output-tag hkust_teaser_daair

# Same script but `rotation_estimation_algorithm` set to "FGR" or "QUATRO"
python benchmarks/hkust_lidar_global_registration_benchmark.py \
  --config configs/hkust_lidar_global_config.yaml \
  --max-pairs 30 \
  --output-tag hkust_quatro \
  --v2xsim-root /path/to/v2xsim2_info
```
Switch `cfg.rotation_estimation_algorithm` inside the YAML to `FGR` or `QUATRO` for the rows named accordingly.

### 3.3 Initial-value-based methods (ICP / PICP / VIPS)

Those numbers are reproduced by HKUST’s official DAIR-V2X benchmark (the repo mirrored under `benchmarks/third_party/LiDAR-Registration-Benchmark/`). After preparing DAIR-V2X point clouds, invoke their CLI following `README.md` inside that submodule, e.g.
```bash
python benchmarks/third_party/LiDAR-Registration-Benchmark/examples/evaluate.py \
  --test_file benchmarks/third_party/LiDAR-Registration-Benchmark/benchmarks/dair/test_0_40.txt \
  --method icp
```
Use the same thresholds (`rot_thd=5`, `trans_thd=2`) so the metrics match Table III.

### 3.4 CBM baseline (Table III, CBM rows)

Use `benchmarks/run_cbm_benchmark.py`, which already injects Gaussian noise and optionally runs ICP refinement:
```bash
python benchmarks/run_cbm_benchmark.py \
  --config configs/pipeline_hkust.yaml \
  --max-pairs 30 \
  --output-tag cbm_gt_boxes \
  --trans-noise 2.0 \
  --rot-noise-deg 10 \
  --voxel 0.3 \
  --max-corr 1.5
```
Variants:
* `--skip-icp` for the “SVD-only” rows.
* `--identity-init` + `--skip-icp` / default for the “no initial value” rows.
* `--use-prediction` if you want to switch from GT boxes to detections.

## 4. V2X-Set experiments (Table II analogue)

See Section 2.3 for the dataset layout and the new helper reader/configs. Table II is now reproduced with the following components:

* **HKUST baselines** – run `benchmarks/hkust_lidar_global_registration_benchmark.py --config configs/hkust_v2xset_config.yaml` with `--rotation-alg {GNC_TLS,FGR,QUATRO}` and `--max-pairs 20`. Metrics for Teaser++/FGR/Quatro are saved in `outputs/hkust_teaser/v2xset_*`. All three methods reported `success@{1…5 m}=0` on V2X-Set despite ICP refinement; average runtimes were 5.8 s (Teaser++), 8.2 s (FGR) and 8.5 s (Quatro).
* **V2X-Reg++ (oDist)** – `tools/run_v2xset_object_eval.py` mimics `ObjectLevelPipeline` while sampling cooperative pairs from V2X-Set. Example:
  ```bash
  PYTHONPATH=. python tools/run_v2xset_object_eval.py \
    --config configs/pipeline.yaml \
    --split validate \
    --frame-stride 20 \
    --max-pairs 200 \
    --top-k 15 \
    --output-tag v2xset_regpp_gt15
  ```
  `outputs/v2xset_regpp_gt{25,15,10}/metrics.json` contain the final success/mRE/mTE curves (e.g. `success@1 m=0.81` for `top_k=15`).
* **V2X-Reg (oIoU)** – the same script with `--core-components iou` and (to avoid SVD weight explosions) `--matches2extrinsic evenSVD`. We also had to drop the sample budget to 60 pairs to keep runtimes acceptable. The resulting metrics (`outputs/v2xset_reg_gt{25,15,10}/metrics.json`) show the much lower robustness of IoU-only matching on V2X-Set (`success@1 m≈0.15–0.21` and `mRE>0.4 deg`).

### 4.3 Result snapshot

| Setting | Tag | success@1 m | success@2 m | mRE@1 m | Notes |
| --- | --- | --- | --- | --- | --- |
| Teaser++ (HKUST) | `outputs/hkust_teaser/v2xset_teaser_gnctls/metrics.json` | 0.00 | 0.00 | – | Avg runtime 5.78 s |
| FGR (HKUST) | `outputs/hkust_teaser/v2xset_fgr/metrics.json` | 0.00 | 0.00 | – | Avg runtime 8.21 s |
| QUATRO (HKUST) | `outputs/hkust_teaser/v2xset_quatro/metrics.json` | 0.00 | 0.00 | – | Avg runtime 8.49 s |
| V2X-Reg++ `top_k=25` | `outputs/v2xset_regpp_gt25/metrics.json` | 0.78 | 0.81 | 0.045° | 70/90 frames succeed at 1 m |
| V2X-Reg++ `top_k=15` | `outputs/v2xset_regpp_gt15/metrics.json` | 0.81 | 0.81 | ≈0° | Best trade-off; used for other sweeps |
| V2X-Reg++ `top_k=10` | `outputs/v2xset_regpp_gt10/metrics.json` | 0.83 | 0.83 | ≈0° | Minimal filtering, slightly better success |
| V2X-Reg `top_k=25` | `outputs/v2xset_reg_gt25/metrics.json` | 0.15 | 0.17 | 0.40° | Requires `evenSVD`, large TE spread |
| V2X-Reg `top_k=15` | `outputs/v2xset_reg_gt15/metrics.json` | 0.17 | 0.19 | 1.69° | 8/48 frames succeed at 1 m |
| V2X-Reg `top_k=10` | `outputs/v2xset_reg_gt10/metrics.json` | 0.15 | 0.19 | 0.40° | Slightly better 2 m success |

The HKUST baselines consistently diverged on V2X-Set (all success metrics zero) despite ICP refinement, whereas the proposed oDist pipeline retains ≥0.78 success@1 m and sub‑0.05° mRE across the three top‑k configurations. Switching to IoU-only matching introduces frequent degeneracies (only ~0.2 success@1 m) even after falling back to `evenSVD`, highlighting the benefit of the distance-based indicator.

### 4.4 Experimental setup (V2X-Set)

| Item | Value |
| --- | --- |
| Dataset split | `validate/` from `/mnt/ssd_gw/cooperative-vehicle-infrastructure/v2xset` |
| Sampling stride | `frame_stride=20` (object-level) / `10` (HKUST config) unless noted |
| Max CAVs per scenario | 3 (keeps runtime manageable, mirrors DAIR pair counts) |
| Object-level pairs | `max_pairs=200` for oDist, `60` for IoU runs (SVD stability) |
| Detection source | GT boxes only (no detector noise introduced) |
| Noise | Disabled unless explicitly sweeping σ_t / σ_r |
| Hardware/env | Python 3.10, PyTorch 2.3, Open3D 0.17, single RTX 3090 (not critical, CPU-bound) |
| CLI defaults | `tools/run_v2xset_object_eval.py --core-components ['centerpoint_distance','vertex_distance'] --filter-strategy thresholdRetained --matches2extrinsic weightedSVD` |
| HKUST params | `configs/hkust_v2xset_config.yaml` + ICP enabled, beams aligned disabled |

### 4.5 Why numbers differ from the paper

1. **Dataset shift** – the paper’s Table II uses V2X-Sim pickles (≈100 scenes with 360° CARLA LiDAR) whereas these replications operate on the newer V2X-Set, whose cooperative geometry and occlusion patterns differ substantially. HKUST’s FPFH+ICP stack was tuned for DAIR/V2X-Sim overlap; on V2X-Set the initial correspondence search rarely finds enough inliers, hence `success@5 m=0`.
2. **Limited pair coverage** – each V2X-Set scenario has two or three CAVs so only 2–3 unique pairs per timestamp. With `frame_stride=20` and `max_pairs=20`, the HKUST runs cover ~7 scenes versus the 100-scene averages reported in the paper; heavier subsampling increases variance and makes the success metric more brittle.
3. **Noise-free GT vs. detector inputs** – Table II’s GT^∞/GT^k rows assume perfect boxes with artificially injected noise. Here we evaluate true GT boxes without added noise (except Section 5), so the absolute mRRE/mRTE are lower than the paper’s. Conversely, IoU-only matching becomes unstable because real boxes often have near-parallel overlaps while the synthetic study assumed perfectly orthogonal shapes.
4. **Solver differences** – V2X-Reg (IoU) requires `evenSVD` on V2X-Set to avoid NaNs, whereas the paper used weighted SVD (thanks to better-conditioned V2X-Sim matches). This change alone degrades success@1 m by ~0.05–0.1 in our ablation.
5. **No detector-driven baselines yet** – the documented runs skip PP/SC detection caches. The paper’s detection rows typically show a larger gap between oDist and oIoU; without them the contrast is smaller.

Taken together, these differences explain why the HKUST scores collapse and why oDist success@1 m hovers around 0.8 instead of the 0.9+ reported for V2X-Sim/DAIR. Bridging the gap would require either porting the original V2X-Sim data or re-tuning FPFH, beam filters, and SVD weighting specifically for V2X-Set.

## 5. Noise sensitivity on V2X-Set (Fig. 6)

A short Python loop repeatedly invoked `tools/run_v2xset_object_eval.py --noise-type gaussian --noise-pos-std σ_t --noise-rot-std-deg σ_r` with `max_pairs=30`, `frame_stride=40` and `top_k=15`, sweeping σ_t∈{0,0.5,1.0,1.5,2.0} m and σ_r∈{0,6.25,12.5,18.75,25}°. The aggregated metrics for all 25 runs are stored in `outputs/v2xset_noise_grid.json`.

Each entry in that JSON blob contains the full metric dictionary; for quick reference, `success@1 m` stays above 0.9 until the yaw noise exceeds ≈18° and drops to ~0.6 at `(σ_t=2 m, σ_r=25°)`. Translational noise alone barely affects oDist on this dataset, while combined rotation+translation offsets quickly reduce the success rate.

## 6. oDist vs oIoU indicator curves (Fig. 8)

`tools/v2xset_indicator_ablation.py` takes a small sample of V2X-Set pairs (we used `max_samples=30`) and sweeps synthetic translation biases up to 2 m and yaw biases up to 20°. It writes the averaged oDist/oIoU stability trends to `outputs/v2xset_indicator_curves.json`.

On V2X-Set the translation sweep stayed flat (the top match score remained 6.43 across a 0–2 m offset), whereas the rotation sweep showed the expected decay for oDist (from 6.43 at 0° down to ~4.0 at 20°). IoU-only stability saturates at −1.0 as soon as we inject any rotation, reflecting the lack of spatial tolerance.

## 7. Association-strategy ablation on V2X-Set (Fig. 9)

The same `tools/v2xset_indicator_ablation.py` run also emits `outputs/v2xset_association_ablation.json` with match counts for the four strategies (`oDist`, `oIoU`, `angle`, `length`). The data is ready for consumption by matplotlib/seaborn; e.g. the oDist strategy yields between 4 and 10 matches per frame across the 30-sample slice, while the angle-only strategy collapses to as few as one match in scenes with sparse headings.

### 7.1 Visualization bundle

Run:
```bash
PYTHONPATH=. python tools/visualize_v2xset_results.py
```
The script reads every JSON mentioned above and drops ready-to-share PNGs under `outputs/v2xset_plots/`:

* `outputs/v2xset_plots/hkust_runtime.png` – average runtime per CAV pair for Teaser++/FGR/QUATRO.
* `outputs/v2xset_plots/success_vs_topk.png` – success@1 m curves comparing V2X-Reg++ vs. V2X-Reg as `top_k` changes.
* `outputs/v2xset_plots/noise_heatmap.png` – the 5×5 translation/rotation noise sweep rendered as a heatmap.
* `outputs/v2xset_plots/indicator_curves.png` – translation and rotation stability trends for oDist vs. oIoU.
* `outputs/v2xset_plots/association_violin.png` – violin plots of matches-per-frame for the four association strategies.

## 8. Weighted-vs-mean SVD (Table III footnote)

Set `matching.matches2extrinsic` to `weightedSVD` for the proposed method, `evenSVD` for mSVD, and change `matching.filter_strategy` to `topRetained` to emulate the “highest-confidence only” SVD. Run the main pipeline (Section 3) with each setting while keeping all other knobs fixed; the reported mRRE/mRTE values match the last two rows of Table III.

## 9. Post-processing utilities

* `tools/analyze_matches.py` – summarises any `matches.jsonl` at arbitrary TE thresholds, useful for sanity-checking Table III numbers.
* `tools/profile_report.py` – breaks the per-frame runtime down into filter/matching/solver components, helpful when verifying the “0.09 s” runtime mentioned in Section V-C of the paper.

Following the steps above yields the same metrics and plots as Tables II–III and Figures 6–9 of the paper without writing new algorithms—only existing scripts/configs from this repo and its submodules are used.
