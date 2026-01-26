# Table III Reproduction (DAIR-V2X paper3737)

This document explains how to reproduce the Table III results for DAIR-V2X (paper3737 subset): V2X-Reg, V2X-Reg++, and the baselines listed in the paper.

## 1. Environment

```bash
conda create -n v2xreg python=3.10 -y
conda activate v2xreg
pip install -r requirements.txt
```

Optional (needed for some baselines in `benchmarks/`):
```bash
pip install open3d==0.17.* torch==2.3.*
```

TEASER++ is required for HKUST baselines (FGR/Quatro/GNC-TLS). Build the submodule under `benchmarks/third_party/TEASER-plusplus` and make sure `teaserpp_python` imports in the same environment as Open3D.

## 2. Data (paper3737 subset)

- Dataset root: `data/DAIR-V2X/cooperative-vehicle-infrastructure/`
- Subset list: `data/data_info_dair_paper3737.json`

If you keep the dataset elsewhere, symlink it or update `data.data_root` / `data.data_info_path` in your YAML configs.

## 3. Detection caches for PP/SC

Table III PP/SC rows require dual-agent detection caches covering all 3737 pairs:
- PP: `data/DAIR-V2X/detected/paper3737_pp_dual/stage1_boxes.json`
- SC: `data/DAIR-V2X/detected/paper3737_sc_dual/stage1_boxes.json`

Each cache is a JSON dict keyed by frame index with per-agent `pred_corner3d_np_list` (or `pred_box3d_np_list`) entries. The loader expects infra/vehicle boxes in local frames; use the conversion script below if your detector exports in ego/world coordinates.

### 3.1 Export dual-agent stage1 boxes (HEAL)

Use HEAL stage-1 export in per-agent mode and a pair-level split list to avoid duplicated vehicle IDs. Generate the pair list from `data/data_info_dair_paper3737.json`:

```bash
python - <<"PY"
import json
from pathlib import Path

src = Path("data/data_info_dair_paper3737.json")
items = json.loads(src.read_text())
pairs = []
for item in items:
    pairs.append({
        "veh_frame_id": Path(item["vehicle_pointcloud_path"]).stem,
        "inf_frame_id": Path(item["infrastructure_pointcloud_path"]).stem,
    })
Path("data/paper3737_pairs.json").write_text(json.dumps(pairs, indent=2))
print("Wrote", len(pairs), "pairs to data/paper3737_pairs.json")
PY
```

Then export PP/SC stage1 boxes with HEAL (paths/checkpoints are placeholders):

```bash
cd HEAL
python opencood/tools/pose_graph_pre_calc.py \
  --per_agent --splits test \
  --test_dir_override ../data/paper3737_pairs.json \
  --vehicle_hypes path/to/pp_vehicle_config.yaml \
  --vehicle_checkpoint path/to/pp_vehicle_ckpt.pth \
  --infra_hypes path/to/pp_infra_config.yaml \
  --infra_checkpoint path/to/pp_infra_ckpt.pth \
  --merged_output ../data/DAIR-V2X/detected/paper3737_pp_dual/stage1_boxes.json

python opencood/tools/pose_graph_pre_calc.py \
  --per_agent --splits test \
  --test_dir_override ../data/paper3737_pairs.json \
  --vehicle_hypes path/to/sc_vehicle_config.yaml \
  --vehicle_checkpoint path/to/sc_vehicle_ckpt.pth \
  --infra_hypes path/to/sc_infra_config.yaml \
  --infra_checkpoint path/to/sc_infra_ckpt.pth \
  --merged_output ../data/DAIR-V2X/detected/paper3737_sc_dual/stage1_boxes.json
```

### 3.2 Convert stage1 export to detection cache

If the stage1 export is in ego/world coordinates, convert it back to each CAV local frame:

```bash
python tools/heal_stage1_to_detection_cache.py \
  --stage1 data/DAIR-V2X/detected/paper3737_pp_dual/stage1_boxes.json \
  --output data/DAIR-V2X/detected/paper3737_pp_dual/stage1_boxes.json \
  --deproject-to-local \
  --require-two-cavs \
  --require-cav-substrings vehicle infrastructure \
  --data-info data/data_info_dair_paper3737.json

python tools/heal_stage1_to_detection_cache.py \
  --stage1 data/DAIR-V2X/detected/paper3737_sc_dual/stage1_boxes.json \
  --output data/DAIR-V2X/detected/paper3737_sc_dual/stage1_boxes.json \
  --deproject-to-local \
  --require-two-cavs \
  --require-cav-substrings vehicle infrastructure \
  --data-info data/data_info_dair_paper3737.json
```

If your stage1 export lacks `cav_id_list`, add `--swap-order` when the vehicle/infra order is reversed.

Sanity check coverage:
```bash
python tools/summarize_detection_cache.py --path data/DAIR-V2X/detected/paper3737_pp_dual/stage1_boxes.json
python tools/summarize_detection_cache.py --path data/DAIR-V2X/detected/paper3737_sc_dual/stage1_boxes.json
```

### 3.3 Notes on HEAL single-agent detection

Single-agent HEAL exports are not sufficient for Table III: overlap is low and many pairs are missing, which collapses success rates. The reproduction setup uses:
- Pair-level split keyed by infra+veh IDs to avoid duplicated vehicle IDs in DAIR-V2X.
- Dual-agent exports with both CAVs present and frame IDs preserved in the cache.
- Conversion back to each CAV local frame (`--deproject-to-local`) and strict cache filtering (`--require-two-cavs`, `--require-cav-substrings`).
- PP/SC configs that apply confidence filtering, consistency filtering, and ICP refinement to stabilize RE tails.

## 4. Table III methods

All commands below assume the `paper3737` subset and the output tags shown (used by `tools/generate_table3_repro_table.py`).

### 4.1 V2X-Reg++ GT variants (GTinf/25/15/10)

```bash
python tools/run_dair_pipeline_experiments.py \
  --config configs/pipeline_paper3737.yaml \
  --tags paper3737_v2xregpp_gt_inf paper3737_v2xregpp_gt25 \
        paper3737_v2xregpp_gt15 paper3737_v2xregpp_gt10
```

### 4.2 V2X-Reg (oIoU)

```bash
python tools/run_calibration.py \
  --config configs/pipeline_paper3737_oiou_gt15_thr4_top6_iter2.yaml --print
```

### 4.3 V2X-Reg++ PP/SC (detections)

```bash
python tools/run_calibration.py \
  --config configs/pipeline_paper3737_pp15_pcalwh_topkCand15_25_30_35_confexp2_consistency1p5_icp.yaml --print

python tools/run_calibration.py \
  --config configs/pipeline_paper3737_sc15_pcalwh_gate1_confexp2_icp.yaml --print
```

### 4.4 HKUST baselines (Teaser++ / FGR / Quatro)

Teaser++ (GNC-TLS):
```bash
python benchmarks/hkust_lidar_global_registration_benchmark.py \
  --config configs/hkust_lidar_global_paper3737_table3.yaml \
  --rotation-alg GNC_TLS \
  --output-tag hkust_teaser_paper3737_gnctls_keepall_full
```

FGR / Quatro:
```bash
python benchmarks/hkust_lidar_global_registration_benchmark.py \
  --config configs/hkust_lidar_global_paper3737_table3_budget0p1_voxel2p0_maxfeat2000_corr1500.yaml \
  --rotation-alg FGR \
  --output-tag paper3737_hkust_budget0p1_maxfeat2000_fgr_full

python benchmarks/hkust_lidar_global_registration_benchmark.py \
  --config configs/hkust_lidar_global_paper3737_table3_budget0p1_voxel2p0_maxfeat2000_corr1500.yaml \
  --rotation-alg QUATRO \
  --output-tag paper3737_hkust_budget0p1_maxfeat2000_quatro_full
```

### 4.5 ICP / PICP (noise 0/1/2)

ICP:
```bash
python benchmarks/run_dair_lidar_benchmark.py \
  --project-config configs/hkust_lidar_global_paper3737_table3.yaml \
  --method icp \
  --trans-noise 0 \
  --rot-noise-deg 0 \
  --output-tag paper3737_icp_noise0

python benchmarks/run_dair_lidar_benchmark.py \
  --project-config configs/hkust_lidar_global_paper3737_table3.yaml \
  --method icp \
  --trans-noise 1 \
  --rot-noise-deg 1 \
  --init-source unadjusted \
  --noise-mode left \
  --max-iter 1 \
  --max-corr 1.0 \
  --output-tag paper3737_icp_noise1_it1_vx0p3_mc1p0_unadj_left

python benchmarks/run_dair_lidar_benchmark.py \
  --project-config configs/hkust_lidar_global_paper3737_table3.yaml \
  --method icp \
  --trans-noise 2 \
  --rot-noise-deg 2 \
  --init-source unadjusted \
  --noise-mode left \
  --max-iter 1 \
  --max-corr 1.0 \
  --output-tag paper3737_icp_noise2_it1_vx0p3_mc1p0_unadj_left
```

PICP:
```bash
python benchmarks/run_dair_lidar_benchmark.py \
  --project-config configs/hkust_lidar_global_paper3737_table3.yaml \
  --method picp \
  --trans-noise 0 \
  --rot-noise-deg 0 \
  --output-tag paper3737_picp_noise0

python benchmarks/run_dair_lidar_benchmark.py \
  --project-config configs/hkust_lidar_global_paper3737_table3.yaml \
  --method picp \
  --trans-noise 1 \
  --rot-noise-deg 1 \
  --init-source unadjusted \
  --noise-mode left \
  --max-iter 1 \
  --max-corr 1.0 \
  --output-tag paper3737_picp_noise1_it1_vx0p3_mc1p0_unadj_left

python benchmarks/run_dair_lidar_benchmark.py \
  --project-config configs/hkust_lidar_global_paper3737_table3.yaml \
  --method picp \
  --trans-noise 2 \
  --rot-noise-deg 2 \
  --init-source unadjusted \
  --noise-mode left \
  --max-iter 1 \
  --max-corr 1.0 \
  --output-tag paper3737_picp_noise2_it1_vx0p3_mc1p0_unadj_left
```

### 4.6 VIPS (noise 0/1/2)

```bash
python benchmarks/run_vips_benchmark.py \
  --config configs/pipeline_paper3737.yaml \
  --max-pairs 3737 \
  --trans-noise 0 \
  --rot-noise-deg 0 \
  --init-source gt \
  --match-distance-thr 1.5 \
  --output-tag paper3737_noise0_thr1p5_initgt_icp

python benchmarks/run_vips_benchmark.py \
  --config configs/pipeline_paper3737.yaml \
  --max-pairs 3737 \
  --trans-noise 1 \
  --rot-noise-deg 1 \
  --init-source gt \
  --match-distance-thr 1.5 \
  --output-tag paper3737_noise1_thr1p5_initgt_icp

python benchmarks/run_vips_benchmark.py \
  --config configs/pipeline_paper3737.yaml \
  --max-pairs 3737 \
  --trans-noise 2 \
  --rot-noise-deg 2 \
  --init-source unadjusted \
  --match-distance-thr 1.5 \
  --output-tag paper3737_noise2_thr1p5_initunadj_icp
```

### 4.7 CBM (noise 0/1/2)

```bash
python benchmarks/run_cbm_benchmark.py \
  --config configs/pipeline_paper3737.yaml \
  --max-pairs 3737 \
  --trans-noise 0 \
  --rot-noise-deg 0 \
  --init-source gt \
  --min-matches 20 \
  --match-distance-thr 4 \
  --skip-icp \
  --output-tag paper3737_cbm_noise0_min20_thr4_initgt_noicp_full \
  --voxel 0.3 \
  --max-corr 1.5

python benchmarks/run_cbm_benchmark.py \
  --config configs/pipeline_paper3737.yaml \
  --max-pairs 3737 \
  --trans-noise 1 \
  --rot-noise-deg 1 \
  --init-source unadjusted \
  --min-matches 20 \
  --match-distance-thr 4 \
  --skip-icp \
  --output-tag paper3737_noise1_min20_thr4_initunadj_noicp \
  --voxel 0.3 \
  --max-corr 1.5

python benchmarks/run_cbm_benchmark.py \
  --config configs/pipeline_paper3737.yaml \
  --max-pairs 3737 \
  --trans-noise 2 \
  --rot-noise-deg 2 \
  --init-source gt \
  --min-matches 20 \
  --match-distance-thr 4 \
  --skip-icp \
  --output-tag paper3737_cbm_noise2_min20_thr4_initgt_noicp_full \
  --voxel 0.3 \
  --max-corr 1.5
```

Note: Long baselines can be sharded with `--start` / `--max-pairs`. Merge the JSONL outputs to cover all 3737 pairs before summarizing.

## 5. Reproduction results (paper3737)

The table below summarizes the runs used for Table III reproduction. Metrics are computed from `matches.jsonl` / `details.jsonl` with success gate `te_re` on the full 3737 pairs.

| Method | Success@1/2/3 (%) | mTE@1/2/3 (m) | mRE@1/2/3 (°) | Time (s) | Output |
|---|---|---|---|---|---|
| ICP-noise0 | 54.91/85.34/97.08 | 0.628/0.849/1.007 | 0.681/0.886/1.017 | 0.497 | `outputs/paper3737_icp_noise0/details.jsonl` |
| ICP-noise1 | 4.58/41.56/78.06 | 0.700/1.316/1.763 | 0.657/1.166/1.374 | 0.067 | `outputs/paper3737_icp_noise1_it1_vx0p3_mc1p0_unadj_left/details.jsonl` |
| ICP-noise2 | 0.27/3.67/19.11 | 0.754/1.367/2.066 | 0.757/1.308/1.972 | 0.070 | `outputs/paper3737_icp_noise2_it1_vx0p3_mc1p0_unadj_left/details.jsonl` |
| PICP-noise0 | 51.43/80.31/94.78 | 0.602/0.837/1.043 | 0.665/0.879/1.048 | 0.609 | `outputs/paper3737_picp_noise0/details.jsonl` |
| PICP-noise1 | 6.05/43.40/78.81 | 0.678/1.276/1.726 | 0.674/1.155/1.365 | 0.174 | `outputs/paper3737_picp_noise1_it1_vx0p3_mc1p0_unadj_left/details.jsonl` |
| PICP-noise2 | 0.35/4.28/19.75 | 0.677/1.359/2.044 | 0.769/1.324/1.970 | 0.174 | `outputs/paper3737_picp_noise2_it1_vx0p3_mc1p0_unadj_left/details.jsonl` |
| VIPS-noise0 | 49.91/78.14/91.81 | 0.603/0.840/1.039 | 0.662/0.877/1.045 | 1.714 | `outputs/vips/paper3737_noise0_thr1p5_initgt_icp/matches.jsonl` |
| VIPS-noise1 | 26.47/40.62/47.44 | 0.589/0.827/1.025 | 0.665/0.874/1.035 | 1.394 | `outputs/vips/paper3737_noise1_thr1p5_initgt_icp/matches.jsonl` |
| VIPS-noise2 | 6.58/9.87/11.13 | 0.588/0.813/0.973 | 0.675/0.858/0.989 | 1.423 | `outputs/vips/paper3737_noise2_thr1p5_initunadj_icp/matches.jsonl` |
| CBM-noise0 | 16.91/22.61/23.15 | 0.591/0.729/0.764 | 0.641/0.760/0.779 | 0.419 | `outputs/cbm/paper3737_cbm_noise0_min20_thr4_initgt_noicp_full/matches.jsonl` |
| CBM-noise1 | 15.44/20.71/21.17 | 0.597/0.729/0.764 | 0.641/0.758/0.775 | 0.615 | `outputs/cbm/paper3737_noise1_min20_thr4_initunadj_noicp/matches.jsonl` |
| CBM-noise2 | 8.48/11.02/11.16 | 0.600/0.723/0.742 | 0.648/0.750/0.758 | 0.415 | `outputs/cbm/paper3737_cbm_noise2_min20_thr4_initgt_noicp_full/matches.jsonl` |
| FGR | 21.43/35.40/42.23 | 0.690/0.915/1.127 | 0.664/0.849/1.021 | 0.150 | `outputs/hkust_teaser/paper3737_hkust_budget0p1_maxfeat2000_fgr_full/matches.jsonl` |
| Quatro | 22.83/38.35/46.00 | 0.689/0.915/1.133 | 0.666/0.864/1.044 | 0.297 | `outputs/hkust_teaser/paper3737_hkust_budget0p1_maxfeat2000_quatro_full/matches.jsonl` |
| Teaser++ | 24.91/36.02/43.06 | 0.592/0.797/1.028 | 0.654/0.865/1.052 | 162.690 | `outputs/hkust_teaser/hkust_teaser_paper3737_gnctls_keepall_full/matches.jsonl` |
| V2X-Reg | 24.62/55.69/71.31 | 0.530/0.863/1.088 | 0.639/1.045/1.294 | 0.146 | `outputs/paper3737_dair_v2xreg_oiou_gt15_thr4_top6_iter2/matches.jsonl` |
| V2X-Reg++ GT∞ | 24.70/52.05/66.79 | 0.490/0.833/1.080 | 0.622/1.003/1.259 | 0.144 | `outputs/paper3737_dair_v2xregpp_gt_inf/matches.jsonl` |
| V2X-Reg++ GT25 | 32.57/67.84/83.46 | 0.517/0.844/1.044 | 0.633/1.008/1.236 | 0.096 | `outputs/paper3737_dair_v2xregpp_gt25/matches.jsonl` |
| V2X-Reg++ GT15 | 26.81/62.70/80.04 | 0.520/0.873/1.095 | 0.641/1.063/1.308 | 0.053 | `outputs/paper3737_dair_v2xregpp_gt15/matches.jsonl` |
| V2X-Reg++ GT10 | 18.97/53.47/70.94 | 0.552/0.929/1.161 | 0.665/1.132/1.399 | 0.037 | `outputs/paper3737_dair_v2xregpp_gt10/matches.jsonl` |
| V2X-Reg++ PP15 | 26.57/57.27/72.60 | 0.538/0.865/1.070 | 0.634/1.021/1.274 | 0.660 | `outputs/paper3737_pp15_pcalwh_topkCand15_25_30_35_confexp2_consistency1p5_icp/matches.jsonl` |
| V2X-Reg++ SC15 | 25.42/55.07/70.81 | 0.532/0.849/1.067 | 0.660/1.028/1.297 | 0.106 | `outputs/paper3737_sc15_pcalwh_gate1_confexp2_icp/matches.jsonl` |

## 6. Regenerate the table

Outputs for each run live under `outputs/<tag>/` and include `metrics.json` and `matches.jsonl` / `details.jsonl`. To regenerate the table:

```bash
python tools/generate_table3_repro_table.py --output table3_results.md
```

Use `--success-gate te_re` (default) and `--allow-incomplete` if you are summarizing partial runs. You can also point to a different outputs directory with `--outputs-root`.
