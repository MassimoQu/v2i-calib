# V2I-Calib++ Feature-Level Extension Log

**Owner:** Codex (autonomous agent)  
**Kick-off:** 2025-11-22  
**Objective:** Extend the V2I-Calib++ (V2X-Reg++) pipeline so that it can consume HEAL-generated feature objects — not just 3D detection boxes — for extrinsic calibration across LiDAR–LiDAR, camera–LiDAR, and camera–camera cases. This document tracks design decisions, implementation details, experiments, and conclusions.

## 1. Background & Feasibility Assessment

- **Current state:** V2X-Reg++ operates purely on 3D bounding boxes (GT or detections). HEAL’s stage-1 export provides per-agent boxes (`pred_corner3d_np_list`) plus poses, but no low-level features. All calibration stages live under `calib/`, while HEAL handles detection/export flows.
- **Why feature-level?:** Detection overlap between infrastructure and vehicle is sparse (Section 2 of `docs/operations/heal_detection_status.md`). Relying solely on boxes leaves many frames unsolved. Prominent BEV/local features (peaks, corners, stable map points) are denser and can act as “virtual objects”.
- **Feasibility check:**
  1. HEAL’s stage-1 scripts already traverse BEV logits per agent, so extracting top-K anchor responses is straightforward.
  2. V2X-Reg++ downstream modules only assume inputs implement the `BBox3d` interface (8 corners + type/confidence). Any feature that can be wrapped as a pseudo box fits into existing filters/matching/solvers.
  3. Coordinate alignment remains the main hurdle — exported features must be reprojected into each sensor’s local frame, similar to the current detection cache conversion. The existing `heal_stage1_to_detection_cache.py` pipeline can be extended instead of reinvented.
- **Planned scope for the first milestone:**
  - Focus on LiDAR–LiDAR features derived from HEAL stage-1 BEV logits.
  - Treat each high-response anchor cell as a “micro box” (type `feature`) so the rest of the calibration stack remains untouched.
  - Add plumbing in `calib.data.DatasetManager` to load either GT boxes, detection boxes, or feature boxes.
  - Validate on DAIR-V2X test split and target V2X-Reg++-level metrics.
  - Document every action and outcome here; iterate until performance matches detection-box baseline.

## 2. Execution Plan

1. **Tooling upgrades (HEAL side):**
   - Extend `opencood/tools/pose_graph_pre_calc.py` to optionally dump feature boxes per agent (`feature_corner3d_np_list`, `feature_score_list`).
   - Parameters: top-K anchors, minimum score, micro-box dimensions.
   - Ensure merged stage-1 exports carry the feature metadata forward.
2. **Conversion pipeline:**
   - Update `tools/heal_stage1_to_detection_cache.py` to transform feature corners back to each CAV’s local frame (same as detections).
   - Support dict-based box entries so we can preserve type/confidence in JSON.
3. **Calibration data ingestion:**
   - Enhance `calib/data/detection_adapter.py` to parse dict objects (`{'type': 'feature', 'corners': ..., 'score': ...}`) and to expose `get(..., field='feature_corner3d_np_list')`.
   - Add feature-specific knobs to `DatasetManager` (choose GT/detection/feature source).
   - Create/modify pipeline configs (e.g., `configs/pipeline_features.yaml`) with feature-friendly filter priorities and distance thresholds.
4. **Verification:**
   - Export a dual-agent stage-1 run with feature dumps.
   - Convert to detection/feature caches.
   - Run `tools/run_calibration.py` with the new pipeline config; compare metrics vs. V2X-Reg++ detection baseline.
   - Iterate on filter/matching hyper-parameters if feature-only performance lags.

Each milestone will capture commands, intermediate artefacts, and observations in the following sections.

## 3. Current Findings (2025-11-22)

- `HEAL/opencood/tools/pose_graph_pre_calc.py` already has `--feature_topk`, `--feature_min_score`, and `--feature_box_dims` arguments. Internally it calls `_extract_feature_boxes` to wrap the top-K anchor responses into `{'type': 'feature', 'score': s, 'corners': [...]}` blobs per agent and writes them under `feature_corner3d_np_list`. Merged dual-agent exports also propagate this field.
- `tools/heal_stage1_to_detection_cache.py` normalizes those records: when `--deproject-to-local` is set, it applies the same pose-based back-transformation to `feature_corner3d_np_list`, so the JSON already ends up in each CAV’s local frame. Swapping vehicle/infra order works for both detections and features.
- `calib/data/detection_adapter.py` is feature-ready: `DetectionAdapter._convert_bbox` accepts dict entries with custom `type` and `score`, meaning a cache with `type: feature` produces valid `BBox3d` objects. Missing piece: `DatasetManager` still has a single `use_detection` flag and field hard-coded to `pred_corner3d_np_list`, so the pipeline cannot yet opt into feature caches.
- Config dataclass (`calib/config.py`) already defines `feature_cache`, `use_features`, and `feature_field`, so wiring them through `DatasetManager` + `ObjectLevelPipeline` should unblock the ingestion path without touching matching or solver logic.

## 4. Milestone #1 – Feature export & ingestion smoke test (2025-11-22)

### 4.1 Commands & artefacts

1. **Stage-1 dual-agent export with feature taps (10-frame pilot, CPU-only).**
   ```bash
   cd HEAL
   PYTHONPATH=. python opencood/tools/pose_graph_pre_calc.py \
     --per_agent \
     --vehicle_hypes ../checkpoints/stage1/Pyramid_DAIR_m1_base_2023_08_14_11_42_29/config.yaml \
     --vehicle_checkpoint ../checkpoints/stage1/Pyramid_DAIR_m1_base_2023_08_14_11_42_29/net_epoch_bestval_at23.pth \
     --vehicle_output ../data/DAIR-V2X/detected/heal_stage1_vehicle_feature \
     --vehicle_force_ego vehicle \
     --infra_hypes ../checkpoints/stage1/Pyramid_DAIR_m1_base_2023_08_14_11_42_29/config.yaml \
     --infra_checkpoint ../checkpoints/stage1/Pyramid_DAIR_m1_base_2023_08_14_11_42_29/net_epoch_bestval_at23.pth \
     --infra_output ../data/DAIR-V2X/detected/heal_stage1_infra_feature \
     --infra_force_ego infrastructure \
     --merged_output ../data/DAIR-V2X/detected/heal_stage1_dual_feature \
     --splits test \
     --single_agent_comm_range 0 \
     --feature_topk 64 \
     --feature_min_score 0.30 \
     --max_export_samples 10
   ```
   Output: `data/DAIR-V2X/detected/heal_stage1_dual_feature/stage1_boxes.json` with `feature_corner3d_np_list` populated (64 micro-boxes per agent).

2. **Conversion to calibration cache (local coordinates).**
   ```bash
   python tools/heal_stage1_to_detection_cache.py \
     --stage1 data/DAIR-V2X/detected/heal_stage1_dual_feature \
     --output data/DAIR-V2X/detected/heal_stage1_dual_feature_cache.json \
     --require-two-cavs \
     --deproject-to-local \
     --ego-index 0
   ```

3. **Pipeline plumbing.**
   - `calib/data/interfaces.py`: `CalibrationSample` now carries `features_infra` / `features_vehicle`.
   - `calib/data/dataset_manager.py`: keeps a dedicated `DetectionAdapter` for feature caches and forwards both detections and features to callers.
   - `calib/pipelines/object_level.py`: selects feature boxes whenever `data.use_features=true`, falling back to detections or GT otherwise.
   - New config `configs/pipeline_features.yaml` (feature cache path, `filters.top_k=15`, `matching.distance_thresholds.feature=1.0`, `output.tag=heal_features_smoke`).

### 4.2 Smoke-test run

Command:
```bash
python tools/run_calibration.py --config configs/pipeline_features.yaml --print
```

Cache: `data/DAIR-V2X/detected/heal_stage1_dual_feature_cache.json` (10 frames, 64 features/agent, filtered to top-15 in `FilterPipeline`).  
Metrics (`outputs/heal_features_smoke/metrics.json`):

| Metric | Value |
| --- | --- |
| `success@1m` | 0% |
| `success@2m` | 0% |
| `success@3m` | 10% (1/10) |
| `mRE@3m` | 2.71° |
| `mTE@3m` | 2.24 m |
| `avg_time` | 0.89 s / frame (dominated by matching on dense features) |

Observations:
- Initial attempt with `filters.top_k=50` caused `MatchingEngine` to spend ~150 s/frame (Hungarian on 50×50 KP); scaling top-K down to 15 gives usable latency.
- Stability scores saturate at 49 because feature boxes are nearly identical in both sensors—translational ambiguity remains and solver collapses to high-RE solutions. Need to inject additional cues (e.g., pseudo categories, anchor yaw) or switch to IoU-style matching for features.
- The plumbing is functional end-to-end: features flow from HEAL → cache → dataset manager → matcher/solver, and bookkeeping in `matches.jsonl` now tags the bbox source as `feature`.

### 4.3 Next actions

1. Increase export coverage from 10 pilot frames to the full DAIR test split (1.7k frames) once runtime knobs are tuned (may require even smaller feature_topk or pre-filtering by BEV cell clustering).
2. Experiment with different matching settings tailored for dense pseudo points:
   - IoU-only core component vs. distance metrics.
   - Aggressive thresholding (`filter_threshold`, `distance_thresholds.feature`) plus per-category quotas that mimic spatial tiling.
3. Fuse feature and detection queues (e.g., append top-K HEAL detections to the feature list) to see whether sparse boxes help disambiguate yaw for the dense anchors.

## 5. 深度特征抽头方案（2025-11-23）

- **候选特征层**  
  - `heter_feature_2d`：每个 agent 在 `encoder → backbone → aligner` 之后的 BEV feature（64 通道、BEV 范围 204.8 m × 102.4 m，以 0.4 m voxel 尺寸编码，经过 backbone downsample×2 → 实际网格约 256×128）。这是中融合前的“自车视角”语义图，具备跨 agent 一致性。  
  - `PyramidFusion.get_multiscale_feature` 输出的多尺度 feature pyramid（level0≈256×128，level1≈128×64，level2≈64×32）以及每层的 `occ_map`（单通道 logits）。`occ_map` 通过 sigmoid 后可以视作稠密的 BEV occupancy cues，比最终 NMS 后的检测框更细粒度。  
  - `fused_feature`：多 agent 权重融合后的 BEV 特征，更多反映协同后的一致感知，但若直接用于标定会丢失“双方独立观测”这一约束，暂不作为匹配输入。

- **抽头方式**  
  - `HeterPyramidCollab.forward` 并不会把上述特征写入 `output_dict`，也无法通过 `register_forward_hook` 直接捕获 `pyramid_backbone.forward_collab` 的输出（调用的是自定义方法而非 `forward`).  
  - 计划在 `pose_graph_pre_calc.export_stage1_boxes` 内对 `stage1_model.pyramid_backbone.forward_collab` 做猴子补丁：包一层 wrapper，缓存 `(feature_list, occ_map_list, fused_feature)` 并在单帧导出完成后写入磁盘。  
  - 对于 pre-fusion 的 `heter_feature_2d`，可在 `HeterPyramidCollab.forward` 中设置 `self._latest_agent_features = heter_feature_2d.detach()`，或者在导出脚本里注册 `forward_pre_hook`，读取 `stage1_model.__dict__['modality_feature_dict']`（需轻量代码改动）。

- **空间对齐 & 伪观测构造**  
  - 每个 BEV cell 对应物理坐标 `x = x_min + (j+0.5)*voxel_size_x*ds`, `y = y_min + (i+0.5)*voxel_size_y*ds`（`ds=2` 为 backbone 下采样因子）。  
  - 我们可以从 `occ_map` 或 `||feature||` 中挑选 top-K（或基于 soft-argmax 进行 peak clustering）的位置，生成带方向的“微 patch”。方向可由 anchor yaw（`[0, π/2]`）或通过局部梯度估计。  
  - 将这些 patch 封装为 `FeatureBox`：`corners` 由中心点 + 固定尺寸（0.6×0.6×0.5）生成，`descriptor` 存储对应 BEV feature 向量（例如 64D 或降维到 16D）以供后续匹配使用。

- **匹配与最优传输**  
  - 现有的 `MatchingEngine/BoxesMatch` 只比较几何（中心距/顶点距/IoU）。为了让深度特征发挥作用，需要扩展 `similarity_utils` 支持 descriptor 相似度（如余弦或 L2），并在 `matching.strategy` 中新增 `descriptor` 分支。  
  - 梯度强的 patch 之间可以先用局部 SVD 获取候选姿态，再借助 descriptor cost 构造 OT 问题（类似原论文的匹配打分），最后仍用全局 SVD 求解外参。

- **下一步行动**  
  1. 在 `pose_graph_pre_calc.py` 中加 `--dump-feature-map` 和 `--feature_levels` 之类的开关，输出 `{sample_idx}/{agent}/{level}/occ_map.npy` 或聚合好的 peak 列表。  
  2. 实现 `FeaturePeakExtractor`（输入 occupancy/feature tensor，输出中心坐标 + descriptor），并在导出阶段直接写成 JSON/NPZ，避免保存全量 64×256×128 浮点图。  
  3. 扩展 `calib.matching` 以使用 descriptor cost，重新设计 `distance_thresholds`、`filter_threshold`，匹配 pipeline 将同时考虑几何与语义一致性。

## 6. Dense BEV Feature Export – Pilot（2025-11-23）

### 6.1 实现要点

- `HeterPyramidCollab` 暴露 `record_runtime_features`，触发时会缓存 `heter_feature_2d`、`occ_map_list`、`record_len` 等 CPU tensor，供导出脚本调用。  
- `pose_graph_pre_calc.py` 新增 `--dump_bev_features`、`--bev_feature_topk` 等参数，调用 `_extract_bev_feature_peaks` 在 PyramidFusion 的 occupancy map 上做 sigmoid+topK，直接生成 “micro-box + descriptor” 结构并写入 `feature_corner3d_np_list`。  
- `BBox3d` / `DetectionAdapter` 支持 `descriptor` 字段，后续匹配可读取深度描述子。

### 6.2 命令 & 产物

```bash
cd HEAL
PYTHONPATH=. python opencood/tools/pose_graph_pre_calc.py \
  --per_agent \
  --vehicle_hypes ../checkpoints/stage1/Pyramid_DAIR_m1_base_2023_08_14_11_42_29/config.yaml \
  --vehicle_checkpoint ../checkpoints/stage1/Pyramid_DAIR_m1_base_2023_08_14_11_42_29/net_epoch_bestval_at23.pth \
  --vehicle_output ../data/DAIR-V2X/detected/heal_stage1_vehicle_bev \
  --vehicle_force_ego vehicle \
  --infra_hypes ../checkpoints/stage1/Pyramid_DAIR_m1_base_2023_08_14_11_42_29/config.yaml \
  --infra_checkpoint ../checkpoints/stage1/Pyramid_DAIR_m1_base_2023_08_14_11_42_29/net_epoch_bestval_at23.pth \
  --infra_output ../data/DAIR-V2X/detected/heal_stage1_infra_bev \
  --infra_force_ego infrastructure \
  --merged_output ../data/DAIR-V2X/detected/heal_stage1_dual_bev \
  --splits test \
  --single_agent_comm_range 0 \
  --feature_topk 0 \
  --dump_bev_features \
  --bev_feature_topk 32 \
  --bev_feature_score_min 0.35 \
  --bev_descriptor_dim 32 \
  --max_export_samples 5
python tools/heal_stage1_to_detection_cache.py \
  --stage1 data/DAIR-V2X/detected/heal_stage1_dual_bev \
  --output data/DAIR-V2X/detected/heal_stage1_dual_bev_cache.json \
  --require-two-cavs \
  --deproject-to-local \
  --ego-index 0
```

`feature_corner3d_np_list` 现在由 32 个 peak 组成，每个 peak 记录 `corners/score/descriptor/grid/level/type`。Descriptors 为 32 维 BEV embedding，后续可直接喂入匹配模块。

### 6.3 Calibration 烟测

- `configs/pipeline_features.yaml` → `use_detection=false`，`feature_cache=data/DAIR-V2X/detected/heal_stage1_dual_bev_cache.json`，`filters.top_k=20`，`distance_thresholds.feature=1.0`，输出目录 `outputs/heal_features_bev_smoke/`。  
- `python tools/run_calibration.py --config configs/pipeline_features.yaml --print`

结果（5 帧 pilot）：

| Metric | Value |
| --- | --- |
| `success@1m` / `@2m` | 0 % |
| `success@3m` | 20 % (1/5) |
| `mTE@3m` | 2.24 m |
| `avg_time` | 3.20 s/frame |

**观察**  
1. 匹配耗时显著上升（20×20 Hungarian + descriptor尚未参与过滤）。  
2. 仍依赖纯几何分数，stability 很高但 RE/TE 仍大，说明需要 descriptor cost 或额外空间先验才能区分密集 BEV patch。  
3. Peaks 之间存在大量重复（邻格响应），需要局部 NMS 或 score margin 控制。

**后续计划**  
- 在 `MatchingEngine` 增加 descriptor similarity（余弦 / L2），将其纳入 `matching.strategy` 并对 `BoxesMatch` / `similarity_utils` 做 OT 级别的加权。  
- 对 `bev_feature_topk` 做自适应（按 score 阈值 + NMS），把每帧候选压到 8–12 提升实时性。  
- 探索 detection+feature 混合列表，或将少量 HEAL 检测框 append 到这些 dense features 中，缓解对 descriptor 的依赖。

## 7. Descriptor Matching 尝试（2025-11-23）

### 7.1 代码更新

- `legacy/v2x_calib/corresponding/BoxesMatch.py`: 支持 `descriptor_weight`、`descriptor_metric`，当 `matching.strategy` 包含 `descriptor` 时，将 `similarity_utils.cal_descriptor_similarity` 的输出叠加到 KP 矩阵。  
- `legacy/v2x_calib/corresponding/similarity_utils.py`: 新增 `cal_descriptor_similarity`，同时实现 cosine / L2 指标。  
- `calib/matching/engine.py`:  
  - 引入 descriptor-only 匹配路径（`matching.strategy: ['descriptor_only']`）→ 直接对 descriptor 相似度矩阵做 Hungarian，按阈值筛选后输出配对与稳定度。  
  - 普通路径可叠加 descriptor cost，并暴露 `descriptor_min_similarity`、`descriptor_max_pairs`。  
- `legacy/v2x_calib/reader/BBox3d.py` / `calib/data/detection_adapter.py`: `BBox3d` 持 descriptor 属性，adapter 会将 JSON 中的 `descriptor` 数组转为 `np.float32`。  
- `HEAL/opencood/tools/pose_graph_pre_calc.py`:  
  - `--bev_feature_min_separation`：BEV peak 层内的栅格 NMS，默认 0。  
  - `--bev_descriptor_on_detections`：为 `pred_corner3d_np_list` 中的检测框采样 BEV descriptor（最近邻）。  
  - 引入 `_annotate_detection_descriptors`，并确保 `record_runtime_features` 在 “提峰” 或 “标注 detection” 任一开启时都生效。  
- `tools/heal_stage1_to_detection_cache.py`: 去 local 坐标时能处理 `{'corners': ..., 'descriptor': ...}` 结构。

### 7.2 配置 / 命令

1. **重导 5 帧 sample（dense feature 关闭，仅保留 detection + descriptor）**
   ```bash
   cd HEAL
   PYTHONPATH=. python opencood/tools/pose_graph_pre_calc.py \
     --per_agent \
     ...  # 同 §6.2，但设置 --feature_topk 0 --bev_feature_min_separation 4 \
     --bev_descriptor_on_detections \
     --max_export_samples 5
   python tools/heal_stage1_to_detection_cache.py \
     --stage1 data/DAIR-V2X/detected/heal_stage1_dual_bev \
     --output data/DAIR-V2X/detected/heal_stage1_dual_bev_cache.json \
     --require-two-cavs \
     --deproject-to-local \
     --ego-index 0
   ```

2. **Descriptor-only 匹配测试**（初期将 features+detections 合并后完全依赖 descriptor OT，失败）
   ```bash
   python tools/run_calibration.py --config configs/pipeline_features.yaml --print
   # matching.strategy = ['descriptor_only'], descriptor_min_similarity=0.7
   ```

3. **回退到 detection-only（带 descriptor）**  
   ```bash
   python tools/run_calibration.py --config configs/pipeline_features.yaml --print
   # matching.strategy = ['core', 'descriptor'], use_features=false, use_detection=true
   ```

### 7.3 结果

| 场景 | 描述 | `success@3m` | `avg_time` | 备注 |
| --- | --- | --- | --- | --- |
| Dense feature + descriptor-only | `use_features=true`, `bev_feature_topk=32→10 after NMS` | 20% | 0.20 s | descriptor 仍将“本地相似”映射成自配对，SVD 得到 identity → TE≈55 m。 |
| detection+feature 混合 | 同上但保留检测框 | 20% | 1.79 s | 稳定度 39，但仍错配。 |
| detection-only + descriptor | `use_features=false` | 20% | 2.62 s | 即便启用 descriptor，少量检测帧依旧没有共享目标，导致 V2X-Reg++ 也失败。 |

结论：descriptor 向量（截取 32 维 BEV embedding）在不同 object 间高度相似（off-diagonal cosine ≈0.96），难以单独区分目标；若没有额外几何先验，求出的姿态会退化到 identity。接下来需要：

1. 让 descriptor 与几何共同作用：在 BoxesMatch 里沿用纯几何候选，但依据 descriptor score 调整 KP 权重（已支持）。关键是确保输入的 detection 实际存在 overlap，否则任何方法都会失败。  
2. 将 descriptor 采样改成 “按检测框范围统计 feature map patch” 或 “多尺度 pooling”，增加区分度。  
3. 准备更大规模的样本（至少数百帧）评估 descriptor 对 V2X-Reg++ baseline 的实际增益；当前 5 帧 sample 本身 overlap 极低，无法代表真实性能。  
4. 在 detection pipeline 内引入 descriptor-aware filtering / matching（例如 per-object OT + Weighted SVD），并用完整 DAIR 测试集与论文 baseline 对比。
