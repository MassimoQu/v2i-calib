# HEAL 检测与 V2I-Calib++ 集成状态（2025-11-22）

## 为什么要做
- **目标**：让 `v2i-calib++` 能利用 HEAL 的检测结果（车端 vs 路端）获得稳定的外参估计。  
- **痛点**：HEAL 官方提供的 `stage1` 导出把所有 CAV 感知堆在一个 ego 坐标系下，无法满足 V2I-Calib++ 对“双方各自在本地坐标下的检测框”的要求；并且路端（RSU）缺少可直接部署的 checkpoint。

## 目前已完成
1. **工具链改造**  
   - `HEAL/opencood/tools/pose_graph_pre_calc.py` 新增 `--per_agent` 模式，可分别指定车端/路端 hypes & checkpoint、强制 ego、限制通讯距离、自动合并双端 JSON。  
   - `tools/heal_stage1_to_detection_cache.py` 可把 Stage1 导出转换为 V2I-Calib++ 的 detection cache；`configs/pipeline_detection.yaml` 指向该缓存，并设置 `use_detection=true`。
2. **实验设置**  
   - **检测导出**：暂用 HEAL stage1 点云模型在车端/路端点云上分别推理（路端缺少专用模型，因此仍是同一模型复用），`single_agent_comm_range=0`，强制不同 ego。  
   - **检测缓存**：`data/DAIR-V2X/detected/heal_stage1_dual_detection_cache.json`，共 1,789 帧双端检测，平均车端 17.4 个框、路端 11.5 个框。  
   - **标定**：`python tools/run_calibration.py --config configs/pipeline_detection.yaml`，`max_samples=1800`，输出目录 `outputs/heal_detection/`。
3. **结果与分析**  
   - `outputs/heal_detection/metrics.json`：`success@{1,2,3,4,5}m=[0.191, 0.192, 0.192, 0.193, 0.194]`，`mRE@{1…5}m=[0.0041, 0.0088, 0.0109, 0.0244, 0.0355]`，`mTE@{1…5}m=[0.0025, 0.0081, 0.0155, 0.0342, 0.0483]`，`avg_time=0.155s`。  
   - `matches.jsonl`：每帧有效匹配均值 1.62，343/1800 帧无任何匹配；TE/RE 中位数依旧在 60–80 m/deg，说明：  
     1. 虽然双方都有检测框，但匹配质量低，常被几何过滤淘汰。  
     2. 路端检测模型仍是“车端模型的替身”，缺乏真正的 RSU 学习能力。

## 接下来要做什么
1. **训练独立的车端/路端单端检测模型**  
   - 车端：基于 `opencood/hypes_yaml/dairv2x/Single/DAIR_single_m1.yaml` 或现有 stage1 YAML，训练 PointPillars 模型。  
   - 路端：基于 `checkpoints/stage2_and_final_infer/m3_alignto_m1/config.yaml` 或新建 YAML，专门针对 RSU 点云（spconv 2.x 需注意通道数）。  
   - 输出：`logs/veh_single_m1/net_epoch_bestval_atXX.pth`、`logs/rsu_single_m3/net_epoch_bestval_atYY.pth`。
2. **重新导出检测与统计共有框**  
   ```bash
   cd HEAL
   PYTHONPATH=. python opencood/tools/pose_graph_pre_calc.py \
     --per_agent \
     --vehicle_hypes logs/veh_single_m1/config.yaml \
     --vehicle_checkpoint logs/veh_single_m1/net_epoch_bestval_atXX.pth \
     --vehicle_output ../data/DAIR-V2X/detected/veh_single \
     --infra_hypes logs/rsu_single_m3/config.yaml \
     --infra_checkpoint logs/rsu_single_m3/net_epoch_bestval_atYY.pth \
     --infra_output ../data/DAIR-V2X/detected/rsu_single \
     --splits test \
     --merged_output ../data/DAIR-V2X/detected/veh_rsu_dual \
     --single_agent_comm_range 0 \
     --vehicle_force_ego vehicle \
     --infra_force_ego infrastructure
   python tools/heal_stage1_to_detection_cache.py \
     --stage1 data/DAIR-V2X/detected/veh_rsu_dual \
     --output data/DAIR-V2X/detected/veh_rsu_dual_detection_cache.json \
     --require-two-cavs
   ```
   - 统计共有框（可快速检查是否 ≥3/帧）：见 `docs/operations/heal_detection_status.md` 的 Python 片段或直接调用 `matches.jsonl`。
3. **再跑 V2I-Calib++ 并分析**  
   - `python tools/run_calibration.py --config configs/pipeline_detection.yaml`。  
   - 关注成功率是否显著提升；若仍不足，需要从 `configs/pipeline_detection.yaml` 中调整 `filters.top_k`、`matching.distance_thresholds`、`solver.stability_gate` 等参数，并结合 `matches.jsonl` 定位匹配失败原因。

## 交接要点
- 所有脚本/配置已更新到仓库；只要提供新的车端/路端 checkpoint，即可直接复用现有流程。  
- `pose_graph_pre_calc.py --per_agent` 默认会生成 `stage1_boxes.json` 并可自动合并；不再需要手工拆/合。  
- 需要确保新模型在 spconv 2.x 环境可加载（若用旧 checkpoint 需转换权重）。  
- 新实验的重要结果统一放在 `outputs/heal_detection/` 下并更新此文档，便于跟踪进度。  
- 若在训练或导出过程中遇到问题，优先检查：路径是否添加 `PYTHONPATH=.`、数据软链是否存在、`single_agent_comm_range` 是否已置 0。
