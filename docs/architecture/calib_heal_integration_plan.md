# V2X-Reg++ 与 HEAL 协同感知融合规划

本文档给出将 V2X-Reg++ 外参标定流程融入 HEAL 协同感知框架的整体思路、关键接口以及阶段性计划，旨在让两套系统在保证标定准确率的同时，不破坏 HEAL 现有的多模态中融合能力（若需复现旧版 V2I-Calib，可在关联模块切换回 oIoU）。

## 1. 目标与基本假设

1. **实时性需求**：在 HEAL 推理循环中周期性获取最新车-路外参，延迟控制在单帧推理预算内（或可异步滞后一帧）。
2. **数据源**：标定以 HEAL 模型推理得到的 3D 检测框（corner 坐标）为主要输入，可选接入真值框或历史缓存。
3. **输出形式**：返回 6DOF 姿态矩阵 + 稳定度评分，并告知数据来源（GT / 检测）和匹配摘要，便于协同感知端做置信度判决。
4. **多 Agent 支持**：至少覆盖 1 RSU + 1 自车的典型 V2I 组合，后续扩展 n-agent → ego 的批处理。

## 2. 整体架构

```
HEAL 推理循环
 ├─ Late/Intermediate Dataset (提取 CAV 数据)
 │    └─ 新增 Detection Exporter  →  Calibration Adapter 队列
 ├─ 模型推理 / NMS
 └─ Pose Updater
      ├─ 从 Calibration Adapter 读取最新外参 + 稳定度
      ├─ 覆盖 transformation_matrix / pairwise_t_matrix
      └─ 记录回退/质控信息

Calibration Adapter
 ├─ 统一 JSON schema（infra_boxes / veh_boxes / poses / meta）
 ├─ 向 V2X-Reg++ Pipeline 提供 Sample（可离线或服务化）
 └─ 将输出写回共享缓存（含时间戳/帧号）
```

关键思想：在 HEAL 数据集类中插入“导出 → 标定 → 回注”闭环，而标定内部仍沿用现有 `DatasetManager → FilterPipeline → MatchingEngine → ExtrinsicSolver` 结构，只需扩展数据来源。

## 3. 数据通路设计

### 3.1 导出检测框
- **插桩位置**：`opencood/data_utils/datasets/late_fusion_dataset.py` 与 `intermediate_fusion_dataset.py` 的 `get_item_test`/`collate_batch`。
- **内容**：每个 agent 的 `pred_corner3d_np_list`、 NMS 前 logits、当前噪声姿态 `lidar_pose`、帧索引等。
- **Schema**：复用 `calib/data/detection_adapter.py` 期望的 `{ pred_corner3d_np_list: [...], infra_id, veh_id }`，必要时附加分数/类别/置信度。
- **实现形式**：
  1. **离线文件**：保存为 JSONL，供标定流程批量读取，先验证精度。
  2. **内存通道**：使用 `multiprocessing.Queue` / Redis / Socket 在推理循环中实时发送，满足在线需求。

### 3.2 标定输入重组
- `CalibrationSample` 需扩展可选字段（传感器类型、多 agent 列表、时间戳）。
- 当 HEAL 场景存在多个协作方时，Adapter 负责拆分为一对多样本或批量样本：`{ego, cav_i}`。
- 支持“上一帧 prior + 当前帧匹配”模式，以减少计算压力（利用 `stability_gate` 逻辑）。

## 4. 标定模块适配

### 4.1 运行模式
1. **离线批处理**：对齐 `configs/pipeline.yaml`，指定 detection cache 路径为 HEAL 导出的 JSON；用于验证精度收益。
2. **在线服务化**：
   - 将 `ObjectLevelPipeline` 封装为长驻进程（可选 FastAPI/gRPC）。
   - 输入：`{sample_id, infra_boxes, veh_boxes, poses}`。
   - 输出：`{sample_id, T6, stability, RE/TE}`，并写入共享缓存供 HEAL 消费。

### 4.2 算法改动
- **多 Agent 扩展**：在 `MatchingEngine` 中支持“1 对 N”匹配，或迭代调用现有逻辑。
- **质量门控**：根据 `stability`、匹配数量、TE 上限决定是否采用结果；失败时返回回退标记。
- **日志**：保持 `matches.jsonl` 与 HEAL 推理日志同步，以便定位问题。

## 5. HEAL 侧回注方案

1. **Late Fusion**：
   - 在 `LateFusionDataset.get_item_test` 中新增 `transformation_matrix_online` 字段。
   - `post_process` 选择在线矩阵进行投影；若在线矩阵缺失或稳定度不足，则回落到噪声姿态或上一帧缓存。
2. **Intermediate Fusion**：
   - 在构造 `pairwise_t_matrix` 前，将 `lidar_pose_list` 替换为标定输出（或做增量更新）。
   - `box_align` 模块可改写为“调用标定服务 + 结果融合”，保留旧逻辑作为 backup。
3. **稳定度策略**：
   - 低稳定度 → 保持旧外参并记录告警。
   - 连续多帧失败 → 触发 fallback（例如使用数据集 GT 或初始标定）。

## 6. 多 Agent、调度与缓存

- **配对策略**：默认以 `ego` 与每个 `cav` 形成独立样本；如需 RSU↔RSU，可在 Adapter 中配置路侧列表。
- **缓存一致性**：通过 `sample_id`（`infra_id + veh_id + timestamp`）对齐导出与回注，避免帧错位。
- **性能**：优先使用异步标定（上一帧结果用于当前帧投影），并为高帧率场景提供 batching（多对样本一起送入 SVD）。

## 7. 验证与运维

1. **A/B 测试**：比较 HEAL 在固定噪声下启用/关闭在线标定的 mAP、延迟。
2. **指标看板**：将标定模块输出的 `RE/TE/stability` 与 HEAL 检测指标联合存档，形成长期趋势。
3. **回退机制**：提供 CLI 或配置项快速切换到“纯 HEAL”/“标定 + HEAL”模式；标定服务异常时自动降级。
4. **部署建议**：标定服务与 HEAL 推理最好运行在同一节点或高速网络，减少 IPC 延迟；若用 GPU 加速，可复用 HEAL 的环境。

## 8. 阶段规划

| 阶段 | 目标 | 关键产出 |
| --- | --- | --- |
| P0 | 离线验证 | HEAL 导出检测 → V2X-Reg++ 批处理 → 生成精度对比报告（见 `docs/architecture/p0_calibration_report.md`） |
| P1 | 在线数据流 | Detection Adapter + Calibration Service MVP，支撑单对车-路实时回注 |
| P2 | 多 Agent / 中融合 | 扩展匹配器，替换 `box_align`，验证多协作节点稳定性 |
| P3 | 性能与鲁棒性 | 异步缓存、批量求解、稳定度策略、回退机制完善 |
| P4 | 交付与文档 | 集成测试、README/操作手册更新、监控告警落地 |

按照以上规划，可以逐步把 V2X-Reg++ 的外参估计能力融入 HEAL，使协同感知模型在噪声环境下保持稳定的空间对齐。下一步建议先完成 P0/P1 的离线验证与接口打通，再逐步优化实时性与多 Agent 支持。
