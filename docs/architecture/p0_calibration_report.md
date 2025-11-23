# P0 阶段：HEAL 检测 → V2X-Reg++ 离线融合验证

本报告记录了“导入 HEAL 检测结果，离线运行 V2X-Reg++ 标定流水线，并输出精度对比”的 P0 打通过程。V2X-Reg++ 默认使用距离（oDist）关联检测框；若需要回滚到旧版 V2I-Calib（oIoU 关联），可在 `configs/pipeline*.yaml` 中切换 `matching.strategy`。验证数据为 DAIR-V2X 测试集，硬件为单机 CPU（RTX GPU 未使用）。

## 1. 数据准备与工具

1. **导出 HEAL 检测框**  
   - 运行 `opencood/tools/pose_graph_pre_calc.py` 等脚本得到 `stage1_boxes.json`。文件结构中包含 `pred_corner3d_np_list`、`uncertainty_np_list`、`lidar_pose_clean_np` 等字段。
   - 使用更新后的脚本 `tools/heal_stage1_to_detection_cache.py` 将框重新“反投影”到各自传感器的本地坐标系，再按 `DetectionAdapter` 预期顺序写回缓存。例如：
     ```bash
     python tools/heal_stage1_to_detection_cache.py \
       --stage1 /path/to/HEAL/logs/test/stage1_boxes.json \
       --output data/DAIR-V2X/detected/detected_boxes_test.json \
       --swap-order --require-two-cavs --deproject-to-local
     ```
     `--deproject-to-local` 使用 `lidar_pose_clean_np` 先把所有 corner 从 “ego/vehicle” 坐标系转换回各自传感器坐标；`--swap-order` 用于把路端放在列表首位，满足 `DetectionAdapter` 的 `infra/vehicle` 假设。

2. **运行标定流水线**  
   - 使用 `tools/run_calibration.py --config configs/pipeline.yaml` 获取 GT 基准。
   - 使用 `tools/run_calibration.py --config configs/pipeline_detection.yaml` 评估 HEAL 检测（`use_detection: true`）。
   - 输出位于 `outputs/<tag>/metrics.json` 与 `matches.jsonl`，默认包含 RE/TE、运行时间、匹配日志等。

## 2. 实验结果

| 输入来源 | 样本数 | mRE@1m | mTE@1m | Success@1m | Success@2m | 平均耗时 (s/frame) | 全局 RE 均值 (°) | 全局 TE 均值 (m) | RE 中位 (°) | TE 中位 (m) | 平均匹配数 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GT 框 (`configs/pipeline.yaml`, `outputs/20251122-024208`) | 3000 | 0.378 | 0.215 | 0.565 | 0.739 | 0.061 | 13.58 | 12.52 | 0.99 | 0.73 | 1.71 |
| HEAL 检测（反投影后；`configs/pipeline_detection.yaml`, `outputs/detection`） | 500 | 0.000* | 0.000* | 0.180† | 0.180† | 0.129 | 78.28‡ | 75.33‡ | 86.05‡ | 74.98‡ | 1.64‡ |

\* `mRE@1m` 与 `mTE@1m` 仅统计 TE < 1m 的样本；在检测数据中这部分样本全部来自“无匹配 → 默认 0” 的 fallback，因此指标为 0。  
† `success_at_{1,2}m` 同样受 fallback 影响，不能反映真实成功率，需参考“全局均值/中位值”。  
‡ 只统计有匹配结果的 410 帧（其余 90 帧无匹配，RE/TE 皆 0）。

补充统计（来自 `matches.jsonl` 与离线分析脚本）：

- **GT**：平均稳定度 1.78，TE 中位数 0.73m，长尾由难场景造成；匹配数量多为 1～2 个即可得到稳定解。
- **HEAL 检测**：只在 410/500 帧里找到匹配，TE 中位值 74.98m；90 帧完全无匹配。对前 200 帧进一步分析发现：
  - 使用真实外参对齐的情况下，1 m 阈值内只有 **1 帧** 存在公共检测框；5 m 阈值下也只有 **69/200** 帧；
  - 路端检测到的每个目标在车端最近检测的中心距离中位值 **32.6 m**（10% 分位仍有 8.5 m）。
  - 说明当前检测数据几乎没有可用的共同目标，无法满足 V2X-Reg++ 的基础假设。

## 3. 结果分析

1. **打通程度**：通过“反投影 → DetectionAdapter → ObjectLevelPipeline”的流程，现已确认 HEAL 导出的 `stage1_boxes.json` 可转换回本地坐标并供标定流水线消费。
2. **检测重叠性极低**：在转换后的数据上，即便使用真实外参对齐两个检测框集合，1 m 半径内存在公共目标的帧<1%，即便放宽到 5 m 也只有 ~34%（200 帧样本中 69 帧）。多数场景中，路端/车端检测到的对象集合基本互不重叠，导致匹配/求解不可行。
3. **算法输出失真**：当前 `ExtrinsicSolver` 在无匹配时返回 `[0]*6`，聚合统计会把 90 帧“无解”样本计为成功，掩盖了失败比例。P1 需在 `matches==[]` 时显式标记失败，或保留上一帧外参。
4. **运行时间**：GT 模式单帧约 61 ms；检测模式因匹配失败触发多次降级，约 129 ms，仍满足离线分析需求。

## 4. 下一步建议

1. **指标修正**：为 `aggregate_metrics` 添加“无解 → 计为失败”逻辑，并输出 `success_at_{thr}` 的真实值。
2. **质量门控**：引入稳定度 / 匹配数量阈值，当检测质量过低时直接回退到上一帧可信外参，避免无意义的 0 结果。
3. **检测链路排查**：
   - 若要进行真实校准评估，需让 HEAL 输出“尚未对齐”的检测框，或在导出时标记更多元信息（当前检测多数由 stage1 投影得来，重叠率极低）。
   - 对 `matches.jsonl` 中有匹配/无匹配的帧做可视化，核实是模型召回不足、阈值过严，还是场景本身缺少共视区域。
4. **自动化报告**：将上表写入 CI 报告（CSV/Markdown），并把 `tools/run_calibration.py` 的输出路径作为 artifacts，方便 A/B 对比。

P0 阶段完成后，可以开始 P1：修复指标、改进稳定度门控，并探索在 HEAL 侧加噪声/前处理以提升匹配可靠性。
