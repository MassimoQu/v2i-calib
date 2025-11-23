# DAIR-V2X 检测框 Benchmark 汇总（P2 阶段）

日期：2025-11-22  
配置：`configs/pipeline_detection.yaml`（use_detection=true, max_samples=1000, data_info=data/DAIR-V2X/cooperative/.../data_info.json，过滤/匹配与 GT 版本一致），输出脚本：`tools/run_detection_bench.py`

| 检测源 | Success@1m | Success@2m | success_frames | mRE@1m | mTE@1m | avg_time (s) | 输出目录 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| dairv2x-second_uncertainty/test/stage1_boxes.json | 0.18 | 0.18 | 180 | 0.0° | 0.0 m | 0.153 | outputs/det-dairv2x-second_uncertainty-test-stage1_boxes-* |
| dairv2x-second_uncertainty/train/stage1_boxes.json | 0.083 | 0.084 | 83 | 0.0° | 0.0 m | 0.213 | outputs/det-dairv2x-second_uncertainty-train-stage1_boxes-* |
| dairv2x-second_uncertainty/val/stage1_boxes.json | 0.18 | 0.18 | 180 | 0.0° | 0.0 m | 0.152 | outputs/det-dairv2x-second_uncertainty-val-stage1_boxes-* |
| detected_boxes_test.json | 0.18 | 0.18 | 180 | 0.0° | 0.0 m | 0.152 | outputs/det-detected_boxes_test-* |
| detected_boxes_train.json | 0.083 | 0.084 | 83 | 0.0° | 0.0 m | 0.215 | outputs/det-detected_boxes_train-* |
| detected_boxes_val.json | 0.18 | 0.18 | 180 | 0.0° | 0.0 m | 0.151 | outputs/det-detected_boxes_val-* |
| given_boxes_val.json | 0.138 | 0.138 | 138 | 0.0° | 0.0 m | 0.144 | outputs/det-given_boxes_val-* |

> 注：每个检测源仅跑前 1000 个样本。`mRE@1m`/`mTE@1m` 为成功帧（TE<1 m）的平均值；由于检测框匹配后大多返回接近零的变换，这里数值接近 0。完整原始数据见 `outputs/detection_bench_summary.json` 和各自的 `matches.jsonl`。

## 结果分析

1. **检测质量差异**  
   - val/test 集合（`detected_boxes_{val,test}`、`second_uncertainty` 的 val/test）成功率均为 0.18，说明这些检测源的几何质量相近。  
   - train 集合成功率骤降至 ~0.08，意味着该目录的检测框与当前 `data_info.json` 的 ID/分布存在较大差异（可能是旧版本或未对齐帧）。  
   - `given_boxes_val.json` 介于两者之间（0.138），符合“高质量 baseline”预期。

2. **耗时特征**  
   - 匹配阶段主导了总耗时（`tools/profile_report.py` 对 `outputs/det-*/matches.jsonl` 的统计显示，成功帧平均匹配 0.065 s，失败帧 0.142 s）。  
   - train 集合耗时 >0.21 s/帧，主要因为匹配失败占比高，需要更多尝试/回退。

3. **下一步建议**  
   - 对 train 检测数据检查 ID 对齐和置信度，必要时过滤低置信框。  
   - 针对零匹配场景考虑 fallback/回退策略，否则整体成功率会被大量空帧稀释。  
   - 使用提供的 `matches.jsonl` + `tools/analyze_matches.py` 可快速比较其他阈值（1–5 m），检查是否在更宽松条件下有明显改善。
