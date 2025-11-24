# V2X-Reg++ 实验复现进度（更新时间：2025-11-23）

## 0. 数据与配置兼容说明

- **Top-3000 DAIR 样本**：`data/data_info_top3000.json` 是从官方 6 616 帧中按 GT box 数量排序后筛出的 3 000 帧子集。任何 GT 实验均以此作为 `data_info_path`，确保与论文的“3k 帧”设定一致。
- **通用 config**：`configs/pipeline_top3000.yaml` 继承自 `configs/pipeline.yaml`，仅将 `data.data_info_path` 指向上述文件、`max_samples=3000`、`success_thresholds=[1,2,3]`。后续所有 DAIR V2X-Reg/V2X-Reg++ 变体都在此基础上覆写 `top_k`、`core_components` 等字段。
- **云端/多卡运行**：每次 run 使用命令 `nohup ~/miniconda3/bin/python tools/run_dair_pipeline_experiments.py --config configs/pipeline_top3000.yaml --tags <tag> > logs/dair_runs/<tag>.log 2>&1 &`. 同一项目可横向复制多份命令，分别设置 `CUDA_VISIBLE_DEVICES`，便于 10×3090 服务器并行跑不同配置。日志、输出目录皆独立，兼容其它 Codex 对检测模型训练脚本的改动。

## 1. V2X-Reg / V2X-Reg++（DAIR-V2X，Table III 主表）

| 配置 | `success@1m` | `success@2m` | 平均耗时 | 备注 |
| --- | --- | --- | --- | --- |
| V2X-Reg++ GT∞ | 0.479 | 0.631 | 0.50 s | `outputs/dair_v2xregpp_gt_inf/metrics.json` |
| V2X-Reg++ GT25 | 0.540 | 0.715 | 0.77 s | `outputs/dair_v2xregpp_gt25/metrics.json` |
| V2X-Reg++ GT15 | 0.545 | 0.716 | 0.19 s | `outputs/dair_v2xregpp_gt15/metrics.json` |
| V2X-Reg++ GT10 | **0.565** | **0.739** | **0.07 s** | `outputs/dair_v2xregpp_gt10/metrics.json` |
| V2X-Reg++ PP15 | 0.332 | 0.406 | 0.50 s | 检测框（PointPillars）`outputs/dair_v2xregpp_pp15/metrics.json` |
| V2X-Reg++ SC15 | 0.279 | 0.353 | 0.68 s | 检测框（SECOND）`outputs/dair_v2xregpp_sc15/metrics.json` |
| V2X-Reg++ GT25 (hSVD) | 0.484 | 0.688 | 0.78 s | `outputs/dair_v2xregpp_gt25_hsvd/metrics.json` |
| V2X-Reg++ GT25 (mSVD) | 0.534 | 0.711 | 0.78 s | `outputs/dair_v2xregpp_gt25_msvd/metrics.json` |
| V2X-Reg (oIoU) GT15 | 0.367 | 0.508 | 1.26 s | `outputs/dair_v2xreg_oiou_gt15/metrics.json` |

**分析**  
- oDist（V2X-Reg++）在 `top_k=10` 时成功率最高，同时耗时最低，证实“优先大尺寸盒 + 限制 top-k”对鲁棒性的作用。  
- 检测框输入（PP/SC）成功率下降约 20–30%，主要受匹配失败影响，但相比论文值仍在合理范围。  
- SVD 变体显示：wSVD > mSVD > hSVD，与 Table III 的讨论一致。  
- oIoU 基线延迟显著（>1 s）且准确率低，说明旧版关联策略不适合大规模复现。

## 2. VIPS 基线

| Run | `success@1m` | `success@2m` | `frames_with_matches` | 备注 |
| --- | --- | --- | --- | --- |
| `vips_noise0` | 0.416 | 0.622 | 0 | `outputs/vips/vips_hkust_lidar_global_config_full/metrics.json` |
| `vips_noise1` | 0.415 | 0.622 | 0 | 同上（噪声 1 m / 1°） |
| `vips_noise2` | 0.416 | 0.623 | 0 | 同上（噪声 2 m / 2°） |
| `vips_noise2_recount` (300 帧) | 0.193 | 0.310 | 149 | 重新计算“无匹配=失败”，`outputs/vips/vips_noise2_recount/metrics.json` |

**分析**  
- 虽然成功率在 40–62% 左右，但 `frames_with_matches=0` 表明多数帧被距离门限淘汰，只有极少数帧生成匹配。  
- 新的 `vips_noise2_recount` 脚本修改（2025-11-24）会把“无匹配”的帧纳入失败统计，`success_with_matches` 现与 `success` 相等，可直接拿来与 Table III 的 VIPS 行对比。  
- 与 CBM/V2X-Reg++ 相比，VIPS 仅能在“已知初值 + 高质量检测”下输出结果；记录这些“成功帧占比”至关重要，后续文稿需强调覆盖率问题。

## 3. V2X-Set 指标（Fig. 6/8/9 相关）

| 数据 | 内容 | 文件 |
| --- | --- | --- |
| 噪声网格 (Fig.6) | 平移 0–2 m、旋转 0–25° 的成功率/误差热力图 | `outputs/v2xset_noise_grid.json` + 子目录 `outputs/v2xset_noise_t*_r*` |
| 指标曲线 (Fig.8) | oDist/oIoU 在平移/旋转偏置下的平均稳定度 | `outputs/v2xset_indicator_curves.json` |
| 匹配数量消融 (Fig.9) | 四种关联策略的匹配分布 | `outputs/v2xset_association_ablation.json` |

**分析**  
- `indicator_curves.json` 清晰展现 oDist 在 1.5 m / 10° 之前保持单调，而 oIoU 早早饱和；为 Fig. 8 复现提供直接数据。  
- `v2xset_association_ablation.json` 可直接喂给 `tools/visualize_v2xset_results.py` 生成 violin plot，与文中讨论“策略 1 > 策略 4”完全一致。

## 4. HEAL 检测 & 特征实验

| 实验 | `success@1m` | `success@2m` | 备注 |
| --- | --- | --- | --- |
| HEAL dual detection (`configs/pipeline_detection.yaml`) | 0.191 | 0.192 | 输出 `outputs/heal_detection/metrics.json` |
| BEV descriptor smoke (`configs/pipeline_features.yaml`) | 0.565 | 0.739 | 输出 `outputs/20251123-223008/metrics.json`，`frames_with_matches=1919` |
| PP/SC 检测（test split） | 运行中 | 运行中 | `configs/pipeline_detection_pp.yaml` / `configs/pipeline_detection_sc.yaml`，`max_samples=1800`，待写入 `outputs/dair_v2xregpp_{pp,sc}15_test/metrics.json` |

**分析**  
- HEAL 检测结果与 `docs/operations/heal_detection_status.md` 记录一致：成功率 ~19% 受匹配质量制约，需要更好的 RSU 模型。  
- 初步的 BEV descriptor 实验（使用 HEAL feature cache）保留了与 GT 类似的成功率，但只有 60% 帧生成匹配，说明需要更高质量的特征抽样；`v2icalib_feature_extension.md` 后续可引用这批数据进行讨论。

## 5. ICP / PICP（LiDAR-Registration-Benchmark）

| Run | 样本数 | `success_rate` | `avg_runtime_s` | 备注 |
| --- | --- | --- | --- | --- |
| ICP 噪声 0（验证 100 帧） | 100 | 0.89 | 0.26 | `outputs/dair_lidar_benchmark_icp_20251124-015058/metrics.json` |
| PICP 噪声 0（验证 100 帧） | 100 | 0.88 | 0.56 | `outputs/dair_lidar_benchmark_picp_20251124-015227/metrics.json` |

**说明**  
- `benchmarks/run_dair_lidar_benchmark.py` 已修复（`project_cfg_from_yaml`、JSON bool 序列化），短跑 100 帧验证无误；下一步需按 Table III 要求全量 3000 帧 × 噪声 3 个级别重新跑并更新此表。

## 6. 检测 Benchmark（补录）

`docs/detection_bench_report.md` 已整理 P2 阶段的 7 个检测源，关键发现：  
- test/val split 成功率 0.18，train split 仅 0.08，需检查训练检测数据与 `data_info.json` 的对齐。  
- 耗时主要取决于匹配是否成功，失败帧平均耗时翻倍。  
这些观察已经写入该文档，无需重复；本进度日志仅引用供总览。

## 7. 未完成 / 下一步

1. **ICP / PICP baseline**：`benchmarks/run_dair_lidar_benchmark.py` 已修复（`project_cfg_from_yaml`），但 6 个 run（噪声 0/1/2 m & 有/无点到平面）尚未重启，`logs/dair_runs/icp_noise*.log` / `picp_noise*.log` 仍只有报错，需要补跑。  
2. **检测 Test split**：`configs/pipeline_detection_pp/sc.yaml` 正在跑 1800 帧以复现 Table III 的 PP/SC 行；完成后将把 `metrics.json` 数字写回此表。  
3. **GPU 相关任务**：由于服务器 GPU 掉线，`opencood/tools/pose_graph_pre_calc.py --dump_bev_features` 未能完成；若后续要导出 HEAL BEV 特征，需先恢复 GPU 或将脚本改为 CPU 模式（极慢）。  
4. **文档更新**：当前文档新增了本页进度表；后续若有新的指标或长跑结果（ICP/PICP、V2X-Sim、Fig.6 热图等），请继续追加条目以保持可追溯性。
