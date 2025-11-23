# V2X-Reg++ 重构总体方案

> 目标：在保留 oIoU/oDist 语义的前提下，重构整体架构，使标定精度、成功率、运行速度达到 README / 论文所述水平，并为后续与协同感知框架融合、引入更多算法打下基础。

## 1. 数据与输入层

| 模块 | 主要职责 | 说明 |
| --- | --- | --- |
| `DatasetManager` | 统一管理 DAIR-V2X / V2X-Sim / 其他协同感知数据集的读取、缓存、划分 | 支持真值框（GT）与检测框（PointPillars/SECOND 等）双入口；提供样本索引、元信息查询接口；缓存 `/mnt/ssd_gw/v2i-calib/data/DAIR-V2X/detected/detected_boxes_test.json` 等检测输出 |
| `SampleLoader` | 根据任务描述加载单帧/批次的 box/点云/外参真值 | 需要可选噪声注入；对接现有 `CooperativeReader` 但抽象为无状态函数；预留“上帧 prior + 扰动”入口 |
| `DetectionAdapter` | 与外部协同感知框架接口 | 当前由协同感知框架（Python）向本文提供检测框；第一版缓存检测结果，第二版计划与 OpenCOOD 感知模块实时联动 |

行动项：
1. 梳理 detected_boxes_test.json 的字段定义（坐标系、类别 ID、置信度等），形成统一 JSON schema。
2. 设计适配层，使 OpenCOOD/其他框架输出可以按 schema 写入缓存或实时输入。

## 2. 预处理与过滤层

### 重写 `FilterPipeline`

- 以配置驱动（YAML/JSON）组合下列策略：
  - 类别/置信度过滤（兼容检测框）。
  - 体积/距离/可见性（支持自适应阈值）。
  - 动态目标过滤（依赖前一帧或速度先验；时间同步误差导致的动态目标偏移需降权）。
- 提供“优先类别/优先尺寸”接口（例如车辆尺寸大的优先，默认优先级：bus/truck > car > others）。
- 统计接口：输出每个策略前后的框数量、类别分布，用于调参。
- 可选：与数据集管理组件集成，生成可视化（histogram/heatmap）。

## 3. 匹配 / 相似度层

| 组件 | 功能 | 备注 |
| --- | --- | --- |
| `MatchEngine` | 管理多种匹配策略（oIoU、oDist、未来扩展） | 面向接口编程：`compute_matches(infra_boxes, veh_boxes, prior_T, sensor_combo) -> MatchSet`，支持点云-点云、图片-图片、图片-点云三类组合 |
| `SimilarityKernel` | 具体实现：oIoU、oDist（center/vertex distance）、可选的新指标 | 本阶段保证 oIoU/oDist 与论文一致；为图像相关匹配实现特定算子（需相机内参）并留插件扩展点 |
| `MatchFilter` | trueRetained/threshold/top/all 的标准实现，修复现有阈值被覆盖的问题 | 支撑“稳定性阈值”“匹配数量阈值”等 |
| `MatchQuality` | 负责稳定性评分、匹配统计、可视化 | 对齐论文中的稳定性定义，同时提供 RE/TE 混合指标接口 |

关键任务：
1. 重新整理 oIoU/oDist 的数学定义，保证与论文一致（需要查原稿/补实验）。
2. 评估“只用 TE”与“RE+TE”策略的影响，保留可配置接口。

约束与扩展：
- 真实项目允许使用上帧外参作为 prior，并在严重扰动时降权或回退。
- 需要在匹配过程中考虑“动态目标 + 时间同步误差”的影响，提供可配置的降权策略。

## 4. 外参求解层

### 现有功能分拆
1. `SvdSolver`：封装加权/非加权 SVD，支持“with match / without match”两种策略。
2. `RobustOptimizer`：对接 RANSAC、NDT、ICP 等细化器（可选），接口统一为 `refine(T_init, data) -> T`.
3. `QualityEvaluator`：计算 RE/TE 及扩展指标（RMSE、协方差、匹配残差）。

行动项：
- 与匹配层解耦，`MatchSet` 中保留权重信息，使 SVD 层无需重新计算。
- 设计面向协同感知的输出（例如 6DOF + 置信度 + 匹配摘要）。
- 为图片参与的场景预留相机模型（内参/畸变）接口，支持图片-图片、图片-点云配准。

## 5. 评价与日志层

### 指标体系
- 必选：`RRE(°)、RTE(m)、SuccessRate@λ(基于 TE)`。
- 可选：`SuccessRate@λ(RE+TE 联合)、稳定性均值/方差、匹配精度召回、时间分解（匹配/求解/总计）`。
- 支持持续集成中的 regression test（与 README 基准比对）。

### 日志与分析
- 统一输出目录（如 `outputs/<timestamp>/`），包含：
  - `matches.jsonl`：逐帧结果。
  - `metrics.json`：汇总指标。
  - `perf.json`：时间统计。
  - `vis/`：可选的可视化数据。
- 日志可重新定义格式，不兼容旧版本；需记录“输入类型（GT/检测）”“传感器组合”“prior 设置”“冗余外参”等元信息。
- 提供新的 `tools/analyze.py`（或整合为 CLI），可生成 CSV/Markdown/可视化图表。

注意事项：
- 需要新老日志不兼容的标记（如在 README 中注明）。

## 6. 性能与工程方案

### Python 端短期优化
- 使用 NumPy/Numba/KDTree（scikit-learn 或自写）减少 Python 层循环。
- 避免 `sys.path` hack，重构为真正的包以便 Cython/pybind11 编译。
- 采用 `multiprocessing.shared_memory` 或 `ray` 保存共享的 bbox 数据，减少 pickle。

### C++/GPU 方案
- 优先级：
  1. 匹配得分计算（尤其是 oDist 的点距）；
  2. KDTree / 最近邻搜索；
  3. 加权 SVD（Eigen/Ceres）。
- 方案：
  - 子模块 `cpp/core`，使用 CMake 构建；
  - pybind11 暴露 Python 接口；
  - 可选：CUDA 实现批量距离计算；
  - 允许“在 C++ 中预加载数据，通过 gRPC/IPC 调用”。
- 性能目标：当前 CPU 环境达到 0.13 s/帧以内；0.01 s/帧为长期目标，需要结合检测规模、C++/GPU pipeline 进一步验证。
- 硬件仅需支持当前 CPU + RTX 3090。

## 7. 协同感知融合计划

1. 在 `DetectionAdapter` 中定义外参输出 schema，与协同感知框架对接。
2. 设计“外参在线更新 + 感知性能验证”实验：外参由本方法输出，评估感知精度（mAP 等）。
3. 若需要实时性，评估在协同感知管线中插入 C++ 外参模块的延迟。
4. 外参输出添加安全冗余（置信度、备份外参/回退策略），供协同感知框架低置信度时选择旧外参。
5. 预留与 OpenCOOD/其他框架的接口（ROS2/HTTP/gRPC 均可），便于下一版本集成。

待确认：
- 协同感知框架的接口/语言栈（ROS2? Apollo? 自研？）。
- 是否需要在线同步多车、多路端外参？

## 8. 实验与 README 更新

1. 制定 Benchmark checklist：
   - DAIR-V2X（GT/检测框）、V2X-Sim（GT/检测框）、协同感知集成场景。
   - 每套配置记录硬件、线程数、帧数、平均/中位时间。
2. 对照 README 的表格，逐项复现/更新数据。
3. 运行 `tools/gen_readme_tables.py` 自动写入 README（减少手工错误）。

## 9. 时间与阶段划分（建议）

| 阶段 | 目标 | 产出 |
| --- | --- | --- |
| P0 | 架构搭建、配置/日志系统就绪 | 新的包结构、配置模板、分析工具草稿 |
| P1 | oIoU / oDist 在 GT 数据上对齐 | DAIR-V2X GT 精度/成功率达到论文水平 |
| P2 | 检测框 + 加速（Python 优化/C++ 原型） | 性能评估报告，定位瓶颈 |
| P3 | C++/GPU 实现，达到目标速度 | C++ 子模块、pybind 接口、CI 脚本 |
| P4 | 协同感知集成、README 更新 | 外参输出接口、协同感知评测结果、新 README |

（实际排期需结合资源调整）

## 10. 待确认 / 风险点

1. 协同感知框架：现阶段为 Python，外参模块作为服务消费检测结果；后续需对接 OpenCOOD。
2. 额外先验：车辆尺寸越大越优先；时间同步暂不处理但需在文档中提示其影响。
3. 硬件：仅需面向当前 CPU + RTX 3090。
4. 性能：0.01 s/帧为长期方向，需结合检测规模、prior 策略实验确认可行性。
5. 安全冗余：外参输出必须携带置信度/回退逻辑。
6. 传感器类型：本项目核心涵盖相机与激光雷达，需为图片-图片、图片-点云、点云-点云配准提供基础接口和示例。

欢迎补充或修正上述计划，确认后我将按照阶段推进并在关键节点产出详细文档/代码。
