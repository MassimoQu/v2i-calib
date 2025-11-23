# 项目代码与文档清单（清理前评估）

本文档盘点了当前 `v2i-calib` 仓库中所有顶层文件 / 目录以及核心子模块，解释各自的用途、依赖关系与删除风险，方便后续决策是否保留。若无特殊说明，路径均相对于仓库根目录。

## 顶层文件与目录

| 路径 | 类型 | 主要内容 / 用途 | 依赖关系与引用 | 删除影响 / 备注 |
| --- | --- | --- | --- | --- |
| `README.md` | Markdown | 项目介绍、论文链接、可视化、实验表格。 | 对外宣传、GitHub 首页、文档引用图片 `static/images`、`static/visuals/thumbnail_*.png`。 | **核心**：删掉会失去对外说明。 |
| `benchmarks/` | 目录 | 包含 `hkust_lidar_global_registration_benchmark.py`、`run_dair_lidar_benchmark.py` 及 `third_party/LiDAR-Registration-Benchmark` 子模块。 | 依赖 `configs/hkust_lidar_global_config.yaml`、`legacy.v2x_calib.*`。 | 保留；用于对接 HKUST 基准。 |
| `calib/` | 包 | 新版模块化 pipeline（数据、filters、matching、solvers、pipelines 等）。 | `configs/pipeline*.yaml`、`tools/run_calibration.py`、CI 未来引用。 | **核心代码**；删除将失去重构目标实现。 |
| `config/` | 包 | 兼容层（将 `config.config` 指向 `configs.legacy_api`）。 | 仅当旧路径不可避免时使用。 | 建议逐步淘汰，首选 `configs.*`。 |
| `configs/` | 配置 | 新 pipeline 的 YAML（`pipeline.yaml`, `pipeline_detection.yaml`）。 | `tools/run_calibration.py`、`calib/pipelines`。 | 如改用其他配置方式可整合；否则需保留。 |
| `data/` | 数据 | 本地 DAIR-V2X / V2X-Sim 样本、检测缓存。 | 几乎所有 reader（`v2x_calib.reader`、`calib.data`）需要。 | **必要**。删除需另行存储数据。 |
| `docs/` | 文档 | 细分为 `architecture/`（如 `refactor_plan.md`）与 `operations/`（如本文件）。 | 产品规划参考、代码清理依据。 | 文档性，可视需要保留。 |
| `.git/` | Git 元数据 | Git 历史。 | 版本控制。 | 不能删除。 |
| `.gitignore` | 配置 | 忽略规则。 | Git；防止临时文件入库。 | 删除会导致噪声文件被跟踪。 |
| `.gitmodules` | 配置 | 记录 `LiDAR-Registration-Benchmark` 子模块 URL。 | `git submodule` 命令。 | 不保留会导致子模块信息丢失。 |
| `.vscode/` | IDE | VS Code 工作区配置。 | 本地开发辅助。 | 不影响运行，可按需删除。 |
| `__init__.py` | Python | 空文件，让仓库可作为包导入。 | 一些脚本 `sys.path.append` 后 import 根包。 | 删除可能导致 `import v2i_calib` 失败。 |
| `benchmarks/third_party/LiDAR-Registration-Benchmark/` | Git 子模块 | HKUST LiDAR 注册基准的源码（benchmarks、misc、examples 等）。 | `benchmarks/run_dair_lidar_benchmark.py`、后续对比实验。 | 删除将无法复现 HKUST 基准；若不需要可移除子模块。 |
| `outputs/` | 输出 | 运行 pipeline 时的 `metrics.json`、`matches.jsonl`、benchmark 结果等。 | `tools/analyze_matches.py`、结果复查。 | 可清理旧实验；保留结构。 |
| `outputs/logs/` | 输出 | `Logger` 的历史日志，已归档至 `outputs/` 下。 | 旧 CLI、基准脚本。 | 仅保留必要日志，可定期清理。 |
| `static/` | 静态资源 | README 使用的图片（workflow）。 | README、文档。 | 删除会导致 README 图片裂开。 |
| `tools/` | 工具 | 运行/分析脚本：`run_calibration.py`（调用新 pipeline）、`run_detection_bench.py`、`analyze_matches.py`、`profile_report.py`。 | 日常实验 & 生成 README 表格。 | 建议保留；可逐一评估。 |
| `v2x_calib/` | 包 | 兼容层，指向 `legacy/v2x_calib`（reader、preprocess、matching、search、utils）。 | `benchmarks/*`、部分工具依赖其中的核心逻辑。 | **迁移完成前不要删除**。 |
| `visualize/` | 包 | 兼容层，指向 `legacy/visualize`（Open3D / matplotlib 可视化脚本）。 | 原用于 `test.py` 调试。 | 若不再可视化可移除，但会失去调试工具。 |
| `static/visuals/` | 媒资 | README 视频、缩略图。 | README。 | 删除会导致 README 链接失效。 |
| `legacy/` | 归档 | 存放 `config.yaml`、`analysis_results.csv`、`selected_pairs_*.json`、`setup.sh`、`v2x_calib/`、`visualize/` 等历史资产。 | 仅保留以便复现旧实验。 | 删除前请确认不再需要旧数据/脚本。 |

## 关键子目录深入说明

### `calib/`

- `config.py`：集中读取 `configs/pipeline*.yaml`，提供 dataclass 样式配置对象。
- `data/`：`dataset_manager.py`、`sample_loader` 等新式数据读写封装，逐步替换 `v2x_calib.reader`。
- `filters/`：数据筛选逻辑（置信度、Top-K、距离等），对应重构方案第 2 节。
- `matching/`：`match_engine.py`, `similarity.py` 等组件，封装 oIoU/oDist。
- `solvers/`：SVD、RANSAC 等外参求解器。
- `pipelines/`：高阶 orchestration（如 `pipeline.py`）直接驱动完整流程。

> 删除风险：这是“重构版”实现，直接关系到 README 中宣传的 pipeline，强烈建议保留并继续合并旧逻辑。

### `config/`

- `config.py`：兼容层，重定向到 `legacy/config.py` 以支持旧脚本 (`cfg`, `Logger` 等)。
- `hkust_lidar_global_config.yaml`：HKUST benchmark 运行用到的路径 + FPFH 超参。

> 新代码推荐使用 `calib.config.*`；若完全停用 HKUST 对比，可考虑删除该兼容层。

### `configs/`

与新 `calib` pipeline 强绑定：包含运行数据 split、检测缓存开关、匹配策略、输出目录等。若未来转到 CLI 参数，可考虑生成/导入，但现阶段是唯一配置来源。

### `legacy/`

- `config.yaml`、`analysis_results.csv`、`selected_pairs_olddatarange.json`：早期实验输出。
- `legacy/setup.sh`：旧依赖安装脚本。
- `v2x_calib/`、`visualize/`：原始实现与可视化脚本，供兼容层调用。

> 若明确不再需要旧实验，可清理该目录；当前将其视为“冷备”。

### `benchmarks/`

- `hkust_lidar_global_registration_benchmark.py`：直接在 DAIR/V2X-Sim 上调用 FPFH+TEASER。
    - `run_dair_lidar_benchmark.py`：新脚本，读取 DAIR 数据并对 LiDAR-Registration-Benchmark 子模块进行评测，输出 `outputs/dair_lidar_benchmark_*`.

> 若后续聚焦 V2X-Reg++ 主线，可考虑拆分到单独 `benchmarks/` 目录或存档，但在未跑完 HKUST 对比前建议保留。

### `tools/`

- `run_calibration.py`：封装 `calib/pipelines`，批量运行配置（含 argparse）。
- `run_detection_bench.py`：类似 CLI，但针对检测版配置。
- `analyze_matches.py`：读取 `outputs/*/matches.jsonl` 做统计/CSV 导出。
- `log_analyze.py`：原 `Log/analyze.py`，用于批量日志对比。
- `profile_report.py`：性能分析工具。

> 若后续整合到统一 CLI，可逐步淘汰单用途脚本，但现在它们构成操作入口。

### `v2x_calib/`

历史版本核心：包括
- `reader/`（`CooperativeReader`, `CooperativeBatchingReader`, `V2XSim_Reader` 等）；
- `preprocess/Filter3dBoxes.py`；
- `corresponding`, `search`, `utils`.

历史 CLI（旧版 `test.py`）和当前 `benchmarks`、`calib` 逻辑都依赖这里实现 oIoU/oDist、SVD、点云读写，也是新 pipeline 背后的数据处理实现。彻底迁移前无法移除。

## 已移除的遗留脚本

- `test.py`：早期的批处理入口，已被 `tools/run_calibration.py` / `calib.pipelines` 取代。
- `batchtest.py`：围绕 `test.py` 的参数扫描脚本，随同移除。
- `benchmarks/test.py`、`benchmarks/initial_value_method_*.py`、`benchmarks/ransac.py`：旧的点云初值实验脚本；保留 HKUST 基准后已清理。

### `visualize/` 与 `static/visuals/`

提供了点云/框体可视化脚本以及 README 中的 demo 视频。对开发和宣传都仍有价值；只有在完全不再需要可视化或媒资时才考虑删。

## 其他说明

- `outputs/logs/`、`outputs/` 等运行产物建议设定保留策略（如仅保留最近 N 次），但不要直接删除目录。
- `data/` 与 `benchmarks/third_party/LiDAR-Registration-Benchmark/` 体积较大；若需减小仓库，可将数据移出仓库，通过符号链接或配置指向外部路径。
- `legacy/setup.sh` 目前未覆盖所有依赖（例如 `teaserpp_python`, `open3d>=0.17`），如果保留建议同步更新。

如需进一步细化（例如 `calib/matching` 内各脚本的必要性），可在本文件基础上添加子章节或表格。
