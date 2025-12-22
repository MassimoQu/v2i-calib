# V2I-Calib / V2X-Reg++: Object-Level Online Registration for V2I/V2X

<p align="center">
  <a href="https://ieeexplore.ieee.org/abstract/document/10802098"><img src="https://img.shields.io/badge/IROS24-Paper-blue?logo=paper" alt="IROS 2024"/></a>
  <a href="https://arxiv.org/abs/2410.11008"><img src="https://img.shields.io/badge/TITS-arXiv_2410.11008-b31b1b?logo=arxiv" alt="arXiv"/></a>
</p>

<p align="center">
  <img src="./static/images/V2I-CALIB++_workflow_v2.png" alt="V2I-Calib / V2X-Reg++ workflow" width="88%">
</p>

## Introduction

This repository provides an **object-level, online registration / calibration solver** for vehicle–infrastructure (V2I) and multi-terminal sensing systems, using **3D bounding boxes** as geometric primitives.

Two variants are included:
- **V2I-Calib**: oIoU-based association (IROS 2024).
- **V2X-Reg++**: distance-based association (IEEE T-ITS).

## Installation

```bash
conda create -n v2xreg python=3.10 -y
conda activate v2xreg
pip install -r requirements.txt
```

Optional (only needed for baselines under `benchmarks/`):
```bash
git submodule update --init --recursive
pip install open3d==0.17.* torch==2.3.*
```

## Data Preparation (DAIR-V2X)

1) Download the official **DAIR-V2X cooperative** split.

2) Place it at (recommended):
- `data/DAIR-V2X/cooperative-vehicle-infrastructure/`

If you keep the dataset elsewhere, create a symlink:
```bash
ln -s /path/to/cooperative-vehicle-infrastructure data/DAIR-V2X/cooperative-vehicle-infrastructure
```

Minimal files used by the default **object-level pipeline** (GT 3D boxes from labels):
```text
cooperative-vehicle-infrastructure/
  cooperative/
    data_info.json
    calib/lidar_i2v/{vehicle_frame_id}.json
  infrastructure-side/label/virtuallidar/{infrastructure_frame_id}.json
  vehicle-side/label/lidar/{vehicle_frame_id}.json
```

Notes:
- The ground-truth transform is loaded from `cooperative/calib/lidar_i2v/*.json` (no extra post-processing in the public pipeline).
- Images/point clouds are not required for the default object-level evaluation, but are needed by some baselines in `benchmarks/`.

## Run (DAIR-V2X)

- Single run on the official test split (GT boxes):
  ```bash
  python tools/run_calibration.py --config configs/pipeline.yaml --print
  ```

- Table III GT sweeps (Top-3000 subset):
  ```bash
  python tools/run_dair_pipeline_experiments.py --config configs/pipeline_top3000.yaml
  ```

Outputs are written to `outputs/<tag>/`:
- `metrics.json`: aggregated metrics + average runtime
- `matches.jsonl`: per-pair RE/TE, timing and match details

## Documentation

- Minimal entrypoint: `docs/operations/experiment_progress_public.md`
- Reproduction guide: `docs/operations/experiment_reproduction.md`

## Citation

If you find our work or this repo useful, please cite:
```bibtex
@inproceedings{qu2024v2i,
  title={V2I-Calib: A novel calibration approach for collaborative vehicle and infrastructure lidar systems},
  author={Qu, Qianxin and Xiong, Yijin and Zhang, Guipeng and Wu, Xin and Gao, Xiaohan and Gao, Xin and Li, Hanyu and Guo, Shichun and Zhang, Guoying},
  booktitle={2024 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  pages={892--897},
  year={2024},
  organization={IEEE}
}

@article{zhang2025v2x,
  title={V2X-Reg++: A Real-Time Global Registration Method for Multi-End Sensing System in Urban Intersections},
  author={Zhang, Xinyu and Qu, Qianxin and Xiong, Yijin and Xia, Chen and Song, Ziqiang and Peng, Qian and Liu, Kang and Li, Jun and Li, Keqiang},
  journal={IEEE Transactions on Intelligent Transportation Systems},
  year={2025},
  publisher={IEEE}
}
```

## Acknowledgements

This project is built upon the following codebases:
- [DAIR-V2X](https://github.com/AIR-THU/DAIR-V2X)
- [LiDAR-Registration-Benchmark](https://github.com/HKUST-Aerial-Robotics/LiDAR-Registration-Benchmark)
