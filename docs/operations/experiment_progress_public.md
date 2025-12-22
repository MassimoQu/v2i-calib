# V2X-Reg++ Public Reproduction Notes

This page is the **public-facing** experiment entrypoint (minimal + reproducible).

## 1. DAIR-V2X Table III (GT sweeps)

The paper’s GT sweeps can be run using the Top-3000 subset config:
- Base config: `configs/pipeline_top3000.yaml`
- Subset list: `data/data_info_top3000.json`

Run all GT experiments (V2X-Reg++ / V2X-Reg and SVD variants):
```bash
python tools/run_dair_pipeline_experiments.py --config configs/pipeline_top3000.yaml
```

Or run a single config:
```bash
python tools/run_calibration.py --config configs/pipeline_top3000.yaml --print
```

All outputs are written to `outputs/<tag>/` with:
- `metrics.json`: aggregated success/mRE/mTE + avg runtime
- `matches.jsonl`: per-frame RE/TE, timing and match details

## 2. Data layout expectation

The pipeline expects the official DAIR-V2X cooperative split under:
- `data/DAIR-V2X/cooperative-vehicle-infrastructure/`

If you keep data elsewhere, either symlink it into place or edit `data.data_root` / `data.data_info_path` in the YAML config. See `docs/operations/experiment_reproduction.md`.
