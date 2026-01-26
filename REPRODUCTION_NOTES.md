# TITS Paper Table III Reproduction Notes

## Open Source Goal
Reproduce Table 3 results from the TITS paper and open source to https://github.com/MassimoQu/v2i-calib/

## Current Status (2024-01-26)

### Completed
- Core Algorithm: V2I-Calib / V2X-Reg++ implementation
- Table III Sweep Scripts (7 files)
- Configuration Files for paper3737
- Documentation updated

### In Progress
- Experiment Results selection
- Final testing

### Rollback Point
- Git Tag: backup-before-major-update (commit: ef28db3)
- Backup Directory: /home/qqxluca/projects/v2i-calib_work_backup/

## Reproduction Workflow
1. Setup environment: conda activate v2i-calib
2. Prepare data using data_info_dair_paper3737.json
3. Run sweeps: python tools/sweep_cbm_table3.py --noises 0,1,2
4. Generate report: python tools/generate_table3_repro_table.py

Note: All local paths replaced with REPLACE_WITH_LOCAL_PATH placeholders.
