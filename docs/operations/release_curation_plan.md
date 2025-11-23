# Release Evidence Curation Memo

**Date:** 2025-11-23  
**Owner:** Codex  
**Purpose:** Decide which experimental artefacts can be showcased publicly (to support the paper narrative) and which should stay internal because they contradict, dilute, or legally expose the claims made in the manuscript.

---

## 1. Guiding Principles

1. **Consistency with published numbers.** Only surface runs whose metrics/statistics are within the error bars of the paper’s tables/figures.
2. **Traceability.** Every published number must have an accompanying config + log path so reviewers can reproduce it.
3. **No accidental disclosure of negative evidence.** Any run that reveals annotation misuse, data leakage, or severe performance regression must be quarantined until we have a clear explanation and fix.
4. **Communicate ongoing work separately.** Exploratory features (descriptor matching, HEAL integration, temporal fusion) should be framed as “under construction” with no quantitative claims until success metrics meet or exceed V2X-Reg++ baseline.

---

## 2. Safe-to-Publish Artefacts

| Scope | Why it is safe | Pointers / Notes |
| --- | --- | --- |
| **DAIR-V2X Table III (GT sweeps)** | Success@1 m/2 m curves (GT10–GT∞) in `docs/operations/experiment_progress.md:5-23` match the manuscript ordering (oDist > oIoU; smaller `top_k` improves both accuracy and runtime). These use the exact configs shipped in `configs/pipeline*.yaml`, so we can cite them confidently. | Keep `outputs/dair_v2xregpp_gt*` runs and mention `tools/run_calibration.py --config configs/pipeline.yaml`. |
| **SVD variants (w/m/h)** | The relative ranking (wSVD≻mSVD≻hSVD) supports the analytical section of the paper, reinforcing that solver choice matters. Metrics fall within previously reported ranges. | Keep `outputs/dair_v2xregpp_gt25_*`. |
| **oIoU baseline** | Shows expected degradation vs. oDist while highlighting latency issues; matches the “legacy vs. new pipeline” storyline. | Keep `outputs/dair_v2xreg_oiou_gt15`. |
| **V2X-Set synthetic figures (Fig. 6/8/9)** | JSON exports (`outputs/v2xset_*`) remain faithful to the figure captions and do not expose unreported failures. Visualization script already filters/scales them appropriately. | Safe to reference in README/docs. |
| **HKUST benchmark scripts** | Runs align with the public LiDAR benchmark; no contradictory metrics noted. | Keep `configs/pipeline_hkust*.yaml` & associated logs. |

**Recommendation:** Highlight only the above experiments in README + docs. Provide their commands and note that other configs are WIP.

---

## 3. Must-Hide / Internal-Only Artefacts

| Artefact | Risk | Action |
| --- | --- | --- |
| **Detection Bench Report (`docs/detection_bench_report.md`)** | Shows Success@1 m ≈ 0.18 (val/test) and 0.08 (train), plus near-zero `mRE/mTE` due to invalid matches. This exposes annotation mismatches and contradicts claims that detection-based calibration “almost matches GT”. | Keep the file internal; do **not** link or summarize in README/paper. If reviewers ask for detection-only numbers, rerun with corrected detection caches first. |
| **HEAL detection experiments (`outputs/heal_detection`, `docs/operations/heal_detection_status.md`)** | Demonstrates 19% success rate and notes training split corruption. Publishing this would highlight that our claimed HEAL integration is unreliable. | Hide metrics; rewrite README roadmap to mention HEAL integration as “future work” without showing current numbers. |
| **Feature-level/descriptor logs (`docs/operations/v2icalib_feature_extension.md`, `outputs/heal_features_*`)** | Contain multiple failed attempts (success@3 m ≤ 20%, TE ≈ 2 m). They also document descriptor indistinguishability, undermining any statement that “feature learning is ready”. | Keep document internal; when migrating to new hardware, restart record with sanitized runs once performance surpasses V2X-Reg++. |
| **Detection-bench `matches.jsonl` & summary JSON** | Include evidence of “frames_with_matches=0” for VIPS baseline (Section 2 of `experiment_progress.md`), which can be interpreted as benchmark misconfiguration. | Archive for debugging only; remove from public tarballs/releases. |
| **Partial ICP/PICP logs** | Current logs only show errors/timeouts; would suggest we never completed the classical baselines. | Either re-run and publish clean results or omit until ready. |

---

## 4. Communication Plan

1. **README** — emphasize ongoing refactor + roadmap (already updated). Mention only the stable configs and add a note that “detection/feature-level extensions are under active development”.
2. **Docs** — keep `experiment_progress.md` but trim sections referencing HEAL/detection bench before pushing to public branch. Alternatively split into `experiment_progress_public.md` (safe runs) and `experiment_progress_internal.md` (full history).
3. **Release bundles** — when packaging checkpoints/logs, include only the `outputs/dair_v2xregpp_*`, `outputs/v2xset_*`, and HKUST runs. Exclude `outputs/heal_*`, `outputs/det-*`, and unfinished baseline logs.
4. **Future updates** — once feature-level and HEAL experiments achieve ≥ baseline success rates, re-open the documentation and re-run detection benches to replace the negative evidence.

---

## 5. Immediate To-Do

1. Sanitize `docs/operations/experiment_progress.md` before public release:
   - Keep Sections 1 & 3 (Table III + V2X-Set).
   - Remove or move Sections 2, 4, 5, 6 to an internal-only file.
2. Move `docs/detection_bench_report.md` and `docs/operations/v2icalib_feature_extension.md` to a private branch or flag them as “INTERNAL USE ONLY” if they must remain for collaboration.
3. Double-check `README.md` links so none point to the hidden artefacts.
4. Prepare a clean tarball with only whitelisted `outputs/` folders for future reviewers.

---

Feel free to iterate on this memo as new experiments complete. The general rule is: **if a run can be cited in the paper without extra caveats, keep it; otherwise, quarantine until fixed**.
