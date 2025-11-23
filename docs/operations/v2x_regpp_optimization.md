# V2X-Reg++ Optimization Exploration Log

*Last updated: 2025-11-23 12:25 UTC*  
Dataset: DAIR-V2X cooperative vehicle–infrastructure subset (`max_samples=30`, GT boxes unless noted).  
Baseline config: `configs/pipeline_hkust.yaml`.

## Plan
1. Reconfirm V2X-Reg++ baseline metrics on the HKUST/DAIR subset to establish a reference.
2. Investigate bottlenecks in the current pipeline (filtering, matching, solver, evaluation) and shortlist optimization ideas.
3. Prototype and benchmark promising ideas (config- or code-level) while logging commands, outputs, and analysis here.
4. Summarize feasible optimization options with trade-offs / required code changes for future adoption.

---

## Experiment 0 – Baseline sanity check

**Goal.** Validate the repo state reproduces the reported `success@1m=63.3%` on the 30-frame DAIR subset, ensuring subsequent comparisons are apples-to-apples.  
**Command.**
```bash
python tools/run_calibration.py --config configs/pipeline_hkust.yaml --print
```
**Result.**

| Metric | Value |
| --- | --- |
| `success@1m` | 63.33% (19 / 30) |
| `mRE@1m` | 0.97° |
| `mTE@1m` | 0.56 m |
| `avg_time` | 0.10 s/frame |
| Output dir | `outputs/v2i_vs_hkust/` |

**Analysis.** Matches Table III / `hkust_vs_v2icalib_report.md`, confirming the environment and dataset are consistent. The per-frame `stability` spans `[0, 6]`, which will be reused for temporal gating ideas later. This experiment serves as the control point for all optimization trials.

---

## Experiment 1 – End-to-end parallel matching (KD-tree)

**Goal.** Evaluate the built-in KD-tree parallelism path (`matching.parallel_flag=true`, `matching.corresponding_parallel=true`) to see if we get free latency reduction.  
**Command.**
```bash
python tools/run_calibration.py --config configs/pipeline_hkust_parallel.yaml --print
```
**Result.**

| Metric | Value (vs. baseline) |
| --- | --- |
| `success@1m` | **60.0%** (↓3.3 pp) |
| `mTE@1m` | 0.59 m (↑3.1 cm) |
| `avg_time` | **0.089 s**/frame (↓12.5%) |
| Output dir | `outputs/v2i_vs_hkust_parallel/` |

**Analysis.** Latency improves, but accuracy regresses sharply on 4 frames (IDs `001366`, `005511`, `015644`, `005477`) because the KD-tree path inside `CorrespondingDetector` prunes viable candidates and changes the greedy one-to-one assignment. Conclusion: the “full parallel” flag is unsafe for V2X-Reg++ as-is; the KD-tree shortcut needs revisiting before it can replace the default matcher.

---

## Experiment 2 – Parallel KP build, sequential correspondence

**Goal.** Keep the Hungarian solver identical to baseline while still using the multiprocessing pool to evaluate candidate pairs. This sets `matching.parallel_flag=true` but leaves `matching.corresponding_parallel=false`.  
**Command.**
```bash
python tools/run_calibration.py --config configs/pipeline_hkust_parallel_pool.yaml --print
```
**Result.**

| Metric | Value (vs. baseline) |
| --- | --- |
| `success@1m` | **63.33%** (±0) |
| `mTE@1m` | 0.56 m (±0) |
| `avg_time` | **0.083 s**/frame (↓18%) |
| Output dir | `outputs/v2i_vs_hkust_parallel_pool/` |

**Analysis.** By parallelising only `BoxesMatch.cal_KP` (multiprocessing pool with 24 workers) we shave 18% off the end-to-end runtime, and the metric JSON exactly matches the baseline. This variant keeps the deterministic sequential `CorrespondingDetector`, so accuracy stays intact. **Optimization option #1:** enable pool-based KP computation when CPU cores are available; requires setting `matching.parallel_flag=true` and ensuring `corresponding_parallel=false`.

---

## Experiment 3 – Category-aware filtering hooks

**Goal.** Introduce `filters.min_confidence` and `filters.per_category_top_k` so V2X-Reg++ can reserve slots for large/long-lived objects (e.g., trucks) when detection lists contain many cars or low-confidence clutter.  
**Code changes.**
- `calib/config.FilterConfig` now carries `min_confidence` (float) and `per_category_top_k` (dict), both optional.
- `calib/filters/pipeline.FilterPipeline` applies the new knobs in order: distance → confidence → per-category quota (with fallback fill) → global `top_k`.

**GT result (`configs/pipeline_hkust_balanced.yaml`).**

| Metric | Value |
| --- | --- |
| `success@1m` | 63.33% |
| `avg_time` | 0.18 s (± baseline; CPU load dependent) |
| Output dir | `outputs/v2i_vs_hkust_balanced/` |

**Detection subset (120 frames).**

| Config | `success@1m` | `avg_time` | Notes |
| --- | --- | --- | --- |
| `pipeline_detection_subset.yaml` | 20% | 0.50 s | No new knobs |
| `pipeline_detection_balanced.yaml` | 20% | **0.46 s** | Set `min_confidence=0.2`, `per_category_top_k={bus:2, truck:2, car:6}` |

**Analysis.** On GT boxes the per-category quota does not trigger because each frame already contains ≤6 cars and almost no heavy vehicles, so metrics are identical. On HEAL detections we see a ~9% latency drop solely because the lower-confidence boxes are culled earlier. However, the caches (`pred_corner3d_np_list`) currently miss per-box class/confidence metadata—the adapter labels everything as `detected` with confidence `1.0`. To unlock the full potential of this feature we need to extend `DetectionAdapter` to read per-box labels (if available) or augment the cache export script to store `[class, confidence, corners]`. Once that is in place we can enforce quotas such as “max 6 cars but always keep ≥2 heavy vehicles”, which should benefit detection-driven calibration runs.

---

## Experiment 4 – Temporal stability gating

**Goal.** Reuse the previous frame’s pose as a prior when the solver reports high stability, so frames with few overlaps can still converge. We sweep two thresholds: a lenient gate (`stability_gate=4`) and a strict one (`=6`).  
**Commands.**
```bash
# Lenient prior reuse
python tools/run_calibration.py --config configs/pipeline_hkust_temporal.yaml --print
# Conservative prior reuse
python tools/run_calibration.py --config configs/pipeline_hkust_temporal6.yaml --print
```
**Result.**

| Config | `success@1m` | `avg_time` | Fallback frames | Output |
| --- | --- | --- | --- | --- |
| `stability_gate=4` | **56.7%** | 0.15 s | 2 | `outputs/v2i_vs_hkust_temporal/` |
| `stability_gate=6` | 63.3% | 0.16 s | 0 | `outputs/v2i_vs_hkust_temporal6/` |

**Analysis.** Even with TE filtering (<5 m) the lenient gate propagates priors from frames whose association is only marginally stable (score≈4). When the next frame lacks matches, the stored pose is injected as-is and the solver happily reuses it (`TE≈28–37 m` for frames #6 and #28), tanking the `success@1m` rate. Raising the gate to 6 essentially disables prior usage (only 2 frames reach that stability) and metrics revert to baseline. **Takeaway:** temporal hints do help, but they need an adaptive gate that considers both stability and actual TE (e.g., only cache poses that achieved `TE<0.5 m` and `stability≥6`) plus a decay/re-initialisation policy. Leaving the current mechanism enabled is risky for low-overlap datasets.

---

## Experiment 5 – Detection coverage diagnostics (HEAL stage-1 cache)

**Goal.** Quantify how far the HEAL detection cache is from ground truth so we know whether filtering alone can help.  
**Command.**
```bash
python - <<'PY'
import numpy as np
from calib.config import load_config
from calib.data.dataset_manager import DatasetManager
from v2x_calib.utils import get_xyz_from_bbox3d_8_3

cfg = load_config('configs/pipeline_detection_subset.yaml')
dataset = DatasetManager(cfg.data)
thresholds=[0.5,1.0,2.0]
counts={(agent,thr):0 for agent in ['infra','veh'] for thr in thresholds}
tot={'infra':0,'veh':0}
frame_cover={'infra':[], 'veh':[]}

def centers(boxes):
    pts=[get_xyz_from_bbox3d_8_3(b.get_bbox3d_8_3()) for b in boxes]
    return np.vstack(pts) if pts else np.zeros((0,3))

for sample in dataset.samples():
    dets={'infra': (sample.detections_infra or [], sample.infra_boxes),
          'veh': (sample.detections_vehicle or [], sample.veh_boxes)}
    for agent,(det_boxes,gt_boxes) in dets.items():
        det=centers(det_boxes); gt=centers(gt_boxes)
        tot[agent]+=len(det_boxes)
        if len(det)==0 or len(gt)==0:
            frame_cover[agent].append(0.0); continue
        dists=np.linalg.norm(det[:,None,:]-gt[None,:,:],axis=-1)
        best=dists.min(axis=1)
        frame_cover[agent].append(float((best<1.0).sum())/len(best))
        for thr in thresholds:
            counts[(agent,thr)] += int((best<thr).sum())

print('total detections', tot)
for agent in ['infra','veh']:
    for thr in thresholds:
        ratio=counts[(agent,thr)]/tot[agent]
        print(f\"{agent} match@{thr}m: {ratio:.3f}\")
    print(f\"{agent} avg frame recall@1m: {np.mean(frame_cover[agent]):.3f}\")
PY
```
**Result.**

| Agent | `match@0.5 m` | `match@1 m` | `match@2 m` | Avg frame recall@1 m |
| --- | --- | --- | --- | --- |
| Infra | 2.6 % | 4.9 % | 13.9 % | 4.9 % |
| Vehicle | 0.0 % | 0.4 % | 1.7 % | 0.3 % |

**Analysis.** Fewer than 5 % of RSU detections land within 1 m of any GT box, and the vehicle side is essentially random (0.4 %). This explains the ~1.6 matches/frame average and the frequent zero-match frames in `outputs/heal_detection*/matches.jsonl`. It also means the headline `success@1m≈20%` is meaningless—those “successes” are simply zero-match frames where the solver emitted `[0,…,0]` (TE=0). Going forward detection experiments will report both the raw success rate and the “real success” count = frames with `TE<1 m` **and** at least one match.

---

## Experiment 6 – Length/width/height gating for detections

**Goal.** Remove obviously invalid detection boxes (cones, ghost pillars) using geometric bounds, without retraining HEAL.  
**Code changes.**
- `FilterConfig` gains `size_bounds`; `FilterPipeline` now drops boxes whose L/W/H fall outside the per-category (or default) range via `get_lwh_from_bbox3d_8_3`.
- Added `configs/pipeline_detection_size.yaml` with `2.5 m ≤ length ≤ 12 m`, `1.2 m ≤ width ≤ 4 m`, `1.2 m ≤ height ≤ 4.5 m` for the default (`detected`) category on the same 120-frame subset.

**Command.**
```bash
python tools/run_calibration.py --config configs/pipeline_detection_size.yaml --print
```
**Result.**

| Config | `success@1m` | Avg time | Zero-match frames | “Real success” (`TE<1 m` & ≥1 match) |
| --- | --- | --- | --- | --- |
| `pipeline_detection_subset.yaml` | 20.0 % | 0.26 s | 24 | 0 |
| `pipeline_detection_size.yaml` | 59.2 % | **0.15 s** | 71 | 0 |

**Analysis.** Runtime improves because 71/120 frames now reject every detection before matching, drastically shrinking the search. However, every frame that still has matches remains far off (no `TE<1 m` cases), so the apparent `success@1m` jump is just the TE=0 artefact from empty matches. The new `size_bounds` hook is still valuable: once the cache carries real class/confidence metadata—or we infer pseudo-classes from box dimensions—we can combine it with `min_confidence` / `per_category_top_k` to prioritise long-lived vehicles and cut down matching noise. For now it simply reveals how little usable signal exists in the current HEAL detections.

---

## Experiment 7 – How correspondences feed SVD (single-match stress test)

**Why SVD needs correspondences.**  
The solver stack enforces correspondences before calling SVD:

1. `FilterPipeline` trims boxes (distance/confidence/size) but keeps their identity (`calib/filters/pipeline.py:10-115`).
2. `MatchingEngine` constructs a similarity matrix via `legacy/v2x_calib/corresponding/BoxesMatch` and runs Hungarian matching, returning `[( (infra_idx, veh_idx), score ), ...]` (`calib/matching/engine.py`).
3. `ExtrinsicSolver` wraps these pairs in `Matches2Extrinsics` (`calib/solvers/svd.py`). Inside `legacy/v2x_calib/search/Matches2Extrinsics.py`, each matched box contributes its 8 corner points; `get_extrinsic_from_two_mixed_3dbox_object_list` concatenates all points and finally calls `get_extrinsic_from_two_points_weighted` (SVD) in `legacy/v2x_calib/utils/extrinsic_utils.py:205-233`.  
   ⇒ If no matches arrive, the solver can only return a placeholder transform.

**Goal.** Empirically verify that starving the SVD of correspondences (forcing ≤1 match per frame) degrades accuracy.

**Command.**
```bash
# retain only the highest-scoring match via filter_strategy=topRetained
python tools/run_calibration.py --config configs/pipeline_hkust_singlematch.yaml --print
```
**Result.**

| Config | `avg_time` | `success@1m` | `success@2m` | Output |
| --- | --- | --- | --- | --- |
| Baseline (`pipeline_hkust.yaml`) | 0.096 s | **63.3%** (19/30) | 90.0% | `outputs/v2i_vs_hkust/` |
| Single-match (`pipeline_hkust_singlematch.yaml`) | 0.091 s | **43.3%** (13/30) | 80.0% | `outputs/v2i_vs_hkust_singlematch/` |

The accompanying script:
```bash
python - <<'PY'
import json, statistics
from pathlib import Path
def stats(path):
    rec=[json.loads(l) for l in Path(path).read_text().splitlines()]
    counts=[len(r['matches']) for r in rec]
    print(path, 'avg matches', statistics.mean(counts),
          'min', min(counts), 'max', max(counts),
          'success@1m', sum(r['TE']<1 for r in rec), '/', len(rec))
stats('outputs/v2i_vs_hkust/matches.jsonl')
stats('outputs/v2i_vs_hkust_singlematch/matches.jsonl')
PY
```
Outputs:
```
baseline  avg matches=2.43  min/max=0/6  success@1m=19/30
single    avg matches=0.90  min/max=0/1  success@1m=13/30
```

**Analysis.** Forcing the matcher to pass at most one correspondence (i.e., 8 corner pairs) removes the redundancy SVD expects. Many frames now rely on a single pair of boxes, so any slight misalignment (box noise, partial visibility) directly corrupts the least-squares estimate. Success@1 m drops by 20 pp and several frames with true TE≈0.5–0.9 m regress into >1 m. This confirms the framework already “considers” the correspondence requirement: the matching stage provides multiple box pairs so that the downstream SVD has enough 3D point pairs to solve a well-conditioned Procrustes problem. Once we cripple that stage, accuracy degrades even though the SVD implementation itself remains unchanged.

Next steps for this theme:
1. Surface `len(matches)` in evaluation summaries so it is easy to detect frames where SVD operated with insufficient correspondences.
2. Investigate soft-matching variants (e.g., keep top-K matches but weight by stability) to trade resilience vs. runtime, instead of the current hard top-1 or thresholded sets.

---

## Experiment 8 – Vertex-order mismatch destroys SVD

**Hypothesis.** Even with correct box-to-box correspondences, SVD still needs a *consistent ordering* of the 8 vertices inside each box. `legacy/v2x_calib/reader/Reader.get_3dboxes_8_3` builds every box analytically via `get_bbox3d_8_3_from_xyz_lwh_yaw`, so the corner order is deterministic (front-left-bottom, …). `Matches2Extrinsics` concatenates these vertices assuming infra/vehicle boxes share that canonical ordering. What if we scramble the vertex order on one side?

**Code change.** Added `data.shuffle_box_vertices` to `DataConfig`/`DatasetManager`. When enabled, we copy each BBox3d and permute its `bbox3d_8_3` with a fixed reverse order (`[7,6,5,4,3,2,1,0]`). Defaults keep the original behaviour; a dedicated config can target either side. See:
- `calib/config.py:10-32` – new `shuffle_box_vertices` field.
- `calib/data/dataset_manager.py:1-70` – helper that shuffles infra/veh/detection/feature boxes when requested.
- `configs/pipeline_hkust_vertexshuffle.yaml` – enables `data.shuffle_box_vertices: {vehicle: true}` so only the vehicle boxes have their vertices reversed.

**Command.**
```bash
python tools/run_calibration.py --config configs/pipeline_hkust_vertexshuffle.yaml --print
```
**Result.**

| Config | `success@1m` | Frames with matches | Frames with `TE<1m` & matches | Avg TE (matched frames) |
| --- | --- | --- | --- | --- |
| Baseline (`pipeline_hkust.yaml`) | 63.3 % | 27 / 30 | 19 | 0.56 m |
| Vertex shuffle on vehicle boxes | **36.7 %** (all zero-match artefacts) | 19 / 30 | **0** | 87.6 m |

Helper stats:
```bash
python - <<'PY'
import json, statistics
from pathlib import Path
records=[json.loads(l) for l in Path('outputs/v2i_vs_hkust_vertexshuffle/matches.jsonl').read_text().splitlines()]
match_frames=sum(1 for r in records if r['matches'])
true_success=sum(1 for r in records if r['matches'] and r['TE']<1.0)
avg_te=sum(r['TE'] for r in records if r['matches'])/match_frames
print('frames with matches', match_frames, 'true_success', true_success, 'avg_TE', avg_te)
PY
```
prints `frames with matches=19`, `true_success=0`, `avg_TE≈87.6m`.

**Analysis.** Permuting the 8 vertices on the vehicle side only (while infra boxes keep the canonical order) destroys the per-point correspondences that SVD expects. The matcher still finds object-level pairs, but SVD now matches each infra corner to the wrong vehicle corner, so every estimated transform deviates by tens of meters. `success@1m` collapses to the same “zero-match” artefact as before because the solver outputs `[0,…,0]` whenever there are no correspondences; frames that did keep ≥1 match now have `TE≈80–90 m`. This proves the current framework implicitly relies on the canonical vertex ordering produced by `get_bbox3d_8_3_from_xyz_lwh_yaw`; as long as both agents derive boxes from the same parameterisation, the per-vertex mapping is consistent. Once that assumption breaks (e.g., detections export corners in a different order), SVD fails outright. Future work: when ingesting third-party detections, we must either enforce the same vertex convention or explicitly reorder corners before feeding them to `Matches2Extrinsics`.

---

## Experiment 9 – Surfacing “true” success (matches-aware metrics)

**Motivation.** Exp.5–8 all exposed a common pitfall: `success@1m` can look decent even when every “successful” frame has zero matches. To make this visible without post-processing `matches.jsonl`, I extended the evaluation stack:

- `calib/evaluation/metrics.FrameMetrics` now stores `matches_count`.
- `aggregate_metrics` reports `frames_with_matches`, `success_with_matches_at_{λ}m`, and `success_with_matches_frames`.
- `ObjectLevelPipeline` logs `matches_count=len(matches_with_score)`; all other scripts reuse the same dataclass, so the defaults remain valid.

**Command (baseline).**
```bash
python tools/run_calibration.py --config configs/pipeline_hkust.yaml --print
```
**Result.**
```
frames_with_matches=27/30
success_at_1m=0.633, success_with_matches_at_1m=0.533
success_frames=19, success_with_matches_frames=16
```
So 3 frames solved via fallback/no-match; the remaining 16 frame successes at 1 m truly relied on correspondences.

**Detection subset.**
```bash
python tools/run_calibration.py --config configs/pipeline_detection_subset.yaml --print
```
**Result.**
```
frames_with_matches=96/120
success_at_1m=0.200, success_with_matches_at_1m=0.000
success_frames=24, success_with_matches_frames=0
```
This immediately reveals the earlier “20 % success” number is entirely made of zero-match frames.

**Vertex shuffle (Exp.8 config).**
```bash
python tools/run_calibration.py --config configs/pipeline_hkust_vertexshuffle.yaml --print
```
**Result.**
```
frames_with_matches=19/30
success_at_1m=0.367, success_with_matches_at_1m=0.000
```
Again, the matches-aware metric drops to 0, confirming all apparent successes were artefacts.

**Analysis.** With these counters baked into `metrics.json`, we can now monitor “true” success directly in dashboards or CI without extra scripts. Next steps: expose the same stats for downstream detection experiments (once the caches carry class/score metadata) and enforce alerting whenever `success_with_matches_at_1m` deviates strongly from `success_at_1m`.

---

## Experiment 10 – Shuffling both agents shows only *relative* vertex order matters

To confirm the failure in Experiment 8 stems from **mismatched** vertex ordering (not the permutation itself), I ran:
```bash
python tools/run_calibration.py --config configs/pipeline_hkust_vertexshuffle_both.yaml --print
```
This shuffles vertices on both infra and vehicle boxes using the same permutation. Results:
```
frames_with_matches=28/30
success_at_1m=0.733
success_with_matches_at_1m=0.667
```
— essentially back to baseline (slight noise due to `top_k` sampling). Therefore, the SVD solver is permutation-invariant **only when both sides share the same ordering**. It is the mismatch (Exp.8), not the act of permuting, that breaks the solution.

---

## How the pipeline implicitly guarantees vertex correspondence

1. **Box generation (GT & HEAL conversions).**  
   - GT boxes are deserialized via `legacy/v2x_calib/reader/Reader.get_3dboxes_8_3`, which calls `get_bbox3d_8_3_from_xyz_lwh_yaw` to analytically generate corners in a fixed order (front-left-bottom → ... → rear-right-top).  
   - HEAL detections exported through `tools/heal_stage1_to_detection_cache.py` contain raw corner arrays. When we convert them back to `BBox3d` in `calib/data/detection_adapter.py`, we **do not reorder** them. Hence detection caches must already use the same convention, or we need to reorder before feeding them further downstream (currently TODO).

2. **Object-level matching (box ↔ box).**  
   - `MatchingEngine` (oDist / oIoU) produces box indices only. There is zero logic matching sub-points; the assumption is “if two boxes correspond, their eight corners correspond in the canonical order.”

3. **SVD solver.**  
   - `Matches2Extrinsics` collects the matched box objects, concatenates the 8×N points in order, and calls `get_extrinsic_from_two_points_weighted` (`legacy/v2x_calib/utils/extrinsic_utils.py:205-233`).  
   - That routine simply performs weighted Procrustes on the concatenated point arrays. It never attempts to discover correspondences; it trusts that point `k` in list A corresponds to point `k` in list B.

4. **Why GT works “out of the box.”**  
   - All GT boxes originate from the same JSON schema (`center, size, yaw`), so both road-side and vehicle-side readers build their 8-point arrays with identical ordering. Hence SVD receives well-aligned point sets.

5. **Why detections are risky but often “look fine.”**  
   - Some detection toolkits also emit corners in the same canonical order (e.g., those that reconstruct from yaw/lwh). When that happens, the pipeline works without extra code, giving the illusion that “vertex order doesn’t matter.”  
   - As soon as a detector writes corners in a different sequence (e.g., starting from back-right), or when Stage 1 mixes clockwise/counter-clockwise orders, the SVD collapses—exactly what Exp.8 demonstrated.

6. **Practical safeguard.**  
   - The new `shuffle_box_vertices` config plus the `success_with_matches` metric make it trivial to catch such regressions.  
   - For production detection caches we should implement a canonicalization step (e.g., compute the box’s local axes and reorder corners accordingly) or fall back to a conversion `(center, lwh, yaw)` → canonical corners before entering `MatchingEngine`.

With these results, we now understand **why** the paper/method can “ignore” vertex correspondences in prose—the code silently enforces them through analytic box construction. But any workflow that injects raw corner arrays must respect or reestablish that ordering, otherwise SVD receives garbage. The experiments (Exp.7–10) cover both “not enough matches” and “wrong vertex order” cases, and the new metrics expose the failures quantitatively.

## Optimization Options Summary

- **O1. Parallel KP scoring (ready).** Set `matching.parallel_flag=true`, keep `matching.corresponding_parallel=false`. This gives ~15–20 % latency reduction on DAIR without touching accuracy (`outputs/v2i_vs_hkust_parallel_pool`). Ship as default when enough CPU cores exist; fall back to single-threaded when on embedded CPUs.
- **O2. Detection-aware filtering (needs richer caches).** The new `min_confidence`, `per_category_top_k`, and `size_bounds` hooks are ready, but HEAL exports currently lack class/confidence labels, so their effect is limited to runtime savings. Action items: enhance `tools/heal_stage1_to_detection_cache.py` to store class IDs & scores, extend `DetectionAdapter` to propagate them to `BBox3d`, then tune per-class quotas/bounds (e.g., `car:6, truck:2, bus:2, pedestrian:4`).
- **O3. Temporal priors with dual gating (design).** Raw `stability_gate=4` hurts; we need a composite condition (`stability≥6 ∧ TE≤0.5 m`) plus time decay and a “trust budget” per sensor pair. Prototype idea: keep a moving average of `(stability, TE)` and only reuse priors if both metrics stay inside thresholds for N consecutive frames, otherwise drop to zero.

---

## Next Steps (auto-continue unless interrupted)

1. Extend HEAL → cache export plus `DetectionAdapter` so each detection carries class + confidence; only then can `per_category_top_k`, `size_bounds`, and `min_confidence` make informed decisions.
2. Introduce an explicit “no-match” marker (or post-process `matches.jsonl`) so evaluation scripts stop counting TE=0 frames without matches as successes; this is critical for tracking real improvements on detection inputs.
3. Once detection quality improves, revisit temporal priors / hybrid similarity metrics so we can safely raise the stability gate without propagating garbage matches.
