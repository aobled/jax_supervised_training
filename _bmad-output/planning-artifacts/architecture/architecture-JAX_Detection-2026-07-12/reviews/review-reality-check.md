# Reality-Check Review — ARCHITECTURE-SPINE.md (JAX_Detection refactor)

**Reviewed artifact:** `_bmad-output/planning-artifacts/architecture/architecture-JAX_Detection-2026-07-12/ARCHITECTURE-SPINE.md`
**Lens:** Verify every committed decision was web-researched or reality-checked rather than asserted from training data — with the review scope narrowed (per task brief) to codebase-reality claims rather than library/version claims, since this spine introduces no new external technology.
**Method:** Direct inspection of the source files the spine makes factual claims about (`model_library.py`, `dataset_configs.py`, `bounding_boxes_with_classification_from_benchmark.py`, `bounding_boxes_with_classification_from_video_generation.py`, `tools/bounding_boxes_with_classification_from_images_generation.py`, `heatmap_generation.py`, `heatmap_contouring.py`, `tools/audit_dataset_classification.py`, `checkpoint_manager.py`), cross-referenced against the spine's own cited source `docs/dead-code-and-duplication-audit.md`.

## Overall Verdict: PASS WITH FINDINGS

The spine's factual claims about the codebase are **substantially accurate** — every headline number and named-entity claim I spot-checked against the actual source files checked out. There is no evidence of assertions invented from training-data priors about this specific codebase (which would be impossible anyway — this is a private brownfield repo, not something in a training set). The two findings below are not fabrications; they are **incomplete reconciliation** of divergences that the spine's own cited source document (`docs/dead-code-and-duplication-audit.md`) had already flagged as needing design work, not mechanical cleanup.

## Finding counts
- Critical: 0
- High: 0
- Medium: 2
- Low: 0
- Confirmed-accurate (positive checks, listed for completeness): 7

---

## Medium Findings

### M1 — AD-2's canonical `predict_crop` signature doesn't mechanically compose with the "batch of 1" delegation it claims, and ignores that a conflicting `predict_crop` already exists (as dead code) in a file the spine designates as an `inference_utils.py` consumer

**File:** `ARCHITECTURE-SPINE.md`, AD-2 (lines 34-38); reality in `bounding_boxes_with_classification_from_video_generation.py:251-256` and `tools/bounding_boxes_with_classification_from_images_generation.py:75-96`

AD-2 asserts a single canonical signature: `predict_crop(crop_img, model, variables, mean, std, config)`, and states it "peut être implémenté en interne comme un appel à `predict_crops_batch` avec un batch de taille 1."

Reality check:
- `tools/bounding_boxes_with_classification_from_images_generation.py:128` already has `predict_crop(crop_img, model, variables, mean, std, config)` — matches AD-2's chosen signature. This function calls `model.apply(variables, ...)` directly (no JIT), confirmed by direct read.
- `bounding_boxes_with_classification_from_video_generation.py:251` — a file the spine's own structural seed lists as a designated `inference_utils.py` consumer — already has **a different `predict_crop`**: `predict_crop(crop_img, predict_fn, mean, std, config)`, and its body is literally `return predict_crops_batch([crop_img], predict_fn, mean, std, config)[0]` (confirmed by direct read, line 256). This is exactly the "batch of 1" delegation pattern AD-2 describes — but it operates on a `predict_fn`, not on `model, variables`.
- Grep across the whole repo for call sites of `predict_crop(` shows this `video_generation.py` version is **never called anywhere in the file or the repo** — it is dead code today, but the spine's structural seed doesn't flag it for removal; it just says the file "importe depuis inference_utils.py."

Consequence: AD-2's chosen `(model, variables, ...)` signature cannot literally delegate to `predict_crops_batch(crops, predict_fn, ...)` without first building a `predict_fn` (e.g. wrapping `model.apply` in `jax.jit`) on every call — which is either a perf regression (re-jit per call) or requires caching logic the spine never mentions. The "can be implemented as a call to predict_crops_batch" claim is asserted but not shown to actually work with the chosen signature; it was true of the *existing* predict_crop in `video_generation.py` (which uses `predict_fn`), not of the signature AD-2 actually picked.

This matters because the spine's own cited source, `docs/dead-code-and-duplication-audit.md:50`, explicitly warned: *"La mutualisation ne peut donc pas être un simple 'supprimer les doublons et importer' : il faut d'abord choisir/réconcilier un comportement canonique par fonction, ce qui est un vrai travail de conception."* AD-2 does attempt this reconciliation for `predict_crop`, but the reconciliation has a gap that direct source reading surfaces immediately.

**Recommendation before implementation:** either (a) change AD-2's `predict_crop` signature to take a `predict_fn` (matching `video_generation.py`'s already-existing, currently-dead implementation, and explicitly deprecate/delete the `model,variables`-based version from `images_generation.py`'s call site), or (b) keep `(model, variables, ...)` and specify explicitly how it builds/caches a `predict_fn` internally without a per-call JIT cost — and note in the structural seed that `video_generation.py`'s existing `predict_crop` (dead code) is deleted, not migrated.

### M2 — AD-1's "7 canonical functions, single source of truth" framing is incomplete: one designated consumer file duplicates the same functionality under different names/shapes, and a whole family of near-duplicate helper functions isn't in the inventory at all

**File:** `ARCHITECTURE-SPINE.md`, AD-1 (lines 28-32) and Structural Seed (lines 92-108); reality in `tools/audit_dataset_classification.py:24-103` and `bounding_boxes_with_classification_from_video_generation.py:179-193`

AD-1 states the "single source of truth" set is exactly 7 functions: `load_jax_model`, `load_detection_model`, `predict_crop`, `predict_crops_batch`, `get_iou`, `non_max_suppression`, `_preprocess_crop_to_hwc`, and that "tout fichier qui en a besoin importe ; aucune redéfinition locale n'est autorisée."

Reality check on `tools/audit_dataset_classification.py` — one of the 5 files the structural seed lists as importing from `inference_utils.py`:
- It defines `_preprocess_crop_to_hwc` (matches AD-1's list — fine).
- It does **not** define `load_jax_model`. Instead it defines `load_classification_model(config)` (line 50), which does the same job (load checkpoint, resolve model via `get_model`, load mean/std) but with a different signature (single `config` arg vs `checkpoint_path, config`), a different path-fallback strategy (single script-parent fallback, not the CWD → parent-of-CWD → parent-of-script chain AD-3 describes for the sibling detection loader), and — critically — a **different return shape**: `predict_fn, mean, std` (line 103), not `model, variables, mean, std` as `load_jax_model` returns everywhere else.
- To build that `predict_fn`, it defines its own private `build_predict_fn(model, variables)` (line 42) — a JIT-wrapping helper functionally identical to `build_det_predict_fn`/`build_clf_predict_fn` already defined separately in `bounding_boxes_with_classification_from_video_generation.py:179-193`. None of these three near-identical helpers appear anywhere in AD-1's 7-function inventory.

Consequence: under the letter of AD-1's rule ("ces 7 fonctions existent en un seul exemplaire... aucune redéfinition locale n'est autorisée"), this file's `load_classification_model` and `build_predict_fn` are **not covered** — they're neither one of the 7 named functions (so the "no local redefinition" rule doesn't technically apply to them) nor addressed anywhere else in the spine. A refactor executed to the letter of AD-1 could pass its own acceptance rule while leaving `audit_dataset_classification.py` with a bespoke, un-mutualized loader and JIT-builder — directly undercutting the stated purpose of the module ("Mutualisation des fonctions d'inférence dupliquées," scope line 7) for exactly the kind of divergence the spine's own source audit flagged as real (`docs/dead-code-and-duplication-audit.md` §2 lists `_preprocess_crop_to_hwc` as shared by these same two files but says nothing about `load_classification_model`/`build_predict_fn` — the audit itself didn't catch this pair, and the spine inherited the gap).

**Recommendation before implementation:** either fold `load_classification_model` + `build_predict_fn` into the AD-1 inventory explicitly (deciding whether `load_jax_model` should also return a pre-built `predict_fn`, or whether `audit_dataset_classification.py` should be changed to call `load_jax_model` + a shared `build_predict_fn` that also gets added to `inference_utils.py`), or explicitly scope them out with a stated reason.

---

## Confirmed-accurate claims (verified directly against source, no issues found)

1. **AD-5 / `decode_grid_and_detect`** — confirmed present in `bounding_boxes_with_classification_from_benchmark.py:151-199`, grid-cell decode + vectorized-numpy NMS, exactly as described. No other file imports from this module (`grep` across repo confirms zero external references), so "supprimé, pas réconcilié" is safe.
2. **`model_library.py` — 4 survivors / 18 removed** — `MODELS` dict (lines 2450-2474) has exactly 22 entries; `aircraft_detector_unet`, `aircraft_detector_miniunet`, `sophisticated_cnn_128_plus`, `kepler_1d_cnn` are all present as claimed. 22 − 4 = 18 matches the "18 supprimées" claim exactly.
3. **`dataset_configs.py` — 3 survivors / 4 removed** — `DATASET_CONFIGS` has exactly 7 top-level keys (`FIGHTERJET_CLASSIFICATION`, `FIGHTERJET_VIT`, `FIGHTERJET_HYBRID_VIT`, `FIGHTERJET_LETTERBOX`, `FIGHTERJET_DETECTION`, `FIGHTERJET_DETECTION_SOPHISTICATED`, `JAX_KEPLER`). The 3 claimed survivors (`FIGHTERJET_CLASSIFICATION`, `FIGHTERJET_DETECTION`, `JAX_KEPLER`) are present; 7 − 3 = 4 matches.
4. **AD-3 — degraded `load_detection_model` in `heatmap_generation.py` / `heatmap_contouring.py`** — confirmed both files' `load_detection_model` (lines 18-38 and 19-39 respectively) lack both the 3-level path fallback (they only `raise FileNotFoundError` immediately) and the `batch_stats` re-init fallback (they leave `batch_stats = {}` silently if absent, no warning, no re-init), exactly the fragility AD-3 describes. The "robust" versions in `video_generation.py` and `tools/images_generation.py` do have both, confirmed by direct read.
5. **AD-4 — dead `model_name` fallbacks** — confirmed all 4 files that have a `load_detection_model` default a `model_name` to a since-deleted model: `video_generation.py` and `tools/images_generation.py` default to `'aircraft_detector_v3'`; `benchmark.py`, `heatmap_generation.py`, and `heatmap_contouring.py` default to `'aircraft_detector_v7_advanced'`. Both are in the 18-model removal set from finding 2 above, so AD-4's stated risk (a `ValueError` from `get_model()` if this fallback ever fires post-refactor) is real and would trigger against any of 5 files today if left unfixed.
6. **Box format convention — `[x1,y1,x2,y2,score]` list format, "seul survivant après AD-5"** — confirmed: the surviving `non_max_suppression(boxes, iou_threshold)` implementations in `video_generation.py:335` and `tools/images_generation.py:245` are identical (list of `[x1,y1,x2,y2,score]`, score embedded, sorted internally). The only other variant — `non_max_suppression(boxes, scores, iou_threshold)` with separate numpy score array — lives solely in `bounding_boxes_with_classification_from_benchmark.py:140`, which AD-5 deletes. So AD-5's deletion incidentally resolves this particular divergence (flagged in `docs/dead-code-and-duplication-audit.md:47`) for free — a positive, verifiable side effect not explicitly claimed by the spine but consistent with it.
7. **Structural seed file list (5 consumer files)** — confirmed no other `.py` file in the repo (outside `tokens/`, which is deleted wholesale by FR8 and separately duplicates several of these same functions with yet more signature variants — not in scope since the whole directory goes away) defines any of the 7 AD-1 function names. The 5-file consumer list is complete for those exact 7 names (modulo the M2 gap above, which is about functions *not* on that list).

## Scope note

Per the task brief, I did not chase library/framework version currency (JAX/Flax/NumPy/OpenCV pins) or starter defaults — the spine names no version numbers and introduces no new dependency, so that axis of the lens is inapplicable here, and I confirmed by reading the spine that it makes no version-specific claims to check. The two findings above are the only reality-check gaps surfaced by direct comparison against the source files and the spine's own cited audit document.
