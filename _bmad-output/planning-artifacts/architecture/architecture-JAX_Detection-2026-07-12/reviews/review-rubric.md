---
title: Review — ARCHITECTURE-SPINE.md (good-spine checklist)
reviewed_artifact: ../ARCHITECTURE-SPINE.md
reviewed_against: good-spine checklist (divergence coverage, AD enforceability, Deferred safety, tech currency, brownfield ratification, spec coverage, parent-spine non-contradiction, dimension completeness)
method: read PRD + spine + memlog + dead-code audit; cross-checked AD claims against actual source (bounding_boxes_with_classification_from_video_generation.py, tools/bounding_boxes_with_classification_from_images_generation.py, heatmap_generation.py, heatmap_contouring.py, tools/audit_dataset_classification.py, bounding_boxes_with_classification_from_benchmark.py, model_library.py, live .pkl checkpoints)
created: 2026-07-12
---

# Review — ARCHITECTURE-SPINE.md

## Overall verdict

The spine is well-grounded where it engaged the code: AD-3/AD-4 (`load_detection_model` robustness, dead-model fallback) and AD-5 (benchmark.py deletion) are verified accurate against the actual source, and AD-2's canonical `predict_crop` signature correctly ratifies the majority existing pattern rather than inventing one. However, the spine misses the single largest real divergence point in its own charter's domain — the box-decoding logic (`decode_segmentation_and_detect` vs `decode_segmentation_and_detect_batch`) — and explicitly defers resolving it rather than deciding it, which contradicts FR3's own requirement. It also carries a safety-relevant spec gap (FR6) and a couple of minor internal-consistency issues. Net: usable as a build substrate for AD-1 through AD-5 as scoped, but not safe to treat as complete for FR1-FR3's stated goal (full reconciliation of the 5 in-scope files) without closing Finding 1 first.

## Findings

### Finding 1 — Severity: High
**Location:** AD-1 (Rule, list of 7 functions), Deferred bullet 4, Consistency Conventions ("Format des boxes")

The spine's canonical-function list (AD-1: `load_jax_model`, `load_detection_model`, `predict_crop`, `predict_crops_batch`, `get_iou`, `non_max_suppression`, `_preprocess_crop_to_hwc`) omits the box-decoding functions — `decode_segmentation_and_detect` (`tools/bounding_boxes_with_classification_from_images_generation.py:269`) and `decode_segmentation_and_detect_batch` (`bounding_boxes_with_classification_from_video_generation.py:362`) — and Deferred bullet 4 explicitly punts them: *"Comportement exact de `decode_segmentation_and_detect(_batch)` (au-delà des 7 fonctions listées dans AD-1) — non auditée fonction par fonction dans ce cycle ; si une divergence y est découverte en implémentation, elle suit la même règle que AD-1."*

I read both function bodies. They are **not** a name-matching duplicate pair (hence invisible to the identical-name search that produced `dead-code-and-duplication-audit.md` and, in turn, AD-1's list) — they diverge substantially in behavior:
- `decode_segmentation_and_detect_batch` (video path) binarizes the **low-resolution** (224×224) prediction mask with a single threshold, applies fixed `MORPH_CLOSE` + dilate kernels (`_CLOSING_KERNEL`, `_DILATE_KERNEL`), extracts contours at low-res, then projects box coordinates to full resolution via `scale_x`/`scale_y`.
- `decode_segmentation_and_detect` (image-tooling path) first resizes the mask to **full resolution** with `cv2.INTER_CUBIC`, then does a **dual-threshold** binarization (`strong_mask` at `conf_threshold` AND `weak_mask` at `conf_threshold*0.4`, intersected), with three different morphological kernels (9×9 dilate, 21×21 close, 11×11 dilate) before contour extraction.

These are materially different pipelines that will produce different boxes for the same model/image between the video tool and the image tool — both of which are in-scope FR2 files (`bounding_boxes_with_classification_from_video_generation.py`, `tools/bounding_boxes_with_classification_from_images_generation.py`) that Goal 1's baseline/diff non-regression check is supposed to cover. This is exactly the class of "real divergence point" AD-1/FR1-FR3 exist to fix, and the spine leaves it explicitly unresolved rather than choosing a canonical behavior — directly at odds with FR3: *"Pour chaque fonction ayant des comportements divergents entre fichiers, un comportement canonique est choisi explicitement avant fusion (pas de fusion mécanique aveugle)."* It also means the Consistency Conventions table's claim that box handling is "cohérent avec le paradigme segmentation, seul survivant après AD-5" is true only of the output *tuple format*, not of the underlying decode computation — potentially misleading if read as "the boxes will match."

**Why it matters for build substrate:** an implementer following AD-1 literally will move the 7 listed functions into `inference_utils.py` and leave the two decode functions untouched and diverging — satisfying the letter of AD-1 while leaving the actual behavioral risk this refactor was chartered to close. Since the divergence was findable with a few minutes of direct reading (not deep archaeology), treating it as "to be discovered during implementation" undersells known risk in a spine whose whole purpose is to fix real divergence points before implementation.

### Finding 2 — Severity: Medium
**Location:** frontmatter `binds: [... FR6, NFR1 ...]`; spine body (absence)

The frontmatter claims the spine binds FR6 and NFR1, but neither is engaged anywhere in the spine body — no AD, Rule, Convention, or Deferred entry addresses them (verified: `grep -n "NFR1"` and `grep -n "FR6"` against the spine file return no body hits, only the frontmatter/scope lines).

- **FR6** — *"Avant suppression d'une architecture, vérifier qu'aucun `.pkl` actuellement versionné (`best_model.pkl`, `best_model_detection.pkl`) n'en dépend."* This is the safety gate for the FR5 mass-deletion (18 of 22 architectures) that the Structural Seed prescribes concretely (`model_library.py # réduit ... 18 supprimées`). AD-4 covers an adjacent but narrower risk (a stale fallback `model_name` string), not "does a currently-live `.pkl` actually depend on one of the architectures about to be deleted." I spot-checked the live checkpoints directly: `best_model.pkl` → `sophisticated_cnn_128_plus`, `best_model_detection.pkl` → `aircraft_detector_unet` — both among the 4 kept architectures, so the gap is currently latent, not active. But nothing in the spine instructs an implementer to run this check before executing FR5's deletion, so FR6 is silent rather than decided/deferred/open, which the checklist flags directly.
- **NFR1** (genericity, validated by `JAX_KEPLER`) — never named in the spine body, unlike NFR2/NFR3/NFR4 which each get an explicit citation tied to a specific AD or the Design Paradigm paragraph. It is very likely preserved in practice (the Structural Seed keeps `kepler_1d_cnn`, and no AD touches `task_strategies.py`/`data_management.py`), but the PRD treats NFR1 as a primary non-negotiable ("Contre-métriques: Ne pas casser la généricité... `JAX_KEPLER` doit continuer à fonctionner") and the spine gives it zero explicit reasoning, unlike the other three NFRs.

FR4 (dataset_configs.py purge) is a milder version of the same pattern — bound in frontmatter, not cited by tag in any AD — but is adequately covered descriptively by the Structural Seed ("réduit : 3 configs ... 4 supprimées"), so it doesn't rise to the same severity as FR6/NFR1.

### Finding 3 — Severity: Low
**Location:** Deferred bullet 1

*"Réconciliation formelle avec le PRD : FR2/Goal 1 du PRD listent encore 6 fichiers (avant la découverte AD-5) — à synchroniser en Update PRD juste après ce spine (voir Finalize)."*

This appears stale. The PRD at `_bmad-output/planning-artifacts/prds/prd-JAX_Detection-2026-07-12/prd.md` (same `updated: 2026-07-12` date) already reflects the post-AD-5 state: FR2 explicitly lists 5 files, Goal 1 explicitly says "sur les 5 fichiers concernés... pas seulement 2," FR9 (benchmark.py deletion) is already present, and Open Questions/OQ1 is already marked resolved with a direct pointer back to this same spine's AD-1..AD-5. The synchronization this Deferred item describes as still-pending has apparently already happened, but the spine still lists it as outstanding work. Minor documentation-freshness issue — doesn't affect build correctness, but a reader relying on the spine's Deferred list as a live to-do would waste a cycle re-doing already-done work.

### Finding 4 — Severity: Low
**Location:** AD-2 (Rule)

AD-2's rule states the public `predict_crop(crop_img, model, variables, mean, std, config)` "peut être implémenté en interne comme un appel à `predict_crops_batch` avec un batch de taille 1." But `predict_crops_batch`'s own signature (per AD-2 itself, and per the existing `video_generation.py:214` implementation) takes `predict_fn` — a precompiled JIT function — not `model, variables`. Bridging the two requires first building a `predict_fn` from `model`/`variables` (e.g., via `build_det_predict_fn`/`build_clf_predict_fn`, which exist in `video_generation.py` but aren't part of AD-1's canonical-function list, or an ad-hoc `jax.jit` wrap). Not wrong, just underspecified: the Rule asserts an internal-delegation relationship between two functions whose signatures don't actually compose without an unstated intermediate step.

### Finding 5 — Severity: Low (informational)
**Location:** whole spine (absence of an operational/environmental section)

The spine says nothing about deployment, environments, or infra/runtime (Python/JAX/GPU version pinning, what "before/after" environment stability means for NFR3's timing comparisons). This is almost certainly legitimate: the project is a personal, locally-run script pipeline with no deployment surface, and `docs/architecture.md` already notes "Pas de CI/CD, pas de packaging, pas de suite de tests" as a known, accepted gap that the PRD's Non-Goals section explicitly carries forward ("Pas de suite de tests automatisée ni de CI/CD dans ce cycle"). The gap is that the spine itself doesn't say this — a reader of the spine alone, without cross-referencing the PRD's Non-Goals, can't distinguish "silently skipped" from "deliberately inherited and out of scope." A one-line pointer to the PRD Non-Goals would close this cheaply. Flagged per the checklist's explicit instruction to check this dimension even when it looks like a natural fit for silence.

## Checklist items with no findings (verified, not just assumed)

- **AD enforceability / actually prevents its divergence** — AD-3 and AD-4 checked directly against source: `heatmap_generation.py`/`heatmap_contouring.py`'s `load_detection_model` genuinely lacks both the path fallback and the `batch_stats` reinit (raises immediately, silently defaults to `{}`), confirming AD-3's stated risk; `video_generation.py`/`images_generation.py`'s `load_detection_model` genuinely defaults `model_name` to the dead `'aircraft_detector_v3'`, confirming AD-4's stated risk.
- **Ratifies rather than contradicts brownfield** — AD-2's chosen `predict_crop(crop_img, model, variables, mean, std, config)` signature matches the *existing* signature already used in 2 of 3 source files (`images_generation.py`, `benchmark.py`), not an invented one.
- **AD-5 grounding** — `bounding_boxes_with_classification_from_benchmark.py` genuinely contains `decode_grid_and_detect` and a `non_max_suppression(boxes, scores, iou_threshold)` variant tied to grid-based anchor decoding, distinct from the list-based NMS shared by the other files — confirms AD-5's rationale for deletion-not-reconciliation.
- **Named tech verified-current** — the spine names no external libraries/versions to verify; N/A.
- **Parent spine non-contradiction** — no initiative/product-altitude spine exists in the repo (this is the only file under `_bmad-output/planning-artifacts/architecture/`); nothing to contradict.
- **`get_iou`/`non_max_suppression`/`_preprocess_crop_to_hwc` "identical everywhere" claim** — verified byte-for-byte identical between `video_generation.py` and `images_generation.py`.
