# Reconcile Review — ARCHITECTURE-SPINE.md (JAX Single-Pass, 2026-07-15)

Sources checked against:
1. `ARCHITECTURE-SPINE.md` (the spine under review)
2. `notes-jax-single-pass.md` (primary source, full exploration log)
3. `jax-single-pass.mmd` (diagram source)
4. Parent spine `architecture-JAX_Detection-2026-07-12/ARCHITECTURE-SPINE.md` (AD-1..AD-8)
5. `.memlog.md` (authoritative distilled decision record)

## Overall verdict: minor gaps

The spine is a faithful, well-consolidated distillation. Every `[ADOPTED]` decision in the memlog maps to an AD or Deferred entry, the two motivating threads from the notes (eliminate python/cv2, fix box-fusion on close-formation aircraft) both land in AD-1's Prevents clause, and the "train modular / infer unified" split (AD-6) plus fixed-size-output-with-explicit-mask (AD-7) are captured precisely, including the late corrections Aymeric requested on 2026-07-15 (valid_mask and box coordinates surfaced in final output, symmetric frozen-weights handling). No content is contradicted. Findings below are omissions/weakenings, not errors.

## Findings

### 1. Inherited Invariants table omits parent AD-4 and AD-5 (moderate, structural)

The parent spine (`architecture-JAX_Detection-2026-07-12`) defines eight invariants, AD-1 through AD-8. The new spine's "Inherited Invariants" table (lines 29-36) lists only AD-1, AD-2, AD-3, AD-6, AD-7, AD-8 — AD-4 ("Pas de référence résiduelle à un modèle mort" — any default `model_name` fallback in `inference_utils.py` must point to `aircraft_detector_unet`, never to the deleted `aircraft_detector_v3`/`v7_advanced`) and AD-5 (`bounding_boxes_with_classification_from_benchmark.py` stays deleted, not reconciled) are silently absent.

This matches the `.memlog.md`, which also only cites AD-1/2/3/6/7/8 as "[Hérité, ...]" constraints (lines 28-33) — so the omission was decided upstream of the spine write, not introduced by the distillation step. Neither AD-4 nor AD-5 is contradicted by anything in this initiative (no new default `model_name` fallback is added; the benchmark file isn't touched), so there's a reasonable argument the omission is legitimate scoping rather than a dropped requirement. Still, per this reconcile pass's own brief, all eight parent ADs are expected to appear (even if only to note "not touched by this initiative"), and right now a reader of the new spine alone cannot tell these two invariants still exist and still bind the codebase.

### 2. "Ressources — conclusion" (performance/memory reassurance) has no home in the spine (minor)

`notes-jax-single-pass.md` records a question Aymeric asked twice — "le pipeline unifié sera-t-il plus lourd/plus consommateur ?" — resolved explicitly both times ("Ressources — conclusion", line 102-104, and open item #6, line 97): parameter count is unchanged, the only new cost is holding a 1920×1080 grayscale frame on-device (~8.3 MB/image, negligible even at batch=32), and removing the sequential Python crop loop (replaced by `vmap`) should make the pipeline *faster*, not slower.

This resolved conclusion doesn't appear anywhere in the spine — not as a rule, not in Consistency Conventions, not in Deferred. It's arguably non-binding background/reassurance rather than an architectural invariant (nothing in the spine contradicts it, and no future implementer needs it to build correctly), so this is a soft finding, not a design gap. Flagging because it was an explicit, twice-repeated question with a clear resolved answer, and terse AD-block structures are exactly where this kind of "quiet reassurance" tends to get silently dropped.

### 3. Structural argument for point-based vs. grid-based on overlap is narrowed to a data-availability argument (minor)

`notes-jax-single-pass.md` (line 72) records a distinct technical point: "une grille 7×7 classique a elle-même une limite structurelle sur le chevauchement (une prédiction dominante par cellule/ancre) — une approche par point central... s'en sortirait mieux, indépendamment du problème de données" (i.e., point-central beats grid/anchor on overlap *even setting the data-scarcity problem aside*, for a structural reason: one dominant prediction per cell/anchor).

The spine's AD-10 ("Grid-based/YOLO à ancres reporté, pas abandonné") frames the deferral purely as a data-scarcity issue: "le volume global de données n'est plus limitant, mais le signal d'entraînement spécifique au chevauchement... reste quasi inexistant." AD-1's Prevents clause covers the segmentation-vs-point-detection box-fusion argument but not the grid-vs-point structural argument specifically. The independent structural reason for preferring point-central over grid/anchor on overlapping objects is present in the source but not restated in the spine's rationale for AD-10. Low severity: the spine's overall conclusion (defer grid-based, keep point-central) is unchanged, and AD-1 covers the closely related segmentation-vs-point argument, so nothing is contradicted — just one supporting argument thread is thinner than the source.

## Not flagged (checked and confirmed legitimately deferred or out of scope)

- Encoder initialization (random vs. UNet-transfer) — correctly in Deferred, matches open item #9.
- Crop pixel parity (`map_coordinates` vs `cv2.resize`, align-corners convention) — correctly in Deferred, matches "Risques identifiés."
- RESIZE method parity (JAX vs PIL/LANCZOS) — correctly in Deferred, matches the 2026-07-15 "Nouveau" risk entry.
- Overlap/chevauchement as a known, accepted limitation — correctly in AD-10 and Deferred.
- Backbone+FPN deferral — correctly in AD-2 and Deferred, with the same reconsideration trigger ("petits avions distants... test réel") as the source.
- Migration of `tools/audit_dataset_detection.py` / `tools/boxes_process_manual_tkinter.py` — correctly in Deferred (AD-8 verification done, migration explicitly not required for this chantier).
- Deployment/environment (local + Colab, no CI/CD) — correctly in Deferred as inherited-unchanged.
- Minor training/inference downsampling-ratio nuance (~8.6× vs ~41×) — correctly in Deferred, matches the 2026-07-15 "Nuance mineure."
- `data_management.py`'s `'masks'` key breaking on the new format — implicitly resolved by AD-9's "new dedicated loader class" rule (avoids touching the existing `DetectionDataset` path), consistent with the memlog decision.
- `main.py`/`trainer.py` minor-impact items and debug-script items (`heatmap_generation.py`, `heatmap_contouring.py`, `reporting.py`) — source itself labels these "probablement mineur" / not yet decided, reasonable to omit from an invariants-focused spine.
- Diagram's `FROZEN` node removal (symmetric frozen-weights handling) — captured in the "État & transverse" Consistency Convention.
- valid_mask / box-coordinate surfacing in final output (the two 2026-07-15 diagram corrections) — both captured in AD-7's rule.
