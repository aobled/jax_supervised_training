# Rubric Walk — ARCHITECTURE-SPINE.md (JAX Single-Pass, 2026-07-15)

Reviewer: fresh/independent pass against the good-spine checklist. Read in full: the spine itself, the memlog it was distilled from (`.memlog.md`), the earlier working notes (`notes-jax-single-pass.md`), the inherited parent spine (`architecture-JAX_Detection-2026-07-12`), and the spine template. Cross-checked factual claims directly against `task_strategies.py`, `data_management.py`, `dataset_configs.py`, `inference_utils.py`, and a live JAX 0.6.2 interpreter.

## Verdict

Sound spine. All checkable factual claims verified true, the two brownfield-ratification claims hold up to direct code inspection, the inherited invariants are correctly carried and not contradicted, and the operational envelope is explicitly addressed (not silently dropped). One real structural defect (AD-id collision with the parent spine) and one minor omission worth a look; everything else is clean.

## Checklist walk

### 1. Fixes the real divergence points for the level below

Covers the load-bearing divergence points for both units it scopes:
- **JAX_DETECTOR training**: head architecture (AD-1), encoder scope (AD-2), loss targets/dataset script (AD-1, AD-9, Structural Seed), config placement (AD-8).
- **JAX Single-Pass inference composition**: crop mechanism (AD-3), canonical resolution + two-branch structure (AD-4), coordinate-frame symmetry (AD-5), train/infer boundary (AD-6), fixed-size output contract incl. explicit validity mask (AD-7), where the composition function lives and how it must NOT be modeled (AD-8).

No divergence point I could identify from the memlog's resolved discussion threads is missing from the spine. One item that *was* flagged in the working notes as a concrete implementation obstacle — `data_management.py:300` hard-requiring a `masks` key in `.npz` chunks, which would break as-is for the new heatmap+size format — is not named explicitly as a Rule in the spine, but AD-9's "new dedicated loader class, existing `DetectionDataset` untouched" resolves it by construction (verified: `DetectionDataset.create_tf_dataset` at line 300 does `data['masks']`, confirming the new class must be genuinely separate, not a branch — which is exactly what AD-9 mandates). Not a gap in practice, but the Rule text doesn't cite the reason as directly as the memlog does — a implementer reading only the spine won't know *why* a separate class is non-negotiable here specifically (vs. just "pattern consistency").

### 2. Every AD's Rule is enforceable

All 10 local ADs have falsifiable rules (named function signatures, named JAX primitives, a concrete output schema, a concrete file/line precedent). None read as vague aspiration. AD-7's rule ("valid_mask dérivé du score de détection... jamais déduit de la prédiction de classification") is a good example of a rule that actually prevents a specific, easy-to-make mistake (thresholding on softmax confidence instead of detection confidence).

Minor: AD-2 ("Backbone/FPN reporté... Ne reconsidérer que si la détection de petits avions distants s'avère concrètement insuffisante en test réel") has a rule whose trigger condition ("insuffisante") is not quantified — it's a scope-boundary rule (govern what NOT to build) rather than a build rule, so this is a softer bar than the others, but it is consistent with the parent's own AD-2/AD-2(local numbering aside) style of qualitative deferral language, so not a new problem this spine introduces.

### 3. Nothing under Deferred enables incompatible divergence

Walked all 7 Deferred items. Two (resize-method pixel parity, crop pixel parity) are explicitly flagged as **must-validate-before-training/production-use**, with a stated validation method (numeric comparison, not visual) — these are legitimate "cheap to defer, not free to skip" items, not divergence risks, because AD-3/AD-4/AD-5 already pin the *mechanism* (map_coordinates order=1, deterministic RESIZE/RESCALE); only the exact parity constant is open, which by definition cannot cause two builders to diverge on architecture, only on a shared, single-owner validation script's outcome. Encoder-init and Backbone/FPN and grid-based/YOLO deferrals are genuinely inert until reopened (no other AD depends on their outcome). Tools migration deferral is safe per AD-8 (old pipeline untouched, no symbol removed). The deployment/environment deferral is inherited-unchanged and reasoned (see §7 below), not a silent gap.

### 4. Named tech is verified-current; "no new external dependency" claim checked

Verified directly against the installed environment (`jax.__version__ == 0.6.2`):
- `jax.scipy.ndimage.map_coordinates` exists, `order=1` = linear/bilinear interpolation as claimed, `mode='constant'` default — matches AD-3's "interpolation bilinéaire" claim.
- `jax.lax.reduce_window` exists.
- `jax.lax.top_k` exists.

All three live in `jax`/`jax.scipy`/`jax.lax`, already a direct dependency of the project (imported throughout `inference_utils.py`, `task_strategies.py`, etc.) and deliberately excluded from `requirements.txt` (platform-specific, install already correct on Colab/local per that file's own comment). The "no new external dependency" claim is **true** — these are existing-package namespace additions, not new pip installs. Checklist item 4 fully satisfied.

### 5. Ratifies rather than contradicts the brownfield codebase

Directly verified both cited files:
- `task_strategies.py`: `ClassificationStrategy` (line 77) and `DetectionStrategy` (line 167) are indeed separate classes extending a common `TaskStrategy` ABC — no per-format branching inside one class.
- `data_management.py:429-464`: confirmed `if task_type in ["classification","kepler"]: ChunkManager(...) / elif task_type == "detection": DetectionDataset(...)` — a genuine dispatch-to-dedicated-class pattern, not a shared class with internal branches. Line range cited (429-463) matches the actual dispatcher body (429-464, off by one line at most, immaterial).

AD-9's ratification claim is accurate, not aspirational.

### 6. No new AD weakens or contradicts an inherited one

Checked each local AD (1-10) against each inherited parent AD (1,2,3,6,7,8) for substantive conflict — found none:
- Parent AD-1 (inference_utils.py single source) vs. local AD-8 (composition lives in inference_utils.py, reads configs without modifying them) — consistent, extends rather than weakens.
- Parent AD-3 (3-level checkpoint fallback, batch_stats reinit) — inherited row correctly extends this to the new JAX_DETECTOR checkpoint; no local AD relaxes it.
- Parent AD-6 (video throughput priority) — inherited row explicitly preserves the *principle* while acknowledging the underlying functions are replaced; local AD-6 ("training stays modular") is a different topic, no conflict.
- Parent AD-8 (tools/ consumer scope) — inherited row extends the verified consumer list with two more files (`tools/audit_dataset_detection.py`, `tools/boxes_process_manual_tkinter.py`) using the same verification method as the parent used; consistent methodology, not a weakening.

No contradiction found on content. See Finding 1 below for a structural (not content) issue in this same area.

### 7. Every feature-altitude dimension is decided/deferred/open

Swept the dimensions this altitude should own: data/dataset (AD-1, AD-6, AD-9, Structural Seed), model/training composition (AD-1, AD-2, AD-6, AD-9), inference composition (AD-3, AD-4, AD-5, AD-7, AD-8), naming/consistency (Consistency Conventions table), dependency stack (Stack section), consumer migration scope (AD-8 inherited + local verification, Structural Seed, Deferred), and — the one the checklist explicitly flags as easy to drop — **operational/environmental envelope**: explicitly present in Deferred ("Déploiement / environnement — hérité du parent sans changement : exécution locale + Colab (GPU/TPU), pas de CI/CD... Ce chantier n'introduit aucune nouvelle dimension opérationnelle"). This is a substantive claim, not a placeholder, and it's consistent with the memlog's own resource-budget conclusion (2026-07-15, "Ressources" section: params ~unchanged, only new cost is keeping the source image on-device instead of CPU RAM, negligible — reasoned, not hand-waved). No whole dimension found silently missing.

## Findings

### Finding 1 (moderate) — Local AD ids collide with inherited parent AD ids

The spine's own "Invariants & Rules" section numbers its local decisions AD-1 through AD-10. The "Inherited Invariants" table also carries forward the parent's AD-1, AD-2, AD-3, AD-6, AD-7, AD-8 **under their original ids**, per template instruction ("read-only, never renumbered"). The result: within this single document, "AD-1" refers to two unrelated rules depending on which section you're in — the parent's "`inference_utils.py` is the single source of inference helpers" (Inherited Invariants) vs. this spine's own "point-central detection head, not segmentation" (Invariants & Rules). Same collision on AD-2, AD-3, AD-6, AD-7, AD-8 — six of the ten local ids are ambiguous with an inherited id of different content.

This is not a content contradiction (checked, see §6 — none found) but a traceability/enforceability risk (checklist item 2): a future implementer or reviewer citing "AD-3" without qualifying "local" vs. "inherited" can genuinely point at the wrong rule (e.g., confusing the checkpoint-fallback rule with the map_coordinates crop rule). The template doesn't give explicit guidance for this doubly-nested case (the parent spine itself has no "Inherited Invariants" section to set precedent), but a namespaced or offset local numbering (e.g. starting local ids at AD-11, or prefixing inherited refs distinctly in prose) would remove the ambiguity at near-zero cost. Recommend renumbering local ADs to avoid overlap with inherited ids in any future revision of this spine, or in the next spine that inherits from this one.

### Finding 2 (low) — AD-9's rationale for "why a shared class would actually break" isn't carried into the Rule text

The concrete forcing function for AD-9 (that `DetectionDataset.create_tf_dataset` unconditionally reads `data['masks']` from the `.npz`, verified at `data_management.py:300`, and would hard-crash on heatmap+size chunks) lives only in the memlog/notes, not in the spine's AD-9 Rule or Prevents text. The Rule as written reads as a stylistic-consistency argument ("ratifies the established pattern") rather than a load-bearing one ("would crash otherwise"). Functionally harmless since the Rule's prescription (new dedicated class) is correct either way, but a downstream reader relying only on the spine (not the memlog) has weaker grounds to resist "just add an `if` branch for the new format" if the true reason (existing class would hard-fail on a missing key) isn't visible where the rule lives.

## Non-findings (checked and cleared)

- "No new external dependency" claim: verified true against a live jax 0.6.2 install — `map_coordinates`, `reduce_window`, `top_k` all exist in the namespaces cited.
- Brownfield ratification (task_strategies.py / data_management.py per-format dedicated-class pattern): verified true by direct file read.
- Canonical "1920×1080 grayscale" input assumption: consistent with existing `dataset_configs.py` (`FIGHTERJET_CLASSIFICATION` and `FIGHTERJET_DETECTION` both already `grayscale: True`), not a new unstated assumption.
- Operational/environmental envelope: explicitly addressed in Deferred, reasoned rather than silent.
- No AD-8 (`JAX_DETECTOR`/`build_single_pass_predict_fn`) is already present or inconsistently started in the codebase — confirmed absent from `dataset_configs.py` and `inference_utils.py`, so this is genuinely greenfield, matching the spine's framing.
