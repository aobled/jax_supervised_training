---
title: Adversarial Review — ARCHITECTURE-SPINE.md (pairwise-divergence construction)
reviewed_artifact: ../ARCHITECTURE-SPINE.md
lens: >
  Construct two units one level down (one per-file migration story) that each obey every AD to
  the letter yet still build incompatibly — clashing shared-data shapes, two owners of one
  entity, conflicting state-mutation paths. Every pair found is a hole to close with a new or
  tightened AD.
method: >
  Read AD-1..AD-5, Consistency Conventions, Structural Seed against the actual bodies of every
  function AD-1 names, across all 5 in-scope consumer files plus every file in the repo that
  currently imports from them (grep for def/call sites of the 7 AD-1 functions and their
  supporting helpers across *.py and tools/*.py). Then, for each AD, tried to write two
  independently-plausible implementer decisions that both satisfy the AD's literal Rule text
  and checked whether the resulting artifacts (inference_utils.py content, call sites,
  constants) would actually compose.
created: '2026-07-12'
---

# Adversarial Review — ARCHITECTURE-SPINE.md

## Overall verdict

The spine's five ADs are well-targeted at the divergences the audit already knew about
(`load_detection_model` robustness, dead-model fallback, `predict_crop`'s two signatures,
`benchmark.py` deletion), and two prior reviews in this folder (`review-rubric.md`,
`review-reality-check.md`) have already surfaced the biggest *behavioral* gap (the
`decode_segmentation_and_detect(_batch)` divergence) and the biggest *inventory* gap
(`load_classification_model`/`build_predict_fn` duplicating `load_jax_model` under different
names). This review does not re-litigate those. Attacking the spine specifically for **pairs of
independently-working, letter-compliant implementers landing on incompatible artifacts** surfaces
a different class of hole: the spine constrains the *public* shape of 7 functions but says nothing
about who owns writing the *one shared file* all 5 stories must edit, is blind to two real
consumers that reach into a to-be-refactored file's namespace from outside the stated scope, and
leaves at least one throughput-critical constant with no assigned home. All are concrete,
evidenced against current source, not speculative.

## Finding counts
- Critical: 1
- High: 2
- Medium: 0
- Low: 2

---

## Finding 1 — Severity: Critical
### `inference_utils.py` has no assigned author across 5 parallel stories, so AD-1/AD-2 constrain the public signature but not the shared file's actual contents

**Location:** AD-1 (Rule, lines 28-32), AD-2 (Rule, lines 34-38), Structural Seed (lines 92-108)

**The pair:**

- **Story B** (`bounding_boxes_with_classification_from_video_generation.py`, one of the 5
  consumer files) needs `predict_crop` to exist in `inference_utils.py` to satisfy AD-1. Its
  entire codebase already works in terms of pre-built, `@jax.jit`-wrapped `predict_fn` objects
  (`build_clf_predict_fn`, line 186; `build_det_predict_fn`, line 179) — that's the whole point of
  AD-2's real-time branch. Following AD-2's own suggestion ("`predict_crop` peut être implémenté
  en interne comme un appel à `predict_crops_batch`"), this implementer writes `predict_crop`
  inside `inference_utils.py` to build an ad hoc `jax.jit(lambda x: model.apply(variables, x,
  training=False))` closure on every call and delegate to `predict_crops_batch([crop_img],
  jit_fn, mean, std, config)`.
- **Story C** (`tools/bounding_boxes_with_classification_from_images_generation.py`, a different
  one of the 5 consumer files) needs the *same* `predict_crop` to exist in the *same*
  `inference_utils.py`. Its existing, already-correct implementation
  (`tools/bounding_boxes_with_classification_from_images_generation.py:128-166`) calls
  `model.apply(variables, jnp.array(img_jax), training=False)` directly — no JIT, no batching —
  and this implementer, working from their own file with zero JIT-pipeline context, just moves
  that body verbatim into `inference_utils.py`.

Both stories are AD-1-compliant (single file, no local redefinition) and AD-2-compliant (exact
signature `predict_crop(crop_img, model, variables, mean, std, config)`, matching AD-2's Rule
text character-for-character). Both are plausible, uncoerced readings of AD-2's optional "peut
être implémenté... comme" clause. But they produce **two different, incompatible function bodies
for the same name in the same file** — the first re-JITs on every single call (a severe perf
regression if it ever executes on a hot path, exactly what AD-2 exists to prevent), the second is
correct-but-non-batched. Nothing in the spine says which story is authoritative for
`inference_utils.py`'s actual byte-for-byte content, whether the module is created once by a
0th/setup step before the 5 file-stories start, or how a second story's edit to an
already-existing `inference_utils.py::predict_crop` should be reconciled with the first story's.
Since the repo has no CI/tests (confirmed by the PRD's Non-Goals, cited in `review-rubric.md`
Finding 5) and no defined merge/review gate is named in this spine, whichever story lands second
either silently overwrites the first's implementation choice or produces a merge conflict with no
tie-breaking rule in the AD.

This is the general case: **every** one of the 7 AD-1 functions is written into a file that *no*
individual story "owns" — the Structural Seed models `inference_utils.py` as a leaf dependency of
all 5 consumer files but never assigns it to one of them, or to a distinct upstream story.

**Recommendation:** Add an AD (or a Dependencies note) that explicitly designates one story
(e.g., "create `inference_utils.py` with all 7 functions and their canonical bodies, ratified
against the existing images_generation.py/video_generation.py precedent, is Story 0 — it must land
and be frozen before any of the 5 consumer-file stories start") rather than letting 5 stories
co-author the same new file independently. AD-2 should also resolve its own optional clause: pick
one of (a) `predict_crop` always builds a fresh JIT wrapper (accept the perf cost, since it's the
"simple case" path by AD-2's own framing) or (b) `predict_crop` never touches `predict_crops_batch`
and stays a bare `model.apply` call (matching the `images_generation.py` precedent already in
production) — "peut être implémenté... comme" is not a decision, it's a permission slip for two
implementers to diverge.

---

## Finding 2 — Severity: Critical
### The spine's scope and dependency graph don't know about two files that already import AD-1's functions from a to-be-refactored file's namespace

**Location:** `Dépendances` Mermaid graph (lines 58-82), Structural Seed (lines 92-108), scope
line 7 ("Mutualisation ... du PRD refactor")

**The pair:**

- **Story B** (`bounding_boxes_with_classification_from_video_generation.py`) satisfies AD-1 by
  replacing its local `def load_detection_model`, `def load_jax_model`, `def get_iou`, `def
  predict_crops_batch` with `from inference_utils import load_detection_model, load_jax_model,
  get_iou, predict_crops_batch, non_max_suppression, predict_crop, _preprocess_crop_to_hwc` at
  module scope — importing only the subset it still calls directly, which is idiomatic Python and
  not forbidden by AD-1's text ("Tout fichier qui en a besoin importe").
- **The un-scoped, un-storied consumers** `tools/audit_dataset_detection.py:16-18` and
  `tools/boxes_process_manual_tkinter.py:1068-1070,1094-1096` — confirmed by direct read — do
  `from bounding_boxes_with_classification_from_video_generation import load_detection_model,
  decode_segmentation_and_detect_batch, get_iou` and `from
  bounding_boxes_with_classification_from_video_generation import load_detection_model,
  load_jax_model` / `... import decode_segmentation_and_detect_batch, predict_crops_batch,
  get_iou` respectively — i.e. they reach directly into `video_generation.py`'s module namespace
  for exactly the names AD-1 is migrating. Neither file appears anywhere in the spine's Mermaid
  graph, Structural Seed, or scope statement; both are entirely invisible to this refactor.

If Story B's implementer imports `load_detection_model`/`get_iou`/`load_jax_model` etc. at module
level (Python re-binds the name locally, so `from video_generation import load_detection_model`
by the two outside files would keep working) — the two outside files are safe *by accident*. But
nothing requires that specific import style, and it is not the only AD-1-compliant one: an
implementer could equally write `import inference_utils` and call `inference_utils.load_detection_
model(...)` everywhere inside `video_generation.py` (also fully AD-1-compliant — "importe" doesn't
mandate `from X import Y`), which creates **no** module-level `load_detection_model` name in
`video_generation.py` at all, and both outside files break with a hard `ImportError` the next time
anyone runs them. There is no test suite to catch this (per PRD Non-Goals), so it would ship
silently broken until someone manually runs `tools/boxes_process_manual_tkinter.py` or
`tools/audit_dataset_detection.py`.

This is precisely "two units one level down, each individually correct against the spec they were
each given, that build incompatibly" — Story B's spec is AD-1 over the 5 named files; the two
outside files have no spec at all in this cycle, because the spine doesn't know they depend on the
file being changed.

**Recommendation:** Either (a) expand the Mermaid graph and scope to acknowledge
`tools/audit_dataset_detection.py` and `tools/boxes_process_manual_tkinter.py` as transitive
consumers and add a Rule/AD requiring their import statements be repointed at `inference_utils.py`
in the same cycle, or (b) if genuinely out of scope, add an explicit AD stating "`video_generation.
py` must re-export all 7 AD-1 names as bare module-level bindings (`from inference_utils import
*` or an explicit list) specifically to preserve today's incidental external-import compatibility,
until FR-scope is widened to migrate those two files too." Silence, as written, leaves this to
chance.

---

## Finding 3 — Severity: High
### `CLF_BATCH_SIZE`/`BATCH_SIZE` — the constant that makes AD-2's real-time branch actually fast — isn't part of `predict_crops_batch`'s signature or named anywhere in the spine

**Location:** AD-2 (Rule, lines 34-38); reality in
`bounding_boxes_with_classification_from_video_generation.py:58-59,214-248,169-176`

**The pair:**

- **Story B** (`video_generation.py`) needs `predict_crops_batch` to keep its current behavior:
  its inner loop chunks `crop_imgs` in slices of `CLF_BATCH_SIZE` (= 32, a private module-level
  constant, line 59) and pads each chunk to exactly that size via `_pad_batch_np` (lines 169-176)
  specifically *to avoid cuDNN recompilation* — this fixed-shape padding is the actual mechanism
  that protects the "ne pas dégrader le débit temps réel" outcome AD-2 exists to prevent. AD-2's
  own signature, `predict_crops_batch(crops, predict_fn, mean, std, config)`, has no batch-size
  parameter — `CLF_BATCH_SIZE` is closed over as a module global. When this function moves into
  `inference_utils.py`, Story B's implementer, following the literal signature, hardcodes `32` (or
  a new `_CLF_BATCH_SIZE = 32` constant) directly inside `inference_utils.py`.
- **Story via the invisible consumer** `tools/boxes_process_manual_tkinter.py:1130-1132` already
  calls `predict_crops_batch(crop_imgs, self.clf_predict_fn, self.dataset_mean, self.dataset_std,
  self.clf_config)` today, but its `clf_predict_fn` (line 1083: `jax.jit(lambda x:
  clf_model.apply(clf_vars, x, training=False))`) is built ad hoc, with **no warmup call** and no
  awareness that `predict_crops_batch` will pad every batch to 32 regardless of how many crops a
  single manually-annotated image actually produced (typically far fewer than 32). This caller
  never had — and never asked for — the real-time video pipeline's fixed-batch-32 assumption; it
  was written against a version of `predict_crops_batch` that happened to also work for its
  smaller, irregular batches, but the *reason* it works (padding + a pre-warmed JIT cache) is a
  video-pipeline-specific optimization now silently forced onto every caller everywhere the
  moment the constant becomes a fixed, ownerless module global inside `inference_utils.py`.

Both "stories" (the literal FR2 story for `video_generation.py`, and the pre-existing, un-migrated
caller in the tkinter tool — see Finding 2 on why it's invisible to begin with) are compliant with
AD-2's stated signature. Neither is wrong. But the AD never decided whether the chunk/pad size is
(a) a hardcoded constant inside `inference_utils.py` (implicitly binding every future caller to
the video pipeline's GPU-memory-driven batch size of 32), (b) a new keyword parameter with a
default (which changes AD-2's signature from what the Rule text literally states), or (c) left as
an external constant callers must still supply themselves (reintroducing the exact duplication
this refactor exists to remove). Three different, individually reasonable resolutions, zero
guidance on which one is canonical.

**Recommendation:** Tighten AD-2 to state explicitly: `predict_crops_batch`'s chunk/pad size is
[a hardcoded `_CLF_BATCH_SIZE` constant private to `inference_utils.py`, chosen from
`video_generation.py`'s existing `32`] **or** [an explicit `batch_size` parameter with default
`32`] — pick one, and note that callers with irregular/small batch counts (image-tooling,
audit scripts) will still be padded to that size, which is an accepted, named trade-off rather
than an accidental one.

---

## Finding 4 — Severity: Low
### `DETECTION_IMAGE_SIZE` — same ownerless-constant pattern as Finding 3, currently dormant

**Location:** AD-3 (Rule, lines 40-44); reality in
`bounding_boxes_with_classification_from_video_generation.py:38,307` and
`tools/bounding_boxes_with_classification_from_images_generation.py:36,217`

`load_detection_model`'s batch_stats-reinit fallback path (the one AD-3 specifically canonizes)
builds a dummy input via `target_size = config_model.get("image_size", DETECTION_IMAGE_SIZE)` —
`DETECTION_IMAGE_SIZE` is a private module constant, independently defined and currently
identical (`(224, 224)`) in both `video_generation.py:38` and
`tools/bounding_boxes_with_classification_from_images_generation.py:36`. AD-3 locks the *behavior*
(3-level path fallback + batch_stats reinit) but says nothing about this constant's new home once
`load_detection_model` becomes single-sourced. Two implementers picking different homes for it
(hardcoded literal vs. new parameter vs. left importable from one of the consumer files, creating
a reverse dependency edge the Mermaid graph forbids) is low-risk *today* only because both current
copies happen to agree — a coincidence, not a guarantee, and the fallback path itself only fires
when `batch_stats` is genuinely absent from a checkpoint, so a divergence here would surface late
and rarely.

**Recommendation:** Fold this into the same tightened AD-2/AD-3 note as Finding 3: name
`DETECTION_IMAGE_SIZE`'s new home explicitly (a private constant inside `inference_utils.py`,
`(224, 224)`, matching both existing copies) so it isn't left to two independent guesses.

---

## Finding 5 — Severity: Low
### AD-2's own Rule text names a parameter that doesn't match the code it claims to ratify

**Location:** AD-2 (Rule, line 38); reality in
`bounding_boxes_with_classification_from_video_generation.py:214`

AD-2's Rule text writes the signature as `predict_crops_batch(crops, predict_fn, mean, std,
config)`. The only extant implementation — the brownfield precedent AD-2 says it's ratifying —
is `def predict_crops_batch(crop_imgs, predict_fn, mean, std, config):`
(`video_generation.py:214`). This is a trivial mismatch in isolation, but it is exactly the kind
of gap that produces incompatible call sites: an implementer who keyword-calls the migrated
function using the spine's literal parameter name (`predict_crops_batch(crops=my_list, ...)`,
reasonable if they're implementing from the spine document rather than reading
`video_generation.py` first) gets a `TypeError` against the actual code (`crop_imgs`), while an
implementer who reads the source first uses `crop_imgs=`. Both are "AD-2 compliant" under a
literal-positional reading; only one is compliant if either implementer chooses to call by
keyword — plausible, since `non_max_suppression`'s existing call sites already mix positional and
keyword style (`tools/bounding_boxes_with_classification_from_images_generation.py:435`:
`non_max_suppression(detections, iou_threshold=NMS_THRESHOLD)`).

**Recommendation:** Make AD-2's Rule text match the actual parameter name (`crop_imgs`), or state
explicitly that the parameter is renamed to `crops` as part of the migration (in which case say so,
since it's a rename to the brownfield precedent, not a ratification of it).

---

## Note on findings already covered elsewhere in this folder

This review intentionally does not restate `review-rubric.md` Finding 1
(`decode_segmentation_and_detect` vs `_batch` behavioral divergence, High) or
`review-reality-check.md` Finding M2 (`load_classification_model`/`build_predict_fn` duplicating
`load_jax_model` under different names) — both are real, both would also qualify under this
review's "two letter-compliant units, incompatible result" lens, and both should be read alongside
the findings above before treating AD-1 as implementation-ready.
