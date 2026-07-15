# Web-Verify Review — ARCHITECTURE-SPINE.md (JAX Single-Pass, 2026-07-15)

Mandate: verify that committed decisions were web-researched or reality-checked rather than
asserted from training data — specifically the three JAX APIs named load-bearing in AD-1, AD-3,
and the Stack section: `jax.lax.reduce_window`, `jax.lax.top_k`, `jax.scipy.ndimage.map_coordinates`.

## Method

1. Checked the installed environment: `jax==0.6.2` (confirmed via `python3 -c "import jax;
   print(jax.__version__)"`). `requirements.txt` deliberately excludes `jax`/`jaxlib` (pinned to
   whatever the local conda env / Colab already has, per an explicit comment dated 2026-07-12) —
   so 0.6.2 is a representative, current version for this project, not a stale pin.
2. Searched the repo for prior usage of the three APIs:
   `grep -rn "reduce_window|top_k|map_coordinates" --include=*.py` → **zero matches**. None of the
   three APIs are in active use anywhere in the codebase today. The spine's Stack-section claim
   ("no new external dependency — these are already part of JAX already used by the project") is
   about the *library* (`jax` itself is an existing dependency), not about prior call-sites of
   these specific functions — read that way the claim is accurate, but it should not be mistaken
   for "reality-checked via existing usage," because there is none.
3. Executed all three APIs directly against the installed jax 0.6.2 in a throwaway script
   (top_k, reduce_window max-pool, map_coordinates order=1 with `jax.grad` through both the image
   values and the coordinates). All three behaved exactly as the spine assumes:
   - `jax.lax.top_k(x, k)` returns `(values, indices)`, stable, sorted descending — matches AD-1's
     "Top-K=20" usage.
   - `jax.lax.reduce_window(heat, -inf, lax.max, window_dimensions=..., padding='SAME')` performs
     the max-pool used for local-maxima/peak extraction — matches AD-1.
   - `jax.scipy.ndimage.map_coordinates(img, coords, order=1)` is differentiable both w.r.t. the
     image (`jax.grad` w.r.t. `img` returned a finite, non-zero gradient) **and w.r.t. the
     coordinates themselves** (`jax.grad` w.r.t. the coordinate arrays also returned finite
     non-zero values) — this directly confirms AD-3's premise that the crop is a genuinely
     differentiable, pure function of already-known box coordinates, unlike a cv2/PIL crop.
4. Read the official current docstrings (`docs.jax.dev/en/latest`) for all three functions to
   check for gotchas.
5. Web-searched for deprecation/removal signals and known coordinate-convention issues.
6. Cross-checked `notes-jax-single-pass.md` (the primary source behind the spine) to see whether
   the spine's technical claims about these APIs were present in the exploration log, i.e.
   reasoned about before being distilled into the spine, vs. invented at distillation time.

## Findings

### 1. `map_coordinates` has a documented coordinate-convention discrepancy from SciPy — already correctly flagged as Deferred, and now independently confirmed real (informational, not a gap)

The current JAX docstring for `jax.scipy.ndimage.map_coordinates` states explicitly: *"Interpolation
near boundaries differs from the scipy function, because JAX fixed an outstanding bug; see
https://github.com/jax-ml/jax/issues/11097... This function interprets the `mode` argument as
documented by SciPy, but not as implemented by SciPy."* There is a real history of related bugs/
discrepancies (issues #11097 mode='mirror' mismatch, #14819 wrong extrapolation, #5687 mode=
'constant' not properly applied).

The spine's own Deferred section already anticipates exactly this class of problem: *"Parité pixel
du `CROP` (`map_coordinates` vs `cv2.resize`)... convention d'alignement pixel (bord vs centre) à
vérifier précisément, pas juste visuellement."* `notes-jax-single-pass.md:166` records the same
caution dated 2026-07-14 ("le classique problème 'align corners' qui piège souvent les portages
entre bibliothèques"). This is a case where the spine's caution was well-placed and is now
externally corroborated — no correction needed, just confirming the Deferred item is grounded in
a real, documented risk rather than generic hedging.

### 2. `jax.scipy.ndimage.map_coordinates` is flagged for possible future relocation out of JAX core (moderate — not checked by the spine, worth a Deferred note)

JAX's own design document, JEP 18137 ("Scope of JAX NumPy & SciPy Wrappers",
https://docs.jax.dev/en/latest/jep/18137-numpy-scipy-scope.html), concludes that `scipy.ndimage`
should be considered **out-of-scope for JAX core**, and explicitly names `map_coordinates` as a
candidate to move to `dm-pix` or another package. This is a live design decision on JAX's roadmap,
not yet enacted (the function is present and unchanged in the current 0.6.2 install and in the
latest published docs), but it directly concerns the one primitive AD-3 makes structurally
load-bearing for the entire crop step, and the Stack section's "no new external dependency" framing
implicitly depends on this function staying in JAX core. Neither the spine nor
`notes-jax-single-pass.md` shows any awareness of this JEP. Nothing needs to change today, but
this is exactly the kind of fact a web check catches that training-data recall would not (the JEP
and its conclusion are dated after most training cutoffs a base model would rely on). Recommend a
one-line Deferred note: watch for `map_coordinates` relocation out of `jax.scipy.ndimage` in future
JAX releases; if it moves, either pin the JAX version or add `dm-pix` as a real new dependency
(which would then need its own review).

### 3. `jax.lax.reduce_window` and `jax.lax.top_k` — no issues found

Both are stable, long-standing `jax.lax` primitives (thin wraps over XLA `ReduceWindowWithGeneralPadding`
and XLA TopK respectively), present unchanged in the current docs, with no deprecation signal, no
open correctness issues found, and no ambiguity of the kind that afflicts `map_coordinates`'s
boundary/mode handling. Local execution against jax 0.6.2 matches the documented behavior exactly
(`top_k` returns sorted-descending `(values, indices)`, stable on ties; `reduce_window` with
`lax.max` and `-inf` init performs the max-pool peak-extraction AD-1 requires). The spine's use of
these two is sound and requires no correction.

### 4. Technical claims trace back to the exploration notes, not invented at distillation time (informational)

`notes-jax-single-pass.md` (lines 36, 84, 94, 110, 134, 137, 166, 209, 221) shows the three APIs
were identified and reasoned about during the 2026-07-14 exploration session, including the
align-corners caution and the "differentiable w.r.t. pixel values and coordinates" claim for
`map_coordinates` — both independently confirmed correct by direct execution in this review (see
Method §3). This is reality-checking by mechanism (the person doing the exploration evidently knew
or looked up the JAX API surface, and the claims hold up empirically), even though no citation/URL
trail was recorded — worth noting as a gap in *evidence*, not a gap in *correctness*.

## Not flagged

- JAX version currency: 0.6.2, installed and intentionally unpinned per `requirements.txt`'s own
  rationale (matches whatever Colab/local CUDA build is already working) — appropriate for this
  project, not stale.
- No existing repo usage of the three APIs was found, which the spine does not claim either
  (its "already used by the project" language refers to the `jax` library as a whole, which is
  accurate).

## Overall verdict

The three named JAX APIs exist, behave as described, and are correctly used in the spine's design
— confirmed by direct execution against the project's actual installed JAX version, not just
documentation reading. The spine's own Deferred section already correctly anticipates the one real
gotcha (`map_coordinates` boundary/align-corners convention) with a real basis in JAX's issue
history. The one gap this review adds: no awareness in the spine or its source notes of JEP 18137's
proposal to relocate `map_coordinates` out of JAX core — low urgency (not enacted), but it is
exactly the kind of drift a web check surfaces that reasoning from a fixed knowledge cutoff would
miss, and it bears directly on AD-3 and the Stack section's "no new dependency" claim.
