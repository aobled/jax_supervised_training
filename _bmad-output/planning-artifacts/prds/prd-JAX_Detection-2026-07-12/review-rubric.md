# PRD Quality Review — Refactor JAX_Detection — nettoyage code mort et duplications

## Overall verdict

This is a tight, well-grounded brownfield refactor PRD that earns its brevity: every FR names exact files/functions/configs, and spot-checks against the live repo (`dataset_configs.py`, `model_library.py`, `tokens/`, git history) confirmed the PRD's factual claims hold up, including one place where it's more accurate than its own source audit. The two real gaps are a performance NFR with no measurable bound and an implicit rather than explicit scoping of what this refactor is *not* doing (tests/CI, the deferred config restructuring). Nothing here blocks handoff to the architecture phase; OQ1 correctly identifies the one decision that genuinely needs to move downstream.

## Decision-readiness — strong

Trade-offs are named, not smoothed. Goal 1 states an explicit epistemic standard — non-regression "prouvée par comparaison, pas par observation" (§ Goals, item 1) — and applies it to all 6 duplicated-function files, not a subset. OQ1 is a genuinely open question (canonical behavior per divergent function) explicitly deferred to the architecture phase with a named workflow (`bmad-architecture`), not a rhetorical question answered in the next sentence. The Success Metrics section names counter-metrics (§ Success Metrics) so the refactor can't "win" by breaking genericity or performance to hit the dead-code counts.

No findings — this dimension does real work.

## Substance over theater — strong

The Vision statement (§ Vision / Problem Statement) is not swappable boilerplate: it names the exact bloat (`model_library.py`, "15 architectures de modèles mortes et 6 fonctions d'inférence dupliquées-et-divergentes"), the exact healthy core (`main.py`/`Trainer`/`TaskStrategy`), and the exact validation proof (`JAX_KEPLER`). No personas, no UJs — correctly absent for a single-operator hobby tool (see Shape fit). NFR1/NFR2 are concrete (named files, named pattern) rather than generic "must be scalable/modular" boilerplate.

### Findings
- **medium** NFR3 reads as adjective, not bound (§ Non-Functional Requirements) — "aucune régression de performance d'entraînement ou d'inférence" has no measurement method or tolerance, unlike Goal 1's rigorous diff-based methodology for functional non-regression. *Fix:* name a measurement (e.g., wall-clock time/epoch or inference latency/image, captured before/after, with a tolerance band accounting for run-to-run noise).

## Strategic coherence — strong

The thesis is explicit and singular: eliminate dead code and duplication without touching the healthy `main.py`/`Trainer`/`TaskStrategy` core, preserving genericity (proven by `JAX_KEPLER`) and modularity (Strategy/Factory/DI). F1–F3 map directly onto the three problem categories named in the Vision (duplicated inference functions, dead architectures/configs, orphaned files) — no "easy wins" tacked on outside that arc. Success Metrics measure the thesis directly (duplicate-function count, architecture count, config count, diff identity) rather than proxy/activity metrics. MVP scope kind is coherently "problem-solving" (debt elimination) throughout — no scope-kind drift into feature work.

No findings.

## Done-ness clarity — adequate

Most FRs are unusually concrete for a PRD of this size and verified accurate against the live repo:
- FR4/FR5 name exact configs and architectures kept vs. removed; independently checked against `dataset_configs.py` and `model_library.py` — `FIGHTERJET_CLASSIFICATION` → `sophisticated_cnn_128_plus`, `FIGHTERJET_DETECTION` → `aircraft_detector_unet` (with `aircraft_detector_miniunet` commented, exactly as FR5 states), `JAX_KEPLER` → `kepler_1d_cnn`. All correct.
- FR7/FR8 (`train_detection.py`, `tokens/`) confirmed isolated — no other file in the repo imports from either.
- Goal 1's baseline/diff methodology is a real acceptance criterion, not an aspiration.

### Findings
- **medium** NFR3 has no testable consequence (see also Substance over theater finding above) — an engineer can't tell what "done" means for "no performance regression": no benchmark script, no metric, no tolerance. This is the one NFR in the set that doesn't carry its own acceptance criterion the way NFR1/NFR2/NFR4 do (each of those names a concrete verification: `JAX_KEPLER` still runs, Strategy/Factory pattern intact, git history retains deletions).
- **low** Goal 1's "un petit set d'images fixes" leaves the baseline sample size unspecified (§ Goals, item 1). Minor — reasonable to leave to architecture/implementation, but worth a number (e.g., "10-20 images") so the baseline itself is reproducible.
- **low** Goal 4's "Un audit global de code (agent dédié)" doesn't name which tool/process re-runs the audit or what "confirms absence of residual dead code" means operationally (same scan as `bmad-document-project`? a different pass?). Low stakes given hobby/solo context, but it's the one goal without a stated verification method.

## Scope honesty — adequate

The addendum (`addendum.md`) is a good model of explicit de-scoping: it names the deferred idea (per-config-file restructuring), states why it's out of scope for this cycle, and tells a future reader where to resume from ("repartir de `dataset_configs.py` dans son état post-purge"). That's exactly what a `[NON-GOAL for MVP]` callout is supposed to do.

### Findings
- **medium** No explicit non-goal callout for testing/CI (§ whole PRD) — `architecture.md` (the PRD's own source doc) explicitly flags "Pas de CI/CD, pas de packaging, pas de suite de tests" as known gaps "notées pour le backlog, en plus du nettoyage de code" (architecture.md, § Développement / Déploiement). This refactor deletes 18 of 22 model architectures and merges 6 divergent inference functions — exactly the kind of change an automated test suite would normally guard — yet the PRD relies solely on a manual baseline-diff (Goal 1) and never states that adding tests/CI is deliberately out of scope for this cycle. The omission is defensible for a hobby project, but it should be stated rather than left for the reader to infer from a different document.
- **low** Open-item count is thin (1 Open Question, 0 `[ASSUMPTION]` tags, 0 `[NOTE FOR PM]` tags) for a PRD whose own source audit flagged a real nuance requiring verification (pkl-to-architecture dependency, audit § 1 "⚠️ Nuance à trancher"). This is not a defect — the PRD converted that nuance into an actionable FR6 rather than leaving it as an open question, which is arguably better — but it's worth noting for calibration: the near-total absence of hedge tags is consistent with a Coaching-path PRD confirmed section-by-section live with the user (nothing left un-vetted), not with under-elicitation.

## Downstream usability — adequate

OQ1 explicitly hands the one open decision to the architecture phase ("à décider explicitement en phase Architecture (`bmad-architecture`), fonction par fonction, avec justification") — a clean, unambiguous handoff. FR/NFR/Goal IDs are contiguous and unique (FR1–FR8, NFR1–NFR4, Goals 1–5, OQ1). The 6-file list in Goal 1 and FR2 is repeated verbatim in the same order in both places — a real consistency check that passes and matters for a downstream reader extracting either section alone.

### Findings
- **medium** `addendum.md` is never referenced from `prd.md` — no "see addendum" pointer anywhere in the main document. A downstream reader (e.g., the architecture-phase agent, or a future contributor) who opens only `prd.md` has no way to discover that config restructuring was raised and explicitly deferred. *Fix:* add a one-line pointer in `prd.md` (e.g., near NFR1/Goals, or a short "Deferred" note) linking to `addendum.md`.
- **low** No Glossary section. Given the PRD's terms are almost entirely code identifiers (file/function/config names) rather than ambiguous domain nouns, this is lower-stakes than usual, but a Glossary would still help anchor terms like "architecture" (= entry in `MODELS`) vs. "config" (= entry in `DATASET_CONFIGS`) for a reader unfamiliar with the codebase.

## Shape fit — strong

Correctly shaped for a hobby/solo brownfield refactor: no personas, no UJs (rubric explicitly treats these as optional overhead for single-operator tools), capability-spec structure (Features → FRs) instead of user-journey structure. Brownfield accuracy is the strongest part of this PRD — every checkable claim was independently verified against the live repository and held up:
- `model_library.py` has exactly 22 entries in `MODELS` (confirmed by direct read, lines 2450-2473).
- `dataset_configs.py` has exactly 7 configs (confirmed), matching the PRD's "réduit de 7 à 3" — note this is worth flagging in Mechanical notes below, since the PRD's own source audit says 8.
- `tokens/` and `train_detection.py` are confirmed unreferenced by any other file in the repo, supporting FR7/FR8's "vestige"/"jamais raccordé" claims.
- The two currently-versioned checkpoints (`best_model.pkl`, `best_model_detection.pkl`, named in FR6) are indeed the only `.pkl` files tracked in git.

No findings — this is the PRD's strongest dimension.

## Mechanical notes

- **Source-doc discrepancy, not a PRD defect**: `dead-code-and-duplication-audit.md` § 1 states "les 8 configs présentes dans `dataset_configs.py`," but the file actually contains 7 (`FIGHTERJET_CLASSIFICATION`, `FIGHTERJET_VIT`, `FIGHTERJET_HYBRID_VIT`, `FIGHTERJET_LETTERBOX`, `FIGHTERJET_DETECTION`, `FIGHTERJET_DETECTION_SOPHISTICATED`, `JAX_KEPLER` — confirmed by direct read). The PRD's own Success Metrics ("`dataset_configs.py` réduit de 7 à 3 configs") and FR4 (4 removed + 3 kept = 7) use the correct count and silently diverge from their cited source without a callout. Recommend correcting the audit doc for future traceability, since a reader cross-checking PRD against audit will trip on the mismatch and not know which is right (it's the PRD).
- ID continuity: FR1–FR8, NFR1–NFR4, Goals 1–5, OQ1 — all contiguous, unique, no gaps or duplicates.
- No `[ASSUMPTION]` tags present, so no Assumptions Index roundtrip to check — consistent with the Coaching-path production method (each section confirmed live with the user rather than inferred).
- No UJs present — expected and correct for this shape (see Shape fit); no protagonist-naming check applicable.
- Glossary: none present (see Downstream usability finding — low severity given the code-identifier-heavy vocabulary).
- Cross-reference gap: `addendum.md` ↔ `prd.md` is one-directional (addendum references the PRD's config-purge outcome; PRD never references the addendum). See Downstream usability finding.
