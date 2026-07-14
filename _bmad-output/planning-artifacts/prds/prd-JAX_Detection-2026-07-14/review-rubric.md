# PRD Quality Review — Renommage de JAX_Detection en jax_supervised_training

## Overall verdict

This is a well-calibrated, appropriately lightweight PRD for a solo rename task — no persona/UJ theater, honest Non-Goals, and NFR-1/NFR-2 grounded in a real prior incident rather than boilerplate. It fails on its own terms in exactly one place, but that place matters: FR-4 claims an exhaustive, verifiable file enumeration ("les 6 fichiers de code identifiés... ne contiennent plus le nom actuel"), and verification against the actual repo shows the enumeration is not exhaustive — it misses at least one active operational script that is directly in FR-3's blast radius (the Google Drive rename), plus an unaddressed live planning artifact and a borderline docs file. This undermines Done-ness clarity and Scope honesty specifically, not the PRD's overall shape or thesis.

## Decision-readiness — adequate

The PRD states real decisions, not hedged considerations: no dual-name backward compatibility (FR-2), no fallback mechanism on missing rename (explicit "Hors scope" in FR-2, with NFR-2 defining the required failure mode instead), and no retroactive renaming of historical BMAD snapshots or this PRD's own folder (§4, with a `[NOTE FOR PM]` placed at the one genuinely awkward self-referential case rather than a safe checkpoint). NFR-1 and NFR-2 both cite a concrete precedent ("ce mode d'échec s'est déjà produit une fois sur ce projet") rather than asserting generic quality goals — this is a real trade-off surfaced, not smoothed over.

Where it falls short: FR-4's file enumeration reads as a closed, decided list ("Les 6 fichiers de code identifiés... les fichiers de docs/... sont mis à jour"), but verification shows the enumeration is incomplete (see Done-ness clarity below). A decision-maker reading FR-4 would believe the scope of "code actif et documentation" has been fully surveyed when it hasn't — that's a decision presented as closed that isn't actually closed.

### Findings
- **high** FR-4's file list is presented as exhaustive but is not (§3.1 FR-4) — `tools/process rclone GDrive and run collab.txt` (an active rclone sync script targeting `gdrive:JAX_Detection` and `/content/drive/MyDrive/JAX_Detection`) is neither in the 6-file code list nor in the docs/ list, yet it is exactly the kind of active execution path FR-3/NFR-2 are meant to protect. *Fix:* add this file explicitly to FR-4 (or fold it into FR-3, since it's Drive-path territory), and re-run the file survey against `tools/` and any other operational scripts before treating the enumeration as closed.

## Substance over theater — strong

No personas, no UJ scaffolding forced onto a single-operator project — §2 states directly "Pas de parcours utilisateur à documenter." NFR-1 and NFR-2 have product-specific verification methods ("Validé par exécution réelle, pas par lecture de code") instead of "must be reliable" boilerplate. The Vision (§1) is specific to this codebase's actual history (JAX_KEPLER, CIFAR10 proof points, two deferred retro mentions) and could not be swapped into another PRD unchanged. No findings — this dimension is clean.

## Strategic coherence — strong

The thesis is explicit and singular: the name should reflect genericity already proven, without doing the generalization work that would make that genericity complete. Scope logic follows the thesis cleanly — FR-1–4 are pure renaming, and §4 Non-Goals draws a sharp, justified line around the one place scope creep would be tempting (the `Kepler1DConvNet` hardcoding), explaining *why* it's deferred (Strategy+Factory pattern is "flexible but fragile," not a declarative format) rather than just asserting deferral. SM-3 functions as a genuine counter-metric — it validates that the historical-artifact non-goal wasn't silently violated, not just that the rename activity happened. No findings.

## Done-ness clarity — thin

FR-1 through FR-3 are unambiguous and independently testable: folder name, `git remote -v` output, redirect behavior, env var absence, Drive folder name, notebook-by-notebook checklist. This is the strongest part of the PRD.

FR-4 is where done-ness breaks down. It asserts a specific, closed set of files as the complete scope of "code actif et documentation," and that claim was checked against the repo:

- The 6 code files listed (`dataset_configs.py`, `inference_utils.py`, `reporting.py`, `bounding_boxes_with_classification_from_video_generation.py`, `tools/bounding_boxes_with_classification_from_images_generation.py`, `tools/kepler_dataset_tools.py`) match exactly what a repo-wide grep for `JAX_Detection`/`JAX_DETECTION` across `.py` files returns — accurate.
- The docs/ list (6 files) is almost accurate but misses `docs/project-scan-report.json`, which also contains `JAX_Detection` references (`display_name`, `root_path` fields) and sits inside `docs/`, so FR-4's own scope statement ("les fichiers de docs/... sont mis à jour") technically includes it, but the itemized list doesn't. It's plausible this file is meant to be treated like a dated snapshot (it's timestamped 2026-07-12, generated by a scan tool) rather than live docs — but the PRD doesn't make that call the way it does for `sprint-status.yaml` via `[ASSUMPTION]`.
- Outside code/docs, `tools/process rclone GDrive and run collab.txt` (not `.py`, so missed by a naive "code" search) references both `gdrive:JAX_Detection` and `/content/drive/MyDrive/JAX_Detection` — i.e., it is an active script that writes to the exact Drive path FR-3 renames. If left unrenamed, the next rclone sync silently populates a new/orphaned `JAX_Detection` Drive folder alongside the renamed one — a failure mode NFR-2 explicitly exists to prevent, but FR-4 doesn't cover this file, and FR-3 doesn't mention sync tooling at all.
- `_bmad-output/planning-artifacts/epics.md` also contains multiple `JAX_Detection` references and is a live (not dated-folder-scoped) planning artifact, structurally similar to `sprint-status.yaml` — but unlike `sprint-status.yaml`, it isn't mentioned anywhere in FR-4 or the Assumptions Index, so its in/out-of-scope status is simply undecided.

None of this changes the PRD's shape or thesis, but an engineer executing FR-4 literally would call the rename "done" while at least one active execution path (the rclone/Drive sync script) still points at the old name.

### Findings
- **high** FR-4 file enumeration incomplete for the rclone/Drive script (see Decision-readiness finding above — same underlying gap, done-ness angle: FR-4's stated verification condition "ne contiennent plus le nom actuel" is falsifiable against the current file list).
- **medium** `docs/project-scan-report.json` not decided (§3.1 FR-4) — contains the old name, sits in `docs/`, but isn't in the enumerated list, and isn't classified as historical-snapshot-like the way `sprint-status.yaml` was via `[ASSUMPTION]`. *Fix:* either add it to FR-4's list or add an explicit `[ASSUMPTION]`/Non-Goal classifying it as a dated scan artifact, consistent with how `sprint-status.yaml` was handled.
- **medium** `_bmad-output/planning-artifacts/epics.md` not addressed anywhere (§3.1 FR-4, §8 Assumptions Index) — contains several `JAX_Detection` references, is a live planning artifact (not inside a dated snapshot folder like the 2026-07-12 PRD/architecture directories that §4 explicitly excludes). *Fix:* add it to FR-4's scope or explicitly place it alongside the excluded historical artifacts in §4, with reasoning (it documents completed epics under the old name, arguably historical despite not living in a dated folder).

## Scope honesty — adequate

The Non-Goals section (§4) does real work: it explains *why* the Kepler generalization gap is out of scope (not just that it is), and it correctly anticipates the self-referential edge case of this PRD's own folder via `[NOTE FOR PM]`. The `[ASSUMPTION]` on `sprint-status.yaml` is a good example of the right move — flagging a genuinely ambiguous "live vs. historical" classification instead of silently picking one. Open-items density (1 assumption, 1 NOTE FOR PM, 0 open questions) is proportionate to a low-stakes solo PRD.

The gap is that this same "live vs. historical" classification judgment wasn't applied consistently — `sprint-status.yaml` got the `[ASSUMPTION]` treatment, but `epics.md` and `docs/project-scan-report.json`, which pose the identical classification question, got no treatment at all (see Done-ness findings above — same root cause, scope-honesty framing: these are omissions the reader is left to infer rather than omissions made explicit).

### Findings
- **medium** Inconsistent application of the "live vs. historical artifact" judgment call across near-identical cases (§3.1 FR-4, §8) — `sprint-status.yaml` is flagged as an assumption needing confirmation; `epics.md` and `docs/project-scan-report.json`, which raise the same question, are not mentioned at all. *Fix:* apply the same classification pass to all `_bmad-output/` and `docs/` artifacts containing the old name, not just the one that happened to be checked.

## Downstream usability — adequate (lighter weight, as expected)

This PRD explicitly feeds epics/stories directly (§0: "sert directement de base à la liste d'epics/stories") rather than an intervening UX/architecture layer, so this dimension matters but doesn't need the full glossary/cross-reference rigor of a chain-top PRD. FR/NFR/SM IDs are contiguous and unique (FR-1–4, NFR-1–2, SM-1–3), the one `[ASSUMPTION]` round-trips correctly to the Assumptions Index, and terminology ("dossier local," "dépôt distant," "notebook Colab actif") is used consistently across sections. No glossary section exists, but the domain vocabulary here is small enough (essentially: old name, new name, six named files, two Drive paths) that its absence isn't a practical drag on story generation — a story author working from FR-4 alone would still be misled by the enumeration gap described above, but that's a completeness problem, not a cross-reference/terminology problem.

## Shape fit — strong

Correctly shaped as a capability spec for a single-operator internal tool: no UJs, no stakeholder section, no multi-user journeys — and the PRD is explicit about why (§2: "Pas de parcours utilisateur à documenter — un seul opérateur"). Success Metrics (§7) are appropriately qualitative ("critères qualitatifs suffisent, pas de tableau de bord quantitatif") rather than forcing a dashboard onto a one-person rename. This is the right rigor level for the stated stakes — no over- or under-formalization to flag.

## Mechanical notes

- **Glossary drift**: none observed — "JAX_Detection"/"jax_supervised_training" and the file/path names are used identically everywhere they appear.
- **ID continuity**: FR-1–4, NFR-1–2, SM-1–3 are contiguous, unique, and every cross-reference resolves (e.g., "voir NFR-2" in FR-2, "Valide FR-1 à FR-4" in SM-1).
- **Assumptions Index roundtrip**: clean — the single inline `[ASSUMPTION: ...]` in FR-4 matches the single entry in §8 verbatim in substance.
- **Repo-fact accuracy** (brownfield check): FR-1's remote (`aobled/JAX_Detection` via `git@github.com:aobled/JAX_Detection.git`) and FR-2's env var read site (`dataset_configs.py`, sole occurrence) were both verified accurate against the live repo.
