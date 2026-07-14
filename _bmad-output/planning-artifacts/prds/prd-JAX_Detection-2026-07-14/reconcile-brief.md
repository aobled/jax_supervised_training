---
title: "Reconciliation — brief vs PRD (Renommage JAX_Detection → jax_supervised_training)"
created: 2026-07-14
---

# Reconciliation: brief.md vs prd.md

**Inputs:**
- Source brief: `_bmad-output/planning-artifacts/briefs/brief-JAX_Detection-2026-07-14/brief.md`
- Derived PRD: `_bmad-output/planning-artifacts/prds/prd-JAX_Detection-2026-07-14/prd.md`

## Verdict

The PRD captures all factual/scope content from the brief faithfully: the retained name, the in-scope/out-of-scope surfaces, the six code files and docs to update, the silent-failure risk (translated into NFR-2), and the three success criteria (translated into SM-1/SM-2/SM-3). No FR/NFR is missing a corresponding brief clause, and no brief clause is contradicted.

The gaps below are all on the qualitative side — rationale, tone, and process nuance that the FR/NFR structure compresses or drops rather than translates.

## Gaps

1. **Execution-checklist framing softened.** The brief's risk section gives an explicit process instruction — "à traiter comme item de checklist explicite en exécution : vérifier chaque notebook actif avant/après le renommage, pas seulement le code du repo" — but the PRD (FR-2/FR-3) reduces this to "dans le même geste que le renommage," losing the explicit before/after verification-checklist framing that the brief calls out precisely because this failure mode already bit the project once.

2. **Architectural rationale behind the Kepler gap is dropped.** The brief explains *why* the current cross-domain genericity holds despite the hardcoded `Kepler1DConvNet` — "la généricité actuelle repose sur la souplesse du pattern Strategy + Factory, pas sur une configuration déclarative du format de données" — but the PRD's Non-Goals (§4) restates only the symptom (hardcoded 1D conv, no declarative input/output format) without this causal explanation of why it's fragile.

3. **Deeper motivation for renaming "now" is narrowed.** The brief frames the rename as more than cosmetic — it "acte publiquement (pour l'utilisateur lui-même) la vocation configurable du projet" — while the PRD's Vision (§1) states only that the name no longer reflects what the code does, dropping the self-directed symbolic-commitment framing.

4. **Deferral urgency/tone is flattened.** The brief's closing note on the topic having been deferred across two retrospectives — "signe qu'il ne se traitera pas tout seul" — carries a self-aware, slightly wry urgency that the PRD's Vision restates as neutral status reporting ("le sujet a été noté et reporté... sans être traité"), losing the implicit push toward finally acting.

None of these are blocking: they are rationale/tone nuances that don't change scope, FRs, or acceptance criteria, but a reader of the PRD alone would miss the "why now" and "why this is riskier than it looks" texture that the brief supplies.
