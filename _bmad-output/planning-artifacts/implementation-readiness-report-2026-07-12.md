---
stepsCompleted: [1, 2, 3, 4, 5, 6]
documentsUsed:
  prd: _bmad-output/planning-artifacts/prds/prd-JAX_Detection-2026-07-12/prd.md
  architecture: _bmad-output/planning-artifacts/architecture/architecture-JAX_Detection-2026-07-12/ARCHITECTURE-SPINE.md
  epics: _bmad-output/planning-artifacts/epics.md
  ux: null
---

# Implementation Readiness Assessment Report

**Date:** 2026-07-12
**Project:** Refactor JAX_Detection — nettoyage code mort et duplications

## Document Discovery

### PRD Files Found

**Whole Documents:**
- `prds/prd-JAX_Detection-2026-07-12/prd.md` (9081 bytes, 12 juil. 17:38)

**Sharded Documents:** none

### Architecture Files Found

**Whole Documents:**
- `architecture/architecture-JAX_Detection-2026-07-12/ARCHITECTURE-SPINE.md` (14875 bytes, 12 juil. 17:40)

**Sharded Documents:** none

### Epics & Stories Files Found

**Whole Documents:**
- `epics.md` (30645 bytes, 12 juil. 18:04)

**Sharded Documents:** none

### UX Design Files Found

None found. Consistent with the PRD/spine: this refactor has no user-facing interface change.

## Issues Found

- No duplicates (whole vs. sharded) for any document type.
- No missing required documents (PRD and Architecture both present and final).
- No UX document — expected and not a gap for this internal refactor.

## PRD Analysis

### Functional Requirements

FR1: Un module partagé unique (`inference_utils.py`) contient les implémentations canoniques de `load_jax_model`, `load_detection_model`, `predict_crop`, `predict_crops_batch`, `build_predict_fn`, `build_clf_predict_fn`, `get_iou`, `non_max_suppression`, `_preprocess_crop_to_hwc`, `decode_segmentation_and_detect`, `decode_segmentation_and_detect_batch` — 11 fonctions au total.
FR2: Les 5 fichiers du scope initial (`bounding_boxes_with_classification_from_video_generation.py`, `tools/bounding_boxes_with_classification_from_images_generation.py`, `heatmap_generation.py`, `heatmap_contouring.py`, `tools/audit_dataset_classification.py`) importent depuis ce module au lieu de redéfinir localement. Voir FR10 pour 2 fichiers supplémentaires découverts en phase Architecture.
FR3: Pour chaque fonction ayant des comportements divergents entre fichiers, un comportement canonique est choisi explicitement avant fusion (pas de fusion mécanique aveugle). Quand la divergence reflète un vrai compromis, les deux comportements peuvent coexister comme fonctions distinctes.
FR4: `dataset_configs.py` ne conserve que `FIGHTERJET_CLASSIFICATION`, `FIGHTERJET_DETECTION`, `JAX_KEPLER`. Suppression de `FIGHTERJET_VIT`, `FIGHTERJET_HYBRID_VIT`, `FIGHTERJET_LETTERBOX`, `FIGHTERJET_DETECTION_SOPHISTICATED`.
FR5: `model_library.py` ne conserve que les architectures référencées par les 3 configs restantes : `sophisticated_cnn_128_plus`, `aircraft_detector_unet`, `aircraft_detector_miniunet`, `kepler_1d_cnn`. Les 18 autres architectures de `MODELS` sont supprimées.
FR6: Avant suppression d'une architecture, vérifier qu'aucun `.pkl` actuellement versionné (`best_model.pkl`, `best_model_detection.pkl`) n'en dépend.
FR7: `train_detection.py` est supprimé (vestige confirmé de la fusion historique JAX_Detection/JAX_Classification).
FR8: `tokens/` est supprimé intégralement (expérience token-based model vs UNet abandonnée, jamais raccordée au pipeline actif).
FR9: `bounding_boxes_with_classification_from_benchmark.py` est supprimé intégralement (et non fusionné dans le module partagé). Sa logique de décodage de détection (`decode_grid_and_detect`) est spécifique aux architectures grid-based supprimées par FR5.
FR10: `tools/audit_dataset_detection.py` et `tools/boxes_process_manual_tkinter.py` — consommateurs invisibles au scope initial — sont repointés vers `inference_utils.py` dans ce même cycle.

Total FRs: 10

### Non-Functional Requirements

NFR1 — Généricité : le pipeline doit rester utilisable pour une tâche non liée à la détection/classification d'avions sans duplication de code (validé par `JAX_KEPLER`).
NFR2 — Modularité : le pattern Strategy/Factory/Dependency Injection (`task_strategies.py`, `model_library.get_model()`, `Trainer`) n'est pas cassé par le refactor.
NFR3 — Performance : aucune régression de performance d'entraînement ou d'inférence introduite par la mutualisation du code, mesurée par temps/epoch et latence/image, avec une marge de tolérance couvrant le bruit de mesure inter-runs.
NFR4 — Réversibilité : toute suppression reste récupérable via l'historique git.

Total NFRs: 4

### Additional Requirements

- **Goals (critères de succès du PRD)** : (1) non-régression prouvée par diff baseline/après-refactor sur les 7 fichiers concernés — pas seulement observation ; (2) `main.py` tourne sans régression sur `FIGHTERJET_CLASSIFICATION`/`FIGHTERJET_DETECTION` ; (3) régénération des datasets `.npz` fonctionne toujours ; (4) audit global confirme l'absence de code mort/doublons résiduels ; (5) généricité `JAX_KEPLER` démontrable.
- **Success Metrics** : 0 fonction dupliquée restante ; `model_library.py` réduit de 22 à 4 architectures ; `dataset_configs.py` réduit de 7 à 3 configs ; diff baseline/après-refactor identique.
- **Contre-métriques** : ne pas casser la généricité (NFR1) ni sacrifier la performance (NFR3) au nom de la propreté du code.
- **Non-Goals explicites** : pas de suite de tests automatisée ni de CI/CD dans ce cycle (méthode baseline/diff manuelle à la place) ; pas de restructuration du système de configuration (différée, voir `addendum.md`).
- **Open Questions** : OQ1 (comportement canonique par fonction divergente) — résolu en phase Architecture (AD-1 à AD-8).

### PRD Completeness Assessment

Le PRD est marqué `status: final`, daté du 2026-07-12, avec son unique question ouverte (OQ1) explicitement résolue par référence croisée vers le spine d'architecture. Les FR sont concrètes, numérotées, testables (fichiers et fonctions nommés explicitement), et les NFR sont mesurables (temps/epoch, latence/image). Le PRD documente aussi une découverte tardive (FR10, consommateurs invisibles) avec sa justification — signe que le PRD a été mis à jour en cohérence avec l'architecture plutôt que figé prématurément. Aucun gap de clarté détecté à ce stade.

## Epic Coverage Validation

### Coverage Matrix

| FR Number | PRD Requirement (résumé) | Epic Coverage | Status |
| --- | --- | --- | --- |
| FR1 | Module `inference_utils.py`, 11 fonctions canoniques | Epic 1, Story 1.2 | ✓ Covered |
| FR2 | 5 fichiers du scope initial importent depuis le module | Epic 1, Stories 1.3-1.7 | ✓ Covered |
| FR3 | Comportement canonique arbitré pour chaque divergence | Epic 1, Story 1.2 | ✓ Covered |
| FR4 | `dataset_configs.py` réduit à 3 configs | Epic 2, Story 2.2 | ✓ Covered |
| FR5 | `model_library.py` réduit à 4 architectures | Epic 2, Story 2.3 | ✓ Covered |
| FR6 | Vérification `.pkl` avant suppression d'architecture | Epic 2, Story 2.1 | ✓ Covered |
| FR7 | Suppression `train_detection.py` | Epic 3, Story 3.1 | ✓ Covered |
| FR8 | Suppression `tokens/` | Epic 3, Story 3.2 | ✓ Covered |
| FR9 | Suppression `bounding_boxes_with_classification_from_benchmark.py` | Epic 3, Story 3.3 | ✓ Covered |
| FR10 | 2 consommateurs invisibles repointés vers le module | Epic 1, Stories 1.8-1.9 | ✓ Covered |

### Missing Requirements

Aucune. Les 10 FR du PRD sont couvertes par au moins une story, avec correspondance directe entre le texte de la FR et les critères d'acceptation de la story (vérifié par relecture ligne à ligne du PRD et de `epics.md`, pas seulement par la coverage map déclarative d'`epics.md`).

### Coverage Statistics

- Total PRD FRs: 10
- FRs covered in epics: 10
- Coverage percentage: 100%

## UX Alignment Assessment

### UX Document Status

Not Found — no `*ux*.md` or sharded UX folder in `planning_artifacts`.

### Alignment Issues

None. The PRD scope is a backend refactor (shared inference module, config/model pruning, dead-code removal) with no new or changed user interface. The one interactive tool touched by the epics (`tools/boxes_process_manual_tkinter.py`, Story 1.9) is a pre-existing Tkinter GUI whose import source is repointed — its behavior is explicitly required to stay unchanged (Story 1.9 AC), not redesigned.

### Warnings

None. UX is not implied by this PRD's scope — absence of a UX document is expected, not a gap.

## Epic Quality Review

### A. User Value Focus Check

This is an internal tech-debt refactor, not an end-user product — the PRD's own Vision/Problem Statement frames the entire initiative around codebase maintainability, not a new user-facing capability. Consistently, every epic and story already sets the persona to "mainteneur du pipeline/code/fichier" rather than an end-user of the trained model. Applying the literal rubric ("what can users do") with `user = maintainer` (not `user = model consumer`), all three epics pass:

- **Epic 1** (Mutualisation) — mainteneurs get one source of truth for inference logic, provably non-regressive.
- **Epic 2** (Purge configs/modèles) — mainteneurs get a config/model surface limited to what's actually used, safely (FR6 gate).
- **Epic 3** (Suppression orpheline) — mainteneurs get a repo free of vestiges, reversibly (git history, NFR4).

No epic is a bare technical milestone with zero framed outcome (e.g., none read as "Setup Database").

### B. Epic Independence Validation

- Epic 1: stands alone (creates `inference_utils.py`, migrates all 7 consumers, validates by diff) — touches no files Epic 2/3 touch.
- Epic 2: does not require Epic 1 or Epic 3 output — `dataset_configs.py`/`model_library.py` are disjoint from Epic 1's and Epic 3's files.
- Epic 3: does not require Epic 1 or Epic 2 output — deletes 3 standalone vestige files/folders.

No circular or forward epic-level dependency found.

### C. Story Quality Assessment

- **Sizing**: 16 of 17 stories are single-file-scoped, appropriately sized for one dev session. Story 1.2 (create `inference_utils.py`) is the outlier — it arbitrates behavior across 11 functions and 6 architecture decisions (AD-1, AD-2, AD-3, AD-4, AD-6, plus the box-format convention) in one story. This is intentional, not an oversight: AD-7 explicitly mandates a single-author, single story for this file specifically to prevent "7 parallel stories... with silent overwrite or conflict" in the absence of a test suite. Splitting it would violate AD-7. Flagged as justified, not a defect.
- **AC format**: Given/When/Then used consistently across all 17 stories; each AC is independently testable and references the specific FR/AD it implements.
- **No forward dependencies**: verified story-by-story — each story only references previous stories within its own epic (1.3-1.9 depend only on 1.2; 1.10 depends on 1.1 and 1.3-1.9; 2.3 depends on 2.1+2.2; 2.4 depends on 2.2+2.3).

### D. Cross-Epic Cross-References (not blocking, flagged for clarity)

Two stories reference facts/checks that are also the formal subject of a story in a *later* epic. Neither is a blocking dependency — both rely on facts already established in the architecture spine, not on the later story's execution:

1. **Story 1.2** (Epic 1) asserts the `.pkl`↔architecture mapping (AD-4) as a precondition for its `model_name` fallback choice. **Story 2.1** (Epic 2) is where this same verification is formally re-executed as the FR6 gate before Story 2.3's deletions. Story 1.2 can proceed because the fact is already documented in AD-4 ("vérification faite en amont de ce spine"); Story 2.1 remains the authoritative gate for the deletion itself.
2. **Story 1.10** (Epic 1, end-of-epic diff) and **Story 2.4** (Epic 2, end-of-epic validation) both re-run a full `main.py` training pass on `FIGHTERJET_CLASSIFICATION`/`FIGHTERJET_DETECTION`. This is deliberate incremental validation — each epic validates that *its own* changes didn't regress training, isolating the regression source by epic — not accidental duplication.

### E. Database / Starter Template / Brownfield Checks

- No database/entity creation in this project — not applicable.
- No starter template specified in the Architecture Spine (brownfield, existing codebase ratified as-is) — not applicable.
- Brownfield indicators present as expected: Epic 1 is entirely migration stories (1.3-1.9 repoint existing consumers), consistent with a brownfield refactor rather than greenfield setup.

### Findings Summary

🔴 **Critical Violations:** None.

🟠 **Major Issues:** None.

🟡 **Minor Concerns (documented, non-blocking):**
- Epic titles read as technically-framed by product-epic conventions, but are correctly justified for an internal refactor where the "user" is the codebase maintainer (see A).
- Story 1.2 is intentionally larger than typical story sizing, justified by AD-7's single-author constraint.
- Story 1.2 and Story 2.1 both touch the `.pkl`/architecture dependency fact (AD-4/FR6) — Story 2.1 is authoritative for the gate; Story 1.2 only consumes the already-documented fact.
- Story 1.10 and Story 2.4 both re-validate full training runs — intentional per-epic isolation, not redundant waste.

## Summary and Recommendations

### Overall Readiness Status

**READY**

### Critical Issues Requiring Immediate Action

None. Zero critical or major violations were found across document discovery, PRD/FR extraction, epic coverage, UX alignment, and epic/story quality review.

### Recommended Next Steps

1. Proceed to `bmad-sprint-planning` to generate the implementation sequence — the epic order (Epic 1 → 2 → 3) and the strict Story 1.1 → 1.2 → {1.3-1.9} → 1.10 sequencing within Epic 1 (AD-7) should be preserved as-is in the sprint plan.
2. When Story 2.1 (FR6 gate) starts, treat it as re-confirming — not blindly trusting — the AD-4 finding already documented in the Architecture Spine, since it is the last checkpoint before Story 2.3 deletes architectures irreversibly from active code.
3. No artifact changes are required before implementation. The 4 minor concerns in the Epic Quality Review are documented for context, not action items.

### Final Note

This assessment identified 0 critical issues, 0 major issues, and 4 minor (non-blocking, justified) concerns across 5 review categories. The PRD, Architecture Spine, and Epics/Stories are aligned and ready for `bmad-sprint-planning`.

**Assessed by:** Implementation Readiness workflow (bmad-check-implementation-readiness)
**Date:** 2026-07-12
