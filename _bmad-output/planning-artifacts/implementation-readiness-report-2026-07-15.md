---
stepsCompleted: [1, 2, 3, 4, 5, 6]
documentsUsed:
  - _bmad-output/specs/spec-jax-single-pass/SPEC.md
  - _bmad-output/planning-artifacts/architecture/architecture-jax_supervised_training-2026-07-15/ARCHITECTURE-SPINE.md
  - _bmad-output/planning-artifacts/architecture/architecture-JAX_Detection-2026-07-12/ARCHITECTURE-SPINE.md
  - _bmad-output/planning-artifacts/epics.md
---

# Implementation Readiness Assessment Report

**Date:** 2026-07-15
**Project:** jax_supervised_training — JAX Single-Pass (Epic 7 & 8)

## Document Discovery

**Scope note:** this assessment targets the JAX Single-Pass initiative (Epic 7 & 8) only, not the full cumulative `epics.md` history (Epics 1-6, already implemented/closed in prior cycles). This initiative used `bmad-spec` instead of a classic PRD — `SPEC.md` is assessed in the PRD's role.

**Requirements source (in place of PRD):**
- Whole: `_bmad-output/specs/spec-jax-single-pass/SPEC.md` — Capabilities/Constraints/Non-goals kernel

**Architecture Documents:**
- `_bmad-output/planning-artifacts/architecture/architecture-jax_supervised_training-2026-07-15/ARCHITECTURE-SPINE.md` — active spine for this initiative (AD-9 to AD-20, `status: final`)
- `_bmad-output/planning-artifacts/architecture/architecture-JAX_Detection-2026-07-12/ARCHITECTURE-SPINE.md` — parent spine (AD-1 to AD-8), inherited, referenced not duplicated

**Epics & Stories Documents:**
- Whole: `_bmad-output/planning-artifacts/epics.md` — cumulative file; only the "Requirements Inventory — JAX Single-Pass" section plus Epic 7 & Epic 8 are in scope for this assessment

**UX Design Documents:**
- None found — expected, no UI change in this initiative (confirmed as Non-goal in `SPEC.md`)

## Issues Found

- No duplicates (no whole+sharded conflict for any of the above).
- No missing required documents for this initiative's actual chain (SPEC.md + spine + epics.md all present).
- One structural note, not a defect: `epics.md` is a single cumulative file spanning 6 prior epics plus this initiative's 2 — the assessment below reads only the JAX Single-Pass sections.

**Ready to proceed?** [C] Continue after resolving issues

## PRD Analysis (source: SPEC.md, used in the PRD's role — see scope note above)

### Functional Requirements

FR1: Un nouveau modèle de détection par point central (`JAX_DETECTOR`, heatmap de centres + régression de taille) peut être entraîné sur des chunks `.npz` classiques à 224×224 — jamais sur un dataset full-HD (ratio ~41× prohibitif) — remplaçant la segmentation UNet comme méthode de détection.
FR2: Une unique fonction JIT-compilée (`build_single_pass_predict_fn`) produit, à partir d'une image 1920×1080 grayscale, jusqu'à 20 détections (boîte+classe+scores), sans `cv2.findContours` ni boucle de recadrage python sur le chemin critique.
FR3: Deux avions proches/en contact, qui fusionnent aujourd'hui en une seule détection sous segmentation+`findContours`, sont prédits comme deux instances indépendantes sous la nouvelle tête par point central.
FR4: `FIGHTERJET_CLASSIFICATION` est chargée figée dans le nouveau graphe d'inférence et réutilisée sans réentraînement.

Total FRs: 4

### Non-Functional Requirements

NFR1: `FIGHTERJET_CLASSIFICATION` ne doit jamais être réentraînée par ce chantier — chargement figé uniquement.
NFR2: Aucun dataset d'entraînement ne peut nécessiter des images full-HD ; `JAX_DETECTOR` s'entraîne uniquement à sa résolution de config.
NFR3: Non-régression / rollback — l'ancien pipeline complet (`FIGHTERJET_DETECTION`, `AircraftDetectorUNet`, `DetectionStrategy`, `DetectionDataset`, `decode_segmentation_and_detect(_batch)`, `fighterjet_detection_dataset_tools.py`, et leurs consommateurs) reste pleinement fonctionnel, sans modification, pendant toute l'epic et après.
NFR4: La composition d'inférence doit être zéro-python/cv2 sur son chemin critique.
NFR5: La sortie de `build_single_pass_predict_fn` est toujours une structure à 20 slots fixes, slots invalides à zéro, `valid_mask` seule autorité.

Total NFRs: 5

### Additional Requirements

Architecture (AD-9 à AD-20, `ARCHITECTURE-SPINE.md` 2026-07-15, héritant AD-1 à AD-8 du parent) :
- AD-9 tête de détection par point central (anchor-free), AD-10 backbone/FPN reporté, AD-11 crop différentiable, AD-12 résolution canonique + double branche, AD-13 RESCALE + stride à source unique, AD-14 entraînement modulaire, AD-15 sortie 20 slots + seuil en config, AD-16 composition hors `DATASET_CONFIGS`, AD-17 nouvelle classe dédiée + `task_type` à source unique (incl. `main.py`), AD-18 schéma `.npz` à source unique, AD-19 grid-based/YOLO reporté, AD-20 non-régression explicite.
- Pas de nouvelle dépendance externe (Stack, `SPEC.md`).
- Constraints supplémentaires notées dans `SPEC.md` mais non reprises en NFR séparé : aucune (les 5 constraints de `SPEC.md` correspondent 1:1 aux 5 NFR ci-dessus).

### PRD Completeness Assessment (SPEC.md en lieu de PRD)

`SPEC.md` est complet pour son rôle : Why, 4 Capabilities (intent+success chacune), 5 Constraints, 6 Non-goals, Success signal, 5 Open Questions. Auto-validé (2 passes coherence/preservation) au moment de sa création. Les Open Questions (parité pixel resize/crop, init encodeur, valeurs de story non tranchées, risque roadmap `map_coordinates`) ne bloquent pas la lecture des exigences elles-mêmes — elles sont déjà correctement reportées dans les stories concernées (7.8, 8.1) plutôt que laissées invisibles.

## Epic Coverage Validation

### Coverage Matrix

| FR Number | Requirement (résumé) | Epic Coverage | Status |
| --- | --- | --- | --- |
| FR1 | `JAX_DETECTOR` entraîné, point central, pas de dataset full-HD | Epic 7, Stories 7.1-7.7 (implémentation) + 7.8 (preuve d'exécution) | ✓ Covered |
| FR2 | `build_single_pass_predict_fn`, zéro python/cv2, 20 slots | Epic 8, Stories 8.1-8.6 (implémentation/assemblage) + 8.9 (preuve d'exécution) | ✓ Covered |
| FR3 | Fusion de boîtes résolue structurellement | Epic 8, Story 8.9 (validation explicite sur cas de formation serrée) | ✓ Covered |
| FR4 | Classification figée réutilisée | Epic 8, Story 8.5 (appel figé) + 8.6 (assemblage) | ✓ Covered |

### Missing Requirements

Aucune. Les 4 FR ont un chemin d'implémentation traçable jusqu'à une story avec AC explicites, et les 5 NFR sont chacun repris nommément dans au moins une AC (NFR1/NFR2 → 7.x ; NFR3 → 8.9 ; NFR4 → 8.9 ; NFR5 → 8.6). Les 12 items "Additional Requirements" (AD-9 à AD-20) sont chacun cités par leur ID dans au moins une story (vérifié par grep sur `epics.md` — chaque `AD-9`…`AD-20` apparaît au moins une fois dans la section Epic 7/8).

### Coverage Statistics

- Total PRD (SPEC.md) FRs: 4
- FRs covered in epics: 4
- Coverage percentage: 100%

## UX Alignment Assessment

### UX Document Status

Not Found — confirmed absent at Document Discovery (Step 1).

### Alignment Issues

N/A — no UX document to align.

### Warnings

None. UX/UI n'est pas impliqué par ce chantier : les deux scripts consommateurs migrés en Epic 8 (`bounding_boxes_with_classification_from_video_generation.py`, `tools/bounding_boxes_with_classification_from_images_generation.py`) sont des scripts batch, pas des interfaces. Le seul outil GUI existant (`tools/boxes_process_manual_tkinter.py`) reste explicitement sur l'ancien pipeline, non modifié (AD-20, Non-goal confirmé dans `SPEC.md`). Cohérent avec `SPEC.md` § Non-goals et `epics.md` § UX Design Requirements ("N/A").

## Epic Quality Review

Revue rigoureuse des 17 stories (Epic 7 : 7.1-7.8 ; Epic 8 : 8.1-8.9) contre les standards de `bmad-create-epics-and-stories`.

### A. User Value Focus Check

- **Epic 8** passe sans réserve : élimine une orchestration manuelle réelle, corrige un bug réel (fusion de boîtes), valeur observable directement.
- **Epic 7** est un cas limite structurellement proche du pattern "infra technique" (rouge dans le guide : "Setup Database"). Jugement : **acceptable, pas une violation** — Story 7.8 livre une capacité inspectable et démontrable de bout en bout (prédictions réelles sur un jeu de validation), contrairement à "créer les tables" qui ne produit aucun comportement observable. Cadrage cohérent avec le précédent déjà établi dans ce document (Epic 1, "Mutualisation du code d'inférence", formulée de la même façon system-outcome plutôt que "As a user"). Documenté ici comme choix assumé, pas comme un défaut à corriger.

### B. Epic Independence Validation

Epic 7 se suffit à lui-même (Story 7.8 le prouve explicitly). Epic 8 dépend du **livrable** d'Epic 7 (checkpoint entraîné), pas de l'inverse — sens de dépendance autorisé par le framework. Aucune violation.

### C. Story Sizing & Forward Dependencies (vérification ligne par ligne)

Chaque story des deux epics a été vérifiée individuellement : aucune ne référence une story de numéro supérieur dans ses clauses `Given`. Toutes les dépendances remontent uniquement vers des stories antérieures ou vers un epic antérieur (8.2/8.5 → checkpoint Epic 7, autorisé). **Aucune violation trouvée.**

### D. Acceptance Criteria Review

Format Given/When/Then respecté partout, critères majoritairement spécifiques (fichiers, fonctions, lignes de code nommés). Deux observations :

- 🟡 **Minor** — Story 7.8, AC "les heatmaps... sont exploitables (comparaison qualitative a minima)" : critère non quantitatif. Faiblesse héritée du critère de succès de CAP-1 dans `SPEC.md` lui-même ("usable... predictions"), pas introduite par la rédaction de la story. Recommandation : au moment de l'implémentation, si un outil de mesure IoU/précision existe déjà pour l'ancien pipeline, le réutiliser pour un seuil quantitatif — mais ne pas inventer un seuil maintenant, aucune donnée ne le justifie encore.
- Aucune autre AC vague relevée. Les conditions d'erreur/robustesse (chargement de checkpoint corrompu, normalisation d'entrée non conforme) sont correctement déléguées à AD-3 hérité / AD-12 plutôt que dupliquées par story — c'est le comportement voulu (AD-1 hérité, source unique), pas une lacune.

### E. Database/Entity Creation Timing

N/A — aucune base de données dans ce projet.

### F/G. Starter Template & Greenfield/Brownfield

N/A pour le starter template (brownfield existant). Indicateurs brownfield correctement présents : intégration avec l'existant (AD-20 non-régression, dispatch `main.py`, fallback de checkpoint hérité) omniprésente dans les stories, pas de story CI/CD hors sujet introduite.

### Best Practices Compliance Checklist

- [x] Epic délivre une valeur observable (Epic 7 : jugement documenté ci-dessus, pas une exception silencieuse)
- [x] Chaque epic fonctionne indépendamment
- [x] Stories correctement dimensionnées (un fichier/une fonction/un rôle par story)
- [x] Aucune dépendance en avant
- [x] N/A base de données
- [x] Critères d'acceptation clairs (1 souci mineur documenté, non bloquant)
- [x] Traçabilité FR maintenue (100%, voir Coverage Matrix)

### Findings by Severity

**🔴 Critical Violations:** aucune.
**🟠 Major Issues:** aucune.
**🟡 Minor Concerns:** 2 — (1) cadrage "infra" d'Epic 7, jugé acceptable et documenté ; (2) AC qualitative de Story 7.8, recommandation non bloquante pour l'implémentation.

## Summary and Recommendations

### Overall Readiness Status

**READY**

### Critical Issues Requiring Immediate Action

Aucune. 0 critique, 0 majeure sur l'ensemble des 4 étapes de vérification (découverte documentaire, analyse des exigences, couverture FR/NFR/AD, revue qualité des epics/stories).

### Recommended Next Steps

1. Procéder à `bmad-sprint-planning` pour intégrer Epic 7 & 8 au plan de sprint (`sprint-status.yaml` ne les référence pas encore — seuls les Epics 1-6 y figurent).
2. Démarrer l'implémentation par **Story 7.1** (schéma d'échange heatmap+taille) — c'est le prérequis bloquant explicite pour 7.4/7.5, cohérent avec le pattern déjà utilisé pour Story 1.2/AD-7 hérité.
3. Non bloquant, à garder en tête à l'implémentation : ajouter un critère quantitatif à Story 7.8 si un outil de mesure IoU existe déjà pour l'ancien pipeline (sinon, laisser tel quel — ne pas inventer de seuil).

### Final Note

Cette évaluation a identifié 2 points mineurs sur 5 catégories vérifiées (découverte documentaire, analyse des exigences, couverture epics, alignement UX, qualité des epics/stories) — aucun des deux ne bloque le passage à l'implémentation. `SPEC.md` + la spine d'architecture (2026-07-15) + `epics.md` (Epic 7 & 8) forment un ensemble cohérent et traçable à 100% (4/4 FR, 5/5 NFR, 12/12 AD cités).

---
**Date de l'évaluation :** 2026-07-15
**Évaluateur :** `bmad-check-implementation-readiness` (Claude Sonnet 5)
