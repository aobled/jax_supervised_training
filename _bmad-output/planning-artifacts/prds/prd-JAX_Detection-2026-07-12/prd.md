---
title: Refactor JAX_Detection — nettoyage code mort et duplications
status: final
created: 2026-07-12
updated: 2026-07-12
---

# Refactor JAX_Detection — nettoyage code mort et duplications

## Vision / Problem Statement

**Problème** : `model_library.py` et le code d'inférence satellite (`tools/`, scripts racine) ont accumulé, au fil des itérations expérimentales, 15 architectures de modèles mortes et 6 fonctions d'inférence dupliquées-et-divergentes (pas de simples copies). Cette dette rend le code plus difficile à maintenir et à faire évoluer, alors que le pipeline d'entraînement principal (`main.py`/`Trainer`/`TaskStrategy`) reste sain et fonctionnel.

**Vision** : un code base où chaque fonction d'inférence/logique métier existe en un seul endroit canonique, où seules les architectures de modèles réellement utilisées subsistent (ou sont explicitement archivées), sans perdre la généricité multi-typologie du pipeline (prouvée par `JAX_KEPLER`) ni sa modularité (pattern Strategy/Factory) ni ses performances actuelles.

## Goals

Le refactor est considéré réussi quand :

1. **Non-régression fonctionnelle, prouvée par comparaison, pas par observation** : capture d'une baseline (boxes/classes/scores sur un petit set d'images fixes) avant refactor, puis diff après refactor — sur les **7 fichiers concernés** par la duplication de fonctions (`bounding_boxes_with_classification_from_video_generation.py`, `tools/bounding_boxes_with_classification_from_images_generation.py`, `heatmap_generation.py`, `heatmap_contouring.py`, `tools/audit_dataset_classification.py`, `tools/audit_dataset_detection.py`, `tools/boxes_process_manual_tkinter.py`), pas seulement 2. (`bounding_boxes_with_classification_from_benchmark.py` a rejoint la liste des suppressions pures — voir FR9 ; les 2 derniers fichiers étaient des consommateurs invisibles du PRD initial, découverts en phase Architecture — voir FR10.)
2. Un entraînement complet (`main.py`) tourne sans régression sur `FIGHTERJET_CLASSIFICATION` et `FIGHTERJET_DETECTION`.
3. La régénération des datasets `.npz` fonctionne toujours (`fighterjet_classification_dataset_tools.py`, `fighterjet_detection_dataset_tools.py`).
4. Un audit global de code (re-scan `bmad-document-project` ou équivalent) confirme l'absence de méthodes mortes/doublons résiduels.
5. La généricité multi-typologie du pipeline reste démontrable (`JAX_KEPLER` continue de fonctionner via le même code commun).

## Features

### F1 — Mutualisation du code d'inférence

- **FR1** : Un module partagé unique (`inference_utils.py`) contient les implémentations canoniques de `load_jax_model`, `load_detection_model`, `predict_crop`, `predict_crops_batch`, `build_predict_fn`, `build_clf_predict_fn`, `get_iou`, `non_max_suppression`, `_preprocess_crop_to_hwc`, `decode_segmentation_and_detect`, `decode_segmentation_and_detect_batch` — 11 fonctions au total (détail et justification dans le spine d'architecture, `ARCHITECTURE-SPINE.md`).
- **FR2** : Les 5 fichiers du scope initial (`bounding_boxes_with_classification_from_video_generation.py`, `tools/bounding_boxes_with_classification_from_images_generation.py`, `heatmap_generation.py`, `heatmap_contouring.py`, `tools/audit_dataset_classification.py`) importent depuis ce module au lieu de redéfinir localement. Voir FR10 pour 2 fichiers supplémentaires découverts en phase Architecture.
- **FR3** : Pour chaque fonction ayant des comportements divergents entre fichiers, un comportement canonique est choisi explicitement avant fusion (pas de fusion mécanique aveugle). Quand la divergence reflète un vrai compromis (ex. temps réel vs précision) plutôt qu'une incohérence accidentelle, les deux comportements peuvent coexister comme fonctions distinctes plutôt que d'être forcés en un seul (cf. spine AD-2, AD-6).

### F2 — Purge des configs et modèles non utilisés

- **FR4** : `dataset_configs.py` ne conserve que `FIGHTERJET_CLASSIFICATION`, `FIGHTERJET_DETECTION`, `JAX_KEPLER`. Suppression de `FIGHTERJET_VIT`, `FIGHTERJET_HYBRID_VIT`, `FIGHTERJET_LETTERBOX`, `FIGHTERJET_DETECTION_SOPHISTICATED`.
- **FR5** : `model_library.py` ne conserve que les architectures référencées (actives ou en commentaire, ex. `aircraft_detector_miniunet`) par les 3 configs restantes : `sophisticated_cnn_128_plus`, `aircraft_detector_unet`, `aircraft_detector_miniunet`, `kepler_1d_cnn`. Les 18 autres architectures de `MODELS` sont supprimées.
- **FR6** : Avant suppression d'une architecture, vérifier qu'aucun `.pkl` actuellement versionné (`best_model.pkl`, `best_model_detection.pkl`) n'en dépend.

### F3 — Suppression du code orphelin

- **FR7** : `train_detection.py` est supprimé (vestige confirmé de la fusion historique JAX_Detection/JAX_Classification).
- **FR8** : `tokens/` est supprimé intégralement (expérience token-based model vs UNet abandonnée, jamais raccordée au pipeline actif).
- **FR9** : `bounding_boxes_with_classification_from_benchmark.py` est supprimé intégralement (et non fusionné dans le module partagé, cf. FR1/FR2). Sa logique de décodage de détection (`decode_grid_and_detect`) est spécifique aux architectures grid-based `aircraft_detector_v3`/`v7_advanced` supprimées par FR5 — trouvaille faite en phase Architecture, confirmée par l'utilisateur.
- **FR10** : `tools/audit_dataset_detection.py` et `tools/boxes_process_manual_tkinter.py` — des consommateurs invisibles au scope initial, qui importent aujourd'hui des fonctions directement depuis l'espace de noms de `bounding_boxes_with_classification_from_video_generation.py` — sont repointés vers `inference_utils.py` dans ce même cycle, plutôt que de risquer une casse silencieuse (pas de suite de tests) ou de reporter le problème via un shim de compatibilité.

## Non-Functional Requirements

- **NFR1 — Généricité** : le pipeline doit rester utilisable pour une tâche non liée à la détection/classification d'avions sans duplication de code (validé par `JAX_KEPLER`).
- **NFR2 — Modularité** : le pattern Strategy/Factory/Dependency Injection (`task_strategies.py`, `model_library.get_model()`, `Trainer`) n'est pas cassé par le refactor.
- **NFR3 — Performance** : aucune régression de performance d'entraînement ou d'inférence introduite par la mutualisation du code, mesurée par temps/epoch (entraînement) et latence/image (inférence) captés avant/après refactor, avec une marge de tolérance couvrant le bruit de mesure inter-runs.
- **NFR4 — Réversibilité** : toute suppression (modèles, `tokens/`, `train_detection.py`) reste récupérable via l'historique git (pas de perte d'information, seulement de code actif).

## Success Metrics

- 0 fonction dupliquée restante entre les 7 fichiers concernés (vérifié par l'audit final).
- `model_library.py` réduit de 22 à 4 architectures actives (mesurable en lignes de code retirées).
- `dataset_configs.py` réduit de 7 à 3 configs.
- Diff baseline/après-refactor identique sur les images de test (§ Goals).

**Contre-métriques** (pour ne pas optimiser la mauvaise chose) :
- Ne pas casser la généricité pour gagner en "simplicité" : `JAX_KEPLER` doit continuer à fonctionner via le même code commun (NFR1).
- Ne pas sacrifier la performance d'entraînement/inférence au nom de la propreté du code (NFR3).

## Non-Goals

- **Pas de suite de tests automatisée ni de CI/CD dans ce cycle.** `architecture.md` (doc source de ce PRD) liste déjà l'absence de tests/CI comme lacune connue du projet — ce refactor s'appuie volontairement sur la méthode baseline/diff manuelle (Goal 1) plutôt que d'introduire une infrastructure de tests. Ajouter des tests/CI reste une piste pour un cycle futur, pas un livrable de celui-ci.
- **Pas de restructuration du système de configuration** (fichier dédié par config au lieu d'un fichier unique) — piste explicitement différée, voir `addendum.md`.

## Open Questions

- ~~**OQ1** : pour chaque fonction divergente (FR3), quel est le comportement canonique retenu ?~~ **Résolu en phase Architecture** (2026-07-12) — voir `_bmad-output/planning-artifacts/architecture/architecture-JAX_Detection-2026-07-12/ARCHITECTURE-SPINE.md` (AD-1 à AD-8, après relecture indépendante). En résumé : la plupart des "divergences" du premier passage n'en étaient pas (fonctions identiques) ; les vraies divergences (`predict_crop`/`predict_crops_batch`, `decode_segmentation_and_detect`/`_batch`) se sont résolues par une double API plutôt qu'un choix forcé ; `bounding_boxes_with_classification_from_benchmark.py` s'est révélé lui-même un vestige à supprimer (AD-5, FR9) ; et une relecture adversariale a révélé une famille de duplication supplémentaire (`load_classification_model`/`build_predict_fn`, AD-1) et 2 fichiers consommateurs invisibles (AD-8, FR10) que le premier passage avait manqués.

---

_Voir aussi `addendum.md` pour les idées considérées mais explicitement différées (restructuration du système de configuration)._
