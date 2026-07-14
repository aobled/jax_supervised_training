---
stepsCompleted: [1, 2, 3]
inputDocuments:
  - _bmad-output/planning-artifacts/prds/prd-JAX_Detection-2026-07-12/prd.md
  - _bmad-output/planning-artifacts/architecture/architecture-JAX_Detection-2026-07-12/ARCHITECTURE-SPINE.md
  - docs/dead-code-and-duplication-audit.md
  - docs/architecture.md
  - docs/source-tree-analysis.md
  - _bmad-output/planning-artifacts/prds/prd-JAX_Detection-2026-07-14/prd.md
---

# Refactor jax_supervised_training — Epic Breakdown

## Overview

This document provides the complete epic and story breakdown for the jax_supervised_training refactor (nettoyage code mort et duplications), decomposing the requirements from the PRD and the Architecture Spine into implementable stories. No UX design contract applies — this is an internal code refactor with no user-facing interface changes.

## Requirements Inventory

### Functional Requirements

FR1: Un module partagé unique (`inference_utils.py`) contient les implémentations canoniques de `load_jax_model`, `load_detection_model`, `predict_crop`, `predict_crops_batch`, `build_predict_fn`, `build_clf_predict_fn`, `get_iou`, `non_max_suppression`, `_preprocess_crop_to_hwc`, `decode_segmentation_and_detect`, `decode_segmentation_and_detect_batch` — 11 fonctions au total.
FR2: Les 5 fichiers du scope initial (`bounding_boxes_with_classification_from_video_generation.py`, `tools/bounding_boxes_with_classification_from_images_generation.py`, `heatmap_generation.py`, `heatmap_contouring.py`, `tools/audit_dataset_classification.py`) importent depuis ce module au lieu de redéfinir localement. Voir FR10 pour 2 fichiers supplémentaires découverts en phase Architecture.
FR3: Pour chaque fonction ayant des comportements divergents entre fichiers, un comportement canonique est choisi explicitement avant fusion (pas de fusion mécanique aveugle). Quand la divergence reflète un vrai compromis (ex. temps réel vs précision), les deux comportements peuvent coexister comme fonctions distinctes plutôt que d'être forcés en un seul.
FR4: `dataset_configs.py` ne conserve que `FIGHTERJET_CLASSIFICATION`, `FIGHTERJET_DETECTION`, `JAX_KEPLER`. Suppression de `FIGHTERJET_VIT`, `FIGHTERJET_HYBRID_VIT`, `FIGHTERJET_LETTERBOX`, `FIGHTERJET_DETECTION_SOPHISTICATED`.
FR5: `model_library.py` ne conserve que les architectures référencées par les 3 configs restantes : `sophisticated_cnn_128_plus`, `aircraft_detector_unet`, `aircraft_detector_miniunet`, `kepler_1d_cnn`. Les 18 autres architectures de `MODELS` sont supprimées.
FR6: Avant suppression d'une architecture, vérifier qu'aucun `.pkl` actuellement versionné (`best_model.pkl`, `best_model_detection.pkl`) n'en dépend.
FR7: `train_detection.py` est supprimé (vestige confirmé de la fusion historique jax_supervised_training/JAX_Classification).
FR8: `tokens/` est supprimé intégralement (expérience token-based model vs UNet abandonnée, jamais raccordée au pipeline actif).
FR9: `bounding_boxes_with_classification_from_benchmark.py` est supprimé intégralement (et non fusionné dans le module partagé). Sa logique de décodage de détection (`decode_grid_and_detect`) est spécifique aux architectures grid-based supprimées par FR5.
FR10: `tools/audit_dataset_detection.py` et `tools/boxes_process_manual_tkinter.py` — des consommateurs invisibles au scope initial, qui importent aujourd'hui des fonctions directement depuis l'espace de noms de `bounding_boxes_with_classification_from_video_generation.py` — sont repointés vers `inference_utils.py` dans ce même cycle.

### NonFunctional Requirements

NFR1 — Généricité : le pipeline doit rester utilisable pour une tâche non liée à la détection/classification d'avions sans duplication de code (validé par `JAX_KEPLER`).
NFR2 — Modularité : le pattern Strategy/Factory/Dependency Injection (`task_strategies.py`, `model_library.get_model()`, `Trainer`) n'est pas cassé par le refactor.
NFR3 — Performance : aucune régression de performance d'entraînement ou d'inférence introduite par la mutualisation du code, mesurée par temps/epoch (entraînement) et latence/image (inférence) captés avant/après refactor.
NFR4 — Réversibilité : toute suppression (modèles, `tokens/`, `train_detection.py`) reste récupérable via l'historique git.

### Additional Requirements

- **AD-1 (FR1, FR2, FR10)** : `inference_utils.py` est la source unique des 11 fonctions d'inférence — aucune redéfinition locale n'est autorisée, y compris sous un autre nom faisant le même travail. `load_classification_model` n'est pas migré comme 12ᵉ fonction : sa composition (`load_jax_model` + `build_predict_fn`) devient explicite au site d'appel.
- **AD-2 (FR1, FR3, NFR3) `[ADOPTED]`** : `predict_crop` et `predict_crops_batch` sont deux implémentations pleinement indépendantes (pas de délégation interne). `_CLF_BATCH_SIZE = 32` devient une constante privée de `inference_utils.py`.
- **AD-3 (FR1, FR3, NFR4) `[ADOPTED]`** : `load_detection_model` conserve le fallback de résolution de chemin à 3 niveaux **et** la ré-initialisation des `batch_stats` manquants. `DETECTION_IMAGE_SIZE = (224, 224)` devient une constante privée.
- **AD-4 (FR1, FR5, FR6)** : tout fallback `model_name` par défaut dans `inference_utils.py` pointe vers `aircraft_detector_unet` (vivant), jamais vers un modèle supprimé par FR5. FR6 reste un gate bloquant à ré-exécuter avant la suppression effective, pas une simple note informative.
- **AD-5 (FR2, FR5, FR9)** : `bounding_boxes_with_classification_from_benchmark.py` est supprimé intégralement, pas migré/réconcilié vers `inference_utils.py`. Confirmé : aucun autre fichier du repo n'importe depuis lui.
- **AD-6 (FR1, FR3, NFR3) `[ADOPTED]`** : `decode_segmentation_and_detect` et `decode_segmentation_and_detect_batch` restent deux fonctions distinctes et indépendantes, chacune ratifiant son implémentation actuelle sans changement de comportement. Le pipeline vidéo (`_batch`) est prioritaire : aucun changement futur ne doit dégrader son débit temps réel au nom d'une unification.
- **AD-7 (FR1, FR2, FR10)** : Séquencement — une story dédiée ("Story 0") crée `inference_utils.py` avec les 11 fonctions canonisées et leurs corps définitifs, **avant** que toute story de migration de fichier consommateur ne démarre. Les stories suivantes ne font qu'importer ; aucune n'édite `inference_utils.py`.
- **AD-8 (FR2, FR10)** : Périmètre étendu aux consommateurs invisibles — `tools/audit_dataset_detection.py` et `tools/boxes_process_manual_tkinter.py` sont ajoutés à la liste des consommateurs de `inference_utils.py`, imports repointés dans le même cycle (pas de shim de compatibilité).
- **Convention format des boxes** : `[x1, y1, x2, y2, score]`, format liste (pas numpy vectorisé) — seul survivant après AD-5.
- **Convention checkpoints** : tout chargement de checkpoint applique le fallback de chemin 3 niveaux + ré-init des `batch_stats` manquants (AD-3) — jamais un chargement "nu" qui échoue silencieusement.
- **Convention performance** : en cas d'arbitrage entre le pipeline vidéo temps réel et un autre consommateur, le pipeline vidéo est prioritaire (AD-6, NFR3).
- **Stack** : aucune nouvelle dépendance externe introduite — `inference_utils.py` réutilise exclusivement JAX/Flax/NumPy/OpenCV déjà présents dans le projet.
- **Non-Goal rappelé (PRD)** : pas de suite de tests automatisée ni de CI/CD dans ce cycle — la méthode de validation est la comparaison baseline/diff manuelle (Goal 1 du PRD), pas une suite de tests. Ceci renforce l'importance d'AD-7 (auteur unique de `inference_utils.py`).

### UX Design Requirements

N/A — aucun contrat UX applicable. Ce refactor est un chantier interne sans changement d'interface utilisateur (pas de script CLI, GUI ou API publique modifié dans son comportement observable).

### FR Coverage Map

FR1:  Epic 1 - Module partagé inference_utils.py (11 fonctions canoniques)
FR2:  Epic 1 - Migration des 5 fichiers consommateurs du scope initial
FR3:  Epic 1 - Réconciliation des comportements divergents (double API si compromis réel)
FR4:  Epic 2 - Purge dataset_configs.py (7→3 configs)
FR5:  Epic 2 - Purge model_library.py (22→4 architectures)
FR6:  Epic 2 - Gate de vérification .pkl avant suppression
FR7:  Epic 3 - Suppression train_detection.py
FR8:  Epic 3 - Suppression tokens/
FR9:  Epic 3 - Suppression bounding_boxes_with_classification_from_benchmark.py
FR10: Epic 1 - Migration des 2 consommateurs invisibles (audit_dataset_detection.py, boxes_process_manual_tkinter.py)

## Epic List

### Epic 1: Mutualisation du code d'inférence

Le codebase dispose d'un point unique de vérité pour la logique d'inférence transverse — plus aucune fonction critique (chargement de modèle, prétraitement, prédiction, décodage, NMS/IoU) n'est redéfinie-et-divergente dans plusieurs fichiers. Une baseline de non-régression (boxes/classes/scores) est capturée avant et diffée après, sur les 7 fichiers concernés.
**FRs covered:** FR1, FR2, FR3, FR10
**Note d'implémentation :** contient une "Story 0" obligatoire (AD-7) qui crée `inference_utils.py` avec ses 11 fonctions canoniques *avant* toute story de migration des 7 fichiers consommateurs — celles-ci ne font qu'importer, aucune n'édite le module partagé.

### Epic 2: Purge des configs et modèles morts

`dataset_configs.py` et `model_library.py` ne contiennent plus que les configs/architectures réellement utilisées par le pipeline (3 configs, 4 architectures), sans risque de casser le chargement des checkpoints existants ni un entraînement complet.
**FRs covered:** FR4, FR5, FR6
**Note d'implémentation :** FR6 (vérification `.pkl`) est un gate bloquant à exécuter avant la suppression effective des architectures (FR5) — pas une story indépendante après coup.

### Epic 3: Suppression du code orphelin

Les vestiges confirmés (fusion historique, expérimentation abandonnée, script de benchmark devenu obsolète après AD-5) sont retirés du dépôt actif, sans perte d'information (récupérables via git).
**FRs covered:** FR7, FR8, FR9

## Epic 1: Mutualisation du code d'inférence

Le codebase dispose d'un point unique de vérité pour la logique d'inférence transverse — plus aucune fonction critique (chargement de modèle, prétraitement, prédiction, décodage, NMS/IoU) n'est redéfinie-et-divergente dans plusieurs fichiers. Une baseline de non-régression (boxes/classes/scores) est capturée avant et diffée après, sur les 7 fichiers concernés. **FRs covered:** FR1, FR2, FR3, FR10

### Story 1.1: Capture de la baseline de non-régression

As a mainteneur du pipeline d'inférence,
I want une baseline capturée (boxes/classes/scores) sur un petit set d'images fixes, pour les 7 fichiers concernés par la duplication,
So that je peux prouver par diff, pas par observation, l'absence de régression après le refactor (Goal 1 PRD).

**Acceptance Criteria:**

**Given** les 7 fichiers concernés (`bounding_boxes_with_classification_from_video_generation.py`, `tools/bounding_boxes_with_classification_from_images_generation.py`, `heatmap_generation.py`, `heatmap_contouring.py`, `tools/audit_dataset_classification.py`, `tools/audit_dataset_detection.py`, `tools/boxes_process_manual_tkinter.py`) dans leur état actuel (avant toute modification)
**When** j'exécute chacun sur un jeu d'images fixe représentatif (couvrant au moins un cas classification et un cas détection)
**Then** les sorties (boxes, classes, scores) sont capturées et sauvegardées dans un format comparable (JSON/CSV) associé à un identifiant de version baseline
**And** le jeu d'images utilisé est documenté (chemin, nombre d'images, critère de sélection) pour être réutilisé à l'identique en Story 1.10

**Given** `tools/boxes_process_manual_tkinter.py` a un usage interactif (GUI Tkinter)
**When** la capture automatique n'est pas possible pour ce fichier
**Then** une méthode de capture alternative (appel direct aux fonctions internes exposées, ou capture manuelle documentée) est utilisée et le choix est noté dans la baseline

### Story 1.2: Création de inference_utils.py (Story 0, AD-7)

As a mainteneur du pipeline d'inférence,
I want un module partagé inference_utils.py contenant les 11 fonctions canoniques d'inférence avec leurs comportements arbitrés,
So that toute la logique d'inférence dupliquée peut ensuite être remplacée par un import unique, sans ambiguïté sur quel comportement fait foi.

**Acceptance Criteria:**

**Given** le besoin de mutualiser les fonctions d'inférence (FR1, AD-1)
**When** `inference_utils.py` est créé
**Then** il contient exactement les 11 fonctions suivantes avec le comportement canonique arbitré : `load_jax_model`, `load_detection_model`, `predict_crop`, `predict_crops_batch`, `build_predict_fn`, `build_clf_predict_fn`, `get_iou`, `non_max_suppression`, `_preprocess_crop_to_hwc`, `decode_segmentation_and_detect`, `decode_segmentation_and_detect_batch`
**And** aucune 12ᵉ fonction (`load_classification_model`) n'est ajoutée — la composition `load_jax_model` + `build_predict_fn` remplace ce besoin au site d'appel

**Given** `predict_crop` et `predict_crops_batch` (AD-2)
**When** elles sont implémentées
**Then** elles sont deux implémentations pleinement indépendantes (`predict_crop` n'appelle pas `predict_crops_batch` en interne)
**And** `_CLF_BATCH_SIZE = 32` est définie comme constante privée du module

**Given** `load_detection_model` (AD-3)
**When** elle est implémentée
**Then** elle conserve le fallback de résolution de chemin à 3 niveaux (CWD → parent du CWD → parent du fichier appelant) et la ré-initialisation des `batch_stats` manquants
**And** `DETECTION_IMAGE_SIZE = (224, 224)` est définie comme constante privée du module

**Given** tout fallback `model_name` par défaut (AD-4)
**When** il est déclenché sur un checkpoint sans métadonnées `config`
**Then** il pointe vers `aircraft_detector_unet`, jamais vers un modèle supprimé par FR5

**Given** `decode_segmentation_and_detect` et `decode_segmentation_and_detect_batch` (AD-6)
**When** elles sont implémentées
**Then** elles restent deux fonctions distinctes et indépendantes, chacune ratifiant son implémentation actuelle sans changement de comportement

**Given** la convention de format des boxes
**When** une fonction retourne des boxes
**Then** le format est `[x1, y1, x2, y2, score]` en liste Python (pas de tableau numpy vectorisé)

**Given** AD-7 (auteur unique)
**When** cette story est complétée
**Then** aucune story suivante ne modifie `inference_utils.py` pour y ajouter/changer une fonction — seules des imports sont faits par les stories 1.3 à 1.9

**Given** FR6 (gate de sécurité, référencé par AD-4)
**When** le fallback `model_name` par défaut est fixé
**Then** il a été vérifié au préalable que `best_model.pkl` (`sophisticated_cnn_128_plus`) et `best_model_detection.pkl` (`aircraft_detector_unet`) ne dépendent d'aucune architecture supprimée

### Story 1.3: Migration bounding_boxes_with_classification_from_video_generation.py

As a mainteneur du pipeline vidéo,
I want que bounding_boxes_with_classification_from_video_generation.py importe ses fonctions d'inférence depuis inference_utils.py au lieu de les redéfinir localement,
So that le pipeline vidéo temps réel utilise la même logique canonique que les autres consommateurs, sans duplication ni divergence.

**Acceptance Criteria:**

**Given** la Story 1.2 complétée
**When** le fichier est migré
**Then** il importe `load_jax_model`, `load_detection_model`, `predict_crops_batch`, `build_predict_fn`/`build_clf_predict_fn`, `get_iou`, `non_max_suppression`, `decode_segmentation_and_detect_batch` depuis `inference_utils.py`
**And** les définitions locales de ces fonctions sont supprimées du fichier

**Given** `predict_crops_batch` (AD-2, nom de paramètre `predict_fn` ratifié depuis ce fichier)
**When** le fichier est migré
**Then** l'appel à `predict_crops_batch` utilise la signature canonique sans changement d'usage côté appelant

**Given** le pipeline vidéo prioritaire en performance (AD-6, NFR3)
**When** la migration est effectuée
**Then** aucune dégradation du débit temps réel n'est introduite (comparaison avant/après sur un extrait vidéo)

**Given** la baseline capturée en Story 1.1 pour ce fichier
**When** le fichier migré est exécuté sur le même jeu d'images/vidéo
**Then** les boxes/classes/scores produits sont identiques à la baseline

### Story 1.4: Migration tools/bounding_boxes_with_classification_from_images_generation.py

As a mainteneur des outils d'inférence sur images,
I want que ce script importe ses fonctions d'inférence depuis inference_utils.py,
So that le traitement d'image statique utilise la même logique canonique que le pipeline vidéo.

**Acceptance Criteria:**

**Given** la Story 1.2 complétée
**When** le fichier est migré
**Then** il importe `predict_crop` (ratifié depuis ce fichier, ligne ~128, AD-2), `load_jax_model`, `load_detection_model`, `get_iou`, `non_max_suppression`, `decode_segmentation_and_detect` depuis `inference_utils.py`
**And** les définitions locales sont supprimées

**Given** `predict_crop` (AD-2, signature `(crop_img, model, variables, mean, std, config)`)
**When** le fichier est migré
**Then** aucun appel n'utilise l'ancienne signature `(crop_img, predict_fn, mean, std, config)` de `video_generation.py`, confirmée morte et non migrée

**Given** la baseline Story 1.1
**When** le script migré est exécuté sur le même jeu d'images
**Then** les résultats sont identiques à la baseline

### Story 1.5: Migration heatmap_generation.py

As a mainteneur de la génération de heatmaps,
I want que heatmap_generation.py importe load_detection_model depuis inference_utils.py,
So that le chargement de modèle bénéficie du fallback robuste (AD-3) au lieu de l'échec silencieux actuel.

**Acceptance Criteria:**

**Given** la Story 1.2 complétée
**When** `heatmap_generation.py` est migré
**Then** il importe `load_detection_model` depuis `inference_utils.py` et supprime sa propre définition locale (qui n'a aujourd'hui aucun fallback ni ré-init de `batch_stats`)

**Given** AD-3 (robustesse uniformisée)
**When** un checkpoint sans `batch_stats` sauvegardés est chargé via ce fichier migré
**Then** la ré-initialisation de structure s'applique au lieu d'un comportement silencieusement dégradé

**Given** la baseline Story 1.1
**When** le script migré est exécuté sur le même jeu d'images
**Then** les heatmaps produites sont identiques à la baseline (hors correction explicite du bug de robustesse AD-3, à documenter si déclenché)

### Story 1.6: Migration heatmap_contouring.py

As a mainteneur de la génération de heatmaps avec contours,
I want que heatmap_contouring.py importe load_detection_model depuis inference_utils.py,
So that le chargement de modèle bénéficie du fallback robuste (AD-3) au lieu de l'échec silencieux actuel.

**Acceptance Criteria:**

**Given** la Story 1.2 complétée
**When** `heatmap_contouring.py` est migré
**Then** il importe `load_detection_model` depuis `inference_utils.py` et supprime sa propre définition locale

**Given** AD-3
**When** un checkpoint sans `batch_stats` sauvegardés est chargé via ce fichier migré
**Then** la ré-initialisation de structure s'applique

**Given** la baseline Story 1.1
**When** le script migré est exécuté sur le même jeu d'images
**Then** les résultats sont identiques à la baseline (hors correction explicite du bug AD-3, à documenter si déclenché)

### Story 1.7: Migration tools/audit_dataset_classification.py

As a mainteneur des outils d'audit de dataset,
I want que tools/audit_dataset_classification.py compose load_jax_model + build_predict_fn au lieu de sa fonction locale load_classification_model,
So that le chargement de modèle et la construction du predict_fn utilisent la logique canonique partagée (AD-1).

**Acceptance Criteria:**

**Given** la Story 1.2 complétée
**When** `tools/audit_dataset_classification.py` est migré
**Then** sa fonction locale `load_classification_model` est supprimée et remplacée par l'appel explicite : `model, variables, mean, std = load_jax_model(...)` puis `predict_fn = build_predict_fn(model, variables)`
**And** `_preprocess_crop_to_hwc` est importée depuis `inference_utils.py` au lieu d'être redéfinie localement

**Given** la baseline Story 1.1
**When** le script migré est exécuté sur le même jeu de données d'audit
**Then** les résultats d'audit sont identiques à la baseline

### Story 1.8: Migration tools/audit_dataset_detection.py

As a mainteneur des outils d'audit de détection,
I want que tools/audit_dataset_detection.py importe ses fonctions d'inférence depuis inference_utils.py au lieu de l'espace de noms de bounding_boxes_with_classification_from_video_generation.py,
So that ce consommateur invisible ne dépende plus d'un fichier qui va lui-même être migré, évitant une casse silencieuse (FR10, AD-8).

**Acceptance Criteria:**

**Given** la Story 1.2 complétée
**When** `tools/audit_dataset_detection.py` est migré
**Then** ses imports actuels depuis `bounding_boxes_with_classification_from_video_generation.py` (`load_detection_model`, `get_iou`, `decode_segmentation_and_detect_batch`, `load_jax_model`, `predict_crops_batch` selon usage réel) sont repointés vers `inference_utils.py`
**And** aucun shim de compatibilité n'est laissé en place

**Given** la baseline Story 1.1
**When** le script migré est exécuté
**Then** les résultats d'audit sont identiques à la baseline

### Story 1.9: Migration tools/boxes_process_manual_tkinter.py

As a utilisateur de l'éditeur manuel de bounding boxes,
I want que tools/boxes_process_manual_tkinter.py importe ses fonctions d'inférence depuis inference_utils.py au lieu de l'espace de noms de bounding_boxes_with_classification_from_video_generation.py,
So that l'outil d'annotation manuelle continue de fonctionner correctement même après la migration du fichier vidéo dont il dépend actuellement (FR10, AD-8).

**Acceptance Criteria:**

**Given** la Story 1.2 complétée
**When** `tools/boxes_process_manual_tkinter.py` est migré
**Then** ses imports actuels depuis `bounding_boxes_with_classification_from_video_generation.py` sont repointés vers `inference_utils.py`
**And** le comportement de l'interface Tkinter (édition manuelle de boxes) reste inchangé

**Given** la baseline Story 1.1 (capture alternative documentée pour ce fichier interactif)
**When** l'outil migré est utilisé sur le même scénario de test
**Then** le comportement observé est identique à la baseline

### Story 1.10: Diff de non-régression (après migration complète)

As a mainteneur du pipeline d'inférence,
I want comparer les sorties post-migration des 7 fichiers concernés à la baseline capturée en Story 1.1,
So that la non-régression fonctionnelle de l'Epic 1 est prouvée par comparaison, pas par observation (Goal 1 PRD).

**Acceptance Criteria:**

**Given** la baseline capturée en Story 1.1 et les 7 fichiers migrés (Stories 1.3 à 1.9)
**When** chaque fichier est ré-exécuté sur le même jeu d'images/scénario que la baseline
**Then** les sorties (boxes, classes, scores) sont diffées automatiquement contre la baseline
**And** tout écart est soit nul, soit explicitement documenté et justifié (ex. correction de bug AD-3)

**Given** les Goals 2 et 3 du PRD (entraînement complet, régénération datasets)
**When** cette story est complétée
**Then** `main.py` est exécuté sans régression sur `FIGHTERJET_CLASSIFICATION` et `FIGHTERJET_DETECTION`

## Epic 2: Purge des configs et modèles morts

`dataset_configs.py` et `model_library.py` ne contiennent plus que les configs/architectures réellement utilisées par le pipeline (3 configs, 4 architectures), sans risque de casser le chargement des checkpoints existants ni un entraînement complet. **FRs covered:** FR4, FR5, FR6

### Story 2.1: Gate FR6 — vérification des dépendances .pkl

As a mainteneur du pipeline d'entraînement,
I want vérifier explicitement qu'aucun .pkl actuellement versionné (best_model.pkl, best_model_detection.pkl) ne dépend d'une architecture qui sera supprimée,
So that la suppression des architectures mortes (FR5) ne casse pas le chargement des checkpoints existants.

**Acceptance Criteria:**

**Given** `best_model.pkl` et `best_model_detection.pkl` versionnés dans le dépôt
**When** leur métadonnée de config/architecture est inspectée
**Then** `best_model.pkl` est confirmé lié à `sophisticated_cnn_128_plus` et `best_model_detection.pkl` à `aircraft_detector_unet` (architectures vivantes après FR5, cf. AD-4)
**And** si l'un des deux dépendait d'une architecture candidate à suppression, cette architecture est retirée de la liste de suppression de FR5 et documentée comme exception

**Given** AD-4 (gate bloquant, pas une note informative)
**When** cette story est complétée avec succès
**Then** elle constitue la condition préalable explicite au démarrage de la Story 2.3

### Story 2.2: Purge dataset_configs.py

As a mainteneur du fichier de configuration,
I want ne conserver que FIGHTERJET_CLASSIFICATION, FIGHTERJET_DETECTION, JAX_KEPLER dans dataset_configs.py,
So that le fichier de config ne référence plus de scénarios d'entraînement abandonnés.

**Acceptance Criteria:**

**Given** `dataset_configs.py` avec 7 configs actuelles
**When** le fichier est purgé
**Then** `FIGHTERJET_VIT`, `FIGHTERJET_HYBRID_VIT`, `FIGHTERJET_LETTERBOX`, `FIGHTERJET_DETECTION_SOPHISTICATED` sont supprimées
**And** `FIGHTERJET_CLASSIFICATION`, `FIGHTERJET_DETECTION`, `JAX_KEPLER` restent inchangées

**Given** qu'aucun autre fichier du repo ne référence les 4 configs supprimées par leur nom littéral
**When** la purge est effectuée
**Then** aucune régression d'import n'est introduite (vérification par recherche textuelle dans le repo avant suppression)

### Story 2.3: Purge model_library.py

As a mainteneur de la factory de modèles,
I want ne conserver que les 4 architectures référencées par les 3 configs restantes,
So that model_library.py ne porte plus 18 architectures mortes (~68% du fichier).

**Acceptance Criteria:**

**Given** la Story 2.1 complétée (gate FR6 passé) et la Story 2.2 complétée
**When** `model_library.py` est purgé
**Then** seules `sophisticated_cnn_128_plus`, `aircraft_detector_unet`, `aircraft_detector_miniunet`, `kepler_1d_cnn` restent dans `MODELS`
**And** les 18 autres architectures (`aircraft_detector`, `v2`-`v7_advanced`, `aircraft_detector_unet_token`, `sophisticated_cnn`, `sophisticated_cnn_droped_out`, `sophisticated_cnn_128_ultimate`, `sophisticated_cnn_optimized`, `resnet_light`, `tiny_vit_plus`, `tiny_vit_plus_ultimate`, `hybrid_tiny_vit`, `tiny_vit_plus_balanced`, `aircraft_detector_sophisticated_unet`) sont supprimées

**Given** AD-4 (fallback `model_name`)
**When** `get_model()` est appelé sans `model_name` explicite
**Then** le fallback pointe vers `aircraft_detector_unet`

### Story 2.4: Validation post-purge

As a mainteneur du pipeline,
I want confirmer qu'un entraînement complet et la régénération des datasets fonctionnent toujours après la purge,
So that la suppression des configs/modèles morts n'a introduit aucune régression fonctionnelle (Goals 2, 3 PRD).

**Acceptance Criteria:**

**Given** les Stories 2.2 et 2.3 complétées
**When** `main.py` est exécuté sur `FIGHTERJET_CLASSIFICATION` puis `FIGHTERJET_DETECTION`
**Then** l'entraînement tourne sans erreur jusqu'à son terme

**Given** `fighterjet_classification_dataset_tools.py` et `fighterjet_detection_dataset_tools.py`
**When** la régénération des datasets `.npz` est exécutée
**Then** elle fonctionne toujours sans erreur

**Given** `best_model.pkl` et `best_model_detection.pkl`
**When** ils sont rechargés via le pipeline post-purge
**Then** le chargement réussit sans erreur (confirmation finale de la Story 2.1)

## Epic 3: Suppression du code orphelin

Les vestiges confirmés (fusion historique, expérimentation abandonnée, script de benchmark devenu obsolète après AD-5) sont retirés du dépôt actif, sans perte d'information (récupérables via git). **FRs covered:** FR7, FR8, FR9

### Story 3.1: Suppression de train_detection.py

As a mainteneur du pipeline,
I want supprimer train_detection.py, vestige confirmé de la fusion historique jax_supervised_training/JAX_Classification,
So that le point d'entrée du pipeline reste unique et non ambigu (main.py).

**Acceptance Criteria:**

**Given** `train_detection.py` contourne le pattern Strategy actuel et pointe vers `aircraft_detector_v3` (architecture supprimée par FR5)
**When** le fichier est supprimé
**Then** aucun autre fichier du repo ne l'importe (vérifié avant suppression)
**And** le fichier reste récupérable via l'historique git (NFR4)

### Story 3.2: Suppression de tokens/

As a mainteneur du repo,
I want supprimer intégralement le dossier tokens/, expérience token-based model vs UNet abandonnée et jamais raccordée au pipeline actif,
So that j'élimine un dossier au nom trompeur mélangeant des sujets sans rapport.

**Acceptance Criteria:**

**Given** `tokens/` contient `read_token.py`/`write_token.py`, `genetic_algorithm.py`/`genetic_algorithm2.py`, `loss_detection_only_directory.py`/`loss_with_classification_images_directory.py`
**When** le dossier est supprimé intégralement
**Then** aucun autre fichier du repo n'importe depuis `tokens/` (vérifié avant suppression)
**And** le contenu reste récupérable via l'historique git (NFR4)

### Story 3.3: Suppression de bounding_boxes_with_classification_from_benchmark.py

As a mainteneur du code d'inférence,
I want supprimer intégralement bounding_boxes_with_classification_from_benchmark.py sans le fusionner dans inference_utils.py,
So that j'élimine un vestige dont la logique de décodage (decode_grid_and_detect) est spécifique aux architectures grid-based déjà supprimées par FR5.

**Acceptance Criteria:**

**Given** AD-5 (confirmé : aucun autre fichier du repo n'importe depuis lui)
**When** le fichier est supprimé
**Then** il n'est pas migré vers `inference_utils.py` (contrairement aux 7 fichiers de l'Epic 1)
**And** le fichier reste récupérable via l'historique git (NFR4)

**Given** ce fichier faisait initialement partie du scope de duplication détecté (`dead-code-and-duplication-audit.md`)
**When** la suppression est effectuée
**Then** le périmètre des 7 fichiers de l'Epic 1 (Story 1.1) l'exclut bien, cohérent avec le scope révisé du PRD

## Addendum — déviations approuvées post-implémentation

Deux déviations mineures aux AC telles qu'écrites ci-dessus, découvertes et approuvées pendant l'implémentation (voir les fichiers story correspondants dans `_bmad-output/implementation-artifacts/` pour le détail complet) :

- **Story 2.2** dit que les 3 configs survivantes "restent inchangées". En pratique, `FIGHTERJET_DETECTION.output_prefix` a été corrigé (`f"{DATA_ROOT}/chunks/detection"` → `f"{DATA_ROOT}/chunks/detection/dataset_detection"`) — un mismatch pré-existant avec le layout réel des chunks (local et Colab/Drive), révélé par la suppression de `FIGHTERJET_DETECTION_SOPHISTICATED` (dont l'`output_prefix` masquait le problème). Corrigé avec l'accord explicite de l'utilisateur. Voir `2-2-purge-dataset-configs-py.md`.
- **Story 1.3** ne liste pas `DETECTION_IMAGE_SIZE` dans les imports attendus de `bounding_boxes_with_classification_from_video_generation.py`. En pratique, cette constante est importée (au lieu d'être redéfinie localement) pour respecter AD-1 ("aucune redéfinition locale, même sous un autre nom"), plutôt que de dupliquer la valeur `(224, 224)`. Voir `1-3-migration-bounding-boxes-with-classification-from-video-generation-py.md`.

## Epic 4: Nettoyage technique post-refactor

Trois corrections techniques ponctuelles issues du backlog de la rétrospective Epic 1-3 (2026-07-12), traitées avec le même niveau de rigueur (baseline/vérification, dev notes) mais sans cérémonie PRD/Architecture complète — aucune n'introduit de nouvelle décision produit ou d'architecture.
**FRs covered:** aucune (hors scope FR1-FR10 du PRD refactor initial ; items de maintenance identifiés post-cycle)

### Story 4.1: Suppression du code Letterbox résiduel

As a mainteneur du repo,
I want supprimer le code Letterbox devenu obsolète/cassé et corriger la documentation trompeuse restante,
so that le repo ne contienne plus de script non-fonctionnel ni de docstring qui décrit un comportement différent du code réel.

**Acceptance Criteria:**

**Given** `generate_letterbox_dataset.py` importe `process_dataset_letterbox` et `balance_and_split_dataset` depuis `fighterjet_classification_dataset_tools.py`
**When** on vérifie l'existence de ces deux fonctions dans tout le repo
**Then** elles n'existent nulle part — ce script est déjà non-fonctionnel (lèverait une `ImportError` dès sa première ligne exécutée)
**And** le script est supprimé intégralement (récupérable via l'historique git)

**Given** `fighterjet_detection_dataset_tools.py` (docstring de la fonction de traitement détection) affirme "2. Applique Letterbox"
**When** le code réel juste en dessous effectue un "STRETCHED RESIZING (au lieu de Letterbox)"
**Then** le docstring est corrigé pour refléter fidèlement le comportement réel du code

### Story 4.2: Correction des warnings Colab (absl/CUDA)

As a utilisateur lançant l'entraînement sur Colab,
I want ne plus voir de warnings alarmants liés à CUDA/absl au démarrage,
so that je puisse distinguer un vrai problème d'un bruit de log cosmétique.

**Acceptance Criteria:**

**Given** `data_management.py` importe `tensorflow` uniquement pour le pipeline `tf.data` (chargement CPU, aucun calcul GPU requis côté TF — JAX gère seul le calcul GPU pour l'entraînement)
**When** TensorFlow est importé
**Then** TF est explicitement configuré pour ne pas rechercher de GPU (`tf.config.set_visible_devices([], 'GPU')`), rendant ce choix intentionnel plutôt qu'un échec d'initialisation CUDA affiché comme warning ("Could not find cuda drivers", "failed call to cuInit")

**Given** le warning `absl::InitializeLog()` apparaît avant toute sortie applicative
**When** on analyse son origine
**Then** il est documenté comme non-actionnable côté code applicatif — le message indique par construction des logs émis *avant* l'initialisation du logging (déjà en amont de `TF_CPP_MIN_LOG_LEVEL`, déjà positionné, qui ne le supprime pas) ; aucun correctif spéculatif n'est ajouté pour un warning inerte dont la cause est interne à TensorFlow

**Given** cette correction touche l'affichage de logs dans un environnement (Colab) que je ne peux pas reproduire localement
**When** le correctif est appliqué
**Then** il est explicitement marqué comme "à valider par l'utilisateur sur Colab", pas vérifié en local

### Story 4.3: Introduction de requirements.txt (résolution structurelle de l'incident cv2)

As a mainteneur du pipeline,
I want un fichier de dépendances déclaratif installable en une seule commande,
so that un runtime Colab frais (ou tout nouvel environnement) n'échoue plus sur un module manquant comme cv2, découvert un par un au fil des crashs.

**Acceptance Criteria:**

**Given** aucun fichier de dépendances n'existe actuellement (gap déjà documenté dans `docs/source-tree-analysis.md` § "Absence de packaging")
**When** `requirements.txt` est créé
**Then** il liste les dépendances tierces réellement importées dans le repo, vérifiées par recherche exhaustive (pas supposées) : `jax`, `flax`, `optax`, `opencv-python-headless`, `numpy`, `scipy`, `pandas`, `Pillow`, `matplotlib`, `tqdm`, `psutil`, `tensorflow`, `ultralytics`, `imagehash`
**And** `opencv-python-headless` (pas `opencv-python`) est utilisée — cohérent avec un environnement Colab sans display

**Given** cette story est la plus proche d'une décision d'architecture (introduction d'un mécanisme de gestion de dépendances)
**When** elle est prête à être implémentée
**Then** elle est soumise à l'agent architecte (Winston, `bmad-agent-architect`) pour avis avant merge — demande explicite de l'utilisateur

### Story 4.4: Archivage du code PyTorch/YOLO résiduel (pré-JAX)

As a mainteneur du repo,
I want déplacer (pas supprimer) le code lié à l'ancienne approche YOLO/PyTorch vers un dossier `archive/` documenté,
so that ce code historique reste traçable et récupérable sans polluer le pipeline actif ni la liste de dépendances par défaut.

**Acceptance Criteria:**

**Given** `tools/bounding_boxes_from_images_generation.py`, `tools/bounding_boxes_from_images_generation_main.py` (bootstrap d'annotation via YOLOv8n générique, PyTorch/Ultralytics) et `tools/yolov8n.pt` (checkpoint associé) ne sont importés par aucun autre fichier du repo (vérifié)
**And** l'utilisateur confirme qu'ils ont servi à un besoin réel (pré-génération de bounding boxes avant l'existence d'un modèle propriétaire) mais ne sont plus utilisés depuis que `aircraft_detector_unet` (JAX) existe
**When** ils sont déplacés vers `archive/` via `git mv` (historique préservé, pas une suppression)
**Then** `archive/README.md` documente leur contexte d'origine, pourquoi ils sont archivés (pas supprimés), et comment les réactiver

**Given** `tools/YOLOv8-n.py` (reproduction expérimentale de YOLOv8n en Flax/JAX, jamais finalisée, jamais importée non plus)
**When** il est déplacé vers `archive/`
**Then** `archive/README.md` documente que l'objectif était de reproduire en JAX un équivalent du modèle `yolov8n.pt`, jamais entraîné ni intégré à `model_library.py`

**Given** `docs/source-tree-analysis.md` référence l'ancien emplacement (`tools/`) de ces 4 fichiers
**When** l'archivage est effectué
**Then** la doc est mise à jour pour refléter le nouvel emplacement (`archive/`)

**Given** `ultralytics` (dépendance PyTorch de ces scripts) ne devient plus nécessaire par défaut une fois ces fichiers archivés
**When** `requirements.txt` (Story 4.3) est finalisé
**Then** `ultralytics` n'y figure ni en dépendance par défaut ni en note "à la demande" pour le pipeline principal — uniquement documentée dans `archive/README.md` pour la réactivation de ce cas d'usage spécifique

## Epic 5: Dataset CIFAR-10 pour boucle de test rapide

Le pipeline dispose d'un second jeu de données de classification, standard et léger (CIFAR-10, ~180 Mo), permettant d'itérer et de valider des changements de pipeline en quelques minutes sans dépendre du dataset avion complet (~10 Go) ni de Colab/TPU. Sert aussi de première preuve concrète de généricité du pipeline sur un cas image standard (`JAX_KEPLER` prouvait déjà la généricité côté 1D, pas côté image). Investigation préalable (Winston, `bmad-agent-architect`) : `data_management.py`, `loss_functions.py`, `reporting.py` et `model_library.py` (Global Average Pooling, pas de dimension spatiale figée) sont déjà génériques. Un vrai point de couplage a été découvert en creusant plus loin, non détecté à l'investigation initiale : `task_strategies.py` nomme les fichiers `.pkl` (export + training-state resumable) par `task_type` codé en dur, pas par config — sans conséquence tant qu'il n'existait qu'une seule config par `task_type`, mais CIFAR10 introduit la 2ᵉ config `classification` et ferait collision avec `FIGHTERJET_CLASSIFICATION` sur le même fichier de training-state. Traité comme un gate (Story 5.0) avant toute génération de données CIFAR-10.
**FRs covered:** aucune (hors scope FR1-FR10 du PRD refactor initial ; nouvelle capacité identifiée post-cycle, cf. `ARCHITECTURE-SPINE.md` § Deferred "Généralisation complète du pipeline")

### Story 5.0: Nommage des checkpoints dérivé de la config (gate)

As a mainteneur du pipeline,
I want que le nom des fichiers `.pkl` (export final et training-state resumable) soit dérivé du nom de la config utilisée plutôt que du `task_type`,
So that deux configs partageant le même `task_type` (ex. `FIGHTERJET_CLASSIFICATION` et `CIFAR10`, toutes deux `classification`) n'écrasent/ne lisent pas le même fichier de checkpoint.

**Acceptance Criteria:**

**Given** `dataset_configs.py::get_dataset_config(dataset_name)` retourne aujourd'hui un dict sans le nom de la config qu'il contient
**When** la fonction est modifiée
**Then** elle injecte `config["dataset_name"] = dataset_name` dans le dict retourné, avant validation

**Given** les 3 implémentations actuelles de `_get_export_path`/`get_training_state_path` (`ClassificationStrategy`, `DetectionStrategy`, `KeplerStrategy` dans `task_strategies.py`) — dont certaines ignorent totalement `config["checkpoint_path"]` (`DetectionStrategy._get_export_path`) et dont aucune ne lit un équivalent pour le training-state
**When** elles sont uniformisées
**Then** chacune retourne `config.get("checkpoint_path")` (export) ou `config.get("training_state_path")` (training-state) si fourni explicitement, sinon un nom par défaut dérivé de `best_model_{config["dataset_name"].lower()}.pkl` / `best_model_training_state_{config["dataset_name"].lower()}.pkl`

**Given** les 3 configs existantes (`FIGHTERJET_CLASSIFICATION`, `FIGHTERJET_DETECTION`, `JAX_KEPLER`) et les fichiers `.pkl` déjà versionnés/produits sous leurs noms actuels (`best_model.pkl`, `best_model_detection.pkl`, `best_model_training_state_classification.pkl`)
**When** la Story 5.0 est complétée
**Then** `checkpoint_path` reste explicite et inchangé pour les 3 configs (déjà le cas) **et** un `training_state_path` explicite est ajouté à chacune des 3, verrouillant le nom actuellement codé en dur (`best_model_training_state_classification.pkl`, `best_model_training_state_detection.pkl`, `best_model_training_state_kepler.pkl`) — aucun renommage silencieux d'un fichier déjà en usage

**Addendum (découvert pendant la validation locale, Story 5.2)** : le run CIFAR10 a silencieusement écrasé `sophisticated_cnn_128_plus.png` (courbes d'entraînement fighterjet, committé dans `2628ef6`) — même classe de bug, non détectée par le grep initial car dans `trainer.py` (`TrainingVisualizer`) et non dans `task_strategies.py`. `trainer.py:448` sauvegardait sous `f"{self.model_name}.png"` (nom d'**architecture**, partagée par CIFAR10 et FIGHTERJET_CLASSIFICATION), pas sous un nom dérivé de la config. Corrigé sur le même principe : `f"{self.config['dataset_name'].lower()}.png"`. Contrairement aux `.pkl`, aucune préservation de nom n'a été faite pour les configs existantes (fichier régénéré à chaque epoch, pas un artefact critique/versionné à préserver) — `sophisticated_cnn_128_plus.png` reste tel que restauré depuis `2628ef6` mais deviendra orphelin (un futur training `FIGHTERJET_CLASSIFICATION` produira désormais `fighterjet_classification.png`). Fichier CIFAR10 initialement écrasé restauré depuis git puis sa version CIFAR10 sauvegardée séparément (scratchpad) avant correctif.

**Addendum 2 (post-validation, retour utilisateur)** : après le premier run local réussi (`sophisticated_cnn_128_plus`, ~79,5% accuracy val en 10 epochs — **correctif** : une version antérieure de cet addendum citait par erreur 0.9448, qui est en réalité le résultat `FIGHTERJET_CLASSIFICATION` bfloat16/256×2 de `BENCHMARK-TPU-PERFORMANCE.md`, sans rapport avec CIFAR10), l'utilisateur a fait remarquer que réutiliser `sophisticated_cnn_128_plus` tel quel pour du 32×32 était surdimensionné (3 max-pools 32→16→8→4, pic à 512 canaux — taillé pour des silhouettes d'avion 128×128, pas pour CIFAR-10). Une nouvelle architecture `sophisticated_cnn_32_plus` a été ajoutée à `model_library.py` (même famille de blocs : SeparableConv, résiduelles, SE, Spatial Attention, tête GAP), avec 2 max-pools au lieu de 3 (32→16→8, pas de réduction jusqu'à 4×4) et des canaux à peu près divisés par 2 à chaque étage (pic à 256 au lieu de 512) — **318 691 paramètres vs 1 257 041** (~4× moins), vérifié par instanciation directe (forward pass train/eval, shapes de sortie correctes). `CIFAR10.model_name` mis à jour vers `sophisticated_cnn_32_plus`. Les paramètres d'augmentation de données de `CIFAR10` étaient déjà minimaux (seul `flip_h=True` actif, tout le reste à 0.0) — pas de changement nécessaire sur ce point, confirmé à l'utilisateur plutôt que modifié à l'aveugle.

**Addendum 3 (runs Colab successifs `sophisticated_cnn_32_plus`, tuning epochs/LR/dropout)** : premier run Colab (`archive/training_cifar10_log_bfloat_128x1.txt` — malgré son nom, le log confirme `float16`, pas bfloat16 ; `epochs=10`, `decay_steps=2000` hérité du réglage précédent) a plafonné à **73,30%** (val accuracy), stagnation nette dès l'epoch 5 (val loss figé 0.7601-0.7611 de l'epoch 6 à 10). Diagnostiqué comme un bug de LR schedule, pas une limite de capacité : avec 391 steps/epoch, `decay_steps=2000` fait tomber le LR à `end_value=1e-6` au step ~2000 (epoch ~5,1/10), gelant l'apprentissage pour le reste du run. Corrigé : `epochs` 10→30, `patience` 3→5, `decay_steps` 2000→11700 (≈ steps/epoch × epochs, couvre tout l'entraînement). Deuxième run (`archive/training_cifar10_log_float16_128x1.txt`) : **79,49%** (meilleur, epoch 10), mais overfitting net ensuite — train accuracy 86,97%→93,22% (epoch 10→15) pendant que le val loss remonte 0.6325→0.7442 au lieu de continuer à baisser ; early stopping déclenché correctement à l'epoch 15 (patience=5 depuis le meilleur). Diagnostiqué comme absence totale de régularisation (`dropout_rate=0.0` dans les deux runs ci-dessus), pas comme un manque de capacité (le train accuracy grimpant librement au-delà de 93% indique une capacité largement suffisante). Corrigé : `dropout_rate` 0.0→0.3 (`gpu` et `tpu`), aligné sur la valeur déjà utilisée par `JAX_KEPLER`. Résultat de ce 3ᵉ run non encore capturé au moment de la revue de code — à mettre à jour si un futur cycle reprend ce sujet.

### Story 5.1: Génération des chunks CIFAR-10 au format .npz

As a mainteneur du pipeline,
I want un script dédié (`cifar10_classification_dataset_tools.py`) qui télécharge le CIFAR-10 officiel (pickle, cs.toronto.edu) et génère les chunks `.npz` (`image`/`label`) et le fichier `meanstd.npz` dans le format exact attendu par `ChunkManager`,
So that le dataset soit consommable par le pipeline existant sans aucune modification de `data_management.py`.

**Acceptance Criteria:**

**Given** CIFAR-10 n'est disponible ni via un dossier d'images ni via un chargeur déjà présent dans le repo
**When** le script `cifar10_classification_dataset_tools.py` s'exécute
**Then** il télécharge `cifar-10-python.tar.gz` depuis `https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz` (source officielle) s'il n'est pas déjà présent localement, l'extrait, et parse les 5 batches `data_batch_1..5` (train, 50 000 images) et `test_batch` (val, 10 000 images) via `pickle`
**And** aucune fonction de `fighterjet_classification_dataset_tools.py` n'est réutilisée pour le chargement/décodage (conçues pour un dossier `.png`/`.jpg`, format non applicable ici)

**Given** le layout natif des batches CIFAR-10 (chaque image stockée en 3072 octets : 1024 rouge, 1024 vert, 1024 bleu, aplati ligne par ligne — pas HWC)
**When** les images sont décodées
**Then** chaque ligne de 3072 octets est reshape en `(3, 32, 32)` puis transposée en `(32, 32, 3)` (HWC) avant normalisation `/255.0` en float32

**Given** le contrat `.npz` attendu par `ChunkManager` (`data_management.py`)
**When** les chunks sont écrits
**Then** le split `train` produit `{output_prefix}_train_chunk0.npz` (clés `image` float32 `(N, 32, 32, 3)`, `label` int32 `(N,)`) et le split `test` produit `{output_prefix}_val_chunk0.npz` selon le même format

**Given** `ChunkManager` exige un fichier `{output_prefix}_meanstd.npz` (clés `mean`/`std`) au chargement
**When** le script termine la génération
**Then** ce fichier est calculé sur le split `train` et sauvegardé au bon emplacement, sans quoi `get_datasets()` lèverait un `FileNotFoundError`

### Story 5.2: Entrée CIFAR10 dans dataset_configs.py et validation locale

As a mainteneur du pipeline,
I want une nouvelle entrée `CIFAR10` dans `dataset_configs.py` réutilisant `ClassificationStrategy` sans modification,
So that j'obtienne une boucle d'entraînement/test complète, exécutable localement en quelques minutes, pour valider rapidement tout changement de pipeline.

**Acceptance Criteria:**

**Given** les Stories 5.0 (nommage checkpoints) et 5.1 (chunks + `meanstd.npz`) complétées
**When** l'entrée `CIFAR10` est ajoutée à `dataset_configs.py`
**Then** elle définit `num_classes=10`, les 10 `class_names` standards CIFAR-10, `image_size=(32, 32)`, `grayscale=False`, `model_name="sophisticated_cnn_128_plus"` (réutilisé tel quel — architecture à Global Average Pooling, aucune modification requise) ⚠️ **superseded par l'Addendum 2 ci-dessous** : `model_name` a été changé pour `sophisticated_cnn_32_plus` (nouvelle architecture, `model_library.py` modifié) suite au retour utilisateur post-validation — cette puce d'AC ne reflète plus le code livré, `loss_method="cross_entropy"` (dataset équilibré par construction, contrairement à `FIGHTERJET_CLASSIFICATION` qui utilise `focal_loss`)
**And** aucun `checkpoint_path`/`training_state_path` explicite n'est requis — le nommage dérivé de `dataset_name` (Story 5.0) s'applique par défaut (`best_model_cifar10.pkl`, `best_model_training_state_cifar10.pkl`)
**And** aucune modification n'est faite à `model_library.py`, `loss_functions.py`, `reporting.py` ou `data_management.py`

**Given** l'objectif explicite d'une boucle de test locale rapide (contrairement à `FIGHTERJET_CLASSIFICATION`/`FIGHTERJET_DETECTION` dont l'entraînement complet doit rester réservé à Colab/TPU pour éviter un crash mémoire local)
**When** `main.py CIFAR10` est exécuté localement (CPU/GPU) sur un nombre réduit d'epochs
**Then** l'entraînement se déroule sans erreur jusqu'à son terme, confirmant par l'exécution (pas seulement par lecture de code) la généricité du pipeline sur un second cas d'usage image
**And** les fichiers `.pkl` produits (`best_model_cifar10.pkl`, `best_model_training_state_cifar10.pkl`) sont distincts de ceux de `FIGHTERJET_CLASSIFICATION`, confirmant par l'exécution que la Story 5.0 élimine bien la collision

## Requirements Inventory — Renommage jax_supervised_training

Source : `_bmad-output/planning-artifacts/prds/prd-JAX_Detection-2026-07-14/prd.md` (status: final). Initiative distincte du refactor Epic 1-3 — pas de document Architecture dédié (renommage mécanique, pas de nouvelle décision technique), cohérent avec le PRD lui-même.

### Functional Requirements

FR1: Le dossier local du projet et le dépôt distant `aobled/JAX_Detection` sont renommés en `jax_supervised_training` ; le remote git local est mis à jour vers la nouvelle URL.
FR2: La variable d'environnement `JAX_DETECTION_DATA_ROOT` (seul site de lecture : `dataset_configs.py::DATA_ROOT`) est renommée en `JAX_SUPERVISED_TRAINING_DATA_ROOT`, sans compatibilité double-nom. Tout notebook Colab actif définissant cette variable est mis à jour dans le même geste.
FR3: Le dossier Google Drive `MyDrive/JAX_Detection/` est renommé en `MyDrive/jax_supervised_training/`. Tout notebook Colab actif référençant l'ancien chemin est mis à jour dans le même geste. `tools/process rclone GDrive and run collab.txt` (pense-bête manuel) est explicitement hors scope, laissé à la charge de l'utilisateur.
FR4: Toute référence textuelle à "JAX_Detection"/"JAX_DETECTION" dans le code activement importé (6 fichiers), `docs/` (6 fichiers) et les artefacts BMAD **vivants** (`sprint-status.yaml`, `epics.md`) est mise à jour. Les artefacts BMAD historiques datés (dossiers PRD/architecture snapshotés du 2026-07-12, rétrospectives passées, `docs/project-scan-report.json`) restent explicitement non touchés.

### NonFunctional Requirements

NFR1 — Non-régression fonctionnelle : un entraînement lancé après renommage (`FIGHTERJET_CLASSIFICATION` ou `CIFAR10`, sur Colab) produit un comportement identique à avant renommage — seuls les noms/chemins changent. Validé par exécution réelle, pas par lecture de code.
NFR2 — Pas d'échec silencieux : si un point de lecture de l'ancien nom (variable d'environnement, chemin Drive) est oublié, l'échec doit être détectable rapidement (erreur explicite), jamais un fallback silencieux vers un chemin local inexistant sur Colab.

### Additional Requirements (Architecture)

Aucun — pas de document Architecture pour cette initiative (confirmé avec l'utilisateur).

### UX Design Requirements

N/A — aucune interface utilisateur, pas de parcours à documenter (projet solo, un seul opérateur).

## Epic List

### Epic 6 : Renommage du projet en jax_supervised_training

Le projet porte un nom cohérent sur toutes ses surfaces actives (dossier local, dépôt GitHub, variable d'environnement, dossier Google Drive, code, documentation) — reflétant la généricité déjà prouvée du pipeline, sans laisser de référence résiduelle à l'ancien nom dans un chemin actif.
**FRs covered:** FR1, FR2, FR3, FR4

### FR Coverage Map

FR1: Epic 6 — Renommage dossier local + dépôt GitHub
FR2: Epic 6 — Renommage variable d'environnement `JAX_DETECTION_DATA_ROOT`
FR3: Epic 6 — Renommage dossier Google Drive
FR4: Epic 6 — Mise à jour références textuelles (code, docs, artefacts vivants)

## Epic 6: Renommage du projet en jax_supervised_training

Le projet porte un nom cohérent sur toutes ses surfaces actives (dossier local, dépôt GitHub, variable d'environnement, dossier Google Drive, code, documentation) — reflétant la généricité déjà prouvée du pipeline, sans laisser de référence résiduelle à l'ancien nom dans un chemin actif. **FRs covered:** FR1, FR2, FR3, FR4

### Story 6.1: Renommage du dossier local et du dépôt GitHub

As a mainteneur du projet,
I want renommer le dossier local et le dépôt GitHub en `jax_supervised_training`,
So that l'identité du projet reflète sa portée réelle avant toute autre étape de renommage.

**Acceptance Criteria:**

**Given** le dossier local `JAX_Detection` et le dépôt distant `aobled/JAX_Detection`
**When** le dossier est renommé et `gh repo rename` (ou équivalent) est exécuté
**Then** le dossier local répond au nouveau nom et le remote local (`git remote -v`) pointe vers la nouvelle URL
**And** l'ancienne URL GitHub redirige encore vers le nouveau dépôt (comportement natif, vérifié)

### Story 6.2: Renommage de la variable d'environnement et du dossier Google Drive

As a mainteneur du projet,
I want renommer `JAX_DETECTION_DATA_ROOT` en `JAX_SUPERVISED_TRAINING_DATA_ROOT` et le dossier Drive `MyDrive/JAX_Detection/` en `MyDrive/jax_supervised_training/`,
So that le pipeline de données fonctionne sous le nouveau nom sans mécanisme de compatibilité.

**Acceptance Criteria:**

**Given** `dataset_configs.py::DATA_ROOT` (seul site de lecture de la variable) et le dossier Drive utilisé par les notebooks Colab
**When** la variable et le dossier sont renommés
**Then** `dataset_configs.py` ne référence plus `JAX_DETECTION_DATA_ROOT`, sans fallback vers l'ancien nom
**And** chaque notebook Colab actif est mis à jour vers la nouvelle variable et le nouveau chemin Drive, dans le même geste — pas en différé (checklist explicite avant/après, ce mode d'échec s'étant déjà produit une fois sur ce projet)
**And** `tools/process rclone GDrive and run collab.txt` n'est pas touché — laissé à la charge de l'utilisateur (hors scope confirmé)

### Story 6.3: Mise à jour des références textuelles — code, documentation, artefacts vivants

As a mainteneur du projet,
I want mettre à jour toute référence textuelle à "JAX_Detection" dans le code actif, `docs/` et les artefacts BMAD vivants,
So that aucune trace de l'ancien nom ne subsiste dans une source consultée en continu.

**Acceptance Criteria:**

**Given** les 6 fichiers de code identifiés (`dataset_configs.py`, `inference_utils.py`, `reporting.py`, `bounding_boxes_with_classification_from_video_generation.py`, `tools/bounding_boxes_with_classification_from_images_generation.py`, `tools/kepler_dataset_tools.py`) et les 6 fichiers `docs/`
**When** ils sont mis à jour
**Then** aucun ne contient plus "JAX_Detection"/"JAX_DETECTION"
**And** `sprint-status.yaml` (champ `project:`) et `epics.md` sont mis à jour — traités comme artefacts vivants, pas historiques
**And** les artefacts BMAD historiques datés (dossiers PRD/architecture du 2026-07-12, rétrospectives passées, `docs/project-scan-report.json`) restent explicitement non touchés

### Story 6.4: Validation finale par exécution réelle

As a mainteneur du projet,
I want lancer un entraînement Colab complet après renommage,
So that la non-régression fonctionnelle (NFR1) et l'absence d'échec silencieux (NFR2) soient prouvées par l'exécution, pas par la lecture de code.

**Acceptance Criteria:**

**Given** les Stories 6.1 à 6.3 complétées
**When** `main.py` est exécuté sur Colab (`FIGHTERJET_CLASSIFICATION` ou `CIFAR10`) sous le nouveau nom
**Then** l'entraînement démarre et se déroule sans erreur de chemin, avec un comportement identique à avant renommage
**And** si un point de lecture de l'ancien nom avait été oublié, l'échec se serait manifesté immédiatement et explicitement (pas un fallback silencieux) — confirmé a posteriori par l'absence d'un tel incident
