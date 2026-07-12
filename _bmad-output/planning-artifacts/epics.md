---
stepsCompleted: [1, 2, 3]
inputDocuments:
  - _bmad-output/planning-artifacts/prds/prd-JAX_Detection-2026-07-12/prd.md
  - _bmad-output/planning-artifacts/architecture/architecture-JAX_Detection-2026-07-12/ARCHITECTURE-SPINE.md
  - docs/dead-code-and-duplication-audit.md
  - docs/architecture.md
  - docs/source-tree-analysis.md
---

# Refactor JAX_Detection — Epic Breakdown

## Overview

This document provides the complete epic and story breakdown for the JAX_Detection refactor (nettoyage code mort et duplications), decomposing the requirements from the PRD and the Architecture Spine into implementable stories. No UX design contract applies — this is an internal code refactor with no user-facing interface changes.

## Requirements Inventory

### Functional Requirements

FR1: Un module partagé unique (`inference_utils.py`) contient les implémentations canoniques de `load_jax_model`, `load_detection_model`, `predict_crop`, `predict_crops_batch`, `build_predict_fn`, `build_clf_predict_fn`, `get_iou`, `non_max_suppression`, `_preprocess_crop_to_hwc`, `decode_segmentation_and_detect`, `decode_segmentation_and_detect_batch` — 11 fonctions au total.
FR2: Les 5 fichiers du scope initial (`bounding_boxes_with_classification_from_video_generation.py`, `tools/bounding_boxes_with_classification_from_images_generation.py`, `heatmap_generation.py`, `heatmap_contouring.py`, `tools/audit_dataset_classification.py`) importent depuis ce module au lieu de redéfinir localement. Voir FR10 pour 2 fichiers supplémentaires découverts en phase Architecture.
FR3: Pour chaque fonction ayant des comportements divergents entre fichiers, un comportement canonique est choisi explicitement avant fusion (pas de fusion mécanique aveugle). Quand la divergence reflète un vrai compromis (ex. temps réel vs précision), les deux comportements peuvent coexister comme fonctions distinctes plutôt que d'être forcés en un seul.
FR4: `dataset_configs.py` ne conserve que `FIGHTERJET_CLASSIFICATION`, `FIGHTERJET_DETECTION`, `JAX_KEPLER`. Suppression de `FIGHTERJET_VIT`, `FIGHTERJET_HYBRID_VIT`, `FIGHTERJET_LETTERBOX`, `FIGHTERJET_DETECTION_SOPHISTICATED`.
FR5: `model_library.py` ne conserve que les architectures référencées par les 3 configs restantes : `sophisticated_cnn_128_plus`, `aircraft_detector_unet`, `aircraft_detector_miniunet`, `kepler_1d_cnn`. Les 18 autres architectures de `MODELS` sont supprimées.
FR6: Avant suppression d'une architecture, vérifier qu'aucun `.pkl` actuellement versionné (`best_model.pkl`, `best_model_detection.pkl`) n'en dépend.
FR7: `train_detection.py` est supprimé (vestige confirmé de la fusion historique JAX_Detection/JAX_Classification).
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
I want supprimer train_detection.py, vestige confirmé de la fusion historique JAX_Detection/JAX_Classification,
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
