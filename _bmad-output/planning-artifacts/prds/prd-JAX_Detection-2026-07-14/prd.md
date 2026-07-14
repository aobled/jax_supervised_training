---
title: Renommage de JAX_Detection en jax_supervised_training
created: 2026-07-14
updated: 2026-07-14
status: final
---

# PRD : Renommage de JAX_Detection en jax_supervised_training

## 0. Document Purpose

Ce PRD traduit en exigences le brief produit déjà validé (`_bmad-output/planning-artifacts/briefs/brief-JAX_Detection-2026-07-14/brief.md`) : renommer le projet `JAX_Detection` en `jax_supervised_training` pour que le nom reflète la généricité déjà prouvée du pipeline, sans engager de nouveau travail de généralisation (voir §4 Non-Goals). Projet solo, enjeux bas — ce document reste court et sert directement de base à la liste d'epics/stories.

## 1. Vision

`JAX_Detection` a démarré comme un pipeline de détection/classification d'avions. Deux preuves de concept — `JAX_KEPLER` (classification 1D de courbes de lumière d'exoplanètes) et `CIFAR10` (Epic 5, classification d'images standard) — tournent déjà sur le même `Trainer`/`TaskStrategy` sans duplication de code. Le nom du projet ne reflète plus ce qu'il fait.

Le code est jugé suffisamment stable pour absorber ce changement maintenant ; le sujet a été noté et reporté à deux rétrospectives consécutives sans être traité. Ce cycle renomme le projet — dossier local, dépôt GitHub, variable d'environnement, dossier Google Drive, références documentaires.

## 2. Contexte d'usage

Usage strictement personnel, pas de portage ni de partage prévu. Job à faire : *"en tant qu'auteur unique du projet, je veux que son nom, ses chemins et ses variables reflètent ce qu'il fait réellement, pour ne plus avoir à réexpliquer (même à moi-même) pourquoi un pipeline générique s'appelle `JAX_Detection`."* Pas de parcours utilisateur à documenter — un seul opérateur, pas d'interface, pas de surface produit au sens habituel du terme.

## 3. Fonctionnalités

Toutes les surfaces qui portent le nom `JAX_Detection` aujourd'hui (dossier local, dépôt distant, variable d'environnement, dossier de stockage des données, code actif, documentation) sont renommées en `jax_supervised_training`, de façon à ce qu'aucune référence résiduelle au nom actuel ne subsiste dans un chemin actif.

### FR-1 : Renommage du dossier local et du dépôt GitHub

Le dossier local du projet et le dépôt distant `aobled/JAX_Detection` sont renommés en `jax_supervised_training`.

**Conséquences (vérifiables) :**
- Le dossier local répond au nouveau nom.
- `gh repo rename` (ou équivalent) exécuté ; le remote local (`git remote -v`) pointe vers la nouvelle URL.
- L'ancienne URL GitHub redirige encore vers le nouveau dépôt (comportement natif GitHub, pas d'action requise).

### FR-2 : Renommage de la variable d'environnement `JAX_DETECTION_DATA_ROOT`

La variable d'environnement lue par `dataset_configs.py::DATA_ROOT` est renommée en `JAX_SUPERVISED_TRAINING_DATA_ROOT`, sans compatibilité double-nom.

**Conséquences (vérifiables) :**
- `dataset_configs.py` ne référence plus `JAX_DETECTION_DATA_ROOT` (seul site de lecture actuel).
- Tout notebook Colab actif définissant cette variable est mis à jour vers le nouveau nom, dans le même geste que le renommage du code — pas en différé. Checklist explicite avant/après renommage : lister les notebooks actifs, vérifier chacun après le renommage — ce mode d'échec s'est déjà produit une fois sur ce projet (voir NFR-2).

**Hors scope de ce FR :** aucun mécanisme de fallback vers l'ancien nom.

### FR-3 : Renommage du dossier Google Drive

Le dossier `MyDrive/JAX_Detection/` utilisé par les notebooks Colab est renommé en `MyDrive/jax_supervised_training/`.

**Conséquences (vérifiables) :**
- Le dossier Drive répond au nouveau nom.
- Chaque notebook Colab actif référençant l'ancien chemin est mis à jour vers le nouveau chemin, dans le même geste que FR-2 (même variable d'environnement, même risque d'échec silencieux si dissocié).

**Hors scope de ce FR :** `tools/process rclone GDrive and run collab.txt` — pense-bête manuel de commandes (pas du code exécuté par le pipeline), explicitement laissé à la charge de l'utilisateur, à traiter en dehors de ce cycle. Contient `rclone sync` vers `gdrive:JAX_Detection` et `/content/drive/MyDrive/JAX_Detection` — couplé à ce FR, donc à mettre à jour manuellement le jour où l'utilisateur l'utilise à nouveau, pas avant.

### FR-4 : Mise à jour des références textuelles dans le code actif et la documentation

Toute référence textuelle à "JAX_Detection" / "JAX_DETECTION" dans le code activement importé et dans `docs/` est mise à jour vers le nouveau nom.

**Conséquences (vérifiables) :**
- Les 6 fichiers de code identifiés (`dataset_configs.py`, `inference_utils.py`, `reporting.py`, `bounding_boxes_with_classification_from_video_generation.py`, `tools/bounding_boxes_with_classification_from_images_generation.py`, `tools/kepler_dataset_tools.py`) ne contiennent plus le nom actuel.
- Les fichiers de `docs/` (`architecture.md`, `dead-code-and-duplication-audit.md`, `development-guide.md`, `index.md`, `project-overview.md`, `source-tree-analysis.md`) sont mis à jour.
- Artefacts BMAD **vivants** (définition en §4 Non-Goals) : `_bmad-output/implementation-artifacts/sprint-status.yaml` (champ `project:`) et `_bmad-output/planning-artifacts/epics.md` sont mis à jour. `[ASSUMPTION: distinction "vivant vs. historique" tranchée en §4 — le brief ne précisait pas ces fichiers explicitement ; à confirmer.]`

**Hors scope de ce FR :** les artefacts BMAD historiques datés, y compris `docs/project-scan-report.json` (voir §4 Non-Goals).

## 4. Non-Goals (Explicit)

- **Pas de nouveau travail de généralisation du pipeline.** La limite révélée par `JAX_KEPLER` (modèle `Kepler1DConvNet` avec convolution 1D codée en dur, absence de notion de format d'input/output configurable dans `dataset_configs.py`) n'est pas traitée ici. La généricité actuelle tient à la souplesse du pattern Strategy + Factory, pas à une configuration déclarative du format de données — une base qui fonctionne mais reste fragile face à un futur cas plus exotique. Cette généralisation est différée à un cycle futur, probablement précédée d'un spike architecture.
- **Pas de réécriture des artefacts BMAD historiques datés** — `_bmad-output/planning-artifacts/prds/prd-JAX_Detection-2026-07-12/`, `architecture-JAX_Detection-2026-07-12/`, rétrospectives passées (`epic-1-3-retro-*.md`, `epic-4-5-retro-*.md`), et `docs/project-scan-report.json` (rapport généré une fois le 2026-07-12 par un scan de projet, horodaté comme les dossiers PRD/architecture). **Critère de distinction vivant/historique** : un artefact **vivant** est mis à jour en continu au fil des cycles (`sprint-status.yaml`, `epics.md` — voir FR-4) ; un artefact **historique** est généré une fois à une date donnée et n'est plus jamais modifié après coup. Ce sont des snapshots d'un moment où le projet portait ce nom ; pas une source à maintenir en synchronisation avec le nom courant. *(Ce PRD lui-même, une fois finalisé sous `prd-JAX_Detection-2026-07-14/`, rejoint cette catégorie — son dossier ne sera pas renommé rétroactivement après l'exécution du renommage.)*
- **Pas de renommage des checkpoints `.pkl` existants** — vérifié : leur contenu (`params`, `batch_stats`, `config`) ne référence aucun nom de projet, seulement `dataset_name` (déjà indépendant du nom du repo). Aucun impact.

## 5. Exigences transverses (NFR)

- **NFR-1 — Non-régression fonctionnelle :** un entraînement lancé après renommage (`FIGHTERJET_CLASSIFICATION` ou `CIFAR10`, sur Colab) produit un comportement identique à avant renommage — seuls les noms/chemins changent, pas la logique. Validé par exécution réelle, pas par lecture de code (méthode déjà appliquée sur ce projet).
- **NFR-2 — Pas d'échec silencieux :** si un point de lecture de l'ancien nom (variable d'environnement, chemin Drive) est oublié lors de l'exécution, l'échec doit être détectable rapidement (erreur explicite au démarrage), pas un fallback silencieux vers un chemin local inexistant sur Colab suivi d'une erreur tardive en cours d'exécution — le mode d'échec déjà rencontré une fois sur ce projet.

## 6. Success Metrics

*Enjeux bas (projet solo) — critères qualitatifs suffisent, pas de tableau de bord quantitatif.*

- **SM-1** : Dépôt, dossier local, variable d'environnement et dossier Drive portent tous `jax_supervised_training`. Aucune référence résiduelle au nom actuel dans un chemin actif. Valide FR-1 à FR-4.
- **SM-2** : Un entraînement Colab démarre sans erreur de chemin après renommage. Valide NFR-1, NFR-2.
- **SM-3** : Les artefacts BMAD historiques datés restent intacts et lisibles. Valide §4 Non-Goals.

## 7. Assumptions Index

- FR-4 — `sprint-status.yaml` et `epics.md` traités comme artefacts vivants à mettre à jour (définition complète en §4 Non-Goals). À confirmer si cette distinction ne correspond pas à l'intention.
