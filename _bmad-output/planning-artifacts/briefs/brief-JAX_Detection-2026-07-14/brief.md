---
title: "Brief produit — Renommage de JAX_Detection en jax_supervised_training"
status: validated
created: 2026-07-14
updated: 2026-07-14
---

# Brief produit : Renommage de JAX_Detection en jax_supervised_training

## Résumé exécutif

`JAX_Detection` a démarré comme un pipeline de détection/classification d'avions. Deux preuves de concept ultérieures — `JAX_KEPLER` (classification 1D de courbes de lumière d'exoplanètes) et `CIFAR10` (Epic 5, classification d'images standard 32×32 RGB) — ont montré que le `Trainer`/`TaskStrategy` sous-jacent fonctionne déjà pour des tâches sans rapport avec l'aviation, sans duplication de code. Le nom du projet ne reflète plus ce qu'il fait réellement.

Ce cycle acte cet état de fait : renommer le projet (dossier local, dépôt GitHub, variable d'environnement, dossier Google Drive, références documentaires) pour que le nom corresponde à la réalité actuelle du code — sans engager de nouveau travail de généralisation. Au-delà du cosmétique, cela acte publiquement (pour l'utilisateur lui-même) la vocation configurable du projet, dans la continuité de la démarche de normalisation déjà engagée (Epics 1-5) ; le sujet a été noté et reporté à deux rétrospectives consécutives (Epics 1-3, Epics 4-5) sans être traité, signe qu'il ne se traitera pas tout seul.

Usage strictement personnel (projet solo, pas de portage ni de partage prévu) — ce qui calibre volontairement ce brief : pas de section utilisateurs/marché/différenciation, le seul "utilisateur" étant l'auteur du projet.

## Nom retenu

**`jax_supervised_training`** — validé par l'utilisateur. S'applique au dossier local, au dépôt GitHub, et sert de base au nouveau nom de variable d'environnement (`JAX_SUPERVISED_TRAINING_DATA_ROOT`, à confirmer en PRD) et au dossier Google Drive.

## Ce qui est déjà prouvé

| Cas | Type d'input | Modèle | Duplication de code |
|---|---|---|---|
| Détection/classification avions (cas d'origine) | Images (128×128, 224×224) | `sophisticated_cnn_128_plus`, `aircraft_detector_unet` | — |
| `JAX_KEPLER` | Séries temporelles 1D | `kepler_1d_cnn` | Aucune — même `Trainer`/`ClassificationStrategy` |
| `CIFAR10` (Epic 5) | Images standard 32×32 RGB | `sophisticated_cnn_32_plus` | Aucune — même `Trainer`/`ClassificationStrategy` |

Limite identifiée en cours de route (pas résolue ici, voir "Explicitement hors scope de ce cycle" ci-dessous) : `JAX_KEPLER` a nécessité un modèle dédié avec une couche convolutive 1D codée en dur (`Kepler1DConvNet`) — la config ne porte pas encore de notion explicite de format d'input/output. La généricité actuelle repose sur la souplesse du pattern Strategy + Factory, pas sur une configuration déclarative du format de données.

## Périmètre de ce cycle

**Dans le scope :**
- Renommer le dossier local du projet.
- Renommer le dépôt GitHub (`aobled/JAX_Detection` → nouveau nom), mettre à jour le remote local.
- Renommer la variable d'environnement `JAX_DETECTION_DATA_ROOT` en conséquence — pas de compat double-nom (projet solo, contrôle total sur le code et les notebooks).
- Renommer le dossier Google Drive (`MyDrive/JAX_Detection/`) utilisé par les notebooks Colab.
- Mettre à jour les références textuelles au nom dans le code actif (6 fichiers identifiés : `dataset_configs.py`, `inference_utils.py`, `reporting.py`, `bounding_boxes_with_classification_from_video_generation.py`, `tools/bounding_boxes_with_classification_from_images_generation.py`, `tools/kepler_dataset_tools.py`) et dans `docs/`.
- Mettre à jour tout notebook Colab existant pour pointer vers le nouveau nom de variable d'environnement et le nouveau chemin Drive.

**Explicitement hors scope de ce cycle :**
- Toute notion de format d'input/output configurable dans `dataset_configs.py` (le point révélé par `Kepler1DConvNet`, voir ci-dessus). Identifié, pas priorisé — à reprendre dans un cycle dédié une fois ce renommage clos, probablement sous forme d'un spike architecture (`bmad-agent-architect`) avant tout PRD, étant donné qu'aucune conviction n'existe encore sur la forme que devrait prendre cette configuration.
- Les artefacts BMAD historiques déjà produits (`_bmad-output/planning-artifacts/prds/prd-JAX_Detection-2026-07-12/`, `architecture-JAX_Detection-2026-07-12/`, rétrospectives passées, `sprint-status.yaml` existant) : laissés tels quels, non réécrits (voir Critères de succès). Ce sont des snapshots datés d'un moment où le projet portait ce nom, pas une source à maintenir en synchronisation permanente avec le nom courant.

## Risque identifié

Le principal risque n'est pas technique mais opérationnel : un échec **silencieux**, pas bruyant. Si la variable d'environnement ou le chemin Drive sont renommés côté code/repo mais qu'un notebook Colab existant n'est pas mis à jour en même temps, `dataset_configs.py::DATA_ROOT` retombe silencieusement sur son défaut local (`os.environ.get(..., default)`) — un chemin qui n'existe pas sur Colab, provoquant une erreur "chunks introuvables" en cours d'exécution plutôt qu'un échec immédiat et explicite au démarrage. Ce mode d'échec s'est déjà produit une fois sur ce projet (incident antérieur non lié à ce cycle). À traiter comme item de checklist explicite en exécution : vérifier chaque notebook actif avant/après le renommage, pas seulement le code du repo.

## Critères de succès

- Le dépôt, le dossier local, la variable d'environnement et le dossier Drive portent tous le même nouveau nom — aucune référence résiduelle au nom actuel dans le code actif ou les notebooks en usage.
- Un entraînement Colab lancé après le renommage (sur `FIGHTERJET_CLASSIFICATION` ou `CIFAR10`) démarre sans erreur de chemin — preuve par exécution, pas par lecture de code, cohérent avec la méthode déjà appliquée sur ce projet.
- Les artefacts BMAD historiques (listés ci-dessus en hors scope) restent intacts et lisibles — pas de réécriture rétroactive.
