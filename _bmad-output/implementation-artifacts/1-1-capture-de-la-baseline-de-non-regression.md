# Story 1.1: Capture de la baseline de non-régression

Status: done

## Story

As a mainteneur du pipeline d'inférence,
I want une baseline capturée (boxes/classes/scores) sur un petit set d'images fixes, pour les 7 fichiers concernés par la duplication,
so that je peux prouver par diff, pas par observation, l'absence de régression après le refactor (Goal 1 PRD).

## Acceptance Criteria

1. Sorties (boxes, classes, scores) capturées sur un jeu d'images fixe couvrant classification et détection, sauvegardées en JSON comparable.
2. Jeu d'images documenté pour réutilisation identique en Story 1.10.
3. Méthode de capture alternative documentée pour le fichier interactif (`tools/boxes_process_manual_tkinter.py`).

## Tasks / Subtasks

- [x] Écrire un script de capture appelant les fonctions ACTUELLES (pré-migration) de `tools/bounding_boxes_with_classification_from_images_generation.py` (chemin image unique) et `bounding_boxes_with_classification_from_video_generation.py` (chemin batch/vidéo)
- [x] Capturer sur 6 images statiques fixes (une par classe: a10, b52, f15, f16, f22, f35, depuis `/home/aobled/Downloads/tmp_multi/`)
- [x] Capturer sur 5 frames fixes de `/home/aobled/Downloads/testvid.mp4` (indices 0, 30, 60, 90, 120)
- [x] Sauvegarder en JSON (`baseline_before.json`)

## Dev Notes

- Script: `_bmad-output/implementation-artifacts/baseline/capture_baseline.py`
- Résultat: `_bmad-output/implementation-artifacts/baseline/baseline_before.json`
- Checkpoint détection confirmé lié à `aircraft_detector_unet` (utile pour Story 2.1/AD-4).
- Couverture: chemin image unique (`decode_segmentation_and_detect` + `predict_crop`, image_generation.py) ET chemin batch (`decode_segmentation_and_detect_batch` + `predict_crops_batch`, video_generation.py) — les deux API distinctes ratifiées par AD-2/AD-6.
- `tools/boxes_process_manual_tkinter.py` (Story 1.9) : pas de capture automatisée dédiée (GUI Tkinter) — sa logique interne appelle exactement `decode_segmentation_and_detect_batch`/`predict_crops_batch` de `video_generation.py` avec les mêmes signatures que celles couvertes par la capture vidéo ci-dessus ; considéré comme couvert par transitivité plutôt que dupliqué avec une capture GUI séparée.

### Résultats de référence (baseline_before.json)

- Images statiques: 6/6 avec détections (1 à 3 boxes chacune).
- Frames vidéo: 5/5 avec détections (5 à 6 boxes chacune).

## Dev Agent Record

### Completion Notes List

- Baseline capturée avec succès en réexécutant le code pré-migration tel quel (aucune modification de fichier source à ce stade).
