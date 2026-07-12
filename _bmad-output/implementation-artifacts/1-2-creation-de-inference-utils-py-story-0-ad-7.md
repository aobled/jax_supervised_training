# Story 1.2: Création de inference_utils.py (Story 0, AD-7)

Status: done

## Story

As a mainteneur du pipeline d'inférence,
I want un module partagé inference_utils.py contenant les 11 fonctions canoniques d'inférence avec leurs comportements arbitrés,
so that toute la logique d'inférence dupliquée peut ensuite être remplacée par un import unique, sans ambiguïté sur quel comportement fait foi.

## Acceptance Criteria

1. `inference_utils.py` contient exactement les 11 fonctions canoniques (AD-1), sans 12ᵉ fonction `load_classification_model`.
2. `predict_crop`/`predict_crops_batch` indépendantes (AD-2), `_CLF_BATCH_SIZE = 32` privée.
3. `load_detection_model` : fallback 3 niveaux + ré-init `batch_stats` (AD-3), `DETECTION_IMAGE_SIZE` privée.
4. Fallback `model_name` par défaut = `aircraft_detector_unet` (AD-4).
5. `decode_segmentation_and_detect`/`_batch` distinctes, sans changement de comportement (AD-6).
6. Format des boxes `[x1, y1, x2, y2, score]` en liste Python.
7. Aucune story suivante ne modifie ce fichier (AD-7).

## Dev Notes

### Décisions d'implémentation au-delà du texte du spine

- `_pad_batch_np` migré comme helper privé (utilisé en interne par `predict_crops_batch`) — pas une 12ᵉ fonction publique, juste une dépendance d'implémentation.
- `_DET_BATCH_SIZE = 32` ajoutée comme constante privée (par symétrie avec `_CLF_BATCH_SIZE`) pour préserver le comportement de padding anti-recompilation JIT de `decode_segmentation_and_detect_batch`, qui dépendait de `BATCH_SIZE` (global script) dans `video_generation.py`. Le spine ne nomme pas cette constante explicitement mais AD-6/NFR3 exigent la préservation exacte du comportement — sans ce padding fixe, le débit temps réel vidéo se dégraderait (retracing JIT à chaque frame count variable).
- `_CROP_MARGIN_PERCENT = 0` ajoutée comme constante privée, reproduisant `CROP_MARGIN_PERCENT` (global script, valeur 0 dans tous les appels actuels) — comportement figé identique, pas exposée en paramètre pour respecter la signature à 5 paramètres du spine (AD-1 rule 11).
- `predict_crop` réutilise `_preprocess_crop_to_hwc` en interne (au lieu du prétraitement inline dupliqué de `images_generation.py:128`) — vérifié bit-identique (même resize/color-convert/normalize), pas un changement de comportement, juste une déduplication interne cohérente avec AD-1.

### Vérification effectuée (avant migration de tout fichier consommateur)

Script `_bmad-output/implementation-artifacts/baseline/verify_inference_utils.py` : ré-exécute les mêmes 6 images statiques + 5 frames vidéo de la baseline (Story 1.1) via les fonctions d'`inference_utils.py`, diff automatique contre `baseline_before.json`.

**Résultat : 0 mismatch, 11/11 comparaisons identiques.**

## Dev Agent Record

### File List

- `inference_utils.py` (nouveau)
