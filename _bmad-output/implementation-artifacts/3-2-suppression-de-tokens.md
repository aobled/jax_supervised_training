# Story 3.2: Suppression de tokens/

Status: done

## Acceptance Criteria — verified

1. Aucun autre fichier du repo n'importe depuis `tokens/` (vérifié `grep` avant suppression). ✅
2. Récupérable via l'historique git (NFR4). ✅

## Dev Notes

`tokens/loss_with_classification_images_directory.py` contenait un appel à l'ancienne signature `predict_crop(crop_img, predict_fn, mean, std, config)` (confirmé mort dans l'analyse de Story 1.3) — cohérent avec le constat PRD : ce dossier n'était jamais raccordé au pipeline actif.

## Dev Agent Record

### File List

- `tokens/` (dossier supprimé : `read_token.py`, `write_token.py`, `genetic_algorithm.py`, `genetic_algorithm2.py`, `loss_detection_only_directory.py`, `loss_with_classification_images_directory.py`)
