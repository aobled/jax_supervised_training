# Story 1.3: Migration bounding_boxes_with_classification_from_video_generation.py

Status: done

## Acceptance Criteria — verified

1. Importe `load_jax_model`, `load_detection_model`, `predict_crops_batch`, `build_predict_fn`, `build_clf_predict_fn`, `get_iou`, `non_max_suppression`, `decode_segmentation_and_detect_batch` depuis `inference_utils.py` — définitions locales supprimées. ✅
2. `predict_crops_batch` (AD-2) : signature/usage inchangés côté appelant. ✅
3. Aucune dégradation du débit temps réel (aucun changement à la logique de threading/queue/warmup, migration = imports seulement). ✅ (pas de mesure formelle — pas de suite de perf, cf. PRD Non-Goals)
4. Sorties identiques à la baseline Story 1.1 sur le jeu vidéo. ✅ (confirmé indirectement via Story 1.2 : `decode_segmentation_and_detect_batch`/`predict_crops_batch` d'`inference_utils.py` sont bit-identiques à la baseline ; ce fichier ne fait qu'importer ces mêmes fonctions, aucune logique propre modifiée)

## Dev Notes

- `build_det_predict_fn` (local) supprimée, remplacée par `build_predict_fn` importé (AD-1 consolidation). Site d'appel mis à jour (`main`).
- Ancien `predict_crop(crop_img, predict_fn, mean, std, config)` — confirmé mort (aucun appelant dans tout le repo, `grep` vérifié) — **non migré**, conforme AD-2.
- `_pad_batch_np` supprimée (déplacée dans `inference_utils.py`, détail d'implémentation privé de `predict_crops_batch`).
- `DETECTION_IMAGE_SIZE` local supprimé, remplacé par l'import (évite la redéfinition interdite par la Consistency Convention du spine).
- `_CLOSING_KERNEL`/`_DILATE_KERNEL`/`CROP_MARGIN_PERCENT` locaux supprimés (devenus privés à `inference_utils.py`, plus utilisés localement).
- `BATCH_SIZE`/`CLF_BATCH_SIZE` locaux **conservés** — utilisés par la logique de threading/warmup propre à ce script (buffering de frames, queues), distincts des constantes privées `_DET_BATCH_SIZE`/`_CLF_BATCH_SIZE` d'`inference_utils.py`. Couplage de valeur (32 des deux côtés) préexistant au refactor, non modifié (AD-6 : pas de changement de comportement).
- `get_iou`/`non_max_suppression` importés mais non appelés directement par ce fichier — conservés pour préserver la surface d'export publique du module (consommée transitivement par `tools/audit_dataset_detection.py`/`tools/boxes_process_manual_tkinter.py` jusqu'à leur propre migration en Stories 1.8/1.9).

## Dev Agent Record

### File List

- `bounding_boxes_with_classification_from_video_generation.py` (modifié)
