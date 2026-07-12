# Story 1.8: Migration tools/audit_dataset_detection.py

Status: done

## Acceptance Criteria — verified

1. Imports (`load_detection_model`, `decode_segmentation_and_detect_batch`, `get_iou`) repointés vers `inference_utils.py`. ✅
2. Aucun shim de compatibilité laissé en place. ✅ (import direct, ancien import supprimé, `grep` confirme 0 référence résiduelle à `bounding_boxes_with_classification_from_video_generation`)
3. Résultats identiques à la baseline. ✅ (indirect via Story 1.2 : fonctions bit-identiques)

## Dev Agent Record

### File List

- `tools/audit_dataset_detection.py` (modifié)
