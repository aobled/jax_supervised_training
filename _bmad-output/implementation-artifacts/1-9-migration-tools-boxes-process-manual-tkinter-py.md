# Story 1.9: Migration tools/boxes_process_manual_tkinter.py

Status: done

## Acceptance Criteria — verified

1. Imports (`load_detection_model`, `load_jax_model`, `decode_segmentation_and_detect_batch`, `predict_crops_batch`, `get_iou`) repointés vers `inference_utils.py` (2 sites d'import, méthodes `load_jax_models` et `generate_and_show_predictions`). ✅
2. Comportement de l'interface Tkinter inchangé — seuls les imports sont modifiés, aucune logique d'UI touchée. ✅
3. `grep` confirme 0 référence résiduelle à `bounding_boxes_with_classification_from_video_generation` dans ce fichier. ✅

## Dev Notes

- Fichier interactif (GUI Tkinter) — pas de test end-to-end automatisé possible dans cette session (pas d'affichage). Vérification par lecture de code + syntax check + confirmation que les fonctions importées sont bit-identiques à la baseline (Story 1.2), cf. Story 1.1 Dev Notes (couverture par transitivité).

## Dev Agent Record

### File List

- `tools/boxes_process_manual_tkinter.py` (modifié)
