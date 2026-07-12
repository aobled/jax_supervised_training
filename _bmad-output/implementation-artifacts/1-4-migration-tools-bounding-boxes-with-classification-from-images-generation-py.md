# Story 1.4: Migration tools/bounding_boxes_with_classification_from_images_generation.py

Status: done

## Acceptance Criteria — verified

1. Importe `predict_crop` (AD-2, ratifié depuis ce fichier), `load_jax_model`, `load_detection_model`, `get_iou`, `non_max_suppression`, `decode_segmentation_and_detect` depuis `inference_utils.py` — définitions locales supprimées. ✅
2. Aucun appel n'utilise l'ancienne signature `predict_crop(crop_img, predict_fn, mean, std, config)` (confirmée morte, non migrée). ✅ — cette signature n'a jamais existé dans ce fichier, non applicable.
3. Résultats identiques à la baseline. ✅ (indirect via Story 1.2 : `predict_crop`/`decode_segmentation_and_detect` d'`inference_utils.py` bit-identiques à la baseline)

## Dev Notes

- `DETECTION_IMAGE_SIZE`, `pickle` import : supprimés (devenus inutilisés après migration des fonctions qui les consommaient).
- `get_iou` importé mais non appelé directement par ce fichier (utilisé en interne par `non_max_suppression`) — conservé pour fidélité à FR2/l'AC, pas un import mort au sens strict (préserve la surface du module tel qu'il existait avant, cf. `dead-code-and-duplication-audit.md`).
- Site d'appel `predict_crop(crop, clf_model, clf_vars, dataset_mean, dataset_std, config)` inchangé — signature déjà identique à la version canonique (AD-2 ratifie explicitement cette ligne).

## Dev Agent Record

### File List

- `tools/bounding_boxes_with_classification_from_images_generation.py` (modifié)
