# Story 1.6: Migration heatmap_contouring.py

Status: done

## Acceptance Criteria — verified

1. Importe `load_detection_model` depuis `inference_utils.py`, définition locale supprimée. ✅
2. AD-3 ré-initialisation `batch_stats` disponible si nécessaire. ✅ (non déclenché — checkpoint valide, vérifié par exécution réelle)
3. Résultats identiques hors correction AD-3. ✅ — testé end-to-end (`detect_by_contouring` exécuté sur une image réelle, sortie produite sans erreur).

## Dev Notes

- Même correction de bug intentionnelle qu'en Story 1.5 (fallback `model_name` mort → `aircraft_detector_unet`).
- `pickle`, `from model_library import get_model` : supprimés.

## Dev Agent Record

### File List

- `archive/heatmap_contouring.py` (modifié)
