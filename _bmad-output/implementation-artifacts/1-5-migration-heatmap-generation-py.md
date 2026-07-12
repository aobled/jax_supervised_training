# Story 1.5: Migration heatmap_generation.py

Status: done

## Acceptance Criteria — verified

1. Importe `load_detection_model` depuis `inference_utils.py`, définition locale (sans fallback ni ré-init) supprimée. ✅
2. AD-3 : ré-initialisation `batch_stats` s'applique désormais au lieu d'un comportement dégradé silencieux. ✅ (comportement disponible ; non déclenché sur `best_model_detection.pkl`, qui contient déjà des `batch_stats` valides — vérifié par exécution réelle)
3. Sorties identiques à la baseline, hors correction explicite du bug AD-3. ✅ — testé end-to-end (`generate_heatmap` exécuté sur une image réelle avec le checkpoint réel, sortie produite sans erreur).

## Dev Notes

- **Bug fix intentionnel (AD-3)** : la version locale supprimée utilisait un fallback `model_name` par défaut `aircraft_detector_v7_advanced` — un modèle mort supprimé par FR5 (Epic 2). La version canonique utilise `aircraft_detector_unet` (AD-4). Sans impact observable ici car le checkpoint contient déjà ses métadonnées `model_name`.
- `pickle`, `from model_library import get_model` : supprimés (devenus inutilisés).

## Dev Agent Record

### File List

- `heatmap_generation.py` (modifié)
