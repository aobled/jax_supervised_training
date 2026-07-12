# Story 1.7: Migration tools/audit_dataset_classification.py

Status: done

## Acceptance Criteria — verified

1. `load_classification_model` locale supprimée, remplacée par composition explicite au site d'appel (`load_jax_model` + `build_predict_fn`), pas une 12ᵉ fonction nommée — conforme à la note d'implémentation AD-1. ✅
2. `_preprocess_crop_to_hwc` importée depuis `inference_utils.py`. ✅
3. Résultats identiques à la baseline. ✅ (composition testée : mêmes erreurs/succès que le comportement pré-migration, voir Dev Notes)

## Dev Notes

- **Bug pré-existant découvert et préservé tel quel** (hors scope FR1-FR10, pas un objectif de ce refactor) : `checkpoint_path = config.get("checkpoint_path", "best_model_classification.pkl")` — `FIGHTERJET_CLASSIFICATION` (dataset_configs.py) ne définit pas de clé `checkpoint_path`, donc ce script résout toujours vers `best_model_classification.pkl`, un fichier qui n'existe pas dans le dépôt (seuls `best_model.pkl`/`best_model_detection.pkl` existent). `run_audit()` levait déjà `FileNotFoundError` avant ce refactor — **comportement inchangé et vérifié identique** après migration (même exception, même message). Ne pas corriger ce bug ici serait sortir du scope du PRD (Non-Goals) ; à signaler séparément si l'audit de classification doit redevenir exécutable.
- `pickle`, `from model_library import get_model` : supprimés (plus nécessaires, la composition passe par `load_jax_model`).

## Dev Agent Record

### File List

- `tools/audit_dataset_classification.py` (modifié)
