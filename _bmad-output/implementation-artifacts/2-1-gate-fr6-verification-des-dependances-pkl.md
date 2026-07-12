# Story 2.1: Gate FR6 — vérification des dépendances .pkl

Status: done

## Acceptance Criteria — verified

1. `best_model.pkl` et `best_model_detection.pkl` inspectés directement (métadonnées `config` embarquées dans le pickle, pas seulement déduites du code). ✅
2. Résultat : `best_model.pkl` → `sophisticated_cnn_128_plus` ; `best_model_detection.pkl` → `aircraft_detector_unet`. Les deux sont dans la liste des 4 architectures survivantes de FR5 (`sophisticated_cnn_128_plus`, `aircraft_detector_unet`, `aircraft_detector_miniunet`, `kepler_1d_cnn`). ✅
3. Gate passé → Story 2.3 (purge `model_library.py`) autorisée à démarrer. ✅

## Dev Notes

Vérification indépendante (pas une simple confiance dans AD-4 du spine) : chargement direct des 2 fichiers `.pkl` via `pickle.load`, lecture du champ `config['model_name']` embarqué dans chaque checkpoint — confirme exactement la même conclusion que le spine d'architecture (AD-4), sans dépendre du fallback de `load_jax_model`/`load_detection_model`.

```
best_model.pkl            -> model_name: sophisticated_cnn_128_plus
best_model_detection.pkl  -> model_name: aircraft_detector_unet
```

Aucune exception à documenter — la liste de suppression FR5 (18 architectures) reste inchangée.
