---
baseline_commit: 30c1b47e9e8b3b620319f2cecc1824271f2cc4b7
---

# Story 7.7: Entrée JAX_DETECTOR (dataset_configs.py) + dispatch task_type (main.py, AD-17)

Status: done

<!-- Note: Validation is optional. Run validate-create-story for quality check before dev-story. -->

## Story

As a mainteneur du pipeline de configuration,
I want une nouvelle entrée `JAX_DETECTOR` dans `dataset_configs.py` et son dispatch dans `main.py`,
so that l'entraînement soit lançable via `main.py JAX_DETECTOR`, cohérent avec les configs existantes.

## Acceptance Criteria

1. **Given** `DATASET_CONFIGS` et `validate_config` (`dataset_configs.py:16-42`, exige `num_classes`, `image_size`, `model_name`, et `len(class_names)==num_classes`) **When** `JAX_DETECTOR` est ajoutée **Then** elle définit `model_name="aircraft_detector_centernet"` (Story 7.2), `image_size=(224,224)`, `num_classes=1`, `class_names=['aircraft']` (mono-classe, cohérent avec `FIGHTERJET_DETECTION`), et un `task_type` dédié dont le littéral exact (`"detection_centernet"`) est utilisé identiquement dans `task_strategies.py` (implicite via le dispatch Task 3), `data_management.py` (Task 2) et `main.py` (Task 3).
2. **Given** `main.py:107-143` (dispatch `if/elif` existant : `classification`, `detection`, `kepler`) **When** le nouveau `task_type` `"detection_centernet"` est ajouté **Then** une branche supplémentaire instancie `CenterNetDetectionStrategy` (Story 7.6) **avec sa signature réelle** (`loss_params`, `metric_threshold` — pas `loss_method`/`metric_method`/`report_method`, que cette stratégie n'accepte pas, contrairement aux trois branches existantes), sans modifier le comportement des branches existantes.
3. **Given** `data_management.py::get_datasets` (`data_management.py:418-464`, dispatch `task_type` actuel) **When** le nouveau `task_type` est ajouté **Then** une branche supplémentaire instancie `CenterNetDetectionDataset` (Story 7.5), même structure que la branche `"detection"` existante (lignes 446-461), sans la modifier.
4. **Given** AD-20 (non-régression) **When** cette story est complétée **Then** `FIGHTERJET_DETECTION` reste inchangée dans `dataset_configs.py`, aucun renommage ni suppression, et `output_prefix` de `JAX_DETECTOR` pointe vers un répertoire distinct (jamais celui de `FIGHTERJET_DETECTION`).
5. **Given** `main.py:87-88` (`print(f"...targets={sample_train[1].shape}")`, exécuté pour tous les `task_type` avant le dispatch de stratégie) **When** `JAX_DETECTOR` est exécutée **Then** cette ligne ne plante pas sur `sample_train[1]` étant un dict (`{HEATMAP_KEY, SIZE_KEY}`, Story 7.5) au lieu d'un tenseur unique — trouvé en vérifiant le chemin d'exécution réel avant d'écrire cette story, pas anticipé par le texte initial de l'epic.

## Tasks / Subtasks

- [x] Task 1: Ajouter l'entrée `JAX_DETECTOR` dans `DATASET_CONFIGS` (`dataset_configs.py`), modelée sur `FIGHTERJET_DETECTION` (`dataset_configs.py:120-192`) (AC: 1, 4)
  - [x] `task_type: "detection_centernet"`, `num_classes: 1`, `class_names: ['aircraft']`, `image_size: (224,224)`, `grayscale: True`, `max_boxes: 20`
  - [x] `output_prefix: f"{DATA_ROOT}/chunks/jax_detector/jax_detector_targets"` — le nom de base (`jax_detector_targets`) doit correspondre **exactement** au préfixe de fichier codé en dur dans `fighterjet_detection_dataset_tools_v2.py` (Story 7.4, Task 2) pour que le glob de `CenterNetDetectionDataset` (Story 7.5) trouve les chunks. Le couplage n'est pas que le nom de base : le répertoire compte aussi — la Story 7.4 dérive `OUTPUT_DIR = os.path.dirname(config["output_prefix"])` (même pattern que l'outil actuel, `fighterjet_detection_dataset_tools.py:176`), donc `output_prefix` porte à la fois le répertoire (`chunks/jax_detector/`, distinct de `FIGHTERJET_DETECTION` = `chunks/detection/`, pas de collision) et le nom de base des fichiers
  - [x] `model_name: "aircraft_detector_centernet"` (Story 7.2)
  - [x] `augmentation_params` : copier telle quelle celle de `FIGHTERJET_DETECTION` (`flip_h`/`flip_v`/`zoom_factor=0.35`/`translation_factor=0.25`/`brightness_delta`/`contrast_factor`) — point de départ raisonnable, pas encore tuné pour ce nouveau format (même statut que les hyperparamètres de perte des Stories 7.1/7.3)
  - [x] `loss_params: {"heatmap_weight": 1.0, "size_weight": 0.1, "alpha": 2.0, "beta": 4.0}` (defaults papier CenterNet, Story 7.3) ; `metric_threshold: 0.5` (Story 7.6). **Pas** de clé `loss_method`/`metric_method`/`report_method` — `CenterNetDetectionStrategy` ne les consomme pas (voir Task 3, AC2), les inclure serait trompeur (suggérerait un dispatch interne qui n'existe pas)
  - [x] Hyperparamètres `tpu`/`gpu` (`micro_batch_size`, `learning_rate`, `weight_decay`, `dropout_rate`), `optimizer`, `lr_schedule`, `epochs`, `patience` : copier la structure de `FIGHTERJET_DETECTION` comme point de départ — valeurs à réajuster empiriquement une fois l'entraînement réel lancé (Story 7.8), ne pas sur-optimiser ici
  - [x] **Ne pas** définir `checkpoint_path`/`training_state_path` explicitement — laisser le fallback dérivé de `dataset_name` (Story 5.0, `CenterNetDetectionStrategy._get_export_path`, Story 7.6) produire `best_model_jax_detector.pkl`/`best_model_training_state_jax_detector.pkl`
- [x] Task 2: Ajouter une branche `elif task_type == "detection_centernet":` dans `data_management.py::get_datasets` (`data_management.py:446-461`, même structure que la branche `"detection"`), instanciant `CenterNetDetectionDataset` (Story 7.5) (AC: 3)
- [x] Task 3: Ajouter une branche `elif task_type == "detection_centernet":` dans `main.py` (`main.py:113-142`) (AC: 1, 2)
  - [x] `from task_strategies import CenterNetDetectionStrategy` ; `strategy = CenterNetDetectionStrategy(loss_params=loss_params, metric_threshold=config.get("metric_threshold", 0.5))` — **ne pas copier** l'appel des branches `classification`/`detection`/`kepler` (qui passent `loss_method`/`metric_method`/`report_method`, absents de la signature réelle de `CenterNetDetectionStrategy`, Story 7.6 Task 1) — une copie mécanique lèverait un `TypeError` à l'instanciation
- [x] Task 4: Corriger `main.py:87-88` pour supporter des `targets` au format dict, sans changer le comportement pour les `task_type` existants (AC: 5)
  - [x] Petite fonction locale `_shape_repr(x)`: retourne `x.shape` si tenseur, `{k: v.shape for k, v in x.items()}` si dict ; utilisée dans **les deux** `print(...)` lignes 87 (`sample_train[1]`) **et** 88 (`sample_val[1]`) — les deux sont des dicts avec `JAX_DETECTOR`, les deux plantent sans le correctif, pas seulement le premier
- [x] Task 5: Test — appeler `get_dataset_config("JAX_DETECTOR")`, vérifier que `validate_config` passe, que `config["gpu"]`/`config["tpu"]` existent bien avec leurs clés `dropout_rate`/`micro_batch_size` (`main.py:76,98` les lisent sans garde — une clé manquante ne serait détectée qu'à l'exécution réelle, Story 7.8, sinon), que `get_model(config["model_name"], num_classes=1, dropout_rate=config["gpu"]["dropout_rate"])` instancie sans erreur, et que `CenterNetDetectionStrategy(loss_params=config["loss_params"])` s'instancie sans erreur (AC: 1). Ne nécessite pas de vrais chunks `.npz` (pas encore générés, Story 7.8) — juste la cohérence de la config et l'instanciation des objets

## Dev Notes

### Cohérence de nommage — trois points de dispatch, un seul littéral

AD-17 exige que `"detection_centernet"` soit défini une fois et utilisé identiquement partout. Trois sites : `data_management.py::get_datasets` (Task 2), `main.py` (Task 3), et implicitement `dataset_configs.py::JAX_DETECTOR["task_type"]` (Task 1) qui est la source de la valeur elle-même — les deux autres sites la lisent via `config.get("task_type", ...)`, ils ne la redéfinissent pas littéralement. Vérifier qu'aucun des deux nouveaux `elif` ne contient une chaîne différente (faute de frappe = branche jamais atteinte, silencieuse jusqu'à l'erreur `ValueError` finale du `else`).

### CenterNetDetectionStrategy n'est pas un dispatcher multi-méthode

Contrairement à `ClassificationStrategy`/`DetectionStrategy`/`KeplerStrategy` (qui acceptent `loss_method`/`metric_method`/`report_method` et dispatchent en interne selon leur valeur), `CenterNetDetectionStrategy` (Story 7.6) a une seule méthode de perte (`compute_centernet_loss`) et une seule métrique (`HeatmapRecall`) — pas de dispatch interne, donc pas de paramètres `*_method` dans son `__init__`. Une des erreurs les plus faciles à commettre dans cette story serait de copier-coller le bloc `elif task_type == "detection":` existant et de changer seulement le nom de la classe — ça lèverait un `TypeError: unexpected keyword argument 'loss_method'` à l'exécution. Vérifier la signature réelle de `CenterNetDetectionStrategy.__init__` (Story 7.6, Task 1) avant d'écrire l'appel.

### Le print qui plante (trouvé en lisant le chemin d'exécution réel, pas anticipé)

`main.py:87-88` s'exécute pour **tous** les `task_type`, avant même le dispatch de stratégie — c'est du code générique, pas spécifique à la détection. `sample_train[1].shape` suppose un tenseur unique ; `CenterNetDetectionDataset` (Story 7.5) yield des cibles en dict. Sans le correctif de Task 4, `main.py JAX_DETECTOR` planterait dès l'étape de vérification des datasets, avant même d'atteindre le dispatch de stratégie ou le modèle — un échec précoce et déroutant si non anticipé. Même classe de bug que celui trouvé dans `trainer.py` pendant la Story 7.6 (`jnp.array(labels_np)` sur un dict) — du code générique partagé par tous les `task_type` qui n'avait jamais eu à gérer une structure autre qu'un tenseur unique avant ce chantier.

### Project Structure Notes

- Modification de `dataset_configs.py` (ajout d'une entrée), `data_management.py` (ajout d'une branche `elif`), `main.py` (ajout d'une branche `elif` + correctif de 2 lignes génériques). Aucune modification de `FIGHTERJET_CLASSIFICATION`/`FIGHTERJET_DETECTION`/`JAX_KEPLER`/`CIFAR10`, ni des branches `classification`/`detection`/`kepler` existantes.
- Cette story ne lance aucun entraînement réel (Story 7.8) — elle rend `JAX_DETECTOR` instanciable et cohérente, sans données réelles à charger (les chunks `.npz` n'existent pas encore tant que la Story 7.4 n'a pas été exécutée en pratique sur le dataset réel).

### Testing Standards

Script autonome (Task 5), même esprit que les stories précédentes de l'Epic 7. Pas de test end-to-end avec de vrais chunks — hors de portée de cette story (nécessite un jeu de données réel généré, Story 7.8).

### References

- [Source: `dataset_configs.py:16-42`] — `validate_config`, champs requis
- [Source: `dataset_configs.py:120-192`] — `FIGHTERJET_DETECTION`, modèle structurel complet
- [Source: `main.py:80-88`] — vérification post-chargement des datasets, le print qui plante sur un dict
- [Source: `main.py:107-143`] — dispatch `task_type` actuel complet
- [Source: `data_management.py:418-464`] — `get_datasets`, dispatch `task_type` actuel complet
- [Source: `_bmad-output/implementation-artifacts/7-2-nouvelle-classe-modele-jax-detector-ad-9-ad-10.md`] — `aircraft_detector_centernet`
- [Source: `_bmad-output/implementation-artifacts/7-3-nouvelles-fonctions-de-perte-heatmap-focal-loss-regression-de-taille.md`] — `loss_params` attendus par `compute_centernet_loss`
- [Source: `_bmad-output/implementation-artifacts/7-4-fighterjet-detection-dataset-tools-v2-py-generation-des-chunks-depuis-raw-boxes.md`] — préfixe de fichier `jax_detector_targets` à faire correspondre à `output_prefix`
- [Source: `_bmad-output/implementation-artifacts/7-5-nouvelle-classe-de-chargeur-dediee-data-management-py-ad-17-ad-18.md`] — `CenterNetDetectionDataset`
- [Source: `_bmad-output/implementation-artifacts/7-6-nouvelle-strategie-dediee-task-strategies-py-ad-9-ad-17.md`] — signature réelle de `CenterNetDetectionStrategy.__init__`

## Dev Agent Record

### Agent Model Used

Claude Sonnet 5 (claude-sonnet-5)

### Debug Log References

`python3 test_jax_detector_config.py` — 8/8 tests passés (config valide, `output_prefix` distinct de `FIGHTERJET_DETECTION` + nom de base correspondant exactement à Story 7.4, hyperparamètres `gpu`/`tpu` présents, absence des clés `loss_method`/`metric_method`/`report_method`, modèle instancié via `get_model`, stratégie instanciée avec sa signature réelle, absence de `checkpoint_path`/`training_state_path` explicite, `FIGHTERJET_DETECTION` inchangée).

### Completion Notes List

- Entrée `JAX_DETECTOR` ajoutée dans `DATASET_CONFIGS` (`dataset_configs.py`), modelée sur `FIGHTERJET_DETECTION` : `task_type="detection_centernet"`, `model_name="aircraft_detector_centernet"` (Story 7.2), `output_prefix` pointant vers `chunks/jax_detector/jax_detector_targets` (répertoire distinct de `chunks/detection/`, nom de base identique au préfixe codé en dur par la Story 7.4), `loss_params`/`metric_threshold` (pas de `loss_method`/`metric_method`/`report_method`, `CenterNetDetectionStrategy` ne les consomme pas). Pas de `checkpoint_path`/`training_state_path` explicite — fallback dérivé de `dataset_name` (Story 5.0/7.6).
- `FIGHTERJET_DETECTION` non modifiée (AC4/AD-20, vérifié par `git diff` : zéro ligne supprimée dans `dataset_configs.py`).
- Branche `elif task_type == "detection_centernet":` ajoutée dans `data_management.py::get_datasets`, même structure que la branche `"detection"` existante, instanciant `CenterNetDetectionDataset` (Story 7.5).
- Branche `elif task_type == "detection_centernet":` ajoutée dans `main.py`, important `CenterNetDetectionStrategy` localement (même pattern que `KeplerStrategy`) et l'instanciant avec sa **signature réelle** (`loss_params`, `metric_threshold`) — pas de copie mécanique du bloc `"detection"` qui aurait levé un `TypeError`.
- `main.py:87-88` corrigé via une fonction locale `_shape_repr(x)` (dict-safe), appliquée aux deux lignes de `print` (`sample_train[1]` et `sample_val[1]`) — comportement inchangé pour les `task_type` existants (tenseur unique → `.shape` direct), dict géré pour `detection_centernet`.
- Vérifié par `git diff` sur `main.py`/`data_management.py`/`dataset_configs.py` contre le commit baseline : uniquement les lignes attendues supprimées/remplacées (les 2 lignes `print` de Task 4), tout le reste est additif.
- Test Task 5 : cohérence de config + instanciation modèle/stratégie, sans chunks réels (Story 7.8).

### File List

- `dataset_configs.py` (modifié — ajout entrée `JAX_DETECTOR`)
- `data_management.py` (modifié — ajout branche `elif "detection_centernet"` dans `get_datasets`)
- `main.py` (modifié — ajout branche `elif "detection_centernet"`, correctif `_shape_repr` dict-safe lignes 87-88)
- `test_jax_detector_config.py` (nouveau)

## Senior Developer Review (AI)

**Reviewer:** Claude Opus (contexte neuf, différent du modèle d'implémentation — Sonnet 5)
**Date:** 2026-07-17
**Outcome:** **APPROVE**

### Vérifications effectuées

- Signature réelle de `CenterNetDetectionStrategy.__init__` vérifiée directement dans `task_strategies.py` (pas seulement citée) : `loss_params`/`metric_threshold` uniquement, confirmé que `main.py` ne passe aucun `loss_method`/`metric_method`/`report_method`.
- Littéral `"detection_centernet"` confirmé caractère-pour-caractère identique aux 3 sites de dispatch (`dataset_configs.py`, `data_management.py`, `main.py`).
- `git diff 30c1b47 -- dataset_configs.py` : **0 ligne supprimée** — vérification programmatique du point le plus critique (AD-20).
- Correspondance exacte confirmée entre le nom de base `output_prefix` (`jax_detector_targets`) et le préfixe codé en dur dans `fighterjet_detection_dataset_tools_v2.py:144` — le glob de `CenterNetDetectionDataset` trouvera bien les chunks.
- `git diff 30c1b47 -- main.py` : exactement les 2 lignes `print` attendues modifiées, rien d'autre.
- `python3 test_jax_detector_config.py` ré-exécuté : 8/8 passés.

### Findings

Aucun HIGH/MEDIUM. Un LOW cosmétique (suppression de espaces en fin de ligne vides adjacentes dans `data_management.py`, sans impact comportemental).
