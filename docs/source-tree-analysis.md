# Arborescence annotée — JAX_Detection

_Généré par `bmad-document-project`, 2026-07-12. Reflète l'état réel du code (structure "à plat", pas de packaging Python standard)._

```
JAX_Detection/
├── dataset_configs.py          # Config centralisée (DATASET_CONFIGS dict). Point d'entrée n°1 du pipeline.
├── data_management.py          # Pipeline tf.data : chargement .npz, normalisation, augmentation, batching
├── model_library.py            # Factory de modèles (MODELS dict + get_model()). 466 lignes (post-purge 2026-07-12, était 2614) — 4 architectures actives
├── trainer.py                  # Boucle d'entraînement JAX @jit (Trainer class), agnostique à la tâche
├── task_strategies.py          # Pattern Strategy : ClassificationStrategy / DetectionStrategy / KeplerStrategy
├── main.py                     # Orchestrateur haut niveau (assemble config → data → model → strategy → trainer)
├── reporting.py                # Génération matrices de confusion / visualisations bounding boxes (962 lignes)
├── checkpoint_manager.py       # Sauvegarde/reprise des poids (.pkl)
├── loss_functions.py           # Fonctions de loss custom (focal, segmentation BCE+Dice, etc.)
├── utils.py                    # Utilitaires partagés
├── inference_utils.py                                           # Module partagé d'inférence (post-refactor 2026-07-12) — source unique de load_jax_model/predict_crop/NMS/IoU/décodage détection
│
├── bounding_boxes_with_classification_from_video_generation.py  # Script d'inférence vidéo temps réel — importe depuis inference_utils.py
├── heatmap_generation.py                                        # Importe load_detection_model depuis inference_utils.py
├── heatmap_contouring.py                                        # Importe load_detection_model depuis inference_utils.py
├── check_image_channels.py                                      # Utilitaire ponctuel (diagnostic images)
├── generate_letterbox_dataset.py                                 # Génération dataset letterboxé
├── fighterjet_classification_dataset_tools.py                    # Outils dataset classification (459 lignes)
├── fighterjet_detection_dataset_tools.py                         # Outils dataset détection (217 lignes)
│
├── tools/                       # 24 scripts — mélange d'outils dataset (convert_*, audit_*), et un script d'inférence
│   ├── bounding_boxes_with_classification_from_images_generation.py  # Script d'inférence — importe depuis inference_utils.py
│   ├── audit_dataset_classification.py / audit_dataset_detection.py / audit_dataset_results_pandas.py
│   ├── convert_*.py (×6)         # Convertisseurs de formats de datasets tiers vers le format JSON interne (AeroScan, HRPlanesv2, Military Aircraft Dataset, YOLOv8...)
│   ├── build_fixed_aircraft_dataset.py, duplicate_image_detection_and_normalization.py, duplicate_files_find.py, find_and_deduplicate.py
│   ├── boxes_process_manual_tkinter.py (1285 lignes)  # Éditeur manuel de bounding boxes (GUI Tkinter), le plus gros fichier de tools/
│   ├── inspect_pickle.py         # Script de debug ponctuel (corrigé pour pointer vers ../*.pkl, cf. session du 2026-07-12)
│   └── kepler_dataset_tools.py, reporting_dataset_pandas.py, move_excess_to_detection.py, rename_category_in_json_files.py, YOLOv8-n.py, detect_non_free_images.py
│
├── docs/                         # Documentation existante (strat_*.md) + nouveaux artefacts BMAD (renommé depuis doc/, 2026-07-12)
├── data/                         # Datasets .npz chunkés — chemin piloté par DATA_ROOT (env var), gitignored
├── best_model.pkl, best_model_detection.pkl  # Checkpoints actuels (classification, détection) — les seuls .pkl encore versionnés après la purge d'historique
│
├── _bmad/, _bmad-output/, .claude/   # Tooling BMAD (nouveau, 2026-07-12)
└── __pycache__/                      # Gitignored
```

## Points d'entrée

- **`main.py`** — point d'entrée unique, pilote tout via `dataset_configs.py`. (`train_detection.py`, l'ancien point d'entrée alternatif dédié détection, a été supprimé le 2026-07-12 — vestige confirmé de la fusion historique JAX_Detection/JAX_Classification, récupérable via l'historique git.)

## Absence de packaging

Aucun `requirements.txt`/`pyproject.toml`/`environment.yml` — les dépendances (JAX, Flax, Optax, TensorFlow, NumPy, SciPy, PIL, Matplotlib, psutil, tqdm) ne sont déclarées nulle part dans le repo, seulement présentes dans l'environnement conda `jax_env` de la machine locale.
