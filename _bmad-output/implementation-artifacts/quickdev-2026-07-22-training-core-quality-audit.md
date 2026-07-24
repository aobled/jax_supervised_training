# Audit qualité — noyau d'entraînement (racine du projet)

**Origine** : session Winston (`bmad-agent-architect`) du 2026-07-22, en continuité de la rétro `quickdev-2026-07-22` — item d'action #2 ("repérer les méthodes dupliquées/copier-collées entre programmes avant toute consolidation"). Demande d'Aymeric : le noyau d'entraînement (racine du projet) est délibérément conçu générique — pas seulement détection/classification aviation — CIFAR10 (cas image simple, évite de relancer sur le dataset avion à chaque test) et Kepler (cas hors-image, tabulaire/1D) existent spécifiquement pour prouver et exercer cette généricité.

**Portée** : `main.py`, `trainer.py`, `task_strategies.py`, `model_library.py`, `data_management.py`, `checkpoint_manager.py`, `dataset_configs.py`, `inference_utils.py`, `loss_functions.py`, `detection_target_encoding.py`, `reporting.py`, `utils.py`. Distinct de l'audit qualité 4-groupes déjà fait (`_bmad-output/implementation-artifacts/deferred-work.md`) qui ciblait la correction (bugs numériques/géométriques) — celui-ci cible 4 axes différents : code mort, duplication copier-coller, cohérence de nommage, lisibilité/maintenabilité.

**Méthode** : lecture directe (`task_strategies.py`, par Winston) + sweep systématique sur les 11 autres fichiers (agent dédié, chaque usage vérifié par grep, pas deviné) + vérification finale par grep direct sur tout le nommage (camelCase).

## Verdict sur la cohérence de nommage

**Aucune dérive camelCase/snake_case trouvée** — vérifié par grep sur les 12 fichiers (définitions de fonctions, attributs `self.x`, variables locales, clés de dict). snake_case est appliqué uniformément partout, y compris dans `task_strategies.py`. Le seul hit du grep (`sigmaX=` dans `bounding_boxes_with_classification_from_video_generation.py`) est un paramètre imposé par l'API `cv2.GaussianBlur`, pas une incohérence de ton fait. Ton inquiétude n'est pas confirmée par le code — bonne nouvelle, rien à corriger sur cet axe.

## Code mort confirmé (usages vérifiés par grep, pas supposés)

| Élément | Fichier | Constat |
|---|---|---|
| `compute_grid_loss`, `compute_grid_loss_multilevel`, `compute_v7_loss` | `loss_functions.py` | ~400 lignes sur 557. Atteignables seulement via `loss_method="grid"/"grid_multilevel"/"v7"`, jamais utilisé par aucune entrée de `DATASET_CONFIGS` — confirme AD-19 (grid/YOLO reporté au profit de CenterNet, jamais réactivé) |
| `from loss_functions import compute_grid_loss` | `trainer.py:27` | import mort, jamais appelé |
| `tree_add` (import) | `trainer.py:25` | importé, jamais appelé |
| `create_eval_step()`, `create_train_step()` | `utils.py:119,147` | version pré-refactor, supplantée par les méthodes privées de `Trainer` |
| `list_available_models()`, `get_model_info()` | `model_library.py:653,658` | jamais importés ailleurs |
| `Reporter.confusion_matrix_from_state`, classe `DetectionReporter` (dont `visualize_batch`) | `reporting.py:267,869` | déjà signalé dans un audit antérieur (`deferred-work.md:141-157`), toujours présent, reconfirmé |
| `"yolo_iou"`/`"yolo_boxes"` branches (`NotImplementedError`) | `task_strategies.py:231,272` | lié au même code mort grid/YOLO ci-dessus |

## Duplication copier-coller confirmée

| Duplication | Fichiers | Constat |
|---|---|---|
| `_get_export_path` + `get_training_state_path` | `task_strategies.py`, les 4 classes stratégie | code identique dans `ClassificationStrategy`, `DetectionStrategy`, `CenterNetDetectionStrategy`, `KeplerStrategy` — **devrait être une implémentation par défaut dans `TaskStrategy` (base), pas 4 copies déclarées `@abstractmethod`** |
| `generate_reports` (visu cv2 : colormap, hconcat, imwrite) | `task_strategies.py`, `DetectionStrategy` vs `CenterNetDetectionStrategy` | ~25 lignes quasi identiques, seul le nom du fichier de sortie change |
| `augment_fn` (flip/translation/zoom/luminosité) | `data_management.py`, `DetectionDataset` vs `CenterNetDetectionDataset` | ~90 lignes de logique d'augmentation dupliquée, seul le payload change (masque unique vs heatmap+taille) |
| `decode_segmentation_and_detect` vs `postprocess_frame` (interne à `decode_segmentation_and_detect_batch`) | `inference_utils.py:736-794` vs `842-882` | même pipeline (seuillage→morphologie→`findContours`→filtre aire→score) implémenté deux fois, kernels recréés à chaque appel côté single-image au lieu de réutiliser les constantes module déjà définies côté batch |
| `confusion_matrix_from_pkl` vs `confusion_matrix_from_state` | `reporting.py:33-183` vs `267-372` | ~80 lignes quasi identiques (déjà noté dans l'audit antérieur, reconfirmé) |

## Lisibilité / maintenabilité

- **God functions** : `Reporter.show_predictions_from_dir` (`reporting.py`, ~364 lignes — chargement modèle + scan dossier + inférence + filtrage + logs + plot, tout en un), `ChunkManager.create_tf_datasets` (`data_management.py`, ~130 lignes, fermetures imbriquées sur 4 niveaux), `main()` (`main.py`, ~160 lignes, plus léger — script d'orchestration, moins grave).
- **Avalement d'erreurs silencieux** : `ChunkManager.get_chunk_statistics` (`data_management.py:84,95`) — un chunk `.npz` corrompu est juste `print()`é et ignoré, aucun comptage d'échec remonté à l'appelant.
- **Incohérence d'interface latente** (trouvée par Winston, pas l'agent) : `KeplerStrategy.compute_loss` (`task_strategies.py:417`) n'accepte pas `**kwargs`, contrairement à la méthode abstraite et aux 3 autres implémentations. Ne casse rien aujourd'hui (`trainer.py` n'appelle qu'avec `use_onehot_labels=`), mais rompt la promesse de généricité de `TaskStrategy` — le jour où `Trainer` passe un nouveau kwarg pour une autre stratégie, Kepler lève un `TypeError`.
- **Magic numbers isolés** : `checkpoint_manager.py:114-120` recalcule `warmup_steps=500, decay_steps=5000` en dur dans `resume_training`, alors que ces valeurs vivent normalement dans la config (`trainer.py:176-188`) — sans conséquence aujourd'hui seulement parce que `opt_state` est écrasé juste après (ligne 135).

## Lecture d'architecte (Winston)

Le pattern Strategy (`TaskStrategy` + 4 implémentations concrètes couvrant image-classification, segmentation, détection point-central, et tabulaire/1D) **généralise réellement** — ce n'est pas un hasard heureux, Kepler et CIFAR10 le prouvent par des cas concrets, structurellement différents. Le design tient.

Deux défauts structurels, pas juste stylistiques, avant d'ajouter un futur type de tâche :

1. **La frontière d'abstraction de `_get_export_path`/`get_training_state_path` est mal placée** — déclarées `@abstractmethod` alors qu'elles sont *toujours* implémentées identiquement. Ce n'est pas juste "du code dupliqué à nettoyer", c'est un signal que le contrat d'interface ne correspond pas à l'usage réel. Un futur type de tâche héritera par défaut de ce copier-coller au lieu d'un comportement partagé.
2. **L'augmentation (`data_management.py`) est couplée au type de tâche alors qu'elle ne devrait pas l'être** — AD-17 dit à raison "un nouveau format de tâche = une nouvelle classe dédiée, jamais une branche conditionnelle", mais l'augmentation géométrique (flip/translation/zoom) s'applique aux pixels de la même façon quelle que soit la nature du payload (masque, heatmap+taille, autre chose demain). La dupliquer intégralement plutôt que de la factoriser en un helper agnostique du payload va à l'encontre de l'objectif de généricité que tu vises.

Le code mort (grid/YOLO, `DetectionReporter`, `create_eval_step`/`create_train_step`) n'est pas un problème de généricité — c'est du poids mort résiduel de itérations passées, sans risque à le laisser, mais sans valeur à le garder non plus.

## Prochaine étape

Rien n'est corrigé ici — c'est un inventaire, pas une intervention. À toi de dire ce qui vaut la peine d'être traité maintenant vs déferré (`deferred-work.md`).
