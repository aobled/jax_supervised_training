# Audit — Code mort et duplications (jax_supervised_training)

_Généré par le workflow `bmad-document-project` (scan brownfield), 2026-07-12. Complète `strat_global.md` en documentant l'écart entre l'architecture cible et l'état réel du code._

_**Post-refactor (2026-07-12)** : les éléments recensés dans cet audit ont été traités par le refactor `_bmad-output/planning-artifacts/epics.md` (Epic 1-3). Ce document reste tel quel comme trace de l'analyse d'origine — voir `epics.md` et les fichiers story sous `_bmad-output/implementation-artifacts/` pour l'état après refactor._

## 0. Point d'entrée vestige confirmé : `train_detection.py`

Vestige de l'époque où `jax_supervised_training` et `JAX_Classification` étaient deux projets fusionnés. Contourne complètement le pattern Strategy actuel (`main.py`/`Trainer`/`TaskStrategy`) : `CONFIG` codé en dur, import direct de `AircraftDetector`/`DetectionDataset`, pointe vers `aircraft_detector_v3` (un des 15 modèles morts de la section 1), doublon `from tqdm import tqdm` (l.17-18). **Confirmé par l'utilisateur (2026-07-12) : à archiver puis supprimer, n'apporte plus rien.** → **Supprimé (Story 3.1).**

## 1. Architectures de modèles mortes (`model_library.py`)

`model_library.py` (2614 lignes) enregistre **22 architectures** dans le dict `MODELS` (factory `get_model()`, lignes 2450-2474). Aucun script du projet n'appelle `get_model()` avec une chaîne littérale — le seul point d'entrée est `dataset_configs.py` via le champ `model_name` de chaque config.

En croisant les 22 entrées de `MODELS` avec les `model_name` réellement utilisés par **les 7 configs présentes dans `dataset_configs.py`** (actives + gardées "pour l'exemple") :

_Correction 2026-07-12 (relevée pendant la revue du PRD de refactor) : ce document indiquait initialement "8 configs" par erreur de comptage — il y en a bien 7 (`FIGHTERJET_CLASSIFICATION`, `FIGHTERJET_VIT`, `FIGHTERJET_HYBRID_VIT`, `FIGHTERJET_LETTERBOX`, `FIGHTERJET_DETECTION`, `FIGHTERJET_DETECTION_SOPHISTICATED`, `JAX_KEPLER`)._

**Utilisées (8/22) :**
- `aircraft_detector_unet`, `aircraft_detector_miniunet`, `aircraft_detector_sophisticated_unet`
- `sophisticated_cnn_128`, `sophisticated_cnn_128_plus`
- `hybrid_tiny_vit`, `tiny_vit_plus_balanced`
- `kepler_1d_cnn`

**Jamais référencées par aucune config (15/22 — candidates code mort) :**
- `aircraft_detector` (lambda), `aircraft_detector_v2`, `v3`, `v4`, `v5_highres`, `v6_multilevel`, `v7_advanced`, `aircraft_detector_unet_token`
- `sophisticated_cnn`, `sophisticated_cnn_droped_out`, `sophisticated_cnn_128_ultimate`, `sophisticated_cnn_optimized`
- `resnet_light`, `tiny_vit_plus`, `tiny_vit_plus_ultimate`

Ces classes `nn.Module` + fonctions `create_*` représentent une bonne partie des 2614 lignes du fichier — probablement l'essentiel du gain de lisibilité si elles sont retirées ou archivées.

**⚠️ Nuance à trancher avant suppression** : "jamais référencé par une config actuelle" ne veut pas dire "jamais utile" — certaines de ces classes ont pu servir à charger d'anciens checkpoints (`.pkl`) dont la structure de poids dépend de l'architecture exacte. À vérifier avant suppression : est-ce que `best_model.pkl`/`best_model_detection.pkl` actuels correspondent bien à un des 8 modèles "vivants" ?

## 2. Fonctions dupliquées / divergentes (confirme le signalement "tools/ copie en dur")

Recherche des fonctions définies à l'identique (même nom) dans plusieurs fichiers, racine et `tools/` :

| Fonction | Définie dans (×N fichiers) |
|---|---|
| `load_jax_model` | `bounding_boxes_with_classification_from_video_generation.py`, `bounding_boxes_with_classification_from_benchmark.py`, `tools/bounding_boxes_with_classification_from_images_generation.py` |
| `load_detection_model` | ces 3 fichiers **+** `heatmap_generation.py`, `heatmap_contouring.py` (5×) |
| `predict_crop` | ces 3 mêmes fichiers |
| `get_iou` | ces 3 mêmes fichiers |
| `non_max_suppression` | ces 3 mêmes fichiers |
| `_preprocess_crop_to_hwc` | `bounding_boxes_with_classification_from_video_generation.py`, `tools/audit_dataset_classification.py` |

**Point important, vérifié par diff** : ce ne sont **pas** des copier-coller identiques — les implémentations ont divergé avec le temps :
- `predict_crop` a 2 signatures différentes selon le fichier : `(crop_img, predict_fn, mean, std, config)` vs `(crop_img, model, variables, mean, std, config)`
- `non_max_suppression` aussi : `(boxes, iou_threshold)` (tri interne par score) vs `(boxes, scores, iou_threshold)` (scores séparés, utilise `get_iou` en interne)
- `load_jax_model` : la version dans `bounding_boxes_with_classification_from_video_generation.py` a un fallback de résolution de chemin (parent dir) que la version de `bounding_boxes_with_classification_from_benchmark.py` n'a pas.

→ La mutualisation ne peut donc pas être un simple "supprimer les doublons et importer" : il faut d'abord **choisir/réconcilier un comportement canonique** par fonction, ce qui est un vrai travail de conception, pas juste du nettoyage mécanique.

**Fichiers concernés par cette duplication** (candidats à devenir un module partagé, ex. `inference_utils.py`) :
- `bounding_boxes_with_classification_from_benchmark.py`
- `bounding_boxes_with_classification_from_video_generation.py`
- `tools/bounding_boxes_with_classification_from_images_generation.py`
- `heatmap_generation.py`, `heatmap_contouring.py`
- `tools/audit_dataset_classification.py`
