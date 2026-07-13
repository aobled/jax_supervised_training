# Archive

Code fonctionnel mais plus utilisé dans le pipeline actif — conservé pour référence historique et récupérable via l'historique git de son emplacement d'origine (aucun de ces fichiers n'a été supprimé, seulement déplacé). Aucun de ces fichiers n'est importé par le pipeline d'entraînement (`main.py`) ni par `inference_utils.py`.

## Outil de bootstrap d'annotation via YOLO générique (pré-JAX)

- `bounding_boxes_from_images_generation.py`
- `bounding_boxes_from_images_generation_main.py`
- `yolov8n.pt` (checkpoint YOLOv8n pré-entraîné, PyTorch, via [Ultralytics](https://github.com/ultralytics/ultralytics))

**Contexte** : utilisés à une période où des milliers d'images étaient disponibles mais sans modèle de détection propriétaire entraîné. Ces scripts appelaient YOLOv8n (modèle générique pré-entraîné, PyTorch) pour pré-générer des bounding boxes automatiquement, servant de point de départ à l'annotation manuelle. Depuis, le projet dispose de son propre modèle de détection (`aircraft_detector_unet`, JAX/Flax, entraîné sur les données du projet) — ces scripts ne sont plus nécessaires pour ce flux.

**Dépendance** : `ultralytics` (PyTorch). Volontairement absente de `requirements.txt` (voir Story 4.3) pour ne pas imposer PyTorch comme dépendance par défaut d'un projet 100% JAX — installer à la demande (`pip install ultralytics`) si ces scripts sont réactivés.

**Réactivation** : les deux scripts sont autonomes (pas de dépendance à `inference_utils.py` ou au reste du pipeline) — `pip install ultralytics opencv-python-headless tqdm` suffit pour les relancer tels quels. Attention : chemins d'entrée codés en dur (`/home/aobled/Downloads/...`) à adapter avant réutilisation.

## Reproduction YOLOv8n en JAX/Flax (expérimentale, jamais finalisée)

- `YOLOv8-n.py`

**Contexte** : tentative de réimplémenter l'architecture YOLOv8n (Conv, C2f, SPPF, tête découplée, post-traitement) directement en Flax/Linen — objectif : obtenir un équivalent JAX du modèle utilisé par les scripts de bootstrap ci-dessus, pour rester cohérent avec la stack JAX du projet plutôt que de dépendre de PyTorch/Ultralytics. Non finalisé, jamais entraîné ni intégré à `model_library.py` — un skeleton d'architecture, pas un modèle utilisable en l'état. Jamais importé par aucun autre fichier du repo.

**Réactivation** : nécessiterait un travail d'entraînement complet (le fichier ne contient que la définition d'architecture, aucun poids). À reprendre uniquement si le besoin d'un détecteur style YOLO (par opposition à l'approche segmentation actuelle, `aircraft_detector_unet`) redevient pertinent.
