import ctypes
import gc
import os
import json
import glob
import numpy as np
from PIL import Image
import tqdm
from typing import List, Tuple, Dict

from detection_target_encoding import encode_detection_targets, HEATMAP_KEY, SIZE_KEY

try:
    _LIBC = ctypes.CDLL("libc.so.6")
except OSError:
    _LIBC = None  # non-Linux (ex. Colab reste Linux, mais garde defensive)


def _release_freed_memory_to_os():
    """
    gc.collect() + malloc_trim(0) (glibc) apres chaque chunk : le comptage de
    references Python libere deja les listes/tableaux immediatement (verifie -
    current_chunk_images = [] etc. abandonne bien les references), mais
    l'allocateur C sous-jacent ne rend pas toujours cette memoire liberee a l'OS
    apres de gros blocs alloues/liberes en serie (fragmentation d'arene glibc) -
    la RAM visible (ps/free) peut sembler ne jamais redescendre entre 2 chunks
    sans qu'il y ait de fuite logique. malloc_trim force la restitution.
    """
    gc.collect()
    if _LIBC is not None:
        _LIBC.malloc_trim(0)


def process_detection_dataset_v2(
    root_dirs: List[str],
    output_dir: str,
    split_name: str = "train",
    target_size: Tuple[int, int] = (224, 224),
    max_boxes: int = 20,
    chunk_size: int = 2000,
    grayscale: bool = False
):
    """
    Traite le dataset pour la detection (JAX_DETECTOR, heatmap+taille) :
    1. Regroupe les annotations par image source (meme logique que fighterjet_detection_dataset_tools.py,
       reimplementee ici - AD-20 interdit de modifier ce fichier pour en extraire une fonction)
    2. Redimensionne par etirement (LANCZOS, meme methode que l'outil actuel)
    3. Encode les raw_boxes en cibles heatmap+taille via encode_detection_targets (Story 7.1)
    4. Sauvegarde en chunks .npz sous les cles "images"/HEATMAP_KEY/SIZE_KEY

    Args:
        grayscale: Si True, convertit les images en niveaux de gris (1 canal) au lieu de RGB (3 canaux)
    """
    mode_str = "Grayscale (1 canal)" if grayscale else "RGB (3 canaux)"
    print(f"🚀 Traitement Détection v2 heatmap+taille ({split_name}) : {len(root_dirs)} dossiers -> {target_size} [{mode_str}]")

    # Structure pour regrouper les infos par image
    # Cle: nom_image_source (ex: "8a3ab5634b9ab46c.jpg")
    # Valeur: { 'path': ..., 'boxes': [...] }
    images_data: Dict[str, Dict] = {}

    # 1. Scanner tous les JSONs pour regrouper par image
    json_files = []
    for directory in root_dirs:
        if os.path.exists(directory):
            json_files.extend(glob.glob(os.path.join(directory, "**", "*.json"), recursive=True))
        else:
            print(f"⚠️  Dossier introuvable : {directory}")

    print(f"📚 Analyse de {len(json_files)} fichiers JSON...")

    for json_file in tqdm.tqdm(json_files, desc="Grouping annotations"):
        try:
            with open(json_file, "r") as f:
                data = json.load(f)

            # Nom de l'image source (ex: "8a3ab5634b9ab46c.jpg")
            img_filename = data["image"]["file_name"]

            # Chemin complet de l'image (suppose dans le meme dossier que le JSON)
            img_path = os.path.join(os.path.dirname(json_file), img_filename)

            # Verifier si l'image existe
            if not os.path.exists(img_path):
                continue

            if img_filename not in images_data:
                images_data[img_filename] = {
                    'path': img_path,
                    'boxes': []
                }

            # Recuperer la bbox (x, y, w, h) originale
            bbox = data["annotation"]["bbox"]
            images_data[img_filename]['boxes'].append(bbox)

        except Exception as e:
            pass

    print(f"📊 {len(images_data)} images uniques trouvées avec annotations.")

    # 2. Creer les Chunks
    image_list = list(images_data.values())
    np.random.shuffle(image_list)  # Melanger pour l'entrainement

    processed_count = 0
    chunk_idx = 0

    current_chunk_images = []
    current_chunk_heatmaps = []
    current_chunk_sizes = []

    os.makedirs(output_dir, exist_ok=True)

    for i, item in enumerate(tqdm.tqdm(image_list, desc="Processing images")):
        img_path = item['path']
        raw_boxes = item['boxes']  # Liste de [x, y, w, h]

        try:
            with Image.open(img_path) as img:
                img_mode = 'L' if grayscale else 'RGB'
                img = img.convert(img_mode)
                orig_w, orig_h = img.size

                # --- STRETCHED RESIZING (meme methode que l'outil actuel) ---
                new_img = img.resize(target_size, Image.Resampling.LANCZOS)

                # Normalisation Image
                img_array = np.array(new_img, dtype=np.float32) / 255.0  # [0-1]

                if grayscale and img_array.ndim == 2:
                    img_array = np.expand_dims(img_array, axis=-1)

                # Encodage heatmap+taille (Story 7.1) - rescale, gaussien, plafond max_boxes delegues
                targets = encode_detection_targets(raw_boxes, orig_w, orig_h, target_size, max_boxes)

                current_chunk_images.append(img_array)
                current_chunk_heatmaps.append(targets[HEATMAP_KEY])
                current_chunk_sizes.append(targets[SIZE_KEY])
                processed_count += 1

                # Sauvegarder si chunk plein
                if len(current_chunk_images) >= chunk_size:
                    _save_chunk_v2(output_dir, split_name, chunk_idx, current_chunk_images, current_chunk_heatmaps, current_chunk_sizes)
                    chunk_idx += 1
                    current_chunk_images = []
                    current_chunk_heatmaps = []
                    current_chunk_sizes = []

        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            continue

    # Sauvegarder le dernier chunk partiel
    if len(current_chunk_images) > 0:
        _save_chunk_v2(output_dir, split_name, chunk_idx, current_chunk_images, current_chunk_heatmaps, current_chunk_sizes)


def _save_chunk_v2(output_dir, split_name, chunk_idx, images, heatmaps, sizes):
    n_images = len(images)

    # Empiler puis vider chaque liste source IMMEDIATEMENT (.clear(), mutation en place -
    # le vidage est visible du cote appelant aussi, qui reassigne de toute facon a [] juste
    # apres l'appel) plutot que d'attendre la fin de la fonction : les 3 listes sources et
    # les 3 tableaux empiles ne sont alors JAMAIS tous les 6 vivants simultanement, ce qui
    # reduit le pic memoire reel (pas seulement le nettoyage post-hoc de _release_freed_memory_to_os).
    images_np = np.array(images, dtype=np.float32)      # (N, H, W, C)
    images.clear()
    heatmaps_np = np.array(heatmaps, dtype=np.float32)  # (N, H, W, 1)
    heatmaps.clear()
    sizes_np = np.array(sizes, dtype=np.float32)         # (N, H, W, 2)
    sizes.clear()

    # Prefixe distinct de "dataset_detection_" pour ne pas etre absorbe par le glob de
    # nettoyage de l'outil actuel (fighterjet_detection_dataset_tools.py:182, AD-20).
    out_path = os.path.join(output_dir, f"jax_detector_targets_{split_name}_chunk{chunk_idx}.npz")
    np.savez_compressed(out_path, images=images_np, **{HEATMAP_KEY: heatmaps_np, SIZE_KEY: sizes_np})
    print(f"💾 Chunk {chunk_idx} saved: {n_images} images")

    del images_np, heatmaps_np, sizes_np
    _release_freed_memory_to_os()


if __name__ == "__main__":
    import sys
    sys.path.append(os.path.dirname(__file__))
    from dataset_configs import get_dataset_config

    # Charger la configuration (Story 7.7 - JAX_DETECTOR)
    config = get_dataset_config("JAX_DETECTOR")

    # Configuration des chemins (memes dossiers source que fighterjet_detection_dataset_tools.py)
    TRAIN_DIRS = [
        "/home/aobled/Downloads/Aircraft_DATASET/detection/train",
        #"/home/aobled/Downloads/Aircraft_DATASET/classification/train"
    ]
    VAL_DIRS = [
        "/home/aobled/Downloads/Aircraft_DATASET/detection/val",
        #"/home/aobled/Downloads/Aircraft_DATASET/classification/val"
    ]
    OUTPUT_DIR = os.path.dirname(config["output_prefix"])

    print("🚀 Démarrage de la préparation des données de DÉTECTION (JAX_DETECTOR, heatmap+taille)...")

    # 1. Traiter le Training Set
    process_detection_dataset_v2(
        TRAIN_DIRS,
        OUTPUT_DIR,
        split_name="train",
        target_size=config.get("image_size", (224, 224)),
        max_boxes=config.get("max_boxes", 20),
        # chunk_size lu depuis la config (meme pattern que fighterjet_classification_dataset_tools.py,
        # CHUNK_SIZE = config.get("chunk_size", ...)) plutot qu'code en dur - permet d'ajuster par
        # environnement (ex. Colab, RAM differente) sans toucher au script. Defaut 3000 : 27000
        # (valeur de l'outil actuel, masques uniquement) provoque ici un pic memoire d'environ 40 Go,
        # chaque image portant desormais image+heatmap+size (~784 Ko/image, ~2x le poids image+masque
        # de l'outil actuel), et _save_chunk_v2 gardant la liste Python ET le tableau numpy empile
        # simultanement en memoire au moment de la sauvegarde. 3000 ramene le pic a ~4.5 Go (machine
        # locale : 30 Go RAM + 2 Go swap) - cause identifiee de 2 crashs systeme au chunk_size
        # d'origine. Recalculer ce pic (~chunk_size x 784 Ko x 2) avant d'augmenter cette valeur
        # sur une machine avec plus de RAM (ex. Colab).
        chunk_size=config.get("chunk_size", 3000),
        grayscale=config.get("grayscale", True)
    )

    # 2. Traiter le Val Set
    process_detection_dataset_v2(
        VAL_DIRS,
        OUTPUT_DIR,
        split_name="val",
        target_size=config.get("image_size", (224, 224)),
        max_boxes=config.get("max_boxes", 20),
        # chunk_size lu depuis la config (meme pattern que fighterjet_classification_dataset_tools.py,
        # CHUNK_SIZE = config.get("chunk_size", ...)) plutot qu'code en dur - permet d'ajuster par
        # environnement (ex. Colab, RAM differente) sans toucher au script. Defaut 3000 : 27000
        # (valeur de l'outil actuel, masques uniquement) provoque ici un pic memoire d'environ 40 Go,
        # chaque image portant desormais image+heatmap+size (~784 Ko/image, ~2x le poids image+masque
        # de l'outil actuel), et _save_chunk_v2 gardant la liste Python ET le tableau numpy empile
        # simultanement en memoire au moment de la sauvegarde. 3000 ramene le pic a ~4.5 Go (machine
        # locale : 30 Go RAM + 2 Go swap) - cause identifiee de 2 crashs systeme au chunk_size
        # d'origine. Recalculer ce pic (~chunk_size x 784 Ko x 2) avant d'augmenter cette valeur
        # sur une machine avec plus de RAM (ex. Colab).
        chunk_size=config.get("chunk_size", 3000),
        grayscale=config.get("grayscale", True)
    )
