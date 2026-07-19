import ctypes
import gc
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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


def _generate_fullframe_zoom_variant(
    raw_boxes,
    orig_w: int,
    orig_h: int,
    margin_ratio: float = 0.15,
    min_visible_ratio: float = 0.3,
):
    """
    Calcule un recadrage serre autour de la plus grande boite annotee (par aire, choix
    deterministe), avec une marge, et remappe les autres boites (target incluse) dans le
    repere local de ce crop. Fonction pure - aucune E/S, aucun objet PIL - le remapping
    vers le repere target_size (rescale + gaussien) reste entierement delegue a
    encode_detection_targets (jamais reimplemente ici).

    Ne leve jamais d'exception : retourne None si aucune boite n'est fournie, si la boite
    cible (plus grande aire) est degeneree (w<=0 ou h<=0), ou si AUCUNE boite (cible
    incluse) ne survit au clip du crop (ex. cible elle-meme partiellement hors cadre
    source, deja observe sur ce dataset) - l'appelant doit alors traiter l'image
    normalement, sans generer de variante pour elle.

    Args:
        raw_boxes: liste de (x, y, w, h), pixels de l'image source (meme convention que
            encode_detection_targets : coin haut-gauche + largeur/hauteur)
        orig_w, orig_h: dimensions de l'image source
        margin_ratio: expansion totale (pas par cote) de chaque dimension du crop par
            rapport a la boite cible - crop_w = tw*(1+margin_ratio), la moitie de la
            marge etant ajoutee de chaque cote. Avec le defaut 0.15, la boite cible
            occupe encore ~1/1.15^2 ~= 75% de l'aire du crop (avant clip aux bords image).
        min_visible_ratio: fraction minimale de l'aire d'origine d'une boite (target
            incluse) qui doit rester visible apres clip aux bords du crop pour que la
            boite soit conservee dans `remapped_boxes` ; en-dessous, la boite est ecartee.

    Returns:
        (crop_x0, crop_y0, crop_w, crop_h, remapped_boxes) - crop_x0/crop_y0/crop_w/crop_h
        en pixels entiers du repere image source (crop_w/crop_h > 0 garanti si non-None) ;
        remapped_boxes = liste de [x, y, w, h] en pixels locaux au crop (origine = coin
        haut-gauche du crop, coherent avec ce que attend encode_detection_targets(...,
        orig_w=crop_w, orig_h=crop_h, ...)). None si aucune variante ne peut etre generee.
    """
    if len(raw_boxes) == 0:
        return None

    target = max(raw_boxes, key=lambda b: b[2] * b[3])
    tx, ty, tw, th = target
    if tw <= 0 or th <= 0:
        return None

    margin_x = 0.5 * margin_ratio * tw
    margin_y = 0.5 * margin_ratio * th

    crop_x0 = int(np.floor(max(0.0, tx - margin_x)))
    crop_y0 = int(np.floor(max(0.0, ty - margin_y)))
    crop_x1 = int(np.ceil(min(float(orig_w), tx + tw + margin_x)))
    crop_y1 = int(np.ceil(min(float(orig_h), ty + th + margin_y)))

    crop_w = crop_x1 - crop_x0
    crop_h = crop_y1 - crop_y0
    if crop_w <= 0 or crop_h <= 0:
        return None

    remapped_boxes = []
    for bx, by, bw, bh in raw_boxes:
        if bw <= 0 or bh <= 0:
            continue  # boite degeneree, jamais remappee (meme regle que encode_detection_targets)

        box_x0, box_y0 = bx, by
        box_x1, box_y1 = bx + bw, by + bh

        inter_x0 = max(box_x0, crop_x0)
        inter_y0 = max(box_y0, crop_y0)
        inter_x1 = min(box_x1, crop_x1)
        inter_y1 = min(box_y1, crop_y1)

        if inter_x1 <= inter_x0 or inter_y1 <= inter_y0:
            continue  # aucun recouvrement avec le crop

        inter_area = (inter_x1 - inter_x0) * (inter_y1 - inter_y0)
        orig_area = bw * bh
        visible_ratio = inter_area / orig_area
        if visible_ratio < min_visible_ratio:
            continue

        remapped_boxes.append([
            inter_x0 - crop_x0,
            inter_y0 - crop_y0,
            inter_x1 - inter_x0,
            inter_y1 - inter_y0,
        ])

    if len(remapped_boxes) == 0:
        # Peut arriver si la boite cible elle-meme deborde du cadre source (deja observe
        # sur ce dataset, cf. detection_target_encoding.py) : le clip aux bords du crop
        # peut alors faire passer la cible sous min_visible_ratio. Sans ce garde, on
        # produirait une variante "plein cadre" avec zero boite - un exemple vide
        # silencieusement injecte comme donnee d'entrainement (revue adversariale +
        # edge-case, 2026-07-19). Coherent avec le contrat existant de la fonction : None
        # = aucune variante utilisable pour cette image.
        return None

    return (crop_x0, crop_y0, crop_w, crop_h, remapped_boxes)


def _encode_and_append(img_crop, boxes, w, h, target_size, max_boxes, grayscale, chunk_lists):
    """
    Factorise le corps par-image (resize LANCZOS -> normalisation -> encode_detection_targets
    -> append) partage par l'image originale ET la variante zoom plein-cadre, pour eviter
    toute divergence entre les deux chemins (Boundaries & Constraints du spec augmentation
    zoom - un seul endroit qui redimensionne/normalise/encode).

    Args:
        img_crop: image PIL deja recadree (ou l'image originale non recadree pour le
            chemin "sans augmentation") - PAS encore redimensionnee a target_size
        boxes: raw_boxes [x, y, w, h] dans le repere pixel de img_crop (donc (0,0) = coin
            haut-gauche de img_crop, pas necessairement de l'image source d'origine)
        w, h: dimensions (largeur, hauteur) de img_crop en pixels - orig_w/orig_h attendus
            par encode_detection_targets
        chunk_lists: tuple (images_list, heatmaps_list, sizes_list), mute en place par
            append (meme objets que current_chunk_images/heatmaps/sizes de l'appelant)
    """
    images_list, heatmaps_list, sizes_list = chunk_lists

    resized = img_crop.resize(target_size, Image.Resampling.LANCZOS)
    img_array = np.array(resized, dtype=np.float32) / 255.0  # [0-1]

    if grayscale and img_array.ndim == 2:
        img_array = np.expand_dims(img_array, axis=-1)

    targets = encode_detection_targets(boxes, w, h, target_size, max_boxes)

    images_list.append(img_array)
    heatmaps_list.append(targets[HEATMAP_KEY])
    sizes_list.append(targets[SIZE_KEY])


def process_detection_dataset_v2(
    root_dirs: List[str],
    output_dir: str,
    split_name: str = "train",
    target_size: Tuple[int, int] = (224, 224),
    max_boxes: int = 20,
    chunk_size: int = 2000,
    grayscale: bool = False,
    zoom_augment_probability: float = 0.0
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
        zoom_augment_probability: opt-in (defaut 0.0 = aucun changement de sortie vs. avant
            ce parametre). Pour chaque image, avec cette probabilite, genere EN PLUS de
            l'image originale une 2e variante synthetique recadree serree autour de la plus
            grande boite annotee (voir _generate_fullframe_zoom_variant) - fabrique des
            exemples "objet plein cadre" pour corriger la sous-representation de ce profil
            a l'entrainement (diagnostic tests/diagnose_single_full_size_aircraft.py).
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

                # --- Image originale (STRETCHED RESIZING, meme methode que l'outil actuel) ---
                # Comportement strictement inchange vs. avant ce parametre quand
                # zoom_augment_probability=0.0 (corps par-image factorise dans
                # _encode_and_append, appele une fois ici pour l'original).
                _encode_and_append(
                    img, raw_boxes, orig_w, orig_h, target_size, max_boxes, grayscale,
                    (current_chunk_images, current_chunk_heatmaps, current_chunk_sizes),
                )
                processed_count += 1

                # Sauvegarder si chunk plein
                if len(current_chunk_images) >= chunk_size:
                    _save_chunk_v2(output_dir, split_name, chunk_idx, current_chunk_images, current_chunk_heatmaps, current_chunk_sizes)
                    chunk_idx += 1
                    current_chunk_images = []
                    current_chunk_heatmaps = []
                    current_chunk_sizes = []

                # --- Variante zoom plein-cadre (opt-in) ---
                # np.random.random() n'est meme pas appele quand zoom_augment_probability=0.0
                # (court-circuit) : zero divergence d'etat aleatoire vs. avant ce parametre.
                if zoom_augment_probability > 0.0 and np.random.random() < zoom_augment_probability:
                    variant = _generate_fullframe_zoom_variant(raw_boxes, orig_w, orig_h)
                    if variant is not None:
                        crop_x0, crop_y0, crop_w, crop_h, remapped_boxes = variant
                        img_crop = img.crop((crop_x0, crop_y0, crop_x0 + crop_w, crop_y0 + crop_h))
                        _encode_and_append(
                            img_crop, remapped_boxes, crop_w, crop_h, target_size, max_boxes, grayscale,
                            (current_chunk_images, current_chunk_heatmaps, current_chunk_sizes),
                        )
                        processed_count += 1

                        # Sauvegarder si chunk plein (re-verifie : la variante peut avoir
                        # rempli le chunk juste apres l'original)
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
        grayscale=config.get("grayscale", True),
        zoom_augment_probability=config.get("zoom_augment_probability", 0.0)
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
        grayscale=config.get("grayscale", True),
        zoom_augment_probability=config.get("zoom_augment_probability", 0.0)
    )
