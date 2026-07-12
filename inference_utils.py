"""Module partagé des fonctions d'inférence JAX/Flax (détection + classification).

Source unique de vérité pour le chargement de checkpoints, le prétraitement,
la prédiction et le décodage de détection par segmentation. Tout fichier qui a
besoin d'une de ces fonctions importe depuis ce module — aucune redéfinition
locale (voir ARCHITECTURE-SPINE.md, AD-1 à AD-8).

Auteur unique (AD-7) : aucune autre story du refactor JAX_Detection ne doit
modifier ce fichier pour y ajouter ou changer une fonction.
"""
import os
import pickle
import concurrent.futures

import cv2
import numpy as np
import jax
import jax.numpy as jnp

from model_library import get_model

# Constantes privées (AD-2, AD-3, AD-6) — jamais redéfinies dans un fichier consommateur.
DETECTION_IMAGE_SIZE = (224, 224)
_CLF_BATCH_SIZE = 32
_DET_BATCH_SIZE = 32
_CROP_MARGIN_PERCENT = 0
_CLOSING_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
_DILATE_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))


def _resolve_checkpoint_path(checkpoint_path):
    """Fallback de résolution de chemin à 3 niveaux : CWD -> parent du CWD -> racine du repo.

    Le 3e niveau résout contre le répertoire de inference_utils.py lui-même (la racine du
    repo, où vivent tous les checkpoints) plutôt que contre le fichier appelant : ce module
    n'a pas de moyen fiable de connaître l'emplacement de son appelant, et la racine du repo
    est de toute façon la seule valeur utile ici (tous les checkpoints y résident).
    """
    if os.path.exists(checkpoint_path):
        return checkpoint_path

    parent_checkpoint = os.path.join(os.path.dirname(os.getcwd()), checkpoint_path)
    if os.path.exists(parent_checkpoint):
        return parent_checkpoint

    repo_root = os.path.dirname(os.path.abspath(__file__))
    repo_root_checkpoint = os.path.join(repo_root, checkpoint_path)
    if os.path.exists(repo_root_checkpoint):
        return repo_root_checkpoint

    return None


def load_jax_model(checkpoint_path, config):
    """Charge le modèle JAX de CLASSIFICATION. Retourne (model, variables, mean, std)."""
    resolved = _resolve_checkpoint_path(checkpoint_path)
    if resolved is None:
        raise FileNotFoundError(f"Checkpoint non trouvé: {checkpoint_path}")
    checkpoint_path = resolved

    print(f"🔍 Chargement du modèle CLASSIFICATION depuis {checkpoint_path}...")
    with open(checkpoint_path, 'rb') as f:
        model_data = pickle.load(f)

    if 'model_state' in model_data:
        params = model_data['model_state']['params']
        batch_stats = model_data['model_state'].get('batch_stats', {})
        model_info = model_data.get('model_info', {})
        model_name = model_info.get('model_name', config["model_name"])
    else:
        params = model_data['params']
        batch_stats = model_data.get('batch_stats', {})
        model_name = model_data.get('model_name', config["model_name"])

    num_classes = config["num_classes"]

    model = get_model(model_name, num_classes=num_classes, dropout_rate=0.0)
    variables = {'params': params, 'batch_stats': batch_stats}

    mean_std_path = config.get("mean_std_path", "./data/chunks/dataset_chunked_meanstd.npz")
    if not os.path.exists(mean_std_path):
        repo_root = os.path.dirname(os.path.abspath(__file__))
        mean_std_path_abs = os.path.join(repo_root, mean_std_path)
        if os.path.exists(mean_std_path_abs):
            mean_std_path = mean_std_path_abs

    if os.path.exists(mean_std_path):
        with np.load(mean_std_path) as data:
            mean = data['mean']
            std = data['std']
            print("✅ Stats de normalisation chargées.")

            if config.get("grayscale", False):
                if isinstance(mean, np.ndarray) and mean.size == 3:
                    print("⚠️  Conversion des stats RGB -> Grayscale (mean/std)")
                    mean = np.mean(mean)
                    std = np.mean(std)
    else:
        print("⚠️  ATTENTION: Stats de normalisation non trouvées, utilisation de valeurs par défaut (0.5, 0.5)")
        mean = 0.5
        std = 0.5

    return model, variables, mean, std


def load_detection_model(checkpoint_path):
    """Charge le modèle JAX de DÉTECTION. Retourne (model, variables, config_model).

    AD-3: fallback de chemin 3 niveaux + ré-initialisation des batch_stats manquants.
    AD-4: fallback model_name par défaut = aircraft_detector_unet (jamais un modèle mort).
    """
    resolved = _resolve_checkpoint_path(checkpoint_path)
    if resolved is None:
        raise FileNotFoundError(f"Checkpoint détection non trouvé: {checkpoint_path}")
    checkpoint_path = resolved

    print(f"🔍 Chargement du modèle DÉTECTION depuis {checkpoint_path}...")
    with open(checkpoint_path, 'rb') as f:
        data_model = pickle.load(f)

    params = data_model['params']
    config_model = data_model.get('config', {})
    model_name = config_model.get('model_name', 'aircraft_detector_unet')

    print(f"   Modèle détecté: {model_name}")

    model = get_model(model_name, dropout_rate=0.0)

    batch_stats = data_model.get('batch_stats', {})

    if not batch_stats:
        if 'model_state' in data_model:
            batch_stats = data_model['model_state'].get('batch_stats', {})

    if not batch_stats:
        print("⚠️  ATTENTION: 'batch_stats' non trouvés dans le checkpoint !")
        print("   Le modèle utilise des BatchNorms mais les stats (moyenne/variance) n'ont pas été sauvegardées.")
        print("   🔧 Tentative de ré-initialisation (les stats seront à 0/1, ce qui peut affecter la performance).")

        rng = jax.random.PRNGKey(0)
        target_size = config_model.get("image_size", DETECTION_IMAGE_SIZE)
        grayscale = config_model.get("grayscale", True)
        channels = 1 if grayscale else 3
        dummy_input = jnp.ones((1, *target_size, channels), jnp.float32)

        init_variables = model.init(rng, dummy_input, training=True)
        batch_stats = init_variables.get('batch_stats', {})
        print("   ✅ Structure batch_stats ré-initialisée.")

    variables = {'params': params, 'batch_stats': batch_stats}

    return model, variables, config_model


def _preprocess_crop_to_hwc(crop_img, mean, std, config):
    """Prépare un crop BGR en tenseur (H, W, C) float32."""
    target_size = config["image_size"]
    grayscale = config.get("grayscale", False)
    crop_resized = cv2.resize(crop_img, target_size)
    if grayscale:
        img_input = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2GRAY)
        img_input = img_input.astype(np.float32) / 255.0
        img_normalized = (img_input - mean) / std
        return img_normalized[:, :, np.newaxis]
    img_input = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2RGB)
    img_input = img_input.astype(np.float32) / 255.0
    img_normalized = (img_input - mean) / std
    return img_normalized


def _pad_batch_np(batch_np, target_batch_size):
    """Pad un batch numpy à une taille fixe pour éviter la recompilation JAX."""
    n = batch_np.shape[0]
    if n >= target_batch_size:
        return batch_np[:target_batch_size], min(n, target_batch_size)
    pad_shape = (target_batch_size - n,) + batch_np.shape[1:]
    padding = np.zeros(pad_shape, dtype=batch_np.dtype)
    return np.concatenate([batch_np, padding], axis=0), n


def predict_crop(crop_img, model, variables, mean, std, config):
    """Prédit la classe d'un crop (image OpenCV BGR), image unique, non-JIT.

    Retourne: (nom_classe, confiance).
    Ratifie tools/bounding_boxes_with_classification_from_images_generation.py:128 (AD-2).
    """
    img_hwc = _preprocess_crop_to_hwc(crop_img, mean, std, config)
    img_jax = img_hwc[np.newaxis, ...]

    logits = model.apply(variables, jnp.array(img_jax), training=False)
    probs = jax.nn.softmax(logits, axis=-1)

    pred_idx = int(jnp.argmax(probs))
    confidence = float(probs[0, pred_idx])

    return config["class_names"][pred_idx], confidence


def predict_crops_batch(crop_imgs, predict_fn, mean, std, config):
    """Classifie plusieurs crops en chunks de taille fixe _CLF_BATCH_SIZE, via predict_fn précompilé (JIT).

    Retourne une liste de (nom_classe, confiance), une entrée par élément de crop_imgs.
    Ratifie bounding_boxes_with_classification_from_video_generation.py:214 (AD-2) —
    implémentation pleinement indépendante de predict_crop (pas de délégation interne).
    """
    if not crop_imgs:
        return []

    names = config["class_names"]
    all_results = []

    for start in range(0, len(crop_imgs), _CLF_BATCH_SIZE):
        chunk_crops = crop_imgs[start:start + _CLF_BATCH_SIZE]
        batch_np = np.stack(
            [_preprocess_crop_to_hwc(c, mean, std, config) for c in chunk_crops],
            axis=0,
        )
        padded_np, valid_n = _pad_batch_np(batch_np, _CLF_BATCH_SIZE)
        prediction_output = predict_fn(jnp.array(padded_np))

        if isinstance(prediction_output, tuple) and len(prediction_output) == 2:
            probs, pred_indices = prediction_output
        else:
            logits = prediction_output
            probs = jax.nn.softmax(logits, axis=-1)
            pred_indices = jnp.argmax(probs, axis=-1)

        probs_np = np.array(probs[:valid_n])
        pred_indices_np = np.array(pred_indices[:valid_n])

        for i in range(valid_n):
            idx = int(pred_indices_np[i])
            all_results.append((names[idx], float(probs_np[i, idx])))

    return all_results


def build_predict_fn(model, variables):
    """Wrapper JIT générique, sortie brute (logits). Consolide build_det_predict_fn
    (video_generation.py) et le build_predict_fn local de tools/audit_dataset_classification.py (AD-1)."""
    @jax.jit
    def predict_fn(batch_images):
        return model.apply(variables, batch_images, training=False)
    return predict_fn


def build_clf_predict_fn(model, variables):
    """Wrapper JIT avec softmax+argmax intégrés (contrat de sortie différent de build_predict_fn)."""
    @jax.jit
    def predict_fn(batch_images):
        logits = model.apply(variables, batch_images, training=False)
        probs = jax.nn.softmax(logits, axis=-1)
        pred_indices = jnp.argmax(probs, axis=-1)
        return probs, pred_indices
    return predict_fn


def get_iou(box1, box2):
    """Calcule l'Intersection over Union (IoU) de deux boxes [x1, y1, x2, y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union = area1 + area2 - intersection
    return intersection / union if union > 0 else 0


def non_max_suppression(boxes, iou_threshold):
    """Applique le Non-Maximum Suppression (NMS) pour supprimer les boîtes superposées.

    boxes: liste de [x1, y1, x2, y2, score].
    """
    if not boxes:
        return []

    boxes = sorted(boxes, key=lambda x: x[4], reverse=True)
    kept_boxes = []

    for current_box in boxes:
        overlap = False
        for kept_box in kept_boxes:
            iou = get_iou(current_box[:4], kept_box[:4])
            if iou > iou_threshold:
                overlap = True
                break
        if not overlap:
            kept_boxes.append(current_box)

    return kept_boxes


def decode_segmentation_and_detect(img_bgr, model, variables, config_model, conf_threshold=0.3, box_aera_min=225):
    """Détection par Segmentation Sémantique (U-Net), image unique, pleine résolution (AD-6).

    Retourne une liste de boxes [x1, y1, x2, y2, score].
    Ratifie tools/bounding_boxes_with_classification_from_images_generation.py sans modification de comportement.
    """
    h_orig, w_orig = img_bgr.shape[:2]

    target_size = config_model.get("image_size", DETECTION_IMAGE_SIZE)
    grayscale = config_model.get("grayscale", True)

    img_resized = cv2.resize(img_bgr, target_size)

    if grayscale:
        img_input = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
        img_input = img_input.astype(np.float32) / 255.0
        img_jax = img_input[np.newaxis, :, :, np.newaxis]
    else:
        img_input = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_input = img_input.astype(np.float32) / 255.0
        img_jax = img_input[np.newaxis, :, :, :]

    preds = model.apply(variables, jnp.array(img_jax), training=False)

    pred_mask = np.array(preds[0, :, :, 0])

    mask_resized = cv2.resize(pred_mask, (w_orig, h_orig), interpolation=cv2.INTER_CUBIC)

    strong_mask = (mask_resized > conf_threshold).astype(np.uint8) * 255
    weak_mask = (mask_resized > (conf_threshold * 0.4)).astype(np.uint8) * 255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    expanded_strong = cv2.dilate(strong_mask, kernel, iterations=1)

    binary_mask = cv2.bitwise_and(expanded_strong, weak_mask)

    closing_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, closing_kernel, iterations=1)

    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    binary_mask = cv2.dilate(binary_mask, dilate_kernel, iterations=1)

    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    final_detections = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < box_aera_min:
            continue

        x, y, w, h = cv2.boundingRect(contour)

        sub_mask = mask_resized[y:y + h, x:x + w]
        score = float(np.max(sub_mask)) if sub_mask.size > 0 else 1.0

        final_detections.append([x, y, x + w, y + h, score])

    return final_detections


def decode_segmentation_and_detect_batch(frames_bgr, predict_fn, config_model, conf_threshold=0.3, box_aera_min=225):
    """Détection par Segmentation Sémantique (U-Net) sur un batch d'images (AD-6).

    Post-traitement en basse résolution, projection des boxes en HD. Priorité au débit temps
    réel du pipeline vidéo — ne pas dégrader (AD-6, NFR3).
    Retourne une liste de tuples (final_detections_hd, pred_mask_lr, binary_mask_lr).
    Ratifie bounding_boxes_with_classification_from_video_generation.py sans modification de comportement.
    """
    if not frames_bgr:
        return []

    target_size = config_model.get("image_size", DETECTION_IMAGE_SIZE)
    target_w, target_h = target_size
    grayscale = config_model.get("grayscale", True)

    def preprocess_frame(img_bgr):
        h_orig, w_orig = img_bgr.shape[:2]
        img_resized = cv2.resize(img_bgr, target_size)
        if grayscale:
            img_input = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
            img_input = img_input.astype(np.float32) / 255.0
            return (img_input[:, :, np.newaxis], (h_orig, w_orig))
        else:
            img_input = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
            img_input = img_input.astype(np.float32) / 255.0
            return (img_input, (h_orig, w_orig))

    with concurrent.futures.ThreadPoolExecutor() as executor:
        preprocessed = list(executor.map(preprocess_frame, frames_bgr))

    batch_input = [p[0] for p in preprocessed]
    orig_shapes = [p[1] for p in preprocessed]

    img_jax_batch = jnp.array(np.stack(batch_input, axis=0))
    n_frames = len(frames_bgr)
    if n_frames < _DET_BATCH_SIZE:
        pad_shape = (_DET_BATCH_SIZE - n_frames,) + batch_input[0].shape
        img_jax_batch = jnp.concatenate(
            [img_jax_batch, jnp.zeros(pad_shape, dtype=img_jax_batch.dtype)],
            axis=0,
        )

    preds = predict_fn(img_jax_batch)
    preds_np = np.array(preds)[:n_frames]

    def postprocess_frame(i):
        h_orig, w_orig = orig_shapes[i]
        pred_mask = preds_np[i, :, :, 0]

        binary_mask_lr = (pred_mask > conf_threshold).astype(np.uint8) * 255
        binary_mask_lr = cv2.morphologyEx(binary_mask_lr, cv2.MORPH_CLOSE, _CLOSING_KERNEL, iterations=1)
        binary_mask_lr = cv2.dilate(binary_mask_lr, _DILATE_KERNEL, iterations=1)

        scale_x = w_orig / target_w
        scale_y = h_orig / target_h
        box_area_min_lr = box_aera_min / (scale_x * scale_y)

        contours, _ = cv2.findContours(binary_mask_lr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        final_detections = []

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < box_area_min_lr:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            sub_mask = pred_mask[y:y + h, x:x + w]
            score = float(np.max(sub_mask)) if sub_mask.size > 0 else 1.0

            margin_x = int(w * (_CROP_MARGIN_PERCENT / 100.0))
            margin_y = int(h * (_CROP_MARGIN_PERCENT / 100.0))

            x1_lr = max(0, x - margin_x)
            y1_lr = max(0, y - margin_y)
            x2_lr = min(target_w, x + w + margin_x)
            y2_lr = min(target_h, y + h + margin_y)

            x1 = int(x1_lr * scale_x)
            y1 = int(y1_lr * scale_y)
            x2 = min(w_orig, int(x2_lr * scale_x))
            y2 = min(h_orig, int(y2_lr * scale_y))

            final_detections.append((x1, y1, x2, y2, score))

        final_detections = sorted(final_detections, key=lambda b: (b[1], b[0]))
        return (final_detections, pred_mask, binary_mask_lr)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(postprocess_frame, range(n_frames)))

    return results
