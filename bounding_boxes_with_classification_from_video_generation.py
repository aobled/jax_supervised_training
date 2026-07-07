import sys
import os
import threading
import queue
import concurrent.futures

# Réduit les crashs cuDNN autotune (GTX 1660 Ti) — doit être défini avant import jax
os.environ.setdefault(
    "XLA_FLAGS",
    "--xla_gpu_strict_conv_algorithm_picker=true",
)

# Ajouter le répertoire parent en PRIORITÉ absolue (index 0) pour forcer Python
# à utiliser le model_library.py de JAX_Detection (et non celui de JAX_Classification si exécuté depuis l'autre dossier)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Remplace YOLO import par JAX/Flax helpers si nécessaire
# from ultralytics import YOLO  <-- REMOVED
import cv2
import numpy as np
import jax
import jax.numpy as jnp
import pickle
from tqdm import tqdm

from dataset_configs import get_dataset_config
from model_library import get_model  # Uniquement get_model (pas besoin de la classe directe)

# =================================================================================================
# Input CONFIGURATION
# =================================================================================================
# 1. Configuration du dataset et du modèle de classification
DATASET_NAME = "FIGHTERJET_CLASSIFICATION"     # Nom de la config dans dataset_configs.py
CHECKPOINT_PATH = "best_model.pkl"      # Chemin vers le modèle de CLASSIFICATION

# 2. Configuration du modèle de détection
DETECTION_CHECKPOINT_PATH = "best_model_detection.pkl" # Chemin vers le modèle de DÉTECTION
DETECTION_IMAGE_SIZE = (224, 224)       # Taille d'entrée du modèle de détection

# 3. Configuration de la zone de détection
CROP_MARGIN_PERCENT = 0  # 15 = Ajoute 15% de marge autour de la détection pour le classifieur
BOX_AERA_MIN = 500

CANVAS_WIDTH = 1920
CANVAS_HEIGHT = 1080

# Kernels morphologiques pré-alloués (post-traitement masque basse résolution)
_CLOSING_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
_DILATE_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))


# ==========================================================
# Detection CONFIGURATION
# ==========================================================
OUTPUT_DIR = "/home/aobled/Downloads/video_frames_annotated"
FRAME_STRIDE = 1  # 1 = toutes les frames
DETECTION_CONF_THRESHOLD = 0.8          # Seuil pour considérer une détection valide (objectness + class) target 0.6
BATCH_SIZE = 32                         # Batch détection (réduire si OOM GPU, ex. GTX 1660 Ti 6 Go)
CLF_BATCH_SIZE = 32                     # Batch classification fixe (évite recompilation cuDNN)

#VIDEO_PATH = "/home/aobled/Downloads/testvid.mp4"
VIDEO_PATH = "/home/aobled/Downloads/tmp 250th USA.mp4"
#TARGET_CLASS_LIST = ["f15", "f22", "b1b", "b2", "b52", "a10", "f16"]
TARGET_CLASS_LIST = ["f35", "f18","v22","f22", "b1b", "b2", "f16", "c17"]

# 3. Chargement de la config dataset
try:
    config = get_dataset_config(DATASET_NAME)
    CLASS_NAMES = config["class_names"]
    print(f"✅ Configuration chargée: {DATASET_NAME}")
    print(f"📊 Classes ({len(CLASS_NAMES)}): {CLASS_NAMES}")
    #print(f"🔒 Seuil de confiance (Classification): {CONFIDENCE_THRESHOLD * 100}%")
except Exception as e:
    print(f"❌ Erreur chargement config: {e}")
    sys.exit(1)





# =================================================================================================
# FONCTIONS UTILITAIRES POUR JAX (CLASSIFICATION)
# =================================================================================================
def load_jax_model(checkpoint_path, config):
    """Charge le modèle JAX de CLASSIFICATION."""
    if not os.path.exists(checkpoint_path):
        # Essayer de trouver le checkpoint dans le dossier parent si on est dans tools/
        parent_checkpoint = os.path.join(os.path.dirname(os.getcwd()), checkpoint_path)
        if os.path.exists(parent_checkpoint):
            checkpoint_path = parent_checkpoint
        else:
            # Essayer de trouver le checkpoint dans le dossier parent du script
            script_dir = os.path.dirname(os.path.abspath(__file__))
            parent_dir = os.path.dirname(script_dir)
            parent_checkpoint = os.path.join(parent_dir, checkpoint_path)
            if os.path.exists(parent_checkpoint):
                checkpoint_path = parent_checkpoint
            else: 
                raise FileNotFoundError(f"Checkpoint non trouvé: {checkpoint_path}")

    print(f"🔍 Chargement du modèle CLASSIFICATION depuis {checkpoint_path}...")
    with open(checkpoint_path, 'rb') as f:
        model_data = pickle.load(f)

    # Extraction des paramètres
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
    
    # Création du modèle
    model = get_model(model_name, num_classes=num_classes, dropout_rate=0.0)
    variables = {'params': params, 'batch_stats': batch_stats}
    
    # Chargement des stats de normalisation
    mean_std_path = config.get("mean_std_path", "./data/chunks/dataset_chunked_meanstd.npz")
    # Tenter de résoudre le chemin relatif
    if not os.path.exists(mean_std_path):
        # Essayer chemin relatif au script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(script_dir)
        mean_std_path_abs = os.path.join(parent_dir, mean_std_path)
        if os.path.exists(mean_std_path_abs):
            mean_std_path = mean_std_path_abs
        
    if os.path.exists(mean_std_path):
        with np.load(mean_std_path) as data:
            mean = data['mean']
            std = data['std']
            print("✅ Stats de normalisation chargées.")
            
            # 🚑 CORRECTION AUTOMATIQUE : Si Grayscale mais stats RGB
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


def _preprocess_crop_to_hwc(crop_img, mean, std, config):
    """Prépare un crop BGR en tenseur (H, W, C) float32, identique à l'ancien predict_crop."""
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


def build_det_predict_fn(det_model, det_vars):
    @jax.jit
    def predict_fn(batch_images):
        return det_model.apply(det_vars, batch_images, training=False)
    return predict_fn


def build_clf_predict_fn(clf_model, clf_vars):
    @jax.jit
    def predict_fn(batch_images):
        logits = clf_model.apply(clf_vars, batch_images, training=False)
        probs = jax.nn.softmax(logits, axis=-1)
        pred_indices = jnp.argmax(probs, axis=-1)
        return probs, pred_indices
    return predict_fn


def warmup_jit_predictors(det_predict_fn, det_config, clf_predict_fn, config):
    """Compile les kernels cuDNN une seule fois avant la boucle vidéo."""
    det_size = det_config.get("image_size", DETECTION_IMAGE_SIZE)
    det_gray = det_config.get("grayscale", True)
    det_ch = 1 if det_gray else 3
    det_dummy = jnp.zeros((BATCH_SIZE, *det_size, det_ch), dtype=jnp.float32)

    clf_size = config["image_size"]
    clf_ch = 1 if config.get("grayscale", False) else 3
    clf_dummy = jnp.zeros((CLF_BATCH_SIZE, *clf_size, clf_ch), dtype=jnp.float32)

    print(f"   Warmup détection (batch={BATCH_SIZE})...")
    det_predict_fn(det_dummy).block_until_ready()
    print(f"   Warmup classification (batch={CLF_BATCH_SIZE})...")
    probs, _ = clf_predict_fn(clf_dummy)
    probs.block_until_ready()


def predict_crops_batch(crop_imgs, predict_fn, mean, std, config):
    """
    Classifie plusieurs crops en chunks de taille fixe CLF_BATCH_SIZE.
    Retourne une liste de (nom_classe, confiance), une entrée par élément de crop_imgs.
    """
    if not crop_imgs:
        return []

    names = config["class_names"]
    all_results = []

    for start in range(0, len(crop_imgs), CLF_BATCH_SIZE):
        chunk_crops = crop_imgs[start:start + CLF_BATCH_SIZE]
        batch_np = np.stack(
            [_preprocess_crop_to_hwc(c, mean, std, config) for c in chunk_crops],
            axis=0,
        )
        padded_np, valid_n = _pad_batch_np(batch_np, CLF_BATCH_SIZE)
        probs, pred_indices = predict_fn(jnp.array(padded_np))
        
        probs_np = np.array(probs[:valid_n])
        pred_indices_np = np.array(pred_indices[:valid_n])

        for i in range(valid_n):
            idx = int(pred_indices_np[i])
            all_results.append((names[idx], float(probs_np[i, idx])))

    return all_results


def predict_crop(crop_img, predict_fn, mean, std, config):
    """
    Prédit la classe d'un crop (image OpenCV BGR).
    Retourne: (nom_classe, confiance)
    """
    return predict_crops_batch([crop_img], predict_fn, mean, std, config)[0]


# =================================================================================================
# FONCTIONS UTILITAIRES POUR DÉTECTION (NOUVEAU)
# =================================================================================================
def load_detection_model(checkpoint_path):
    """Charge le modèle JAX de DÉTECTION."""
    if not os.path.exists(checkpoint_path):
        # Essayer de trouver le checkpoint dans le dossier parent si on est dans tools/
        parent_checkpoint = os.path.join(os.path.dirname(os.getcwd()), checkpoint_path)
        if os.path.exists(parent_checkpoint):
            checkpoint_path = parent_checkpoint
        else:
            # Essayer de trouver le checkpoint dans le dossier parent du script
            script_dir = os.path.dirname(os.path.abspath(__file__))
            parent_dir = os.path.dirname(script_dir)
            parent_checkpoint = os.path.join(parent_dir, checkpoint_path)
            if os.path.exists(parent_checkpoint):
                checkpoint_path = parent_checkpoint
            else: 
                raise FileNotFoundError(f"Checkpoint détection non trouvé: {checkpoint_path}")

    print(f"🔍 Chargement du modèle DÉTECTION depuis {checkpoint_path}...")
    with open(checkpoint_path, 'rb') as f:
        data_model = pickle.load(f)

    params = data_model['params']
    config_model = data_model.get('config', {})
    model_name = config_model.get('model_name', 'aircraft_detector_v3')
    
    print(f"   Modèle détecté: {model_name}")
    
    # Création du modèle
    model = get_model(model_name, dropout_rate=0.0) 
    
    # Récupération ou Initialisation des batch_stats
    batch_stats = data_model.get('batch_stats', {})
    
    if not batch_stats:
        # Tenter de trouver dans 'model_state'
        if 'model_state' in data_model:
             batch_stats = data_model['model_state'].get('batch_stats', {})
             
    if not batch_stats:
        print("⚠️  ATTENTION: 'batch_stats' non trouvés dans le checkpoint !")
        print("   Le modèle utilise des BatchNorms mais les stats (moyenne/variance) n'ont pas été sauvegardées.")
        print("   🔧 Tentative de ré-initialisation (les stats seront à 0/1, ce qui peut affecter la performance).")
        
        # On doit ré-initialiser le modèle pour obtenir la structure des batch_stats
        rng = jax.random.PRNGKey(0)
        target_size = config_model.get("image_size", DETECTION_IMAGE_SIZE)
        grayscale = config_model.get("grayscale", True)
        channels = 1 if grayscale else 3
        dummy_input = jnp.ones((1, *target_size, channels), jnp.float32)
        
        # Init pour avoir la structure
        init_variables = model.init(rng, dummy_input, training=True)
        batch_stats = init_variables.get('batch_stats', {})
        print("   ✅ Structure batch_stats ré-initialisée.")

    variables = {'params': params, 'batch_stats': batch_stats}

    return model, variables, config_model

def get_iou(box1, box2):
    """Calcule l'Intersection over Union (IoU) de deux boxes [x1, y1, x2, y2]"""
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
    """
    Applique le Non-Maximum Suppression (NMS) pour supprimer les boîtes superposées.
    boxes: liste de [x1, y1, x2, y2, score]
    """
    if not boxes:
        return []
    
    # Trier les boîtes par score (le dernier élément) décroissant
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




def decode_segmentation_and_detect_batch(frames_bgr, predict_fn, config_model, conf_threshold=0.3, box_aera_min=225):
    """
    Exécute la détection par Segmentation Sémantique (U-Net) sur un batch d'images.
    Post-traitement en basse résolution (224×224), projection des boxes en HD.
    Retourne une liste de tuples (final_detections_hd, pred_mask_lr, binary_mask_lr).
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
    if n_frames < BATCH_SIZE:
        pad_shape = (BATCH_SIZE - n_frames,) + batch_input[0].shape
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

            margin_x = int(w * (CROP_MARGIN_PERCENT / 100.0))
            margin_y = int(h * (CROP_MARGIN_PERCENT / 100.0))

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


def classify_batch_detections(frames_buffer, batch_results, clf_predict_fn, dataset_mean, dataset_std, config):
    """
    Agrège tous les crops du batch de frames et classifie en un seul forward JAX.
    Retourne une liste (par frame) de tuples (box, crop, (classe, confiance)).
    """
    all_crops = []
    per_frame_items = []

    for i, (detections, _, _) in enumerate(batch_results):
        frame_items = []
        for box in detections:
            x1, y1, x2, y2, _ = box
            crop = frames_buffer[i][y1:y2, x1:x2]
            if crop.size == 0:
                continue
            frame_items.append((box, crop))
            all_crops.append(crop)
        per_frame_items.append(frame_items)

    all_preds = predict_crops_batch(all_crops, clf_predict_fn, dataset_mean, dataset_std, config)

    pred_idx = 0
    frame_predictions = []
    for frame_items in per_frame_items:
        frame_preds = []
        for box, crop in frame_items:
            frame_preds.append((box, crop, all_preds[pred_idx]))
            pred_idx += 1
        frame_predictions.append(frame_preds)

    return frame_predictions


def build_quadrant_canvas(target_frame, target_heatmap_lr, target_binary_mask_lr, frame_predictions, config):
    """Construit le canvas 4 quadrants. Les masques sont en basse résolution (upscale lazy pour la viz)."""
    canvas = np.zeros((CANVAS_HEIGHT, CANVAS_WIDTH, 3), dtype=np.uint8)
    h_orig, w_orig = target_frame.shape[:2]

    # 1. Top-Left: Original
    tl_img = cv2.resize(target_frame, (960, 540))
    canvas[0:540, 0:960] = tl_img

    # 2. Bottom-Left: Heatmap (masque basse résolution)
    hm_vis = (target_heatmap_lr * 255).astype(np.uint8)
    hm_color = cv2.applyColorMap(hm_vis, cv2.COLORMAP_JET)
    bl_img = cv2.resize(hm_color, (960, 540))
    canvas[540:1080, 0:960] = bl_img

    # 3. Top-Right: Annotated + overlay heatmap
    # Optimisation 1 : Réutilisation de l'image redimensionnée (tl_img)
    tr_img = tl_img.copy()
    
    # Optimisation 2 : Réutilisation de la heatmap colorée (bl_img)
    # Création rapide d'un masque binaire en 960x540
    heatmap_mask = cv2.resize((target_heatmap_lr > (40/255)).astype(np.uint8), (960, 540), interpolation=cv2.INTER_NEAREST)
    
    heatmap_color_tr = bl_img.copy()
    heatmap_color_tr[heatmap_mask == 0] = 0

    cv2.addWeighted(tr_img, 1.0, heatmap_color_tr, 0.45, 0, dst=tr_img)

    binary_mask_md = cv2.resize(
        target_binary_mask_lr, (960, 540), interpolation=cv2.INTER_NEAREST
    )
    contours, _ = cv2.findContours(binary_mask_md, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(tr_img, contours, -1, (255, 255, 255), 1)

    # 4. Bottom-Right: Classification Crops (prédictions déjà calculées)
    br_canvas = np.zeros((540, 960, 3), dtype=np.uint8)
    crop_idx = 0
    grid_cols = 7
    grid_rows = 4
    cell_w = 128
    cell_h = 128

    scale_x = 960 / w_orig
    scale_y = 540 / h_orig

    for (x1, y1, x2, y2, det_score), crop, (predicted_class, confidence) in frame_predictions:
        color = (0, 255, 0) if predicted_class in TARGET_CLASS_LIST else (0, 0, 255)
        label = f"{predicted_class}"
        
        # Dessin sur Top-Right (tr_img) avec coordonnées redimensionnées
        x1_md, y1_md = int(x1 * scale_x), int(y1 * scale_y)
        x2_md, y2_md = int(x2 * scale_x), int(y2 * scale_y)
        cv2.rectangle(tr_img, (x1_md, y1_md), (x2_md, y2_md), color, 2)
        cv2.putText(tr_img, label, (x1_md, max(y1_md - 10, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Dessin sur Bottom-Right
        if crop_idx < grid_cols * grid_rows:
            col = crop_idx % grid_cols
            row = crop_idx // grid_cols

            crop_resized = cv2.resize(crop, (cell_w, cell_h))
            if config.get("grayscale", False):
                crop_gray = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2GRAY)
                crop_disp = cv2.cvtColor(crop_gray, cv2.COLOR_GRAY2BGR)
            else:
                crop_disp = crop_resized

            x_start = col * cell_w + 10 + (col * 5)
            y_start = row * (cell_h + 20) + 20 + (row * 5)

            if x_start + cell_w <= 960 and y_start + cell_h <= 540:
                br_canvas[y_start:y_start + cell_h, x_start:x_start + cell_w] = crop_disp
                label = f"{predicted_class} ({100 * confidence:.1f}%)"
                cv2.putText(br_canvas, label, (x_start, y_start - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            crop_idx += 1

    canvas[0:540, 960:1920] = tr_img
    canvas[540:1080, 960:1920] = br_canvas

    return canvas


def process_frames_batch(
    frames_buffer,
    det_predict_fn,
    det_config,
    clf_predict_fn,
    dataset_mean,
    dataset_std,
    config,
):
    """Détection + classification batchée + retourne les canvas générés."""
    batch_results = decode_segmentation_and_detect_batch(
        frames_buffer,
        det_predict_fn,
        det_config,
        conf_threshold=DETECTION_CONF_THRESHOLD,
        box_aera_min=BOX_AERA_MIN,
    )
    frame_predictions_list = classify_batch_detections(
        frames_buffer,
        batch_results,
        clf_predict_fn,
        dataset_mean,
        dataset_std,
        config,
    )

    def process_single_canvas(i):
        _, heatmap_lr, binary_mask_lr = batch_results[i]
        return build_quadrant_canvas(
            frames_buffer[i],
            heatmap_lr,
            binary_mask_lr,
            frame_predictions_list[i],
            config,
        )

    with concurrent.futures.ThreadPoolExecutor() as executor:
        canvases = list(executor.map(process_single_canvas, range(len(frames_buffer))))

    return canvases

def reader_thread_func(cap, input_queue, frame_stride):
    frame_id = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            input_queue.put(None)  # Signal de fin
            break
            
        if frame_id % frame_stride == 0:
            input_queue.put(frame.copy())
            
        frame_id += 1

def writer_thread_func(video_writer, output_queue):
    while True:
        canvas = output_queue.get()
        if canvas is None:
            break
        video_writer.write(canvas)

if __name__ == "__main__":
    
    
    print("🏗️ Chargement des modèles...")

    config = get_dataset_config(DATASET_NAME)

    det_model, det_vars, det_config = load_detection_model(DETECTION_CHECKPOINT_PATH)
    clf_model, clf_vars, dataset_mean, dataset_std = load_jax_model(CHECKPOINT_PATH, config)

    print("✅ Modèles chargés.")
    
    print("⚡ Compilation JAX (JIT)...")
    det_predict_fn = build_det_predict_fn(det_model, det_vars)
    clf_predict_fn = build_clf_predict_fn(clf_model, clf_vars)
    warmup_jit_predictors(det_predict_fn, det_config, clf_predict_fn, config)
    print("✅ Compilation terminée.")

    # ======================================================
    # OUVERTURE VIDEO
    # ======================================================
    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        print("❌ Impossible d'ouvrir la vidéo.")
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if FRAME_STRIDE > 1:
        fps = fps / FRAME_STRIDE
    print(f"🎬 {total_frames} frames détectées")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    OUTPUT_VIDEO_PATH = os.path.join(OUTPUT_DIR, "reconstructed_video.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(
        OUTPUT_VIDEO_PATH, fourcc, fps, (CANVAS_WIDTH, CANVAS_HEIGHT)
    )
    if not video_writer.isOpened():
        print("❌ ERREUR : Impossible d'ouvrir VideoWriter.")
        cap.release()
        sys.exit(1)

    # Initialisation des Queues et Threads
    input_queue = queue.Queue(maxsize=BATCH_SIZE * 3)
    output_queue = queue.Queue(maxsize=BATCH_SIZE * 3)

    reader_thread = threading.Thread(target=reader_thread_func, args=(cap, input_queue, FRAME_STRIDE))
    writer_thread = threading.Thread(target=writer_thread_func, args=(video_writer, output_queue))
    
    reader_thread.start()
    writer_thread.start()

    saved_count = 0
    frames_buffer = []
    
    total_to_process = (total_frames + FRAME_STRIDE - 1) // FRAME_STRIDE

    with tqdm(total=total_to_process, desc="Inférence GPU") as pbar:
        while True:
            frame = input_queue.get()
            if frame is None:
                break

            frames_buffer.append(frame)

            if len(frames_buffer) == BATCH_SIZE:
                canvases = process_frames_batch(
                    frames_buffer,
                    det_predict_fn,
                    det_config,
                    clf_predict_fn,
                    dataset_mean,
                    dataset_std,
                    config,
                )
                for canvas in canvases:
                    output_queue.put(canvas)
                    
                saved_count += len(frames_buffer)
                pbar.update(len(frames_buffer))
                frames_buffer.clear()

        if len(frames_buffer) > 0:
            canvases = process_frames_batch(
                frames_buffer,
                det_predict_fn,
                det_config,
                clf_predict_fn,
                dataset_mean,
                dataset_std,
                config,
            )
            for canvas in canvases:
                output_queue.put(canvas)
                
            saved_count += len(frames_buffer)
            pbar.update(len(frames_buffer))
            frames_buffer.clear()

    # Signal de fin pour le writer
    output_queue.put(None)
    
    reader_thread.join()
    writer_thread.join()
    
    cap.release()
    video_writer.release()

    print("\n✅ Terminé !")
    print(f"📸 Frames traitées : {saved_count}")
    print(f"📂 Dossier de sortie : {OUTPUT_DIR}")
    if os.path.exists(OUTPUT_VIDEO_PATH):
        print(f"🎥 Vidéo : {OUTPUT_VIDEO_PATH}")
    else:
        print("❌ La vidéo n'a pas été créée.")