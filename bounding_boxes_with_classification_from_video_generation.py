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
# à utiliser le model_library.py de jax_supervised_training (et non celui de JAX_Classification si exécuté depuis l'autre dossier)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Remplace YOLO import par JAX/Flax helpers si nécessaire
# from ultralytics import YOLO  <-- REMOVED
import cv2
import numpy as np
import jax
import jax.numpy as jnp
from tqdm import tqdm

from dataset_configs import get_dataset_config
from inference_utils import (
    DETECTION_IMAGE_SIZE,
    load_jax_model,
    load_detection_model,
    predict_crops_batch,
    build_predict_fn,
    build_clf_predict_fn,
    get_iou,
    non_max_suppression,
    decode_segmentation_and_detect_batch,
)

# =================================================================================================
# Input CONFIGURATION
# =================================================================================================
# 1. Configuration du dataset et du modèle de classification
DATASET_NAME = "FIGHTERJET_CLASSIFICATION"     # Nom de la config dans dataset_configs.py
CHECKPOINT_PATH = "best_model.pkl"      # Chemin vers le modèle de CLASSIFICATION

# 2. Configuration du modèle de détection
DETECTION_CHECKPOINT_PATH = "best_model_detection.pkl" # Chemin vers le modèle de DÉTECTION
# Taille d'entrée du modèle de détection : voir inference_utils.DETECTION_IMAGE_SIZE (import ci-dessus)

# 3. Configuration de la zone de détection
BOX_AERA_MIN = 500

CANVAS_WIDTH = 1920
CANVAS_HEIGHT = 1080

# ==========================================================
# Detection CONFIGURATION
# ==========================================================
OUTPUT_DIR = "/home/aobled/Downloads/video_frames_annotated"
FRAME_STRIDE = 1  # 1 = toutes les frames
DETECTION_CONF_THRESHOLD = 0.8          # Seuil pour considérer une détection valide (objectness + class) target 0.6
BATCH_SIZE = 32                         # Batch détection (réduire si OOM GPU, ex. GTX 1660 Ti 6 Go)
CLF_BATCH_SIZE = 32                     # Batch classification fixe (évite recompilation cuDNN)

VIDEO_PATH = "/home/aobled/Downloads/testvid.mp4"
TARGET_CLASS_LIST = ["f15", "f22", "b1b", "b2", "b52", "a10", "f16"]
#VIDEO_PATH = "/home/aobled/Downloads/F-16 Falcons Mid-Air Refueling.mp4"
#TARGET_CLASS_LIST = ["f16"]

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
    det_predict_fn = build_predict_fn(det_model, det_vars)
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