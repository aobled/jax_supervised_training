"""
Story 8.7, Task 1 : capture la baseline AVANT migration - execute l'ancien pipeline
(decode_segmentation_and_detect_batch + classify_batch_detections, chemin actuel de
bounding_boxes_with_classification_from_video_generation.py) sur un extrait fixe de
testvid.mp4, sauvegarde boxes/classes/scores par frame dans un format comparable
(baseline_video_8_7.json).

Extrait (pas la video complete, 599 frames) : NUM_FRAMES premieres frames, suffisant
pour la comparaison de cette story et reutilisable pour la Story 8.9.

Usage: python3 capture_baseline_video_8_7.py
"""
import json
import time

import cv2

from dataset_configs import get_dataset_config
from inference_utils import (
    load_jax_model,
    load_detection_model,
    predict_crops_batch,
    build_predict_fn,
    build_clf_predict_fn,
    decode_segmentation_and_detect_batch,
)

DATASET_NAME = "FIGHTERJET_CLASSIFICATION"
CHECKPOINT_PATH = "best_model.pkl"
DETECTION_CHECKPOINT_PATH = "best_model_detection.pkl"
VIDEO_PATH = "/home/aobled/Downloads/testvid.mp4"  # meme fichier que test_media/testvid.mp4
DETECTION_CONF_THRESHOLD = 0.8
BOX_AERA_MIN = 500
NUM_FRAMES = 64  # extrait fixe, pas la video complete (599 frames)
BATCH_SIZE = 32  # meme taille que bounding_boxes_with_classification_from_video_generation.py
BASELINE_PATH = "baseline_video_8_7.json"


def main():
    config = get_dataset_config(DATASET_NAME)
    det_model, det_vars, det_config = load_detection_model(DETECTION_CHECKPOINT_PATH)
    clf_model, clf_vars, dataset_mean, dataset_std = load_jax_model(CHECKPOINT_PATH, config)
    det_predict_fn = build_predict_fn(det_model, det_vars)
    clf_predict_fn = build_clf_predict_fn(clf_model, clf_vars)

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise FileNotFoundError(f"Impossible d'ouvrir la video: {VIDEO_PATH}")

    frames = []
    for _ in range(NUM_FRAMES):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    print(f"📼 {len(frames)} frames lues depuis {VIDEO_PATH}")

    # Warmup (memes lots factices que warmup_jit_predictors) - exclu de la mesure de
    # debit, meme discipline que capture_migrated_video_8_7.py pour une comparaison
    # equitable AD-6. Warmup des DEUX kernels JIT (detection ET classification) -
    # correction post-revue independante Story 8.7 : le warmup initial ne couvrait que
    # la detection, laissant le premier appel a predict_crops_batch absorber son cout de
    # compilation JIT DANS la region chronometree (asymetrie avec capture_migrated_video_8_7.py,
    # qui warme le graphe complet) - gonflait artificiellement le temps de l'ancien
    # pipeline (jamais l'inverse, donc ne remettait pas en cause la conclusion, mais
    # cassait la stricte comparaison "a-armes-egales" documentee).
    import numpy as np
    warmup_frames = [np.zeros_like(frames[0]) for _ in range(BATCH_SIZE)]
    decode_segmentation_and_detect_batch(
        warmup_frames, det_predict_fn, det_config,
        conf_threshold=DETECTION_CONF_THRESHOLD, box_aera_min=BOX_AERA_MIN,
    )
    warmup_crops = [np.zeros((64, 64, 3), dtype=np.uint8) for _ in range(4)]
    predict_crops_batch(warmup_crops, clf_predict_fn, dataset_mean, dataset_std, config)
    print("✅ Warmup JIT termine (detection + classification, exclu de la mesure de debit)")

    total_time = 0.0
    num_batches = 0
    batch_results = []
    for start in range(0, len(frames), BATCH_SIZE):
        batch_frames = frames[start:start + BATCH_SIZE]
        t0 = time.perf_counter()
        results = decode_segmentation_and_detect_batch(
            batch_frames, det_predict_fn, det_config,
            conf_threshold=DETECTION_CONF_THRESHOLD, box_aera_min=BOX_AERA_MIN,
        )
        batch_results.extend(results)
        total_time += time.perf_counter() - t0
        num_batches += 1

    all_crops = []
    per_frame_items = []
    for i, (detections, _, _) in enumerate(batch_results):
        frame_items = []
        for box in detections:
            x1, y1, x2, y2, det_score = box
            crop = frames[i][y1:y2, x1:x2]
            if crop.size == 0:
                continue
            frame_items.append((box, crop))
            all_crops.append(crop)
        per_frame_items.append(frame_items)

    t0 = time.perf_counter()
    all_preds = predict_crops_batch(all_crops, clf_predict_fn, dataset_mean, dataset_std, config)
    total_time += time.perf_counter() - t0

    fps_old = len(frames) / total_time if total_time > 0 else float("nan")

    pred_idx = 0
    baseline_per_frame = []
    total_detections = 0
    for frame_items in per_frame_items:
        frame_entries = []
        for box, crop in frame_items:
            predicted_class, confidence = all_preds[pred_idx]
            pred_idx += 1
            x1, y1, x2, y2, det_score = box
            frame_entries.append({
                "box": [float(x1), float(y1), float(x2), float(y2)],
                "det_score": float(det_score),
                "class": predicted_class,
                "confidence": float(confidence),
            })
        baseline_per_frame.append(frame_entries)
        total_detections += len(frame_entries)

    with open(BASELINE_PATH, "w") as f:
        json.dump({
            "video_path": VIDEO_PATH,
            "num_frames": len(frames),
            "detection_conf_threshold": DETECTION_CONF_THRESHOLD,
            "box_aera_min": BOX_AERA_MIN,
            "detections_per_frame": baseline_per_frame,
            "throughput_fps": fps_old,
            "total_inference_time_s": total_time,
            "num_batches": num_batches,
        }, f, indent=2)

    print(f"✅ Baseline sauvegardee : {BASELINE_PATH} ({total_detections} detections sur {len(frames)} frames)")
    print(f"⏱️  Debit ancien pipeline : {fps_old:.2f} fps ({total_time:.3f}s pour {len(frames)} frames, {num_batches} lots)")


if __name__ == "__main__":
    main()
