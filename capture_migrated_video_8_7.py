"""
Story 8.7, Tasks 6-7 : execute le NOUVEAU pipeline (build_single_pass_predict_fn +
jax.vmap, Story 8.6) sur le meme extrait de testvid.mp4 que capture_baseline_video_8_7.py
(memes 64 premieres frames), sauvegarde boxes/classes/scores par frame dans un format
comparable a la baseline, et mesure le debit (frames/seconde) pour comparaison AD-6.

Usage: python3 capture_migrated_video_8_7.py
"""
import json
import time

import cv2
import jax
import jax.numpy as jnp
import numpy as np

from dataset_configs import get_dataset_config
from inference_utils import build_single_pass_predict_fn

DATASET_NAME = "FIGHTERJET_CLASSIFICATION"
CHECKPOINT_PATH = "best_model.pkl"
DETECTOR_CHECKPOINT_PATH = "best_model_jax_detector.pkl"
VIDEO_PATH = "/home/aobled/Downloads/testvid.mp4"
NUM_FRAMES = 64  # meme extrait que capture_baseline_video_8_7.py
BATCH_SIZE = 32  # meme taille que bounding_boxes_with_classification_from_video_generation.py
CANVAS_WIDTH, CANVAS_HEIGHT = 1920, 1080
MIGRATED_PATH = "migrated_video_8_7.json"


def _to_canonical_gray(frame):
    if frame.shape[:2] != (CANVAS_HEIGHT, CANVAS_WIDTH):
        frame = cv2.resize(frame, (CANVAS_WIDTH, CANVAS_HEIGHT))
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)[..., None]


def main():
    config = get_dataset_config(DATASET_NAME)
    class_names = config["class_names"]

    predict_fn = build_single_pass_predict_fn(
        detector_checkpoint_path=DETECTOR_CHECKPOINT_PATH,
        classifier_checkpoint_path=CHECKPOINT_PATH,
    )
    batched_predict_fn = jax.vmap(predict_fn)

    # Warmup (meme discipline que warmup_jit_predictors, Task 2bis) - exclu de la
    # mesure de debit, sinon le cout de compilation JIT fausserait la comparaison AD-6.
    dummy_batch = jnp.zeros((BATCH_SIZE, CANVAS_HEIGHT, CANVAS_WIDTH, 1), dtype=jnp.float32)
    warmup_result = batched_predict_fn(dummy_batch)
    jax.tree_util.tree_map(lambda x: x.block_until_ready(), warmup_result)
    print("✅ Warmup JIT termine (exclu de la mesure de debit)")

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

    migrated_per_frame = []
    total_detections = 0
    total_time = 0.0
    num_batches = 0

    for start in range(0, len(frames), BATCH_SIZE):
        batch_frames = frames[start:start + BATCH_SIZE]
        gray_batch = np.stack([_to_canonical_gray(f) for f in batch_frames], axis=0)
        batched_images = jnp.asarray(gray_batch)

        t0 = time.perf_counter()
        results = batched_predict_fn(batched_images)
        jax.tree_util.tree_map(lambda x: x.block_until_ready(), results)
        elapsed = time.perf_counter() - t0
        total_time += elapsed
        num_batches += 1

        valid_mask = np.asarray(results["valid_mask"])
        boxes = np.asarray(results["boxes"])
        classes = np.asarray(results["classes"])
        class_scores = np.asarray(results["class_scores"])
        detection_scores = np.asarray(results["detection_scores"])

        for i in range(len(batch_frames)):
            frame_entries = []
            for slot in range(20):
                if not valid_mask[i, slot]:
                    continue
                x1, y1, x2, y2 = boxes[i, slot]
                frame_entries.append({
                    "box": [float(x1), float(y1), float(x2), float(y2)],
                    "det_score": float(detection_scores[i, slot]),
                    "class": class_names[int(classes[i, slot])],
                    "confidence": float(class_scores[i, slot]),
                })
            migrated_per_frame.append(frame_entries)
            total_detections += len(frame_entries)

    fps_new = len(frames) / total_time if total_time > 0 else float("nan")

    with open(MIGRATED_PATH, "w") as f:
        json.dump({
            "video_path": VIDEO_PATH,
            "num_frames": len(frames),
            "batch_size": BATCH_SIZE,
            "detections_per_frame": migrated_per_frame,
            "throughput_fps": fps_new,
            "total_inference_time_s": total_time,
            "num_batches": num_batches,
        }, f, indent=2)

    print(f"✅ Resultats migres sauvegardes : {MIGRATED_PATH} ({total_detections} detections sur {len(frames)} frames)")
    print(f"⏱️  Debit nouveau pipeline : {fps_new:.2f} fps ({total_time:.3f}s pour {len(frames)} frames, {num_batches} lots)")


if __name__ == "__main__":
    main()
