"""
Story 8.8, Task 9 : execute le NOUVEAU pipeline (build_single_pass_predict_fn, Story 8.6)
sur le meme jeu d'images que capture_baseline_images_8_8.py, sauvegarde les resultats
dans un format comparable pour le diff.

Usage: python3 capture_migrated_images_8_8.py
"""
import json
import os

import cv2
import jax.numpy as jnp
import numpy as np

from dataset_configs import get_dataset_config
from inference_utils import build_single_pass_predict_fn, _rescale_boxes

DATASET_NAME = "FIGHTERJET_CLASSIFICATION"
CHECKPOINT_PATH = "best_model.pkl"
DETECTOR_CHECKPOINT_PATH = "best_model_jax_detector.pkl"
CONFIDENCE_THRESHOLD = 0.6
DEFAULT_CLASSE = "unknown"
CANONICAL_WIDTH, CANONICAL_HEIGHT = 1920, 1080

SCRATCH_DIR = os.environ.get(
    "IMAGES_8_8_SCRATCH_DIR",
    "/tmp/claude-1000/-home-aobled-Desktop-Development-jax-supervised-training/"
    "14ef819d-eaba-4615-b0f2-4b80471a5f7d/scratchpad/images_8_8_migrated_input",
)
MIGRATED_PATH = "migrated_images_8_8.json"


def main():
    config = get_dataset_config(DATASET_NAME)
    class_names = config["class_names"]

    predict_fn = build_single_pass_predict_fn(
        detector_checkpoint_path=DETECTOR_CHECKPOINT_PATH,
        classifier_checkpoint_path=CHECKPOINT_PATH,
    )

    image_files = sorted(
        os.path.join(SCRATCH_DIR, f) for f in os.listdir(SCRATCH_DIR)
        if f.lower().endswith((".jpg", ".png", ".jpeg", ".bmp"))
    )
    print(f"📂 {len(image_files)} images dans {SCRATCH_DIR}")

    results_per_image = {}
    total_detections = 0

    for img_path in image_files:
        file_name = os.path.basename(img_path)
        img = cv2.imread(img_path)
        h, w = img.shape[:2]

        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_canonical = cv2.resize(img_gray, (CANONICAL_WIDTH, CANONICAL_HEIGHT))
        img_canonical = img_canonical.astype(np.float32)[..., None]

        result = predict_fn(jnp.asarray(img_canonical))

        boxes_native = np.asarray(_rescale_boxes(
            result["boxes"], detector_size=(CANONICAL_WIDTH, CANONICAL_HEIGHT), original_size=(w, h)
        ))
        valid_mask = np.asarray(result["valid_mask"])
        classes = np.asarray(result["classes"])
        class_scores = np.asarray(result["class_scores"])
        detection_scores = np.asarray(result["detection_scores"])

        entries = []
        for i in np.where(valid_mask)[0]:
            x1, y1, x2, y2 = boxes_native[i]
            predicted_class = class_names[int(classes[i])]
            confidence = float(class_scores[i])
            if confidence < CONFIDENCE_THRESHOLD:
                predicted_class = DEFAULT_CLASSE
            entries.append({
                "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                "category_name": predicted_class,
                "detection_score": float(detection_scores[i]),
                "classification_score": confidence,
            })

        results_per_image[file_name] = {"width": w, "height": h, "detections": entries}
        total_detections += len(entries)

    with open(MIGRATED_PATH, "w") as f:
        json.dump({
            "scratch_dir": SCRATCH_DIR,
            "num_images": len(image_files),
            "results_per_image": results_per_image,
        }, f, indent=2)

    print(f"✅ Resultats migres sauvegardes : {MIGRATED_PATH} ({total_detections} detections sur {len(image_files)} images)")


if __name__ == "__main__":
    main()
