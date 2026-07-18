"""
Story 8.8, Task 1 : capture la baseline AVANT migration - execute l'ancien pipeline
(decode_segmentation_and_detect + non_max_suppression + predict_crop, chemin actuel de
tools/bounding_boxes_with_classification_from_images_generation.py) sur un petit jeu
d'images fixe, sauvegarde les resultats dans un format comparable.

N'exerce PAS l'effet de bord shutil.move (deplacement des images/JSON vers des
sous-dossiers par classe) du script production - deliberement isole dans un repertoire
scratch (pas /home/aobled/Downloads/tmp_multi, le vrai dossier de production) pour ne
jamais risquer de deplacer des donnees reelles de l'utilisateur pendant cette
verification. Task 1 ne demande que les JSON produits comme reference, pas la
reorganisation de dossiers.

Usage: python3 capture_baseline_images_8_8.py
"""
import json
import os

import cv2

from dataset_configs import get_dataset_config
from inference_utils import (
    load_jax_model,
    load_detection_model,
    predict_crop,
    non_max_suppression,
    decode_segmentation_and_detect,
)

DATASET_NAME = "FIGHTERJET_CLASSIFICATION"
CHECKPOINT_PATH = "best_model.pkl"
DETECTION_CHECKPOINT_PATH = "best_model_detection.pkl"
CONFIDENCE_THRESHOLD = 0.6
DETECTION_CONF_THRESHOLD = 0.8
BOX_AERA_MIN = 60
NMS_THRESHOLD = 0.4
DEFAULT_CLASSE = "unknown"

SCRATCH_DIR = os.environ.get(
    "IMAGES_8_8_SCRATCH_DIR",
    "/tmp/claude-1000/-home-aobled-Desktop-Development-jax-supervised-training/"
    "14ef819d-eaba-4615-b0f2-4b80471a5f7d/scratchpad/images_8_8_baseline_input",
)
BASELINE_PATH = "baseline_images_8_8.json"


def main():
    config = get_dataset_config(DATASET_NAME)
    det_model, det_vars, det_config = load_detection_model(DETECTION_CHECKPOINT_PATH)
    clf_model, clf_vars, dataset_mean, dataset_std = load_jax_model(CHECKPOINT_PATH, config)

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

        detections = decode_segmentation_and_detect(
            img, det_model, det_vars, det_config,
            conf_threshold=DETECTION_CONF_THRESHOLD, box_aera_min=BOX_AERA_MIN,
        )
        detections = non_max_suppression(detections, iou_threshold=NMS_THRESHOLD)

        entries = []
        for (x1, y1, x2, y2, score) in detections:
            crop = img[y1:y2, x1:x2]
            if crop.size == 0 or crop.shape[0] == 0 or crop.shape[1] == 0:
                continue
            predicted_class, confidence = predict_crop(crop, clf_model, clf_vars, dataset_mean, dataset_std, config)
            if confidence < CONFIDENCE_THRESHOLD:
                predicted_class = DEFAULT_CLASSE
            entries.append({
                "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                "category_name": predicted_class,
                "detection_score": float(score),
                "classification_score": float(confidence),
            })

        results_per_image[file_name] = {"width": w, "height": h, "detections": entries}
        total_detections += len(entries)

    with open(BASELINE_PATH, "w") as f:
        json.dump({
            "scratch_dir": SCRATCH_DIR,
            "num_images": len(image_files),
            "results_per_image": results_per_image,
        }, f, indent=2)

    print(f"✅ Baseline sauvegardee : {BASELINE_PATH} ({total_detections} detections sur {len(image_files)} images)")


if __name__ == "__main__":
    main()
