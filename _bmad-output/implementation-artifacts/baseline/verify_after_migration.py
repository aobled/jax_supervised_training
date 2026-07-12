"""
Story 1.10 : diff de non-régression après migration complète (Stories 1.3-1.9).
Importe désormais depuis les FICHIERS CONSOMMATEURS migrés eux-mêmes (pas directement
depuis inference_utils.py) pour prouver que la chaîne complète (fichier -> inference_utils.py)
produit des résultats identiques à baseline_before.json.
"""
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

import cv2
from dataset_configs import get_dataset_config

# Chemin image unique : importé depuis le fichier migré tools/bounding_boxes_with_classification_from_images_generation.py
from tools.bounding_boxes_with_classification_from_images_generation import (
    load_detection_model as load_detection_model_img,
    load_jax_model as load_jax_model_img,
    decode_segmentation_and_detect,
    non_max_suppression,
    predict_crop,
)

# Chemin batch/vidéo : importé depuis le fichier migré bounding_boxes_with_classification_from_video_generation.py
from bounding_boxes_with_classification_from_video_generation import (
    load_detection_model as load_detection_model_vid,
    load_jax_model as load_jax_model_vid,
    build_predict_fn,
    build_clf_predict_fn,
    decode_segmentation_and_detect_batch,
    predict_crops_batch,
)

DETECTION_CHECKPOINT = "best_model_detection.pkl"
CLASSIFICATION_CHECKPOINT = "best_model.pkl"
DATASET_NAME = "FIGHTERJET_CLASSIFICATION"
DET_CONF_THRESHOLD = 0.3
BOX_AREA_MIN = 225
NMS_THRESHOLD = 0.4
VIDEO_PATH = "/home/aobled/Downloads/testvid.mp4"
VIDEO_FRAME_INDICES = [0, 30, 60, 90, 120]


def round_boxes(boxes):
    return [[round(float(v), 3) for v in b] for b in boxes]


def main():
    baseline_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "baseline_before.json")
    with open(baseline_path) as f:
        baseline = json.load(f)

    config = get_dataset_config(DATASET_NAME)
    mismatches = 0

    print("== Vérification images statiques (via tools/..._images_generation.py migré) ==")
    det_model, det_vars, det_config = load_detection_model_img(DETECTION_CHECKPOINT)
    clf_model, clf_vars, mean, std = load_jax_model_img(CLASSIFICATION_CHECKPOINT, config)

    for name, expected in baseline["static_images"].items():
        found = None
        for cls in ["a10", "b52", "f15", "f16", "f22", "f35"]:
            candidate = f"/home/aobled/Downloads/tmp_multi/{cls}/{name}"
            if os.path.exists(candidate):
                found = candidate
                break
        img = cv2.imread(found)
        boxes = decode_segmentation_and_detect(img, det_model, det_vars, det_config,
                                                conf_threshold=DET_CONF_THRESHOLD, box_aera_min=BOX_AREA_MIN)
        boxes = non_max_suppression(boxes, iou_threshold=NMS_THRESHOLD)
        actual_boxes = round_boxes(boxes)
        ok = actual_boxes == expected["boxes"]
        print(f"  {name}: {'OK' if ok else 'MISMATCH'}")
        if not ok:
            mismatches += 1

    print("== Vérification frames vidéo (via bounding_boxes_..._video_generation.py migré) ==")
    det_model_v, det_vars_v, det_config_v = load_detection_model_vid(DETECTION_CHECKPOINT)
    clf_model_v, clf_vars_v, mean_v, std_v = load_jax_model_vid(CLASSIFICATION_CHECKPOINT, config)
    det_predict_fn = build_predict_fn(det_model_v, det_vars_v)
    clf_predict_fn = build_clf_predict_fn(clf_model_v, clf_vars_v)

    cap = cv2.VideoCapture(VIDEO_PATH)
    frames = []
    idx = 0
    wanted = set(VIDEO_FRAME_INDICES)
    max_wanted = max(VIDEO_FRAME_INDICES)
    while idx <= max_wanted:
        ret, frame = cap.read()
        if not ret:
            break
        if idx in wanted:
            frames.append((idx, frame))
        idx += 1
    cap.release()

    for frame_idx, frame in frames:
        key = f"frame_{frame_idx}"
        expected = baseline["video_frames"][key]
        batch_results = decode_segmentation_and_detect_batch(
            [frame], det_predict_fn, det_config_v,
            conf_threshold=0.8, box_aera_min=500,
        )
        boxes, _, _ = batch_results[0]
        actual_boxes = round_boxes(boxes)
        ok = actual_boxes == expected["boxes"]
        print(f"  {key}: {'OK' if ok else 'MISMATCH'}")
        if not ok:
            mismatches += 1

    print(f"\n{'✅ EPIC 1 : AUCUNE REGRESSION (0 mismatch)' if mismatches == 0 else f'❌ {mismatches} MISMATCH(ES)'}")
    sys.exit(1 if mismatches else 0)


if __name__ == "__main__":
    main()
