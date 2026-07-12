"""
Vérification isolée de Story 1.2 : inference_utils.py produit des résultats identiques
à baseline_before.json (Story 1.1), AVANT toute migration des fichiers consommateurs.
"""
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

import cv2
from dataset_configs import get_dataset_config
from inference_utils import (
    load_detection_model, load_jax_model, decode_segmentation_and_detect,
    non_max_suppression, predict_crop, build_predict_fn, build_clf_predict_fn,
    decode_segmentation_and_detect_batch, predict_crops_batch,
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
    det_model, det_vars, det_config = load_detection_model(DETECTION_CHECKPOINT)
    clf_model, clf_vars, mean, std = load_jax_model(CLASSIFICATION_CHECKPOINT, config)

    mismatches = 0

    print("== Vérification images statiques ==")
    for name, expected in baseline["static_images"].items():
        # Retrouver le chemin d'origine (mêmes 6 classes que capture_baseline.py)
        found = None
        for cls in ["a10", "b52", "f15", "f16", "f22", "f35"]:
            candidate = f"/home/aobled/Downloads/tmp_multi/{cls}/{name}"
            if os.path.exists(candidate):
                found = candidate
                break
        if found is None:
            print(f"  SKIP {name}: introuvable")
            continue
        img = cv2.imread(found)
        boxes = decode_segmentation_and_detect(img, det_model, det_vars, det_config,
                                                conf_threshold=DET_CONF_THRESHOLD, box_aera_min=BOX_AREA_MIN)
        boxes = non_max_suppression(boxes, iou_threshold=NMS_THRESHOLD)
        actual_boxes = round_boxes(boxes)
        ok = actual_boxes == expected["boxes"]
        print(f"  {name}: {'OK' if ok else 'MISMATCH'} ({len(actual_boxes)} boxes vs {len(expected['boxes'])})")
        if not ok:
            mismatches += 1
            print(f"    expected={expected['boxes']}")
            print(f"    actual  ={actual_boxes}")

    print("== Vérification frames vidéo ==")
    det_predict_fn = build_predict_fn(det_model, det_vars)
    clf_predict_fn = build_clf_predict_fn(clf_model, clf_vars)

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
            [frame], det_predict_fn, det_config,
            conf_threshold=0.8, box_aera_min=500,
        )
        boxes, _, _ = batch_results[0]
        actual_boxes = round_boxes(boxes)
        ok = actual_boxes == expected["boxes"]
        print(f"  {key}: {'OK' if ok else 'MISMATCH'} ({len(actual_boxes)} boxes vs {len(expected['boxes'])})")
        if not ok:
            mismatches += 1
            print(f"    expected={expected['boxes']}")
            print(f"    actual  ={actual_boxes}")

    print(f"\n{'✅ TOUT IDENTIQUE' if mismatches == 0 else f'❌ {mismatches} MISMATCH(ES)'}")
    sys.exit(1 if mismatches else 0)


if __name__ == "__main__":
    main()
