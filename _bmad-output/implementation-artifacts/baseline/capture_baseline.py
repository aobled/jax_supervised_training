"""
Story 1.1 (Epic 1): capture de la baseline de non-régression, AVANT refactor.
Exécute les fonctions d'inférence ACTUELLES (pré-migration) sur un jeu d'images fixe,
et sauvegarde boxes/classes/scores en JSON pour comparaison en Story 1.10.

Usage: python _bmad-output/implementation-artifacts/baseline/capture_baseline.py [--after]
  --after : capture "après refactor", importe depuis inference_utils.py / fichiers migrés,
            et compare directement contre baseline_before.json.
"""
import os
import sys
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

import cv2
import numpy as np

def _first_image(class_dir):
    files = sorted(f for f in os.listdir(class_dir) if f.lower().endswith((".jpg", ".jpeg", ".png")))
    return os.path.join(class_dir, files[0])


# Jeu d'images fixe (6 images statiques, une par classe distincte, chemins absolus stables)
STATIC_IMAGES = [
    _first_image("/home/aobled/Downloads/tmp_multi/a10"),
    _first_image("/home/aobled/Downloads/tmp_multi/b52"),
    _first_image("/home/aobled/Downloads/tmp_multi/f15"),
    _first_image("/home/aobled/Downloads/tmp_multi/f16"),
    _first_image("/home/aobled/Downloads/tmp_multi/f22"),
    _first_image("/home/aobled/Downloads/tmp_multi/f35"),
]

VIDEO_PATH = "/home/aobled/Downloads/testvid.mp4"
VIDEO_FRAME_INDICES = [0, 30, 60, 90, 120]  # 5 frames fixes

DETECTION_CHECKPOINT = "best_model_detection.pkl"
CLASSIFICATION_CHECKPOINT = "best_model.pkl"
DATASET_NAME = "FIGHTERJET_CLASSIFICATION"
DET_CONF_THRESHOLD = 0.3
BOX_AREA_MIN = 225
NMS_THRESHOLD = 0.4


def round_boxes(boxes):
    out = []
    for b in boxes:
        b = list(b)
        out.append([round(float(v), 3) for v in b])
    return out


def capture_before():
    from dataset_configs import get_dataset_config
    from tools.bounding_boxes_with_classification_from_images_generation import (
        load_detection_model, load_jax_model, decode_segmentation_and_detect,
        non_max_suppression, predict_crop,
    )
    from bounding_boxes_with_classification_from_video_generation import (
        load_detection_model as load_detection_model_vid,
        load_jax_model as load_jax_model_vid,
        build_det_predict_fn, build_clf_predict_fn,
        decode_segmentation_and_detect_batch, predict_crops_batch,
    )

    config = get_dataset_config(DATASET_NAME)
    det_model, det_vars, det_config = load_detection_model(DETECTION_CHECKPOINT)
    clf_model, clf_vars, mean, std = load_jax_model(CLASSIFICATION_CHECKPOINT, config)

    results = {"static_images": {}, "video_frames": {}}

    print("== Capture statique (decode_segmentation_and_detect + predict_crop) ==")
    for path in STATIC_IMAGES:
        img = cv2.imread(path)
        if img is None:
            results["static_images"][os.path.basename(path)] = {"error": "unreadable"}
            continue
        boxes = decode_segmentation_and_detect(img, det_model, det_vars, det_config,
                                                conf_threshold=DET_CONF_THRESHOLD, box_aera_min=BOX_AREA_MIN)
        boxes = non_max_suppression(boxes, iou_threshold=NMS_THRESHOLD)
        preds = []
        for (x1, y1, x2, y2, score) in boxes:
            crop = img[int(y1):int(y2), int(x1):int(x2)]
            if crop.size == 0:
                continue
            cls, conf = predict_crop(crop, clf_model, clf_vars, mean, std, config)
            preds.append({"class": cls, "confidence": round(float(conf), 4)})
        results["static_images"][os.path.basename(path)] = {
            "boxes": round_boxes(boxes),
            "predictions": preds,
        }
        print(f"  {os.path.basename(path)}: {len(boxes)} boxes")

    print("== Capture vidéo (decode_segmentation_and_detect_batch + predict_crops_batch) ==")
    det_model_v, det_vars_v, det_config_v = load_detection_model_vid(DETECTION_CHECKPOINT)
    clf_model_v, clf_vars_v, mean_v, std_v = load_jax_model_vid(CLASSIFICATION_CHECKPOINT, config)
    det_predict_fn = build_det_predict_fn(det_model_v, det_vars_v)
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
        batch_results = decode_segmentation_and_detect_batch(
            [frame], det_predict_fn, det_config_v,
            conf_threshold=0.8, box_aera_min=500,
        )
        boxes, _, _ = batch_results[0]
        crops = []
        for (x1, y1, x2, y2, score) in boxes:
            crop = frame[int(y1):int(y2), int(x1):int(x2)]
            crops.append(crop)
        preds = predict_crops_batch(crops, clf_predict_fn, mean_v, std_v, config) if crops else []
        results["video_frames"][f"frame_{frame_idx}"] = {
            "boxes": round_boxes(boxes),
            "predictions": [{"class": c, "confidence": round(float(cf), 4)} for c, cf in preds],
        }
        print(f"  frame_{frame_idx}: {len(boxes)} boxes")

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "baseline_before.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nBaseline sauvegardée: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.parse_args()
    capture_before()
