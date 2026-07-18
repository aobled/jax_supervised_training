"""
Diagnostic (discussion post-Epic 8, 2026-07-18) : compare plusieurs valeurs de
detection_score_threshold (0.3 actuel vs 0.2 propose) sur les 3 images annotees
(test_media/testvid0{1,2,3}.png), SANS modifier dataset_configs.py ni reconstruire le
pipeline - _top_k_boxes(k=20) calcule deja tous les candidats en interne, valid_mask
n'est qu'un filtre post-hoc sur detection_scores. On reproduit ce filtre nous-memes a
plusieurs seuils sur les memes scores bruts.

Mesure pour chaque seuil :
- Rappel : combien de boites de verite terrain sont retrouvees (gain de rappel attendu
  en baissant le seuil)
- Faux positifs : combien de detections valides a ce seuil NE correspondent a AUCUNE
  boite de verite terrain (risque de bruit attendu en baissant le seuil)

Usage: python3 diagnose_threshold_sensitivity.py
"""
import glob
import json
import os

import cv2
import numpy as np
import jax.numpy as jnp

from dataset_configs import get_dataset_config
from inference_utils import build_single_pass_predict_fn, build_predict_fn, _resize_for_detector, _extract_peaks, _top_k_boxes, _rescale_boxes, load_detection_model

IMAGE_NAMES = ["testvid01.png", "testvid02.png", "testvid03.png"]
TEST_MEDIA_DIR = "test_media"
DETECTOR_CHECKPOINT_PATH = "best_model_jax_detector.pkl"
CLASSIFIER_CHECKPOINT_PATH = "best_model.pkl"
CANONICAL_WIDTH, CANONICAL_HEIGHT = 1920, 1080
IOU_MATCH_THRESHOLD = 0.3
THRESHOLDS_TO_TEST = [0.3, 0.2, 0.15, 0.1]


def _xywh_to_xyxy(bbox):
    x, y, w, h = bbox
    return (x, y, x + w, y + h)


def _iou(box_a, box_b):
    xa1, ya1, xa2, ya2 = box_a
    xb1, yb1, xb2, yb2 = box_b
    x1, y1 = max(xa1, xb1), max(ya1, yb1)
    x2, y2 = min(xa2, xb2), min(ya2, yb2)
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, xa2 - xa1) * max(0.0, ya2 - ya1)
    area_b = max(0.0, xb2 - xb1) * max(0.0, yb2 - yb1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _load_ground_truth(image_name):
    base = os.path.splitext(image_name)[0]
    gt_boxes = []
    for path in sorted(glob.glob(os.path.join(TEST_MEDIA_DIR, f"{base}_*.json"))):
        with open(path) as f:
            data = json.load(f)
        ann = data["annotation"]
        gt_boxes.append({"box": _xywh_to_xyxy(ann["bbox"]), "class": ann["category_name"]})
    return gt_boxes


def main():
    config_det = get_dataset_config("JAX_DETECTOR")
    config_clf = get_dataset_config("FIGHTERJET_CLASSIFICATION")
    class_names = config_clf["class_names"]

    # Reconstruit manuellement le chemin detecteur seul (jusqu'a detection_scores brut,
    # AVANT le filtre valid_mask) - reutilise les memes briques que
    # build_single_pass_predict_fn (Story 8.6), sans dupliquer la classification (pas
    # necessaire ici, on ne regarde que le rappel/faux positifs de la DETECTION).
    detector_model, detector_vars, config_model = load_detection_model(DETECTOR_CHECKPOINT_PATH)
    detector_predict_fn = build_predict_fn(detector_model, detector_vars)
    detector_image_size = config_model["image_size"]

    all_gt = {name: _load_ground_truth(name) for name in IMAGE_NAMES}
    all_scores_boxes = {}  # image_name -> (boxes(20,4) repere original, scores(20,))

    for image_name in IMAGE_NAMES:
        img = cv2.imread(os.path.join(TEST_MEDIA_DIR, image_name))
        h, w = img.shape[:2]
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if (h, w) != (CANONICAL_HEIGHT, CANONICAL_WIDTH):
            img_gray = cv2.resize(img_gray, (CANONICAL_WIDTH, CANONICAL_HEIGHT))
        canonical_image = jnp.asarray(img_gray.astype(np.float32)[..., None])

        resized = _resize_for_detector(canonical_image, detector_image_size, "lanczos3")
        resized_norm = resized / 255.0
        heatmap_size = detector_predict_fn(resized_norm[None, ...])
        heatmap = heatmap_size["heatmap"][0]
        size_map = heatmap_size["size"][0]
        filtered_heatmap = _extract_peaks(heatmap)
        boxes_det, scores = _top_k_boxes(filtered_heatmap, size_map, k=20)
        boxes_orig = _rescale_boxes(boxes_det, detector_image_size, original_size=(CANONICAL_WIDTH, CANONICAL_HEIGHT))

        all_scores_boxes[image_name] = (np.asarray(boxes_orig), np.asarray(scores))

    print(f"{'seuil':>6}  {'rappel':>16}  {'faux positifs':>14}  {'total valides':>14}")
    print("-" * 60)
    for threshold in THRESHOLDS_TO_TEST:
        total_gt = 0
        total_found = 0
        total_valid = 0
        total_fp = 0
        details_fp = []

        for image_name in IMAGE_NAMES:
            boxes_orig, scores = all_scores_boxes[image_name]
            valid_idx = np.where(scores > threshold)[0]
            valid_boxes = boxes_orig[valid_idx]
            valid_scores = scores[valid_idx]
            total_valid += len(valid_idx)

            gt_boxes = all_gt[image_name]
            total_gt += len(gt_boxes)

            matched_det_idx = set()
            for gt in gt_boxes:
                best_iou, best_j = 0.0, None
                for j, box in enumerate(valid_boxes):
                    iou = _iou(gt["box"], tuple(box))
                    if iou > best_iou:
                        best_iou, best_j = iou, j
                if best_iou >= IOU_MATCH_THRESHOLD:
                    total_found += 1
                    matched_det_idx.add(best_j)

            for j, box in enumerate(valid_boxes):
                if j not in matched_det_idx:
                    total_fp += 1
                    details_fp.append((image_name, tuple(round(v, 1) for v in box), float(valid_scores[j])))

        print(f"{threshold:>6.2f}  {total_found:>6}/{total_gt:<9}  {total_fp:>14}  {total_valid:>14}")

    print(f"\n(rappel = boites de verite terrain retrouvees / {sum(len(v) for v in all_gt.values())} au total)")
    print("(faux positif = detection valide a ce seuil sans correspondance en verite terrain, IoU<0.3)")

    # Detail des faux positifs au seuil le plus bas teste, pour inspection
    threshold = THRESHOLDS_TO_TEST[-1]
    print(f"\n=== Detail des faux positifs au seuil={threshold} (le plus permissif teste) ===")
    for image_name in IMAGE_NAMES:
        boxes_orig, scores = all_scores_boxes[image_name]
        valid_idx = np.where(scores > threshold)[0]
        gt_boxes = all_gt[image_name]
        for j in valid_idx:
            box = tuple(boxes_orig[j])
            best_iou = max((_iou(gt["box"], box) for gt in gt_boxes), default=0.0)
            if best_iou < IOU_MATCH_THRESHOLD:
                print(f"  {image_name}: box={tuple(round(v,1) for v in box)} score={scores[j]:.3f} (meilleur IoU verite terrain={best_iou:.3f})")


if __name__ == "__main__":
    main()
