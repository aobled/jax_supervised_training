"""
Diagnostic precis (discussion post-Epic 8, 2026-07-18) : compare l'ancien et le nouveau
pipeline contre la VERITE TERRAIN (test_media/testvid0{1,2,3}.png + leurs annotations
_N.json, 21 boites reelles avec bbox+category_name, Story 8.1) plutot qu'en les
comparant seulement l'un a l'autre - permet de mesurer laquelle des deux boites est
REELLEMENT la plus precise (IoU vs verite terrain), et si l'exactitude de la
classification (pas seulement sa confiance) est affectee.

Usage: python3 diagnose_crop_precision_vs_ground_truth.py
"""
import glob
import json
import os

import cv2
import numpy as np
import jax.numpy as jnp

from dataset_configs import get_dataset_config
from inference_utils import (
    load_detection_model, load_jax_model,
    decode_segmentation_and_detect, predict_crop,
    build_single_pass_predict_fn,
)

IMAGE_NAMES = ["testvid01.png", "testvid02.png", "testvid03.png"]
TEST_MEDIA_DIR = "test_media"
DETECTION_CHECKPOINT_PATH = "best_model_detection.pkl"
CLASSIFIER_CHECKPOINT_PATH = "best_model.pkl"
DETECTOR_CHECKPOINT_PATH = "best_model_jax_detector.pkl"
DETECTION_CONF_THRESHOLD = 0.8
BOX_AERA_MIN = 60
CANONICAL_WIDTH, CANONICAL_HEIGHT = 1920, 1080
IOU_MATCH_THRESHOLD = 0.3


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
    pattern = os.path.join(TEST_MEDIA_DIR, f"{base}_*.json")
    gt_boxes = []
    for path in sorted(glob.glob(pattern)):
        with open(path) as f:
            data = json.load(f)
        ann = data["annotation"]
        gt_boxes.append({"box": _xywh_to_xyxy(ann["bbox"]), "class": ann["category_name"]})
    return gt_boxes


def _best_match(gt_box, detections):
    best_iou, best_det = 0.0, None
    for det in detections:
        iou = _iou(gt_box, det["box"])
        if iou > best_iou:
            best_iou, best_det = iou, det
    return (best_iou, best_det) if best_iou >= IOU_MATCH_THRESHOLD else (0.0, None)


def main():
    config_clf = get_dataset_config("FIGHTERJET_CLASSIFICATION")
    class_names = config_clf["class_names"]

    det_model, det_vars, det_config = load_detection_model(DETECTION_CHECKPOINT_PATH)
    clf_model, clf_vars, dataset_mean, dataset_std = load_jax_model(CLASSIFIER_CHECKPOINT_PATH, config_clf)
    predict_fn = build_single_pass_predict_fn(
        detector_checkpoint_path=DETECTOR_CHECKPOINT_PATH,
        classifier_checkpoint_path=CLASSIFIER_CHECKPOINT_PATH,
    )

    rows = []

    for image_name in IMAGE_NAMES:
        image_path = os.path.join(TEST_MEDIA_DIR, image_name)
        img = cv2.imread(image_path)
        h, w = img.shape[:2]
        gt_boxes = _load_ground_truth(image_name)
        print(f"\n{'='*70}\n{image_name} : {len(gt_boxes)} boites de verite terrain\n{'='*70}")

        # --- Ancien pipeline ---
        old_detections = decode_segmentation_and_detect(
            img, det_model, det_vars, det_config,
            conf_threshold=DETECTION_CONF_THRESHOLD, box_aera_min=BOX_AERA_MIN,
        )
        old_dets = []
        for (x1, y1, x2, y2, det_score) in old_detections:
            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            predicted_class, confidence = predict_crop(crop, clf_model, clf_vars, dataset_mean, dataset_std, config_clf)
            old_dets.append({"box": (float(x1), float(y1), float(x2), float(y2)),
                              "class": predicted_class, "confidence": confidence})

        # --- Nouveau pipeline ---
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if (h, w) != (CANONICAL_HEIGHT, CANONICAL_WIDTH):
            img_gray = cv2.resize(img_gray, (CANONICAL_WIDTH, CANONICAL_HEIGHT))
        canonical_image = jnp.asarray(img_gray.astype(np.float32)[..., None])
        result = predict_fn(canonical_image)
        valid_mask = np.asarray(result["valid_mask"])
        boxes = np.asarray(result["boxes"])
        classes = np.asarray(result["classes"])
        class_scores = np.asarray(result["class_scores"])
        new_dets = []
        for i in np.where(valid_mask)[0]:
            x1, y1, x2, y2 = boxes[i]
            new_dets.append({"box": (float(x1), float(y1), float(x2), float(y2)),
                              "class": class_names[int(classes[i])], "confidence": float(class_scores[i])})

        # --- Appariement de CHAQUE boite a la verite terrain ---
        for gt in gt_boxes:
            iou_old, det_old = _best_match(gt["box"], old_dets)
            iou_new, det_new = _best_match(gt["box"], new_dets)
            rows.append({
                "image": image_name, "gt_class": gt["class"],
                "iou_old": iou_old, "class_old": det_old["class"] if det_old else None,
                "conf_old": det_old["confidence"] if det_old else None,
                "iou_new": iou_new, "class_new": det_new["class"] if det_new else None,
                "conf_new": det_new["confidence"] if det_new else None,
            })

    # === Rapport detaille ===
    print(f"\n\n{'='*100}")
    print("DETAIL PAR BOITE DE VERITE TERRAIN")
    print(f"{'='*100}")
    header = f"{'image':12} {'gt_class':12} {'IoU_old':8} {'class_old':11} {'conf_old':9} {'IoU_new':8} {'class_new':11} {'conf_new':9}"
    print(header)
    print("-" * len(header))
    for r in rows:
        c_old = f"{r['conf_old']*100:.1f}%" if r["conf_old"] is not None else "MANQUE"
        c_new = f"{r['conf_new']*100:.1f}%" if r["conf_new"] is not None else "MANQUE"
        cl_old = r["class_old"] or "-"
        cl_new = r["class_new"] or "-"
        marker = ""
        if r["class_old"] and r["class_new"]:
            correct_old = r["class_old"] == r["gt_class"]
            correct_new = r["class_new"] == r["gt_class"]
            if correct_old != correct_new:
                marker = "  <-- CLASSE CORRECTE CHANGE"
        print(f"{r['image']:12} {r['gt_class']:12} {r['iou_old']:.3f}    {cl_old:11} {c_old:9} "
              f"{r['iou_new']:.3f}    {cl_new:11} {c_new:9}{marker}")

    # === Statistiques agregees (seulement les boites detectees des DEUX cotes) ===
    matched = [r for r in rows if r["conf_old"] is not None and r["conf_new"] is not None]
    print(f"\n\n{'='*70}")
    print(f"STATISTIQUES AGREGEES ({len(matched)}/{len(rows)} boites detectees par les deux pipelines)")
    print(f"{'='*70}")

    iou_old_vals = np.array([r["iou_old"] for r in matched])
    iou_new_vals = np.array([r["iou_new"] for r in matched])
    conf_old_vals = np.array([r["conf_old"] for r in matched])
    conf_new_vals = np.array([r["conf_new"] for r in matched])
    acc_old = np.mean([r["class_old"] == r["gt_class"] for r in matched])
    acc_new = np.mean([r["class_new"] == r["gt_class"] for r in matched])

    print(f"IoU moyen vs verite terrain   : ancien={iou_old_vals.mean():.3f}  nouveau={iou_new_vals.mean():.3f}")
    print(f"Confiance moyenne             : ancien={conf_old_vals.mean()*100:.1f}%  nouveau={conf_new_vals.mean()*100:.1f}%")
    print(f"Exactitude classe (top-1)     : ancien={acc_old*100:.1f}%  nouveau={acc_new*100:.1f}%")

    delta_iou = iou_new_vals - iou_old_vals
    delta_conf = conf_new_vals - conf_old_vals
    if len(delta_iou) > 2 and delta_iou.std() > 0 and delta_conf.std() > 0:
        corr = np.corrcoef(delta_iou, delta_conf)[0, 1]
        print(f"\nCorrelation (delta IoU, delta confiance) sur les {len(matched)} boites appariees : r={corr:.3f}")
        print("(r proche de +1 => une boite plus precise vs verite terrain va de pair avec plus de confiance,")
        print(" confirmerait l'hypothese 'precision de boite -> confiance de classification')")

    missing_new = [r for r in rows if r["conf_old"] is not None and r["conf_new"] is None]
    missing_old = [r for r in rows if r["conf_old"] is None and r["conf_new"] is not None]
    print(f"\nDetections manquees par le nouveau (presentes chez l'ancien) : {len(missing_new)}")
    for r in missing_new:
        print(f"  {r['image']} gt_class={r['gt_class']}")
    print(f"Detections manquees par l'ancien (presentes chez le nouveau) : {len(missing_old)}")
    for r in missing_old:
        print(f"  {r['image']} gt_class={r['gt_class']}")


if __name__ == "__main__":
    main()
