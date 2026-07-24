"""
Audit IoU pour JAX_DETECTOR (Single-Pass, CenterNet) - miroir de
tools/audit_dataset_detection.py (AD-20, pipeline FIGHTERJET_DETECTION/UNet, jamais
modifie) pour permettre une comparaison chiffree directe entre les deux pipelines :
meme format de sortie CSV (image_name, split, num_true_boxes, num_pred_boxes,
mean_iou, status, directory, image_path), meme fonction calculate_mean_iou (matching
glouton meilleure-boite, un-a-plusieurs, identique aux deux scripts).

Difference deliberee avec audit_dataset_detection.py::convert_bbox_format : celui-ci
reimplemente independamment une geometrie de rescale (etirement x/y separe) - risque
documente (deferred-work.md, 2026-07-19) si le vrai pretraitement du modele ne
correspond pas exactement a cet etirement. Ce script-ci ne reimplemente RIEN : il
reutilise _rescale_boxes (inference_utils.py, deja valide Story 8.4, meme fonction
que celle utilisee par le pipeline d'inference reel) pour ramener les boites predites
dans le repere de l'image source - jamais de calcul geometrique maison.

2 fichiers independants, aucun couplage : ce script n'importe rien de
audit_dataset_detection.py et reciproquement (AD-20 : le pipeline FIGHTERJET_DETECTION
ne doit jamais dependre du chemin JAX_DETECTOR ni l'inverse).
"""
import os
import sys
import json
from collections import defaultdict

import cv2
import numpy as np
import jax.numpy as jnp
import pandas as pd
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference_utils import build_single_pass_predict_fn, _rescale_boxes, get_iou
from dataset_configs import get_dataset_config

# --- Configuration ---
DATASET_PATH = '/home/aobled/Downloads/Aircraft_DATASET/detection'
CANONICAL_WIDTH, CANONICAL_HEIGHT = 1920, 1080

# Seuil NMS JAX_DETECTOR (meme source que build_single_pass_predict_fn en interne,
# 2026-07-22) - recupere ici uniquement pour tracabilite dans les logs de l'audit,
# sans dupliquer sa valeur en dur (AD-15).
NMS_IOU_THRESHOLD = get_dataset_config("JAX_DETECTOR")["nms_iou_threshold"]


def load_ground_truth(target_root):
    """Charge les vraies boites [x1,y1,x2,y2] par image depuis les JSON d'annotation
    (meme format que le reste du projet : annotation.bbox = [x,y,w,h], repere
    pixels de l'image source - jamais de conversion ici, comparaison directe avec
    les boites predites une fois rescalees au meme repere)."""
    gt_by_image = defaultdict(list)
    meta_by_image = {}
    for split in ["train", "val"]:
        split_dir = os.path.join(target_root, split)
        if not os.path.exists(split_dir):
            continue
        for root, _, files in os.walk(split_dir):
            for filename in files:
                if not filename.endswith('.json'):
                    continue
                json_path = os.path.join(root, filename)
                try:
                    with open(json_path, 'r') as f:
                        annotation = json.load(f)
                    image_info = annotation.get("image", {})
                    ann_info = annotation.get("annotation", {})
                    image_filename = image_info.get("file_name")
                    bbox = ann_info.get("bbox")
                    if not image_filename or not bbox:
                        continue
                    x, y, w, h = bbox
                    image_path = os.path.join(root, image_filename)
                    gt_by_image[image_path].append([x, y, x + w, y + h])
                    meta_by_image[image_path] = {
                        "image_name": image_filename,
                        "split": split,
                        "directory": os.path.basename(root),
                    }
                except Exception as e:
                    print(f"Erreur lecture {json_path}: {e}")
    return gt_by_image, meta_by_image


def calculate_mean_iou(true_boxes, pred_boxes):
    """IoU moyen : pour chaque vraie boite, meilleur IoU parmi les boites predites
    (matching glouton un-a-plusieurs). Identique en esprit a
    audit_dataset_detection.py::calculate_mean_iou (reimplementee ici, pas importee -
    2 fichiers independants par design, voir docstring de module)."""
    if not true_boxes or not pred_boxes:
        return 0.0
    total_iou = 0.0
    matches = 0
    for true_box in true_boxes:
        best_iou = 0.0
        for pred_box in pred_boxes:
            iou = get_iou(true_box, pred_box)
            if iou > best_iou:
                best_iou = iou
        total_iou += best_iou
        matches += 1
    return (total_iou / matches) if matches > 0 else 0.0


def count_unmatched_predictions(true_boxes, pred_boxes, iou_threshold=0.5):
    """Compte les boites PREDITES qui n'ont aucune correspondance suffisante (IoU >=
    iou_threshold) avec une vraie boite - proxy de faux positif (2026-07-22, retour
    utilisateur : abaisser detection_score_threshold a 0.1 a-t-il fait exploser les
    faux positifs ?). calculate_mean_iou part des vraies boites et mesure le rappel -
    elle ne penalise jamais une prediction en trop ; cette fonction mesure l'autre sens
    (precision) : combien de predictions ne correspondent a AUCUN objet reel."""
    if not pred_boxes:
        return 0
    if not true_boxes:
        return len(pred_boxes)  # aucune vraie boite -> toute prediction est en trop
    unmatched = 0
    for pred_box in pred_boxes:
        best_iou = 0.0
        for true_box in true_boxes:
            iou = get_iou(pred_box, true_box)
            if iou > best_iou:
                best_iou = iou
        if best_iou < iou_threshold:
            unmatched += 1
    return unmatched


def run_audit():
    print("🏗️  Chargement du modèle JAX_DETECTOR (Single-Pass)...")
    print(f"   NMS IoU threshold : {NMS_IOU_THRESHOLD}")
    predict_fn = build_single_pass_predict_fn(nms_iou_threshold=NMS_IOU_THRESHOLD)

    print("📚 Chargement des annotations réelles...")
    gt_by_image, meta_by_image = load_ground_truth(DATASET_PATH)
    print(f"📊 {len(gt_by_image)} images annotées trouvées")

    results = []
    for image_path, true_boxes in tqdm(gt_by_image.items(), desc="Audit JAX_DETECTOR"):
        img = cv2.imread(image_path)
        if img is None:
            continue
        h, w = img.shape[:2]

        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_canonical = cv2.resize(img_gray, (CANONICAL_WIDTH, CANONICAL_HEIGHT)).astype(np.float32)[..., None]
        result = predict_fn(jnp.asarray(img_canonical))

        # Rescale canonique -> repere image source (meme fonction que le pipeline
        # d'inference reel, jamais de geometrie reimplementee - voir docstring module).
        boxes_native = np.asarray(_rescale_boxes(
            result["boxes"], detector_size=(CANONICAL_WIDTH, CANONICAL_HEIGHT), original_size=(w, h)
        ))
        valid_mask = np.asarray(result["valid_mask"])  # seule autorite (AD-15/AC3)
        detection_scores = np.asarray(result["detection_scores"])  # scores BRUTS, avant filtrage par seuil
        pred_boxes = [boxes_native[i].tolist() for i in np.where(valid_mask)[0]]

        mean_iou = calculate_mean_iou(true_boxes, pred_boxes)

        # Diagnostic 2026-07-22 (analyse bucket 70-100% : 91.4% de detections nulles,
        # mais IoU=0.75 quand une detection existe - distingue "aucun pic de confiance
        # produit" de "un pic existe mais juste sous detection_score_threshold=0.2").
        # max_raw_detection_score : le plus haut score parmi les 20 slots, SANS filtrage.
        # best_raw_iou : IoU du candidat le plus confiant (meme sous le seuil) contre la
        # verite terrain - teste si la geometrie est correcte independamment du seuil.
        best_idx = int(np.argmax(detection_scores))
        max_raw_detection_score = float(detection_scores[best_idx])
        best_raw_iou = calculate_mean_iou(true_boxes, [boxes_native[best_idx].tolist()])

        # Precision / faux positifs (2026-07-22, retour utilisateur suite a
        # detection_score_threshold 0.2->0.1) : combien de predictions VALIDES
        # (deja filtrees par valid_mask) ne correspondent a AUCUNE vraie boite.
        num_unmatched_predictions = count_unmatched_predictions(true_boxes, pred_boxes)

        meta = meta_by_image[image_path]
        results.append({
            "image_name": meta["image_name"],
            "split": meta["split"],
            "num_true_boxes": len(true_boxes),
            "num_pred_boxes": len(pred_boxes),
            "num_unmatched_predictions": num_unmatched_predictions,
            "mean_iou": round(mean_iou, 4),
            "status": "GOOD" if mean_iou >= 0.5 else "POOR",
            "max_raw_detection_score": round(max_raw_detection_score, 4),
            "best_raw_iou": round(best_raw_iou, 4),
            "directory": meta["directory"],
            "image_path": image_path,
        })

    df = pd.DataFrame(results)
    df.to_csv("audit_results_jax_detector.csv", index=False)

    print(f"\n📊 RÉSUMÉ ({len(df)} images) :")
    print(f"   IoU moyen global : {df['mean_iou'].mean():.4f}")
    good = (df['status'] == 'GOOD').sum()
    poor = (df['status'] == 'POOR').sum()
    print(f"   GOOD (IoU>=0.5) : {good} ({good / len(df) * 100:.1f}%)")
    print(f"   POOR (IoU<0.5)  : {poor} ({poor / len(df) * 100:.1f}%)")
    total_preds = df['num_pred_boxes'].sum()
    total_unmatched = df['num_unmatched_predictions'].sum()
    fp_rate = (total_unmatched / total_preds * 100) if total_preds > 0 else 0.0
    print(f"   Faux positifs (predictions sans vraie boite correspondante) : {total_unmatched}/{total_preds} ({fp_rate:.1f}%)")
    print(f"   Résultats détaillés : audit_results_jax_detector.csv")


if __name__ == "__main__":
    run_audit()
