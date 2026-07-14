
import sys
import os
import time

# Ajouter le répertoire parent en PRIORITÉ absolue (index 0) pour forcer Python
# à utiliser le model_library.py de jax_supervised_training (et non celui de JAX_Classification si exécuté depuis l'autre dossier)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Remplace YOLO import par JAX/Flax helpers si nécessaire
# from ultralytics import YOLO  <-- REMOVED
import cv2
import json
import shutil
import numpy as np
import jax
import jax.numpy as jnp
from PIL import Image
from tqdm import tqdm

from dataset_configs import get_dataset_config
from inference_utils import (
    load_jax_model,
    load_detection_model,
    predict_crop,
    get_iou,
    non_max_suppression,
    decode_segmentation_and_detect,
)

# =================================================================================================
# CONFIGURATION
# =================================================================================================
# 1. Configuration du dataset et du modèle de classification
DATASET_NAME = "FIGHTERJET_CLASSIFICATION"     # Nom de la config dans dataset_configs.py
CHECKPOINT_PATH = "best_model.pkl"      # Chemin vers le modèle de CLASSIFICATION
INPUT_DIR = "/home/aobled/Downloads/tmp_multi"  # Dossier d'entrée (images à traiter)
CONFIDENCE_THRESHOLD = 0.6            # Seuil de confiance pour valider une CLASSIFICATION bet 0.96

# 2. Configuration du modèle de détection
DETECTION_CHECKPOINT_PATH = "best_model_detection.pkl" # Chemin vers le modèle de DÉTECTION
DETECTION_CONF_THRESHOLD = 0.8          # Seuil pour considérer une détection valide (objectness + class) best 0.5

# 3. Configuration de la zone de détection
BOX_AERA_MIN = 60
NMS_THRESHOLD = 0.4

DEFAULT_CLASSE = "unknown"

# 3. Chargement de la config dataset
try:
    config = get_dataset_config(DATASET_NAME)
    CLASS_NAMES = config["class_names"]
    print(f"✅ Configuration chargée: {DATASET_NAME}")
    print(f"📊 Classes ({len(CLASS_NAMES)}): {CLASS_NAMES}")
    print(f"🔒 Seuil de confiance (Classification): {CONFIDENCE_THRESHOLD * 100}%")
except Exception as e:
    print(f"❌ Erreur chargement config: {e}")
    sys.exit(1)


# =================================================================================================
# MAIN
# =================================================================================================

if __name__ == "__main__":
    # 1. Chargement des modèles
    print("\n🏗️  Initialisation...")
    
    # DÉTECTION (Custom JAX)
    try:
        det_model, det_vars, det_config = load_detection_model(DETECTION_CHECKPOINT_PATH)
        print("✅ Modèle de DÉTECTION JAX chargé.")
        print(f"   Config détection: Grid={det_config.get('grid_size', '?')}, Size={det_config.get('image_size', '?')}")
    except Exception as e:
        print(f"❌ Erreur chargement modèle détection: {e}")
        # traceback
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # CLASSIFICATION (JAX Classifier)
    try:
        clf_model, clf_vars, dataset_mean, dataset_std = load_jax_model(CHECKPOINT_PATH, config)
        print("✅ Modèle de CLASSIFICATION JAX chargé.")
    except Exception as e:
        print(f"❌ Erreur chargement modèle classification: {e}")
        sys.exit(1)


    # 2. Préparation des fichiers
    image_files = []
    if not os.path.exists(INPUT_DIR):
         print(f"❌ Le dossier {INPUT_DIR} n'existe pas.")
         sys.exit(1)
         
    for root, _, files in os.walk(INPUT_DIR):
        for file in files:
            if file.lower().endswith((".jpg", ".png", ".jpeg", ".bmp")):
                image_files.append(os.path.join(root, file))

    print(f"\n📂 Traitement de {len(image_files)} images dans {INPUT_DIR}")


    # 3. Boucle de traitement
    processed_count = 0
    planes_detected = 0

    for img_path in tqdm(image_files, desc="Traitement des images"):
        root = os.path.dirname(img_path)
        file_name = os.path.basename(img_path)
        
        # Lire l'image
        img = cv2.imread(img_path)
        if img is None:
            continue
        h, w = img.shape[:2]
        
        # --- DÉTECTION CUSTOM ---
        detections = decode_segmentation_and_detect(
            img, det_model, det_vars, det_config, 
            conf_threshold=DETECTION_CONF_THRESHOLD,
            box_aera_min=BOX_AERA_MIN
        )
        
        # --- NMS ---
        detections = non_max_suppression(detections, iou_threshold=NMS_THRESHOLD)
        
        json_files_created = []
        detected_classes = set()
        planes_in_image = 0
        
        for (x1, y1, x2, y2, score) in detections:
            # Note: Le modèle de détection ne prédit que "avion" (classe unique implicite)
            
            # Extraire le crop pour classification
            crop = img[y1:y2, x1:x2]
            
            if crop.size == 0 or crop.shape[0] == 0 or crop.shape[1] == 0:
                continue
                
            # --- CLASSIFICATION DU CROP ---
            predicted_class, confidence = predict_crop(crop, clf_model, clf_vars, dataset_mean, dataset_std, config)
            
            # Filtrage par confiance
            if confidence < CONFIDENCE_THRESHOLD:
                predicted_class = DEFAULT_CLASSE
            
            detected_classes.add(predicted_class)
            # ------------------------------
            
            # Préparer JSON
            bbox_float = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]
            
            data = {
                "image": {
                    "file_name": file_name,
                    "width": w,
                    "height": h
                },
                "annotation": {
                    "file_name": file_name,
                    "bbox": bbox_float,
                    "category_name": predicted_class,  # Utilisation de la classe prédite !
                    "bbox_id": planes_in_image,
                    "detection_score": float(score), # Ajout du score de détection pour info
                    "classification_score": float(confidence)
                }
            }
            
            # Sauvegarde JSON
            base_name = os.path.splitext(file_name)[0]
            out_name = f"{base_name}_{planes_in_image}.json"
            out_path = os.path.join(root, out_name)
            
            with open(out_path, "w") as f:
                json.dump(data, f, indent=4)
                
            json_files_created.append(out_path)
            planes_in_image += 1
            planes_detected += 1
        
        # Organisation des fichiers (Déplacement)
        if planes_in_image > 0:
            # Déterminer le dossier de destination
            if len(detected_classes) == 1:
                # Une seule classe détectée -> Dossier au nom de la classe
                folder_name = list(detected_classes)[0]
            else:
                # Plusieurs classes détectées -> Dossier "multi"
                folder_name = "multi"
            
            dest_dir = os.path.join(INPUT_DIR, folder_name)
            os.makedirs(dest_dir, exist_ok=True)
            
            # Déplacer l'image
            try:
                shutil.move(img_path, os.path.join(dest_dir, file_name))
            except shutil.Error:
                pass # Évite de planter si l'image existe déjà (overwrite silencieux ou skip)
            
            # Déplacer les JSONs
            for jf in json_files_created:
                try:
                    shutil.move(jf, os.path.join(dest_dir, os.path.basename(jf)))
                except shutil.Error:
                    pass
                    
        processed_count += 1

    print(f"\n✅ Terminé !")
    print(f"   Images traitées: {processed_count}")
    print(f"   Avions détectés et classifiés: {planes_detected}")
