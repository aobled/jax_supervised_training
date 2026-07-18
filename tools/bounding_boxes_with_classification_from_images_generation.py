
import sys
import os

# Ajouter le répertoire parent en PRIORITÉ absolue (index 0) pour forcer Python
# à utiliser le model_library.py de jax_supervised_training (et non celui de JAX_Classification si exécuté depuis l'autre dossier)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Remplace YOLO import par JAX/Flax helpers si nécessaire
# from ultralytics import YOLO  <-- REMOVED
import cv2
import json
import shutil
import numpy as np
import jax.numpy as jnp
from tqdm import tqdm

from dataset_configs import get_dataset_config
from inference_utils import build_single_pass_predict_fn, _rescale_boxes

# =================================================================================================
# CONFIGURATION
# =================================================================================================
# 1. Configuration du dataset et du modèle de classification
DATASET_NAME = "FIGHTERJET_CLASSIFICATION"     # Nom de la config dans dataset_configs.py
CHECKPOINT_PATH = "best_model.pkl"      # Chemin vers le modèle de CLASSIFICATION
INPUT_DIR = "/home/aobled/Downloads/tmp_multi"  # Dossier d'entrée (images à traiter)
CONFIDENCE_THRESHOLD = 0.6            # Seuil de confiance pour valider une CLASSIFICATION bet 0.96

# 2. Configuration du modèle de détection (Story 8.6 : JAX_DETECTOR remplace l'ancien
# best_model_detection.pkl/AircraftDetectorUNet - celui-ci reste disponible et
# fonctionnel pour l'ancien pipeline FIGHTERJET_DETECTION, AD-20, non touché ici).
DETECTOR_CHECKPOINT_PATH = "best_model_jax_detector.pkl"

# Repère canonique attendu par build_single_pass_predict_fn (AD-12) - les images
# d'entrée de ce script sont de résolution arbitraire, jamais garanties 1920x1080.
CANONICAL_WIDTH, CANONICAL_HEIGHT = 1920, 1080

# DETECTION_CONF_THRESHOLD/BOX_AERA_MIN/NMS_THRESHOLD (ancien chemin) n'ont plus
# d'équivalent direct (Task 9) : le nouveau chemin dérive son propre seuil depuis la
# config JAX_DETECTOR (detection_score_threshold, Story 8.3) et n'a plus de NMS
# explicite (AD-9, la tête par point central n'en a pas besoin) - changement de
# comportement réel, documenté, pas une continuité silencieuse.

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
    # 1. Chargement des modèles (JAX Single-Pass, Story 8.6 - un seul callable, une
    # seule fois, hors de la boucle de traitement)
    print("\n🏗️  Initialisation...")

    try:
        predict_fn = build_single_pass_predict_fn(
            detector_checkpoint_path=DETECTOR_CHECKPOINT_PATH,
            classifier_checkpoint_path=CHECKPOINT_PATH,
        )
        print("✅ Modèles DÉTECTION+CLASSIFICATION (Single-Pass) chargés.")
    except Exception as e:
        print(f"❌ Erreur chargement des modèles: {e}")
        import traceback
        traceback.print_exc()
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

        # --- Normalisation d'entrée vers le repère canonique (AD-12) ---
        # Résolution source arbitraire (dossier hétérogène) -> 1920x1080 grayscale,
        # normalisation d'E/S en Python explicitement autorisée par AD-12 (hors du
        # périmètre zéro-Python de la logique d'inférence elle-même). Pixels bruts
        # [0,255], jamais de /255.0 ici - normalisation interne à predict_fn.
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_canonical = cv2.resize(img_gray, (CANONICAL_WIDTH, CANONICAL_HEIGHT))
        img_canonical = img_canonical.astype(np.float32)[..., None]

        # --- SINGLE-PASS (détection + classification unifiées, Story 8.6) ---
        result = predict_fn(jnp.asarray(img_canonical))

        # --- RESCALE vers la résolution propre de l'image source (Task 5) ---
        # result["boxes"] est dans le repère canonique 1920x1080, pas (w,h) source -
        # second appel à _rescale_boxes, différent de son usage interne à predict_fn.
        boxes_native = _rescale_boxes(
            result["boxes"], detector_size=(CANONICAL_WIDTH, CANONICAL_HEIGHT), original_size=(w, h)
        )
        boxes_native = np.asarray(boxes_native)
        valid_mask = np.asarray(result["valid_mask"])
        classes = np.asarray(result["classes"])
        class_scores = np.asarray(result["class_scores"])
        detection_scores = np.asarray(result["detection_scores"])

        json_files_created = []
        detected_classes = set()
        planes_in_image = 0

        for i in np.where(valid_mask)[0]:  # valid_mask seule autorité (AD-15/AC3 8.6)
            x1, y1, x2, y2 = boxes_native[i]
            score = float(detection_scores[i])
            # Note: Le modèle de détection ne prédit que "avion" (classe unique implicite)

            # --- CLASSIFICATION (déjà calculée par predict_fn) ---
            predicted_class = CLASS_NAMES[int(classes[i])]  # indice -> nom (Task 7)
            confidence = float(class_scores[i])

            # Filtrage par confiance
            if confidence < CONFIDENCE_THRESHOLD:
                predicted_class = DEFAULT_CLASSE

            detected_classes.add(predicted_class)
            # ------------------------------

            # Préparer JSON (boîtes déjà rescalées vers la résolution source, Task 5)
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
                    "detection_score": score, # Ajout du score de détection pour info
                    "classification_score": confidence
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
