import os
import sys
import json
import cv2
import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
from tqdm import tqdm
import pickle

# --- Imports locaux ---
# Ajout du répertoire parent pour importer les modules du projet
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset_configs import get_dataset_config
from model_library import get_model

# --- Configuration ---
DATASET_PATH = '/home/aobled/Downloads/Aircraft_DATASET/detection'
CONFIG_NAME = "FIGHTERJET_CLASSIFICATION"
BATCH_SIZE = 64

def _preprocess_crop_to_hwc(crop_img, mean, std, config):
    """Prépare un crop BGR en tenseur (H, W, C) float32."""
    target_size = config["image_size"]
    grayscale = config.get("grayscale", False)
    
    crop_resized = cv2.resize(crop_img, target_size)
    
    if grayscale:
        img_input = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2GRAY)
        img_input = img_input.astype(np.float32) / 255.0
        img_normalized = (img_input - mean) / std
        return img_normalized[:, :, np.newaxis]
        
    img_input = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2RGB)
    img_input = img_input.astype(np.float32) / 255.0
    img_normalized = (img_input - mean) / std
    return img_normalized

def build_predict_fn(model, variables):
    """Compile la fonction d'inférence avec JIT pour le traitement par batch."""
    @jax.jit
    def predict_fn(batch_images):
        logits = model.apply(variables, batch_images, training=False, mutable=False)
        return logits
    return predict_fn

def load_classification_model(config):
    """Charge l'architecture du modèle, ses poids .pkl et ses statistiques de normalisation."""
    pkl_path = config.get("checkpoint_path", "best_model_classification.pkl")
    
    # Résoudre le chemin relatif
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    pkl_path_abs = os.path.join(parent_dir, pkl_path)
    
    if not os.path.exists(pkl_path_abs):
        raise FileNotFoundError(f"Modèle introuvable : {pkl_path_abs}")
        
    print(f"📦 Chargement des poids depuis {pkl_path_abs}...")
    with open(pkl_path_abs, 'rb') as f:
        model_data = pickle.load(f)
        
    # Compatibilité entre les anciens et nouveaux formats de sauvegarde
    if 'model_state' in model_data:
        params = model_data['model_state']['params']
        batch_stats = model_data['model_state'].get('batch_stats', {})
        model_name = model_data.get('model_info', {}).get('model_name', config["model_name"])
    else:
        params = model_data['params']
        batch_stats = model_data.get('batch_stats', {})
        model_name = model_data.get('model_name', config["model_name"])
    
    model = get_model(model_name, num_classes=config["num_classes"], dropout_rate=0.0)
    variables = {'params': params, 'batch_stats': batch_stats}
    
    # Chargement des stats de normalisation
    mean, std = 0.5, 0.5 # Valeurs par défaut
    mean_std_path = config.get("mean_std_path", "./data/chunks/dataset_chunked_meanstd.npz")
    mean_std_path_abs = os.path.join(parent_dir, mean_std_path)
    
    if os.path.exists(mean_std_path_abs):
        with np.load(mean_std_path_abs) as data:
            mean = data['mean']
            std = data['std']
            
            if config.get("grayscale", False):
                if isinstance(mean, np.ndarray) and mean.size == 3:
                     mean = np.mean(mean)
                     std = np.mean(std)
                     
    predict_fn = build_predict_fn(model, variables)
    
    # Warmup JIT
    print("🔥 Compilation JIT du graphe (Warmup)...")
    dummy_shape = (1, config["image_size"][0], config["image_size"][1], 1 if config.get("grayscale", False) else 3)
    dummy_input = jnp.zeros(dummy_shape, dtype=jnp.float32)
    _ = predict_fn(dummy_input)
    print("✅ Modèle prêt pour l'inférence par batch.")
    
    return predict_fn, mean, std

def run_audit():
    config = get_dataset_config(CONFIG_NAME)
    class_names = config["class_names"]
    
    predict_fn, mean, std = load_classification_model(config)
    
    results = []
    crop_batch = []
    meta_batch = []
    
    def flush_batch():
        nonlocal crop_batch, meta_batch, results
        if not crop_batch:
            return
            
        # Conversion du batch et inférence JAX
        batch_np = np.stack(
            [_preprocess_crop_to_hwc(c, mean, std, config) for c in crop_batch],
            axis=0
        )
        logits = predict_fn(jnp.array(batch_np))
        probs = jax.nn.softmax(logits, axis=-1)
        probs = np.array(probs)
        pred_indices = np.argmax(probs, axis=-1)
        
        # Enregistrement des résultats
        for i, meta in enumerate(meta_batch):
            pred_class = class_names[pred_indices[i]]
            conf = float(probs[i, pred_indices[i]])
            true_class = meta["true_class"]
            status = "CORRECT" if pred_class == true_class else "ERROR"
            
            results.append({
                "base_image_name": meta["base_image_name"],
                "split": meta["split"],
                "true_class": true_class,
                "pred_class": pred_class,
                "confidence": round(conf, 4),
                "status": status,
                "directory": meta["directory"],
                "image_path": meta["image_path"]
            })
            
        crop_batch.clear()
        meta_batch.clear()

    print("\n🔍 Début du scan du dataset...")
    for split in ["train", "val"]:
        split_dir = os.path.join(DATASET_PATH, split)
        if not os.path.exists(split_dir):
            continue
            
        # Parcours des sous-dossiers par classe
        for root, _, files in os.walk(split_dir):
            json_files = [f for f in files if f.endswith('.json')]
            
            if not json_files:
                continue
                
            for json_file in tqdm(json_files, desc=f"Traitement {split}/{os.path.basename(root)}", leave=False):
                json_path = os.path.join(root, json_file)
                base_name = json_file.split('_')[0]
                
                try:
                    with open(json_path, 'r') as f:
                        annotation = json.load(f)
                        
                    image_filename = annotation.get("image", {}).get("file_name", f"{base_name}.jpg")
                    image_path = os.path.join(root, image_filename)
                    
                    if not os.path.exists(image_path):
                        continue
                        
                    bbox = annotation.get("annotation", {}).get("bbox", [])
                    true_class = annotation.get("annotation", {}).get("category_name", "unknown")
                    
                    if not bbox or true_class not in class_names:
                        continue
                        
                    # Lecture de l'image
                    img = cv2.imread(image_path)
                    if img is None:
                        continue
                        
                    # Extraction du Crop
                    x, y, w, h = [int(v) for v in bbox]
                    
                    # Protection contre les boîtes hors limites
                    x, y = max(0, x), max(0, y)
                    w = min(w, img.shape[1] - x)
                    h = min(h, img.shape[0] - y)
                    
                    if w <= 0 or h <= 0:
                        continue
                        
                    crop = img[y:y+h, x:x+w]
                    
                    crop_batch.append(crop)
                    meta_batch.append({
                        "base_image_name": base_name,
                        "split": split,
                        "true_class": true_class,
                        "directory": os.path.basename(root),
                        "image_path": image_path
                    })
                    
                    if len(crop_batch) >= BATCH_SIZE:
                        flush_batch()
                        
                except Exception as e:
                    print(f"Erreur sur {json_path}: {e}")
                    
    # Vider le dernier batch restant
    flush_batch()
    
    # --- Création du CSV et reporting Pandas ---
    print("\n💾 Export des résultats vers audit_classification_results.csv...")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, "audit_classification_results.csv")
    
    df = pd.DataFrame(results)
    if len(df) == 0:
        print("❌ Aucune donnée analysée.")
        return
        
    df.to_csv(csv_path, index=False)
    
    total = len(df)
    errors = len(df[df["status"] == "ERROR"])
    accuracy = ((total - errors) / total * 100) if total > 0 else 0
    
    print(f"\n======================================")
    print(f"📊 BILAN DE L'AUDIT DE CLASSIFICATION")
    print(f"======================================")
    print(f"   Total analysé : {total} images (crops)")
    print(f"   Erreurs       : {errors}")
    print(f"   Précision     : {accuracy:.2f} %")
    
    print(f"\n   Répartition de la précision par split :")
    split_acc = df.groupby('split')['status'].apply(lambda x: (x == 'CORRECT').mean() * 100)
    for s, acc in split_acc.items():
        print(f"     - {s.upper()} : {acc:.2f} %")
    
    if errors > 0:
        print("\n⚠️ TOP 10 PIRES ERREURS (Haute confiance du modèle, mais classé ERROR) :")
        worst = df[df["status"] == "ERROR"].sort_values(by="confidence", ascending=False).head(10)
        print(worst[["base_image_name", "true_class", "pred_class", "confidence", "split"]].to_string(index=False))

if __name__ == "__main__":
    run_audit()
