"""
Configuration centralisée pour tous les datasets
Permet de gérer facilement plusieurs datasets avec leurs paramètres spécifiques
"""

import os

import numpy as np

# Racine des datasets chunkés (.npz). Local par défaut ; sur Colab, définir
# JAX_SUPERVISED_TRAINING_DATA_ROOT (ex: /content/drive/MyDrive/jax_supervised_training/data)
# AVANT d'importer ce module.
DATA_ROOT = os.environ.get("JAX_SUPERVISED_TRAINING_DATA_ROOT", "/home/aobled/Documents/data")


def validate_config(config_name, config):
    """Valide la cohérence d'une configuration de dataset"""
    errors = []
    
    # Vérifier que num_classes correspond à len(class_names)
    if "num_classes" in config and "class_names" in config:
        if len(config["class_names"]) != config["num_classes"]:
            errors.append(f"num_classes ({config['num_classes']}) != len(class_names) ({len(config['class_names'])})")
    
    # Vérifier que image_size est un tuple de 2 entiers
    if "image_size" in config:
        if not isinstance(config["image_size"], tuple) or len(config["image_size"]) != 2:
            errors.append(f"image_size doit être un tuple (H, W)")
    
    # Vérifier que les paramètres requis sont présents
    required = ["num_classes", "image_size", "model_name"]
    for key in required:
        if key not in config:
            errors.append(f"Paramètre requis manquant: {key}")
    
    if errors:
        print(f"❌ Erreurs de configuration pour {config_name}:")
        for error in errors:
            print(f"  - {error}")
        return False
    
    return True


DATASET_CONFIGS = {
    "FIGHTERJET_CLASSIFICATION": {
        # === CONFIG OPTIMALE CNN 128×128 GRAYSCALE STRETCHED ===
        # === Données ===
        "num_classes": 32,
        "class_names": ['a10', 'a4', 'a400m', 'alphajet', 'b1b', 'b2', 'b52', 'c130', 'c17', 'f117', 'f14', 'f15', 'f16', 'f18', 'f22', 'f35', 'f4', 'flanker', 'gripen', 'harrier', 'hawk', 'hawkeye', 'mig29', 'mirage2000', 'mustang', 'rafale', 'spitfire', 'sr71', 'su57', 'tornado', 'typhoon', 'v22'],
        "data_dir": "/home/aobled/Downloads/_balanced_dataset_split",
        "output_prefix": f"{DATA_ROOT}/chunks/classification/dataset_classification",
        "chunk_size": 30000,
        "image_size": (128, 128),
        "grayscale": True,  # ✅ GRAYSCALE (3× plus rapide, même accuracy)
        "augmentation_params": {
            "flip_h": True,
            "flip_v": False,              # False par défaut, mais on tente à True
            "rotation_factor": 0.12,      # On tente 0.15, 0.10 avant
            "zoom_factor": 0.10,          # Adouci (était 0.15)
            "translation_factor": 0.06,   # Divisé par 2 (0.12 -> 0.06) : Assez pour garder l'invariance anti-scintillement sans détruire la cible
            "brightness_delta": 0.10,
            "contrast_factor": 0.10,
            "pixelation_factor": 4.0      # Simule un upscale depuis un tout petit crop
        },
        
        "mean": None,
        "std": None,
        "mean_std_path": f"{DATA_ROOT}/chunks/classification/dataset_classification_meanstd.npz",
        
        # === Modèle ===
        "model_name": "sophisticated_cnn_128_plus",  # ✅ OPTIMAL: Version optimisée+ (4M params, 88% val)
        "loss_method": "focal_loss",
        "loss_params": {"gamma": 2.0},


        
        # === Hyperparamètres TPU ===
        "tpu": {
            "micro_batch_size": 128,   # ✅ Optimal testé
            "accum_steps": 4,          # ✅ Batch effectif: 512 (sweet spot)
            "learning_rate": 8e-3,     # 🔥 NOUVEAU: LR augmenté pour apprentissage plus agressif
            "weight_decay": 5e-5,      # ✅ Optimal trouvé
            "dropout_rate": 0.0,       # 🔥 NOUVEAU: Pas de dropout pour exploiter le potentiel
        },
        
        # === Hyperparamètres GPU ===
        "gpu": {
            "micro_batch_size": 128,   # ✅ Testé, fonctionne
            "accum_steps": 4,          # Batch effectif: 512
            "learning_rate": 8e-4,     # 🔥 NOUVEAU: LR/10 pour GPU (augmenté aussi)
            "weight_decay": 5e-5,
            "dropout_rate": 0.0,       # 🔥 NOUVEAU: Pas de dropout pour GPU aussi
        },
        
        # === Entraînement ===
        "optimizer": "adamw",
        "lr_schedule": "cosine",
        "epochs": 40,              # 40
        "patience": 5,
        "warmup_steps": 1200,      # Rendu explicite (était hérité du Trainer)
        "decay_steps": 6000,       # Le LR chute vite et stagne à 0 pour fine-tuning après ~15 epochs
        "label_smoothing": 0.15,    # ✅ Aide légèrement
        "mixup_alpha": 0.05,        # ✅ OPTIMAL: Mixup doux (meilleur compromis trouvé)
        
        # === Évaluation ===
        "metric_method": "accuracy",
        "report_method": "confusion_matrix",
        "eval_batch_size": 128,

        "eval_use_subset": True,
        "eval_max_subset": 100000,  
        
        # === Sauvegarde ===
        "checkpoint_path": "best_model.pkl",
        "training_state_path": "best_model_training_state_classification.pkl",
        "confusion_matrix_path": "confusion_matrix.png",
    },
    
    "FIGHTERJET_DETECTION": {
        # === CONFIG DETECTION D'AVIONS ===
        "task_type": "detection",
        
        # === Données ===
        "num_classes": 1,  # Single Class Object Detection
        "class_names": ['aircraft'],
        "output_prefix": f"{DATA_ROOT}/chunks/detection/dataset_detection",
        "image_size": (224, 224),
        "grayscale": True,
        "max_boxes": 20,  # 🔥 Images avec plus de 20 boxes seront ignorées (évite padding excessif et faux négatifs)
        
        # === Augmentation de Données ===
        "augmentation_params": {
            "flip_h": True,
            "flip_v": True,
            "rotation_factor": 0.0,
            "zoom_factor": 0.35,          # ±25% (était 20%) : Pour apprendre le multi-échelles
            "translation_factor": 0.25,   # Shifting ±15% (était 10%) : Désaxer les cibles
            "brightness_delta": 0.15,     # Beaucoup plus agressif (casser la dominante ciel gris clair)
            "contrast_factor": 0.30       # Très agressif (simule contre-jour et nuages sombres)
        },
        
        # === Modèle ---
        "model_name": "aircraft_detector_unet",
        #"model_name": "aircraft_detector_miniunet",
        "grid_size": 224,      # Segmentation sémantique (output size = input size)
        "loss_method": "segmentation",
        "loss_params": {
            "bce_weight": 0.3,
            "dice_weight": 0.7,
            "false_positive_penalty": 2.0
        },

        
        # === Hyperparamètres TPU ===
        "tpu": {
            "micro_batch_size": 128,    # Batch ok à 65, test avec 128
            "accum_steps": 1,          # Accumulation pour simuler 64
            "learning_rate": 4e-4,     # LR légèrement remonté
            "weight_decay": 5e-5,
            "dropout_rate": 0.0,
        },
        
        # === Hyperparamètres GPU ===
        "gpu": {
            "micro_batch_size": 16,
            "accum_steps": 2,
            "learning_rate": 2e-4,
            "weight_decay": 5e-5,
            "dropout_rate": 0.05,
        },
        
        # === Entraînement ===
        "optimizer": "adamw",
        "lr_schedule": "cosine",
        "epochs": 8,

        "patience": 5,
        "warmup_steps": 400,           # Préchauffage rapide sur ~0.15 epoch
        "decay_steps": 22000,          # Le LR chutera vers zéro autour de la fin de l'epoch 8
        
        # === Évaluation/Visualization ===
        "metric_method": "segmentation_iou",
        "report_method": "segmentation_heatmap",
        "eval_batch_size": 16,

        "vis_freq": 5,
        
        # === Sauvegarde ===
        "checkpoint_path": "best_model_detection.pkl",
        "training_state_path": "best_model_training_state_detection.pkl",
        "save_dir": "./checkpoints_detection",
    },
    
    "JAX_KEPLER": {
        # === CONFIG CLASSIFICATION EXOPLANETES (1D) ===
        "task_type": "kepler",
        
        # === Données ===
        "num_classes": 2,
        "class_names": ['no_exoplanet', 'exoplanet'],
        "output_prefix": "./data/chunks/kepler/dataset_kepler",
        # (longueur, 1) : format (H,W) attendu par la validation et ChunkManager → tenseurs (L, 1, C)
        "image_size": (3197, 1),
        "grayscale": True,       # Force data_management à ne pas chercher de RGB
        
        # === Augmentation de Données ===
        # Pas d'augmentation spatiale (flip_h n'a pas de sens physique direct ici)
        "augmentation_params": {
            "flip_h": False,
            "flip_v": False,
            "rotation_factor": 0.0,
            "zoom_factor": 0.0,
            "translation_factor": 0.0,
            "brightness_delta": 0.0,
            "contrast_factor": 0.0
        },
        
        # === Modèle ===
        "model_name": "kepler_1d_cnn",
        "loss_method": "cross_entropy",

        
        # === Hyperparamètres TPU ===
        "tpu": {
            "micro_batch_size": 128,
            "accum_steps": 1,
            "learning_rate": 1e-4,
            "weight_decay": 1e-4,
            "dropout_rate": 0.3,
            "label_smoothing": 0.0,
            "mixup_alpha": 0.0
        },
        
        # === Hyperparamètres GPU ===
        "gpu": {
            "micro_batch_size": 32,
            "accum_steps": 4,
            "learning_rate": 1e-4,
            "weight_decay": 1e-4,
            "dropout_rate": 0.3,
            "label_smoothing": 0.0,
            "mixup_alpha": 0.0
        },
        
        # === Entraînement ===
        "optimizer": "adamw",
        "lr_schedule": "cosine",
        "epochs": 30,

        "patience": 5,
        "warmup_steps": 500,
        
        "eval_use_subset": False, # On évalue sur tout
        "metric_method": "accuracy",
        "report_method": "lightcurves",
        
        # === Sauvegarde ===
        "checkpoint_path": "best_model_kepler.pkl",
        "training_state_path": "best_model_training_state_kepler.pkl",
        "confusion_matrix_path": "confusion_matrix_kepler.png",
    },

    "CIFAR10": {
        # === CONFIG BOUCLE DE TEST RAPIDE (pipeline, pas optimisation d'accuracy) ===
        # === Données ===
        "num_classes": 10,
        "class_names": ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'],
        "output_prefix": f"{DATA_ROOT}/chunks/cifar10/dataset_cifar10",
        "image_size": (32, 32),
        "grayscale": False,
        "augmentation_params": {
            "flip_h": True,
            "flip_v": False,
            "rotation_factor": 0.0,
            "zoom_factor": 0.0,
            "translation_factor": 0.1,  # random-crop-like shift (~3px/32px) ; annule l'Addendum 2 Epic 5 ("pas de changement nécessaire") - le surapprentissage persiste malgré le fix dropout de l'Addendum 3 (meilleur epoch 19 : train 93.45% vs val 81.10%, archive/training_cifar10_log_GPU_128x1.txt)
            "brightness_delta": 0.0,
            "contrast_factor": 0.0
        },

        "mean": None,
        "std": None,
        "mean_std_path": f"{DATA_ROOT}/chunks/cifar10/dataset_cifar10_meanstd.npz",

        # === Modèle ===
        "model_name": "sophisticated_cnn_32_plus",  # Variante réduite pour 32×32 (128_plus était surdimensionné/trop de pooling pour cette taille)
        "loss_method": "cross_entropy",  # Dataset équilibré par construction (contrairement à FIGHTERJET_CLASSIFICATION)
        "loss_params": {},

        # === Hyperparamètres GPU ===
        "gpu": {
            "micro_batch_size": 128,
            "accum_steps": 1,
            "learning_rate": 1e-3,
            "weight_decay": 5e-5,
            "dropout_rate": 0.3,
        },

        # === Hyperparamètres TPU ===
        "tpu": {
            "micro_batch_size": 128,
            "accum_steps": 1,
            "learning_rate": 1e-3,
            "weight_decay": 5e-5,
            "dropout_rate": 0.3,
        },

        # === Entraînement ===
        "optimizer": "adamw",
        "lr_schedule": "cosine",
        "epochs": 30,          # 10 coupait le decay LR en plein milieu (391 steps/epoch) ; 30 laisse une vraie fenêtre d'apprentissage
        "patience": 8,  # 5 -> 8 : hypothèse (non confirmée, val jamais augmentée) que la régularisation ralentit la convergence val ; sans risque sur le modèle exporté (best-checkpoint indépendant de patience, trainer.py:467-472), coûte au pire quelques epochs de calcul en plus
        "warmup_steps": 200,
        "decay_steps": 11700,  # ≈ steps/epoch (391) × epochs (30) : couvre tout l'entraînement, plus de LR figé à 1e-6 en plein milieu

        # === Évaluation ===
        "metric_method": "accuracy",
        "report_method": "confusion_matrix",
        "eval_batch_size": 128,
        "eval_use_subset": False,  # 10 000 images val, taille déjà raisonnable

        # === Sauvegarde ===
        # Pas de checkpoint_path/training_state_path explicite : nommage dérivé de dataset_name (Story 5.0)
        "confusion_matrix_path": "confusion_matrix_cifar10.png",
    }
}



def get_dataset_config(dataset_name):
    """
    Récupère la configuration d'un dataset
    
    Args:
        dataset_name: Nom du dataset (ex: "FIGHTERJET_9CLASSES")
    
    Returns:
        dict: Configuration du dataset
    
    Raises:
        ValueError: Si le dataset n'existe pas
    """
    if dataset_name not in DATASET_CONFIGS:
        available = list(DATASET_CONFIGS.keys())
        raise ValueError(f"Dataset '{dataset_name}' inconnu. Datasets disponibles: {available}")
    
    config = DATASET_CONFIGS[dataset_name]
    config["dataset_name"] = dataset_name

    # Par défaut, classification si non spécifié
    if "task_type" not in config:
        config["task_type"] = "classification"
    
    # Valider la configuration
    if not validate_config(dataset_name, config):
        raise ValueError(f"Configuration invalide pour {dataset_name}")
    
    return config


def list_available_datasets():
    """Retourne la liste des datasets disponibles"""
    return list(DATASET_CONFIGS.keys())


def print_config(dataset_name):
    """Affiche la configuration d'un dataset"""
    config = get_dataset_config(dataset_name)
    
    print(f"\n📊 CONFIGURATION: {dataset_name}")
    print("=" * 60)
    print(f"Classes: {config['num_classes']}")
    print(f"Noms: {config['class_names']}")
    print(f"Image size: {config['image_size']}")
    print(f"Modèle: {config['model_name']}")
    print(f"Epochs: {config['epochs']}")
    print(f"Patience: {config['patience']}")
    print(f"\nTPU: batch={config['tpu']['micro_batch_size']}×{config['tpu']['accum_steps']}, lr={config['tpu']['learning_rate']}, wd={config['tpu']['weight_decay']}")
    print(f"GPU: batch={config['gpu']['micro_batch_size']}×{config['gpu']['accum_steps']}, lr={config['gpu']['learning_rate']}, wd={config['gpu']['weight_decay']}")
    print("=" * 60)


if __name__ == "__main__":
    # Test de validation
    print("🧪 TEST DES CONFIGURATIONS")
    print("=" * 60)
    
    for dataset_name in list_available_datasets():
        print(f"\n✅ Validation de {dataset_name}...")
        try:
            config = get_dataset_config(dataset_name)
            print(f"   ✓ {config['num_classes']} classes: {config['class_names']}")
        except Exception as e:
            print(f"   ❌ Erreur: {e}")
    
    print("\n🎯 Affichage d'une config complète:")
    print_config("FIGHTERJET_CLASSES")
