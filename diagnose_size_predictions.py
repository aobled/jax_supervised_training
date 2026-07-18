"""
Diagnostic complementaire a diagnose_heatmap_predictions.py : la tete heatmap a ete
verifiee (collapse trouve/corrige, Story 7.2 addendum), mais la tete taille
(compute_size_regression_loss, Story 7.3) n'avait jamais ete inspectee empiriquement.
generate_reports (Story 7.6) ne visualise QUE le heatmap, jamais la taille - ce script
comble ce point aveugle.

Compare, aux vrais pixels-centres (gt_heatmap==1.0, meme critere que la loss
d'entrainement), la taille predite a la taille reelle - erreur absolue/relative, et
comparaison a une base triviale (prediction = moyenne des tailles reelles du batch)
pour verifier que le modele fait mieux qu'une constante.

Usage: python3 diagnose_size_predictions.py [checkpoint_path] [n_batches]
"""
import pickle
import sys

import jax
import jax.numpy as jnp
import numpy as np

from model_library import get_model
from dataset_configs import get_dataset_config
from data_management import CenterNetDetectionDataset
from detection_target_encoding import HEATMAP_KEY, SIZE_KEY

CHECKPOINT_PATH = sys.argv[1] if len(sys.argv) > 1 else "best_model_jax_detector.pkl"
N_BATCHES = int(sys.argv[2]) if len(sys.argv) > 2 else 8

with open(CHECKPOINT_PATH, "rb") as f:
    ckpt = pickle.load(f)

config = ckpt["config"]
print(f"Checkpoint : dataset_name={config.get('dataset_name')}, model_name={config.get('model_name')}")

model = get_model(config["model_name"], num_classes=config["num_classes"], dropout_rate=0.0)
variables = {"params": ckpt["params"], "batch_stats": ckpt["batch_stats"]}

ckpt_dtype = jax.tree_util.tree_leaves(ckpt["params"])[0].dtype
print(f"Dtype des poids du checkpoint : {ckpt_dtype}")

dataset_config = get_dataset_config("JAX_DETECTOR")
ds_manager = CenterNetDetectionDataset(
    output_prefix=dataset_config["output_prefix"],
    image_size=dataset_config["image_size"],
    batch_size=32,
    grayscale=dataset_config["grayscale"],
    augmentation_params={},
)
val_ds = ds_manager.create_tf_dataset("val", augment=False)

apply_fn = jax.jit(lambda v, x: model.apply(v, x, training=False))

gt_w_all, gt_h_all = [], []
pred_w_all, pred_h_all = [], []

for images, targets in val_ds.take(N_BATCHES):
    images_np = images.numpy()
    gt_heatmap = targets[HEATMAP_KEY].numpy()
    gt_size = targets[SIZE_KEY].numpy()

    x = jnp.asarray(images_np, dtype=ckpt_dtype)
    outputs = apply_fn(variables, x)
    pred_size = np.asarray(outputs[SIZE_KEY], dtype=np.float32)

    is_positive = (gt_heatmap[..., 0] == 1.0)  # (B,H,W)

    gt_w_all.append(gt_size[..., 0][is_positive])
    gt_h_all.append(gt_size[..., 1][is_positive])
    pred_w_all.append(pred_size[..., 0][is_positive])
    pred_h_all.append(pred_size[..., 1][is_positive])

gt_w = np.concatenate(gt_w_all)
gt_h = np.concatenate(gt_h_all)
pred_w = np.concatenate(pred_w_all)
pred_h = np.concatenate(pred_h_all)

n = gt_w.size
print(f"\n=== Sur {N_BATCHES} batches de validation (batch_size=32) ===")
print(f"Nombre de vrais pixels-centres : {n}")

if n == 0:
    print("Aucun pixel positif trouve - augmenter N_BATCHES.")
    sys.exit(0)

print(f"\nTailles REELLES (pixels, repere 224x224) :")
print(f"  largeur : min={gt_w.min():.1f} max={gt_w.max():.1f} mean={gt_w.mean():.1f} median={np.median(gt_w):.1f}")
print(f"  hauteur : min={gt_h.min():.1f} max={gt_h.max():.1f} mean={gt_h.mean():.1f} median={np.median(gt_h):.1f}")

print(f"\nTailles PREDITES (aux memes pixels-centres) :")
print(f"  largeur : min={pred_w.min():.1f} max={pred_w.max():.1f} mean={pred_w.mean():.1f} median={np.median(pred_w):.1f}")
print(f"  hauteur : min={pred_h.min():.1f} max={pred_h.max():.1f} mean={pred_h.mean():.1f} median={np.median(pred_h):.1f}")

mae_w = np.mean(np.abs(pred_w - gt_w))
mae_h = np.mean(np.abs(pred_h - gt_h))
rel_w = np.mean(np.abs(pred_w - gt_w) / np.maximum(gt_w, 1e-6)) * 100
rel_h = np.mean(np.abs(pred_h - gt_h) / np.maximum(gt_h, 1e-6)) * 100

print(f"\nErreur absolue moyenne (MAE) : largeur={mae_w:.2f}px, hauteur={mae_h:.2f}px")
print(f"Erreur relative moyenne : largeur={rel_w:.1f}%, hauteur={rel_h:.1f}%")

# Comparaison a une base triviale : predire la moyenne des tailles reelles pour tout le monde
baseline_mae_w = np.mean(np.abs(gt_w.mean() - gt_w))
baseline_mae_h = np.mean(np.abs(gt_h.mean() - gt_h))
print(f"\nBase triviale (predire la moyenne constante) : MAE largeur={baseline_mae_w:.2f}px, hauteur={baseline_mae_h:.2f}px")
print(f"  -> si le MAE du modele est proche ou pire que cette base triviale : le modele n'a")
print(f"     pas encore appris a differencier les tailles, juste une valeur moyenne constante.")
print(f"  -> si nettement meilleur que la base triviale : le modele apprend reellement a")
print(f"     distinguer les tailles selon le contenu de l'image.")

improvement_w = (1 - mae_w / baseline_mae_w) * 100 if baseline_mae_w > 0 else float('nan')
improvement_h = (1 - mae_h / baseline_mae_h) * 100 if baseline_mae_h > 0 else float('nan')
print(f"\nAmelioration relative vs base triviale : largeur={improvement_w:.1f}%, hauteur={improvement_h:.1f}%")

corr_w = np.corrcoef(pred_w, gt_w)[0, 1] if n > 1 else float('nan')
corr_h = np.corrcoef(pred_h, gt_h)[0, 1] if n > 1 else float('nan')
print(f"\nCorrelation predite/reelle : largeur={corr_w:.3f}, hauteur={corr_h:.3f}")
print(f"  (1.0 = parfait, 0.0 = aucune relation, negatif = relation inversee)")

print(f"\n=== Quelques exemples individuels (5 premiers) ===")
for i in range(min(5, n)):
    print(f"  reel=({gt_w[i]:.1f},{gt_h[i]:.1f})  predit=({pred_w[i]:.1f},{pred_h[i]:.1f})")
