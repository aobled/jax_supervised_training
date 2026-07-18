"""
Diagnostic Story 7.8 (post-hoc) : le modele progresse-t-il reellement aux vrais
pixels-centres, meme sous le seuil HeatmapRecall (0.5) qui les masque ?

HeatmapRecall (task_strategies.py::CenterNetDetectionStrategy.compute_metrics) utilise
un seuil dur (pred > 0.5) - si le modele apprend a faire monter ses predictions aux
vrais centres sans encore franchir ce seuil, HeatmapRecall reste a 0.0000 alors qu'un
vrai progres a lieu. Ce script compare directement la distribution des valeurs predites
aux vrais pixels-centres (gt_heatmap == 1.0) contre un echantillon de pixels de fond,
sur plusieurs batches de validation reels, avec le checkpoint deja sauvegarde.

Usage: python3 diagnose_heatmap_predictions.py [checkpoint_path] [n_batches]
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

# Reproduire la precision d'entrainement (float16 sur TPU d'apres les logs) - eviter
# une comparaison biaisee par une promotion de type implicite en float32
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

all_pos_preds = []
all_neg_sample = []
rng = np.random.default_rng(0)

for i, (images, targets) in enumerate(val_ds.take(N_BATCHES)):
    images_np = images.numpy()
    gt_heatmap = targets[HEATMAP_KEY].numpy()

    x = jnp.asarray(images_np, dtype=ckpt_dtype)
    outputs = apply_fn(variables, x)
    pred_heatmap = np.asarray(outputs[HEATMAP_KEY], dtype=np.float32)

    is_positive = (gt_heatmap == 1.0)
    pos_preds = pred_heatmap[is_positive]
    all_pos_preds.append(pos_preds)

    neg_all = pred_heatmap[~is_positive]
    if neg_all.size > 0:
        idx = rng.choice(neg_all.size, size=min(5000, neg_all.size), replace=False)
        all_neg_sample.append(neg_all[idx])

pos_preds = np.concatenate(all_pos_preds) if all_pos_preds else np.array([])
neg_sample = np.concatenate(all_neg_sample) if all_neg_sample else np.array([])

print(f"\n=== Sur {N_BATCHES} batches de validation (batch_size=32) ===")
print(f"Nombre de vrais pixels-centres (gt_heatmap==1.0) : {pos_preds.size}")

if pos_preds.size == 0:
    print("Aucun pixel positif trouve - augmenter N_BATCHES.")
    sys.exit(0)

print(f"\nPredictions AUX VRAIS CENTRES :")
print(f"  min={pos_preds.min():.6f}  max={pos_preds.max():.6f}  mean={pos_preds.mean():.6f}  median={np.median(pos_preds):.6f}")
print(f"  Fraction > 0.5 (= HeatmapRecall a ce seuil) : {(pos_preds > 0.5).mean():.4f}")
for thr in [0.5, 0.3, 0.1, 0.05, 0.01, 0.001]:
    frac = (pos_preds > thr).mean()
    print(f"  fraction > {thr:<6}: {frac:.4f}")

if neg_sample.size > 0:
    print(f"\nPredictions sur un ECHANTILLON DE FOND ({neg_sample.size} pixels) :")
    print(f"  min={neg_sample.min():.6f}  max={neg_sample.max():.6f}  mean={neg_sample.mean():.6f}  median={np.median(neg_sample):.6f}")

    ratio = pos_preds.mean() / (neg_sample.mean() + 1e-12)
    print(f"\nRatio moyenne(positifs) / moyenne(fond) : {ratio:.2f}x")
    print("  -> si ce ratio est proche de 1 : le modele ne distingue PAS encore centres et fond (collapse probable).")
    print("  -> si ce ratio est nettement > 1 (meme sous 0.5) : progres reel invisible pour HeatmapRecall.")
