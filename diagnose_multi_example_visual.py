"""
Complement a generate_reports (Story 7.6), qui ne visualise structurellement qu'UN
seul exemple (val_ds.take(1), premier de premier batch) - insuffisant pour la Task 4
de la Story 7.8 ("confirmer que les pics predits correspondent visuellement a des
avions reels sur AU MOINS QUELQUES exemples"). Ce script visualise plusieurs exemples
distincts, en privilegiant la diversite (dont un exemple multi-avions si disponible
dans l'echantillon - premier apercu informel, pas une validation Epic 8/CAP-3).

Pour chaque exemple : image, heatmap vrai, heatmap predit, et une boite VRAIE (verte)
vs PREDITE (rouge, meme pixel-centre reel + taille predite a ce pixel - teste la tete
de taille isolement, pas l'extraction de pics qui est le travail d'Epic 8/Story 8.3).

Usage: python3 diagnose_multi_example_visual.py [checkpoint_path] [n_examples]
"""
import pickle
import sys

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from model_library import get_model
from dataset_configs import get_dataset_config
from data_management import CenterNetDetectionDataset
from detection_target_encoding import HEATMAP_KEY, SIZE_KEY

CHECKPOINT_PATH = sys.argv[1] if len(sys.argv) > 1 else "best_model_jax_detector.pkl"
N_EXAMPLES = int(sys.argv[2]) if len(sys.argv) > 2 else 4
OUTPUT_PATH = "diagnose_multi_example_visual.png"

with open(CHECKPOINT_PATH, "rb") as f:
    ckpt = pickle.load(f)

config = ckpt["config"]
print(f"Checkpoint : dataset_name={config.get('dataset_name')}, model_name={config.get('model_name')}")

model = get_model(config["model_name"], num_classes=config["num_classes"], dropout_rate=0.0)
variables = {"params": ckpt["params"], "batch_stats": ckpt["batch_stats"]}
ckpt_dtype = jax.tree_util.tree_leaves(ckpt["params"])[0].dtype

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

# Collecter des candidats sur plusieurs batches, avec leur nombre de vrais centres
candidates = []  # (n_boxes, image, gt_heatmap, gt_size, pred_heatmap, pred_size)
for images, targets in val_ds.take(6):
    images_np = images.numpy()
    gt_heatmap = targets[HEATMAP_KEY].numpy()
    gt_size = targets[SIZE_KEY].numpy()

    x = jnp.asarray(images_np, dtype=ckpt_dtype)
    outputs = apply_fn(variables, x)
    pred_heatmap = np.asarray(outputs[HEATMAP_KEY], dtype=np.float32)
    pred_size = np.asarray(outputs[SIZE_KEY], dtype=np.float32)

    for i in range(images_np.shape[0]):
        n_boxes = int((gt_heatmap[i, ..., 0] == 1.0).sum())
        if n_boxes == 0:
            continue
        candidates.append((n_boxes, images_np[i], gt_heatmap[i], gt_size[i], pred_heatmap[i], pred_size[i]))

candidates.sort(key=lambda c: c[0], reverse=True)  # les plus riches en boites d'abord
print(f"{len(candidates)} exemples candidats (avec >=1 vraie box) sur les batches echantillonnes")
print(f"Distribution du nombre de boites/image (top 10) : {[c[0] for c in candidates[:10]]}")

# Selection : le plus multi-avions dispo, puis des exemples varies (pas tous les memes)
selected_indices = []
if candidates:
    selected_indices.append(0)  # le plus riche en boites
step = max(1, len(candidates) // N_EXAMPLES)
for idx in range(0, len(candidates), step):
    if len(selected_indices) >= N_EXAMPLES:
        break
    if idx not in selected_indices:
        selected_indices.append(idx)
selected = [candidates[i] for i in selected_indices[:N_EXAMPLES]]

print(f"\n{len(selected)} exemples selectionnes, nombre de boites : {[s[0] for s in selected]}")

fig, axes = plt.subplots(len(selected), 4, figsize=(16, 4 * len(selected)))
if len(selected) == 1:
    axes = axes.reshape(1, -1)

for row, (n_boxes, img, gt_hm, gt_sz, pred_hm, pred_sz) in enumerate(selected):
    img2d = img[..., 0]
    axes[row, 0].imshow(img2d, cmap="gray")
    axes[row, 0].set_title(f"Image ({n_boxes} avion(s) reel(s))")
    axes[row, 0].axis("off")

    axes[row, 1].imshow(gt_hm[..., 0], cmap="hot", vmin=0, vmax=1)
    axes[row, 1].set_title("Heatmap VRAI")
    axes[row, 1].axis("off")

    axes[row, 2].imshow(pred_hm[..., 0], cmap="hot", vmin=0, vmax=1)
    axes[row, 2].set_title(f"Heatmap PREDIT (max={pred_hm[...,0].max():.3f})")
    axes[row, 2].axis("off")

    axes[row, 3].imshow(img2d, cmap="gray")
    axes[row, 3].set_title("Boites : verte=vraie, rouge=predite (au vrai centre)")
    axes[row, 3].axis("off")
    ys, xs = np.where(gt_hm[..., 0] == 1.0)
    for cy, cx in zip(ys, xs):
        gw, gh = gt_sz[cy, cx, 0], gt_sz[cy, cx, 1]
        pw, ph = pred_sz[cy, cx, 0], pred_sz[cy, cx, 1]
        axes[row, 3].add_patch(patches.Rectangle((cx - gw/2, cy - gh/2), gw, gh, linewidth=1.5, edgecolor="lime", facecolor="none"))
        axes[row, 3].add_patch(patches.Rectangle((cx - pw/2, cy - ph/2), pw, ph, linewidth=1.5, edgecolor="red", facecolor="none", linestyle="--"))
        axes[row, 3].plot(cx, cy, "b+", markersize=8)

plt.tight_layout()
plt.savefig(OUTPUT_PATH, dpi=100)
print(f"\nVisualisation sauvegardee : {OUTPUT_PATH}")
