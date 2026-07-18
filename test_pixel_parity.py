"""
Story 8.1 : validation de parite pixel entre le chemin JAX (RESIZE via jax.image.resize,
CROP via jax.scipy.ndimage.map_coordinates) et le chemin de reference existant (PIL/LANCZOS
pour la preparation des chunks JAX_DETECTOR, cv2.resize pour FIGHTERJET_CLASSIFICATION).
Mesure numerique (MAE, ecart max), pas une comparaison visuelle - AC1/AC2/AC3.

Images de test : test_media/testvid0{1,2,3}.png (1920x1080 reelles), boites reelles
test_media/testvid0{1,2,3}_*.json (meme format que Story 7.1, data["annotation"]["bbox"]).

Usage: python3 test_pixel_parity.py
"""
import glob
import json
import os

import cv2
import numpy as np
from PIL import Image

import jax
import jax.numpy as jnp
from jax.scipy.ndimage import map_coordinates

TEST_MEDIA_DIR = os.path.join(os.path.dirname(__file__), "test_media")
IMAGES = ["testvid01.png", "testvid02.png", "testvid03.png"]


def load_boxes(image_stem):
    boxes = []
    for jf in sorted(glob.glob(os.path.join(TEST_MEDIA_DIR, f"{image_stem}_*.json"))):
        with open(jf) as f:
            d = json.load(f)
        boxes.append(d["annotation"]["bbox"])  # [x, y, w, h]
    return boxes


# =============================================================================
# Task 2 : RESIZE 1920x1080 -> 224x224, grayscale (JAX_DETECTOR chunk prep path)
# =============================================================================

def test_resize_parity():
    print("=" * 70)
    print("TASK 2 : RESIZE 1920x1080 -> 224x224 (grayscale)")
    print("=" * 70)

    global_best = {}  # (method, antialias) -> liste de MAE sur toutes les images

    for img_name in IMAGES:
        path = os.path.join(TEST_MEDIA_DIR, img_name)
        pil_img = Image.open(path).convert("L")
        assert pil_img.size == (1920, 1080), f"{img_name} n'est pas 1920x1080 : {pil_img.size}"

        pil_resized = pil_img.resize((224, 224), Image.Resampling.LANCZOS)
        pil_arr = np.array(pil_resized, dtype=np.float32)

        src_arr = np.array(pil_img, dtype=np.float32)  # (1080, 1920)
        jax_input = jnp.asarray(src_arr)[..., None]  # (1080, 1920, 1)

        print(f"\n--- {img_name} ---")
        for method in ["linear", "lanczos3"]:
            for antialias in [True, False]:
                jax_resized = jax.image.resize(jax_input, (224, 224, 1), method=method, antialias=antialias)
                jax_arr = np.array(jax_resized[..., 0])
                diff = np.abs(jax_arr - pil_arr)
                mae, max_err = float(diff.mean()), float(diff.max())
                key = (method, antialias)
                global_best.setdefault(key, []).append(mae)
                print(f"  method={method:<9} antialias={str(antialias):<5} : MAE={mae:6.3f}  MAX={max_err:6.1f}  (echelle [0,255])")

    print("\n--- Moyenne sur les 3 images ---")
    ranked = sorted(global_best.items(), key=lambda kv: np.mean(kv[1]))
    for (method, antialias), maes in ranked:
        print(f"  method={method:<9} antialias={str(antialias):<5} : MAE moyen={np.mean(maes):6.3f}")

    best_method, best_antialias = ranked[0][0]
    print(f"\n=> Meilleure combinaison : method={best_method}, antialias={best_antialias} (MAE moyen={np.mean(ranked[0][1]):.3f}/255)")
    return best_method, best_antialias, ranked


# =============================================================================
# CROP helper : map_coordinates avec convention demi-pixel (coordonnees GLOBALES,
# pas de slicing prealable - representatif de ce que fera reellement Story 8.5,
# JIT/vmap-compatible, contrairement a un slice Python a bornes dynamiques)
# =============================================================================

def _map_coordinates_crop(image_color, x1, y1, x2, y2, out_size=128, mode="constant"):
    """image_color: (H,W,C) jnp array. x1,y1,x2,y2: coords GLOBALES (peuvent etre flottantes)."""
    scale_x = (x2 - x1) / out_size
    scale_y = (y2 - y1) / out_size
    dst_y, dst_x = jnp.meshgrid(jnp.arange(out_size), jnp.arange(out_size), indexing="ij")
    src_x = x1 + (dst_x.astype(jnp.float32) + 0.5) * scale_x - 0.5
    src_y = y1 + (dst_y.astype(jnp.float32) + 0.5) * scale_y - 0.5

    channels = []
    for c in range(image_color.shape[-1]):
        ch = map_coordinates(image_color[..., c], [src_y, src_x], order=1, mode=mode, cval=0.0)
        channels.append(ch)
    return jnp.stack(channels, axis=-1)


# =============================================================================
# Task 3 : CROP, boite ENTIERE (chemin exact FIGHTERJET_CLASSIFICATION, inference_utils.py:159)
# =============================================================================

def test_crop_parity_integer_box():
    print("\n" + "=" * 70)
    print("TASK 3 : CROP + resize 128x128, boite ENTIERE (chemin cv2 exact)")
    print("=" * 70)
    print("Teste sur TOUTES les boites reelles disponibles (21, 7 par image), pas une seule.")

    all_mae, all_max = [], []
    for img_name in IMAGES:
        stem = img_name.replace(".png", "")
        cv_img = cv2.imread(os.path.join(TEST_MEDIA_DIR, img_name))
        jax_img = jnp.asarray(cv_img.astype(np.float32))
        for bx, by, bw, bh in load_boxes(stem):
            x1, y1, x2, y2 = int(bx), int(by), int(bx + bw), int(by + bh)
            crop_resized_cv = cv2.resize(cv_img[y1:y2, x1:x2], (128, 128))
            ref = crop_resized_cv.astype(np.float32)
            jax_out = np.array(_map_coordinates_crop(jax_img, float(x1), float(y1), float(x2), float(y2), out_size=128))
            diff = np.abs(jax_out - ref)
            all_mae.append(diff.mean())
            all_max.append(diff.max())

    print(f"Sur {len(all_mae)} boites : MAE moyen={np.mean(all_mae):.3f}, MAE max observe={np.max(all_mae):.3f}, "
          f"MAX ecart moyen={np.mean(all_max):.1f}, MAX ecart pire cas={np.max(all_max):.1f}  (echelle [0,255])")

    # Retourne aussi la premiere boite en detail (compatibilite avec le reste du script)
    bx, by, bw, bh = load_boxes("testvid01")[0]
    x1, y1, x2, y2 = int(bx), int(by), int(bx + bw), int(by + bh)
    return np.mean(all_mae), np.max(all_max), (x1, y1, x2, y2)


# =============================================================================
# Task 4 : CROP, boite FLOTTANTE (simule une sortie RESCALE, Story 8.4 - rarement entiere)
# =============================================================================

def test_crop_parity_float_box(int_mae):
    print("\n" + "=" * 70)
    print("TASK 4 : CROP + resize 128x128, boite FLOTTANTE (simule RESCALE)")
    print("=" * 70)
    print("Teste sur les 21 boites reelles, chacune perturbee par un offset sous-pixel")
    print("non-trivial (simule ce qu'un vrai RESCALE, Story 8.4, produirait).")

    rng = np.random.default_rng(0)
    all_float_vs_ref, all_int_vs_ref, all_float_vs_int = [], [], []

    for img_name in IMAGES:
        stem = img_name.replace(".png", "")
        cv_img = cv2.imread(os.path.join(TEST_MEDIA_DIR, img_name))
        jax_img = jnp.asarray(cv_img.astype(np.float32))
        for bx, by, bw, bh in load_boxes(stem):
            offsets = rng.uniform(-0.7, 0.7, size=4)
            x1f, y1f, x2f, y2f = bx + offsets[0], by + offsets[1], bx + bw + offsets[2], by + bh + offsets[3]
            x1i, y1i, x2i, y2i = int(bx), int(by), int(bx + bw), int(by + bh)

            ref = cv2.resize(cv_img[y1i:y2i, x1i:x2i], (128, 128)).astype(np.float32)
            crop_float = np.array(_map_coordinates_crop(jax_img, x1f, y1f, x2f, y2f, out_size=128))
            crop_int = np.array(_map_coordinates_crop(jax_img, float(x1i), float(y1i), float(x2i), float(y2i), out_size=128))

            all_float_vs_ref.append(np.abs(crop_float - ref).mean())
            all_int_vs_ref.append(np.abs(crop_int - ref).mean())
            all_float_vs_int.append(np.abs(crop_float - crop_int).mean())

    mae_float_vs_ref = np.mean(all_float_vs_ref)
    mae_int_vs_ref = np.mean(all_int_vs_ref)
    print(f"\nSur {len(all_float_vs_ref)} boites :")
    print(f"MAE(map_coordinates flottant  vs cv2 reference) moyen = {mae_float_vs_ref:.3f}")
    print(f"MAE(map_coordinates entier    vs cv2 reference) moyen = {mae_int_vs_ref:.3f}  (= Task 3)")
    print(f"MAE(map_coordinates flottant  vs map_coordinates entier) moyen = {np.mean(all_float_vs_int):.3f}")

    ratio = mae_float_vs_ref / max(mae_int_vs_ref, 1e-6)
    print(f"\nRatio (float_vs_ref / int_vs_ref) = {ratio:.2f}x")
    if ratio > 1.5:
        print("=> Les coordonnees flottantes s'ELOIGNENT mesurablement de la reference cv2 (boite arrondie plus fidele).")
        print("   Confirme le piege signale en Dev Notes (ne pas repeter le choix sous-pixel de la Story 7.1 ici) :")
        print("   ARRONDIR les boites a l'entier avant crop (Story 8.5), pour rester fidele a la distribution")
        print("   d'entrainement de FIGHTERJET_CLASSIFICATION (modele fige, entraine sur des crops issus d'un slice entier).")
    else:
        print("=> Ecart flottant/entier proche de la reference - garder les flottants serait acceptable.")
    return ratio


# =============================================================================
# Task 5 : boite en bord de cadre - mode='constant' vs mode='nearest'
# =============================================================================

def test_out_of_bounds_box():
    print("\n" + "=" * 70)
    print("TASK 5 : boite partiellement hors cadre - mode='constant' vs 'nearest'")
    print("=" * 70)

    img_name = "testvid02.png"
    boxes = load_boxes("testvid02")
    bx, by, bw, bh = boxes[2]  # f16, [645,47,105,70] - proche du bord haut (y=47)
    # Construire une boite deliberement hors cadre : decalee pour depasser y=0 et x=1920
    x1, y1 = 1920 - bw * 0.3, -bh * 0.4  # depasse a droite ET en haut
    x2, y2 = x1 + bw, y1 + bh
    print(f"Boite testee (hors cadre) : [{x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f}] (image {img_name}, {1920}x{1080})")

    cv_img = cv2.imread(os.path.join(TEST_MEDIA_DIR, img_name))
    jax_img = jnp.asarray(cv_img.astype(np.float32))

    crop_constant = _map_coordinates_crop(jax_img, x1, y1, x2, y2, out_size=128, mode="constant")
    crop_nearest = _map_coordinates_crop(jax_img, x1, y1, x2, y2, out_size=128, mode="nearest")

    arr_constant = np.array(crop_constant)
    arr_nearest = np.array(crop_nearest)

    n_zero_pixels_constant = int(np.all(arr_constant == 0.0, axis=-1).sum())
    n_zero_pixels_nearest = int(np.all(arr_nearest == 0.0, axis=-1).sum())
    diff = np.abs(arr_constant - arr_nearest)

    print(f"Pixels totalement noirs (0,0,0) : constant={n_zero_pixels_constant}/{128*128}, nearest={n_zero_pixels_nearest}/{128*128}")
    print(f"Ecart moyen constant vs nearest : {diff.mean():.3f}")
    print("=> 'nearest' repete le bord (pas de zone noire artificielle) - recommande pour un avion partiellement hors champ,")
    print("   'constant' introduit une zone noire nette qui ne correspond a aucun contenu reel de l'image.")

    cv2.imwrite("test_pixel_parity_oob_constant.png", arr_constant.astype(np.uint8))
    cv2.imwrite("test_pixel_parity_oob_nearest.png", arr_nearest.astype(np.uint8))
    print("Images sauvegardees : test_pixel_parity_oob_constant.png / test_pixel_parity_oob_nearest.png")


if __name__ == "__main__":
    best_method, best_antialias, resize_ranking = test_resize_parity()
    crop_mae, crop_max, box = test_crop_parity_integer_box()
    test_crop_parity_float_box(crop_mae)
    test_out_of_bounds_box()

    print("\n" + "=" * 70)
    print("RESUME")
    print("=" * 70)
    print(f"RESIZE : meilleure methode = {best_method}, antialias={best_antialias}")
    print(f"CROP (boite entiere) : MAE={crop_mae:.3f}/255, MAX={crop_max:.1f}/255")
