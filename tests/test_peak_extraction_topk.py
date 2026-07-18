"""
Story 8.3 : verifie l'extraction de pics + Top-K JAX-native (_extract_peaks/_top_k_boxes,
inference_utils.py) contre le nombre de pics REELLEMENT injectes (verite terrain
independante de tout chemin de decodage) et contre le decodage NumPy hors-ligne existant
(decode_detection_targets, Story 7.1) - une seule comparaison croisee des deux chemins ne
detecterait pas un sur-comptage sur un plateau de valeurs egales adjacentes (voir
test_peak_plateau_degenerate_case).

Usage: python3 test_peak_extraction_topk.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax.numpy as jnp
import numpy as np

from inference_utils import _extract_peaks, _top_k_boxes
from detection_target_encoding import encode_detection_targets, decode_detection_targets
from dataset_configs import DATASET_CONFIGS

TARGET_SIZE = (224, 224)
ORIG_W, ORIG_H = 1920, 1080
DETECTION_SCORE_THRESHOLD = DATASET_CONFIGS["JAX_DETECTOR"]["detection_score_threshold"]
TOLERANCE_PX = 1.0


def _decode_via_jax_native(heatmap_np, size_np, k=20, score_threshold=DETECTION_SCORE_THRESHOLD):
    """Chemin JAX-natif complet : _extract_peaks -> _top_k_boxes -> valid_mask (Task 4)."""
    heatmap = jnp.asarray(heatmap_np)
    size = jnp.asarray(size_np)
    filtered = _extract_peaks(heatmap)
    boxes, scores = _top_k_boxes(filtered, size, k=k)
    valid_mask = scores > score_threshold
    valid_boxes = np.asarray(boxes)[np.asarray(valid_mask)]
    valid_scores = np.asarray(scores)[np.asarray(valid_mask)]
    return valid_boxes, valid_scores


def _count_true_peaks(heatmap_np):
    """Nombre de pixels a exactement 1.0 (sommet du noyau gaussien, Story 7.1) - verite
    terrain independante de tout chemin de decodage."""
    return int(np.sum(heatmap_np[:, :, 0] == 1.0))


def _boxes_close(box_a, box_b, tol=TOLERANCE_PX):
    return all(abs(a - b) <= tol for a, b in zip(box_a, box_b))


def test_zero_boxes():
    targets = encode_detection_targets([], ORIG_W, ORIG_H, TARGET_SIZE)
    valid_boxes, valid_scores = _decode_via_jax_native(targets["heatmap"], targets["size"])
    assert len(valid_boxes) == 0, f"attendu 0 boite, obtenu {len(valid_boxes)}"
    print("OK - test_zero_boxes")


def test_one_box():
    raw_boxes = [(500, 300, 200, 150)]
    targets = encode_detection_targets(raw_boxes, ORIG_W, ORIG_H, TARGET_SIZE)

    true_peak_count = _count_true_peaks(targets["heatmap"])
    assert true_peak_count == 1, f"attendu 1 pic reel (verite terrain), obtenu {true_peak_count}"

    valid_boxes, valid_scores = _decode_via_jax_native(targets["heatmap"], targets["size"])
    assert len(valid_boxes) == 1, f"attendu 1 boite valide, obtenu {len(valid_boxes)}"

    decoded_numpy = decode_detection_targets(targets["heatmap"], targets["size"], score_threshold=DETECTION_SCORE_THRESHOLD)
    assert len(decoded_numpy) == 1
    assert _boxes_close(valid_boxes[0], decoded_numpy[0][:4]), (
        f"chemin JAX-natif {valid_boxes[0]} vs decode_detection_targets {decoded_numpy[0][:4]} "
        f"hors tolerance {TOLERANCE_PX}px"
    )
    print(f"OK - test_one_box (1 pic reel, boite {valid_boxes[0]} coherente avec decode_detection_targets)")


def test_multiple_close_boxes():
    # 2 avions en formation serree (cas AD-19), memes boites que test_multiple_close_boxes
    # de la Story 7.1 (test_detection_target_encoding.py) - centres distincts au pixel pres.
    raw_boxes = [(500, 300, 120, 90), (560, 320, 120, 90)]
    targets = encode_detection_targets(raw_boxes, ORIG_W, ORIG_H, TARGET_SIZE)

    true_peak_count = _count_true_peaks(targets["heatmap"])
    assert true_peak_count == 2, f"attendu 2 pics reels, obtenu {true_peak_count}"

    valid_boxes, valid_scores = _decode_via_jax_native(targets["heatmap"], targets["size"])
    assert len(valid_boxes) == 2, f"attendu 2 boites valides, obtenu {len(valid_boxes)}"

    decoded_numpy = decode_detection_targets(targets["heatmap"], targets["size"], score_threshold=DETECTION_SCORE_THRESHOLD)
    assert len(decoded_numpy) == 2
    remaining = list(decoded_numpy)
    for box in valid_boxes:
        best = min(remaining, key=lambda d: sum(abs(a - b) for a, b in zip(box, d[:4])))
        assert _boxes_close(box, best[:4]), f"{box} vs {best[:4]} hors tolerance {TOLERANCE_PX}px"
        remaining.remove(best)
    print("OK - test_multiple_close_boxes (2 pics reels, formation serree AD-19, coherent avec decode_detection_targets)")


def test_more_than_20_real_peaks_capped_at_20():
    """AC3 : plus de 20 detections reelles au-dessus du seuil -> les 20 de plus haute
    confiance conservees, le reste ecarte SANS ERREUR (pas de plantage/branche speciale)."""
    # 25 boites espacees pour eviter toute fusion de pics (AD-19). Aires non differenciees
    # volontairement : _draw_gaussian plafonne toujours le pic a 1.0 (max, pas somme), donc
    # tous les pics reels sont a egalite stricte - seul le NOMBRE de detections valides
    # apres plafond nous interesse ici, pas lesquelles specifiquement (le choix parmi des
    # pics a egalite stricte n'est pas un contrat de cette story).
    raw_boxes = [(50 + i * 70, 50 + (i % 5) * 180, 40, 30) for i in range(25)]
    targets = encode_detection_targets(raw_boxes, ORIG_W, ORIG_H, TARGET_SIZE, max_boxes=25)

    true_peak_count = _count_true_peaks(targets["heatmap"])
    assert true_peak_count == 25, f"attendu 25 pics reels injectes, obtenu {true_peak_count}"

    valid_boxes, valid_scores = _decode_via_jax_native(targets["heatmap"], targets["size"], k=20)
    assert len(valid_boxes) == 20, f"attendu 20 boites (plafond Top-K), obtenu {len(valid_boxes)}"
    print("OK - test_more_than_20_real_peaks_capped_at_20 (25 pics reels injectes -> 20 conserves, aucune erreur)")


def test_peak_plateau_degenerate_case():
    """Cas de plateau construit a la main (explicitement demande par la story) : plusieurs
    pixels adjacents a la meme valeur non nulle. La comparaison stricte hm==hmax garde TOUS
    les pixels du plateau comme "pics" (limite connue de la technique peak-NMS par egalite
    stricte, documentee ici plutot que decouverte silencieusement en Story 8.6) - ce test
    verifie que le compte reste borne/raisonnable (4, la taille du plateau), pas qu'il
    explose au-dela."""
    hm = np.zeros((10, 10, 1), dtype=np.float32)
    hm[4:6, 4:6, 0] = 0.5  # plateau 2x2 non nul
    heatmap = jnp.asarray(hm)
    filtered = _extract_peaks(heatmap)
    peak_count = int(jnp.sum(filtered > 0.0))
    assert peak_count == 4, (
        f"plateau 2x2 attendu -> 4 pixels-pics par egalite stricte (limite connue, "
        f"documentee), obtenu {peak_count}"
    )
    print(f"OK - test_peak_plateau_degenerate_case ({peak_count} pixels-pics sur un plateau 2x2, comportement borne et documente)")


if __name__ == "__main__":
    test_zero_boxes()
    test_one_box()
    test_multiple_close_boxes()
    test_more_than_20_real_peaks_capped_at_20()
    test_peak_plateau_degenerate_case()
    print("Tous les tests sont passés.")
