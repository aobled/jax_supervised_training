"""
Story 8.4 : verifie que _rescale_boxes (inference_utils.py) est l'inverse EXACT de la
convention demi-pixel de RESIZE (Story 8.1/8.2), pas une simple multiplication - la
methode de verification simule RESIZE en avant sur une coordonnee source connue puis
verifie que _rescale_boxes la retrouve, plutot que de comparer deux formules
potentiellement identiquement fausses (mise en garde explicite de la story, Task 3).

Usage: python3 test_rescale_boxes.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import inspect

import jax.numpy as jnp

from inference_utils import _rescale_boxes

DETECTOR_SIZE = (224, 224)    # (W, H)
ORIGINAL_SIZE = (1920, 1080)  # (W, H)
TOLERANCE = 1e-3


def _forward_resize_point(src, scale_original_to_detector):
    """Simule RESIZE en avant sur UNE coordonnee (convention demi-pixel, Story 8.1/8.2) -
    calcul independant de _rescale_boxes, pas une reutilisation de son code."""
    return (src + 0.5) * scale_original_to_detector - 0.5


def test_round_trip_recovers_source_coordinate_x():
    detector_w, _ = DETECTOR_SIZE
    original_w, _ = ORIGINAL_SIZE
    scale_fwd = detector_w / original_w

    for src_x in [0.0, 3.79, 500.0, 960.0, 1500.0, 1915.21, 1919.0]:
        dst_x = _forward_resize_point(src_x, scale_fwd)
        box = jnp.array([[dst_x, 0.0, dst_x, 0.0]])  # x1=x2=dst_x, y arbitraire ici
        rescaled = _rescale_boxes(box, DETECTOR_SIZE, ORIGINAL_SIZE)
        recovered_x = float(rescaled[0, 0])
        assert abs(recovered_x - src_x) < TOLERANCE, (
            f"src_x={src_x} -> dst_x={dst_x:.4f} -> recupere {recovered_x:.4f}, "
            f"ecart {abs(recovered_x - src_x):.4f}px hors tolerance {TOLERANCE}px"
        )
    print("OK - test_round_trip_recovers_source_coordinate_x (7 points, dont extremes)")


def test_round_trip_recovers_source_coordinate_y():
    _, detector_h = DETECTOR_SIZE
    _, original_h = ORIGINAL_SIZE
    scale_fwd = detector_h / original_h

    for src_y in [0.0, 1.91, 300.0, 540.0, 900.0, 1077.09, 1079.0]:
        dst_y = _forward_resize_point(src_y, scale_fwd)
        box = jnp.array([[0.0, dst_y, 0.0, dst_y]])
        rescaled = _rescale_boxes(box, DETECTOR_SIZE, ORIGINAL_SIZE)
        recovered_y = float(rescaled[0, 1])
        assert abs(recovered_y - src_y) < TOLERANCE, (
            f"src_y={src_y} -> dst_y={dst_y:.4f} -> recupere {recovered_y:.4f}, "
            f"ecart {abs(recovered_y - src_y):.4f}px hors tolerance {TOLERANCE}px"
        )
    print("OK - test_round_trip_recovers_source_coordinate_y (7 points, dont extremes)")


def test_naive_multiplication_would_fail_this_test():
    """Garde-fou explicite : confirme que le bug identifie en Task 1 (x*scale au lieu de
    (x+0.5)*scale-0.5) aurait ete detecte par le round-trip ci-dessus - documente pourquoi
    ce test est le bon, pas juste une affirmation non verifiee."""
    detector_w, _ = DETECTOR_SIZE
    original_w, _ = ORIGINAL_SIZE
    scale_fwd = detector_w / original_w
    scale_inv = original_w / detector_w

    src_x = 500.0
    dst_x = _forward_resize_point(src_x, scale_fwd)
    naive_recovered = dst_x * scale_inv  # le bug Task 1 : pas de correction +0.5/-0.5
    error = abs(naive_recovered - src_x)
    assert error > 3.0, (
        f"le calcul naif aurait du s'ecarter de plusieurs pixels (~3.8px en x attendu), "
        f"obtenu {error:.4f}px - le garde-fou ne discrimine plus, a revoir"
    )
    print(f"OK - test_naive_multiplication_would_fail_this_test (naif ecarte de {error:.4f}px, round-trip discrimine bien)")


def test_corner_boxes_224_grid():
    """Boites aux coins du cadre detecteur (x=0, x=223), pas seulement le centre (AC1) -
    valeurs attendues calculees independamment via la formule inverse (pas via le code de
    _rescale_boxes lui-meme)."""
    box = jnp.array([
        [0.0, 0.0, 0.0, 0.0],
        [223.0, 223.0, 223.0, 223.0],
    ])
    rescaled = _rescale_boxes(box, DETECTOR_SIZE, ORIGINAL_SIZE)

    expected = jnp.array([
        [3.785714, 1.910714, 3.785714, 1.910714],
        [1915.214286, 1077.089286, 1915.214286, 1077.089286],
    ])
    assert jnp.allclose(rescaled, expected, atol=1e-3), (
        f"coins du cadre detecteur mal reconvertis : obtenu {rescaled}, attendu {expected}"
    )
    print("OK - test_corner_boxes_224_grid (x=0/x=223, y=0/y=223)")


def test_valid_mask_and_scores_not_touched():
    """Task 2 : _rescale_boxes ne prend et ne retourne QUE des coordonnees - confirme que
    sa signature n'a pas de parametre valid_mask/scores (rien ne peut donc etre modifie,
    ces champs ne rentrent jamais dans cette fonction)."""
    sig = inspect.signature(_rescale_boxes)
    param_names = set(sig.parameters.keys())
    assert "valid_mask" not in param_names and "scores" not in param_names, (
        f"_rescale_boxes ne doit prendre que des coordonnees, signature obtenue: {sig}"
    )
    print("OK - test_valid_mask_and_scores_not_touched (signature confirmee coordonnees-seules)")


if __name__ == "__main__":
    test_round_trip_recovers_source_coordinate_x()
    test_round_trip_recovers_source_coordinate_y()
    test_naive_multiplication_would_fail_this_test()
    test_corner_boxes_224_grid()
    test_valid_mask_and_scores_not_touched()
    print("Tous les tests sont passés.")
