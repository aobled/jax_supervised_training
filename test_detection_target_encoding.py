"""
Test de round-trip pour detection_target_encoding.py (Story 7.1, AD-18).

Script autonome - ce projet n'a pas de framework de test formel (voir Dev Notes
de la story 7.1 : validation par script, pas de suite CI/CD). Executer directement :
    python test_detection_target_encoding.py
"""

import os
import tempfile

import numpy as np

from detection_target_encoding import (
    encode_detection_targets,
    decode_detection_targets,
    save_detection_targets_npz,
    load_detection_targets_npz,
    HEATMAP_KEY,
    SIZE_KEY,
)

# Tolerance round-trip : le centre est quantifie au pixel entier le plus proche a
# l'encodage (arrondi), ce qui peut decaler chaque coordonnee (x1,y1,x2,y2) recuperee
# d'au plus 0.5px par rapport a l'original. 1.0px de marge est donc la tolerance
# la plus stricte qui reste toujours valide, sans etre arbitraire.
TOLERANCE_PX = 1.0


def _boxes_close(box_a, box_b, tol=TOLERANCE_PX):
    return all(abs(a - b) <= tol for a, b in zip(box_a[:4], box_b[:4]))


def _rescale_expected(raw_boxes, orig_w, orig_h, target_size):
    W, H = target_size
    expected = []
    for bx, by, bw, bh in raw_boxes:
        x1 = (bx / orig_w) * W
        y1 = (by / orig_h) * H
        x2 = ((bx + bw) / orig_w) * W
        y2 = ((by + bh) / orig_h) * H
        expected.append((x1, y1, x2, y2))
    return expected


def _run_case(name, raw_boxes, orig_w, orig_h, target_size):
    targets = encode_detection_targets(raw_boxes, orig_w, orig_h, target_size)
    decoded = decode_detection_targets(targets["heatmap"], targets["size"], score_threshold=0.3)
    expected = _rescale_expected(raw_boxes, orig_w, orig_h, target_size)

    assert len(decoded) == len(expected), (
        f"[{name}] attendu {len(expected)} boite(s), obtenu {len(decoded)}: {decoded}"
    )

    remaining = list(decoded)
    for exp_box in expected:
        best = min(remaining, key=lambda d: sum(abs(a - b) for a, b in zip(exp_box, d[:4])))
        assert _boxes_close(exp_box, best), (
            f"[{name}] {exp_box} vs {best[:4]} hors tolerance {TOLERANCE_PX}px"
        )
        remaining.remove(best)

    print(f"OK - {name} ({len(expected)} boite(s), tolerance {TOLERANCE_PX}px)")


def test_zero_boxes():
    targets = encode_detection_targets([], 1920, 1080, (224, 224))
    decoded = decode_detection_targets(targets["heatmap"], targets["size"], score_threshold=0.3)
    assert decoded == [], f"attendu 0 boite, obtenu {len(decoded)}: {decoded}"
    assert targets["heatmap"].shape == (224, 224, 1)
    assert targets["size"].shape == (224, 224, 2)
    print("OK - 0 boite (shapes verifiees, aucune detection)")


def test_one_box():
    _run_case("1 boite", [(500, 300, 200, 150)], 1920, 1080, (224, 224))


def test_multiple_close_boxes():
    # 2 avions en formation serree (cas AD-19) : centres separes de ~8px en repere 224x224
    _run_case(
        "boites proches (formation serree, AD-19)",
        [(500, 300, 120, 90), (560, 320, 120, 90)],
        1920, 1080, (224, 224),
    )


def test_peak_window_invariance_on_true_targets():
    # Cibles vraies : chaque pic culmine exactement a 1.0, donc deux pics voisins sont
    # toujours a egalite exacte - la comparaison stricte de decode_detection_targets
    # ne supprime jamais un pic a egalite, quelle que soit la taille de fenetre. Ce test
    # rend cette affirmation (documentee dans decode_detection_targets) verifiee, pas
    # seulement une note manuelle non rejouable.
    targets = encode_detection_targets(
        [(500, 300, 120, 90), (560, 320, 120, 90)], 1920, 1080, (224, 224)
    )
    counts = {
        pw: len(decode_detection_targets(targets["heatmap"], targets["size"], score_threshold=0.3, peak_window=pw))
        for pw in (3, 21, 51)
    }
    assert len(set(counts.values())) == 1, (
        f"peak_window influence le nombre de detections sur des cibles vraies (a egalite) : {counts}"
    )
    assert counts[3] == 2, f"attendu 2 detections, obtenu {counts}"
    print(f"OK - invariance a peak_window sur cibles vraies ({counts})")


def test_npz_save_load_roundtrip():
    # Verifie que save_detection_targets_npz/load_detection_targets_npz (les seules
    # fonctions autorisees a lire/ecrire le format .npz, AD-18) round-trippent
    # correctement - contrat public du module, pas seulement le dict en memoire.
    targets = encode_detection_targets([(500, 300, 200, 150)], 1920, 1080, (224, 224))
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "targets.npz")
        save_detection_targets_npz(path, targets[HEATMAP_KEY], targets[SIZE_KEY])
        loaded = load_detection_targets_npz(path)
        assert np.array_equal(loaded[HEATMAP_KEY], targets[HEATMAP_KEY]), "heatmap alteree par le round-trip .npz"
        assert np.array_equal(loaded[SIZE_KEY], targets[SIZE_KEY]), "size alteree par le round-trip .npz"
    print("OK - round-trip save/load .npz (cles canoniques HEATMAP_KEY/SIZE_KEY)")


def test_max_boxes_cap():
    # 25 boites reelles, max_boxes=20 par defaut : seules les 20 de plus grande aire
    # doivent survivre (decision nouvelle, voir Dev Notes de la story - pas un precedent
    # repris de fighterjet_detection_dataset_tools.py, qui n'applique pas ce plafond).
    orig_w, orig_h, target_size = 1920, 1080, (224, 224)
    raw_boxes = [(50 + i * 60, 50 + i * 30, 40 + i, 30 + i) for i in range(25)]  # aires croissantes
    targets = encode_detection_targets(raw_boxes, orig_w, orig_h, target_size, max_boxes=20)
    decoded = decode_detection_targets(targets["heatmap"], targets["size"], score_threshold=0.3)

    assert len(decoded) == 20, f"attendu 20 boites (plafond), obtenu {len(decoded)}"

    # Les 5 plus petites (indices 0-4, plus petite aire) doivent avoir ete ecartees :
    # verifie en confirmant qu'aucune boite decodee n'est proche de leur centre attendu.
    expected_all = _rescale_expected(raw_boxes, orig_w, orig_h, target_size)
    dropped = expected_all[:5]
    for box in dropped:
        assert not any(_boxes_close(box, d) for d in decoded), (
            f"une boite censee etre ecartee par le plafond a ete retrouvee : {box}"
        )
    print("OK - plafond max_boxes=20 (25 boites en entree, 20 plus grandes conservees, 5 plus petites ecartees)")


if __name__ == "__main__":
    test_zero_boxes()
    test_one_box()
    test_multiple_close_boxes()
    test_peak_window_invariance_on_true_targets()
    test_npz_save_load_roundtrip()
    test_max_boxes_cap()
    print("\nTous les tests round-trip sont passes.")
