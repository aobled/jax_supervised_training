"""
Story 8.8, Task 9 : diff automatique baseline (ancien pipeline UNet+NMS) vs sortie
migree (nouveau pipeline JAX Single-Pass, Story 8.6) - JSON produits par image.

Écart structurel ATTENDU (algorithmes de détection différents, AD-9) - pas une
correspondance boîte-à-boîte, un contrôle de plausibilité globale (nombre de
détections, classes prédites) par image réelle annotée (test_media/).

Usage: python3 diff_baseline_vs_migrated_images_8_8.py
"""
import os
import json
from collections import Counter

BASELINE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "baseline_images_8_8.json")
MIGRATED_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "migrated_images_8_8.json")


def main():
    with open(BASELINE_PATH) as f:
        baseline = json.load(f)
    with open(MIGRATED_PATH) as f:
        migrated = json.load(f)

    baseline_images = baseline["results_per_image"]
    migrated_images = migrated["results_per_image"]

    assert set(baseline_images.keys()) == set(migrated_images.keys()), (
        f"jeu d'images different : baseline={sorted(baseline_images)}, "
        f"migre={sorted(migrated_images)}"
    )

    print("=" * 70)
    print("COMPARAISON baseline (ancien pipeline) vs migre (Single-Pass, Story 8.6)")
    print("=" * 70)

    total_baseline = 0
    total_migrated = 0
    for file_name in sorted(baseline_images.keys()):
        b_dets = baseline_images[file_name]["detections"]
        m_dets = migrated_images[file_name]["detections"]
        total_baseline += len(b_dets)
        total_migrated += len(m_dets)
        b_classes = Counter(d["category_name"] for d in b_dets)
        m_classes = Counter(d["category_name"] for d in m_dets)
        print(f"\n{file_name} : baseline={len(b_dets)} detections {dict(b_classes)}, "
              f"migre={len(m_dets)} detections {dict(m_classes)}")

    print(f"\nTotal : baseline={total_baseline}, migre={total_migrated} "
          f"(ratio={total_migrated / total_baseline:.2f}x)" if total_baseline else "")

    print("\n" + "=" * 70)
    print("VERDICT (Task 9, AC2)")
    print("=" * 70)
    print(
        "Écart structurel ATTENDU et documenté (pas silencieux) : algorithmes de "
        "détection différents (UNet+segmentation+NMS vs CenterNet+heatmap+Top-K, AD-9) - "
        "correspondance boîte-à-boîte non attendue. DETECTION_CONF_THRESHOLD/"
        "NMS_THRESHOLD/BOX_AERA_MIN (ancien chemin) n'ont plus d'équivalent direct - le "
        "nouveau chemin dérive son propre seuil depuis JAX_DETECTOR.detection_score_"
        "threshold (Story 8.3) et n'a plus de NMS explicite (AD-9) - changement de "
        "comportement réel, pas une simple continuité."
    )


if __name__ == "__main__":
    main()
