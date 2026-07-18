"""
Story 8.7, Task 6 : diff automatique baseline (ancien pipeline UNet+findContours) vs
sortie migree (nouveau pipeline JAX Single-Pass, Story 8.6) - boxes/classes/scores,
pas le rendu visuel du canvas (Task 6 explicite).

Écart structurel ATTENDU et documenté, pas une regression a corriger : les deux chemins
utilisent des ALGORITHMES DE DETECTION DIFFERENTS (UNet+segmentation+findContours vs
CenterNet+heatmap+Top-K, AD-9) - le nombre de boites et leurs coordonnees exactes ne
sont structurellement pas censes correspondre 1:1. Ce script mesure des indicateurs de
plausibilite globale (ordre de grandeur du nombre de detections, classes predominantes),
pas une correspondance boite-a-boite.

Usage: python3 diff_baseline_vs_migrated_video_8_7.py
"""
import os
import json
from collections import Counter

BASELINE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "baseline_video_8_7.json")
MIGRATED_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "migrated_video_8_7.json")


def main():
    with open(BASELINE_PATH) as f:
        baseline = json.load(f)
    with open(MIGRATED_PATH) as f:
        migrated = json.load(f)

    assert baseline["num_frames"] == migrated["num_frames"], (
        f"nombre de frames different : baseline={baseline['num_frames']}, "
        f"migre={migrated['num_frames']} - comparaison invalide"
    )
    num_frames = baseline["num_frames"]

    baseline_counts = [len(f) for f in baseline["detections_per_frame"]]
    migrated_counts = [len(f) for f in migrated["detections_per_frame"]]
    total_baseline = sum(baseline_counts)
    total_migrated = sum(migrated_counts)

    print("=" * 70)
    print(f"COMPARAISON baseline (ancien pipeline) vs migre (Single-Pass, Story 8.6)")
    print(f"Extrait : {num_frames} frames, {baseline['video_path']}")
    print("=" * 70)
    print(f"Detections totales : baseline={total_baseline}, migre={total_migrated} "
          f"(ratio={total_migrated / total_baseline:.2f}x)" if total_baseline else "")
    print(f"Detections/frame (moyenne) : baseline={total_baseline / num_frames:.2f}, "
          f"migre={total_migrated / num_frames:.2f}")
    print(f"Debit : baseline={baseline.get('throughput_fps', float('nan')):.2f} fps, "
          f"migre={migrated.get('throughput_fps', float('nan')):.2f} fps")

    baseline_classes = Counter(
        d["class"] for frame in baseline["detections_per_frame"] for d in frame
    )
    migrated_classes = Counter(
        d["class"] for frame in migrated["detections_per_frame"] for d in frame
    )
    print("\nClasses predites (baseline) :", dict(baseline_classes.most_common()))
    print("Classes predites (migre)    :", dict(migrated_classes.most_common()))

    frames_more_detections = sum(1 for b, m in zip(baseline_counts, migrated_counts) if m > b)
    frames_fewer_detections = sum(1 for b, m in zip(baseline_counts, migrated_counts) if m < b)
    frames_equal = sum(1 for b, m in zip(baseline_counts, migrated_counts) if m == b)
    print(f"\nPar frame : {frames_more_detections} avec plus de detections (migre>baseline), "
          f"{frames_fewer_detections} avec moins, {frames_equal} identiques (compte, pas position)")

    print("\n" + "=" * 70)
    print("VERDICT (Task 6, AC2)")
    print("=" * 70)
    print(
        "Écart structurel ATTENDU et documenté (pas silencieux) : les deux pipelines "
        "utilisent des algorithmes de détection différents (UNet+segmentation+findContours "
        "vs CenterNet+heatmap+Top-K, AD-9) - correspondance boîte-à-boîte non attendue, "
        "seul l'ordre de grandeur et la plausibilité des classes prédites sont comparés ici. "
        "Le quadrant heatmap/contours de visualisation (Dev Notes Story 8.7) est un écart "
        "connu et accepté séparé de cette comparaison de données."
    )


if __name__ == "__main__":
    main()
