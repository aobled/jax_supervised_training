"""
Test standalone pour fighterjet_detection_dataset_tools_v2.py (Story 7.4).
Execution: python3 test_fighterjet_detection_dataset_tools_v2.py
"""

import sys
import os

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, os.path.join(_REPO_ROOT, "dataset_builder"))

import json
import os
import shutil
import tempfile

import numpy as np
from PIL import Image

from fighterjet_detection_dataset_tools_v2 import process_detection_dataset_v2
from detection_target_encoding import HEATMAP_KEY, SIZE_KEY, decode_detection_targets


def _make_fake_dataset(root_dir, n_images=3):
    os.makedirs(root_dir, exist_ok=True)
    known_boxes = []
    for i in range(n_images):
        w, h = 640, 480
        img = Image.new("RGB", (w, h), color=(50 + i * 10, 80, 120))
        img_filename = f"fake_{i}.jpg"
        img_path = os.path.join(root_dir, img_filename)
        img.save(img_path)

        bbox = [100 + i * 20, 150 + i * 10, 80, 60]  # x, y, w, h en pixels source
        known_boxes.append(bbox)

        annotation = {
            "image": {"file_name": img_filename},
            "annotation": {"bbox": bbox},
        }
        json_path = os.path.join(root_dir, f"fake_{i}.json")
        with open(json_path, "w") as f:
            json.dump(annotation, f)

    return known_boxes


def test_chunk_shapes_and_keys():
    tmp_dir = tempfile.mkdtemp(prefix="test_v2_src_")
    out_dir = tempfile.mkdtemp(prefix="test_v2_out_")
    try:
        n_images = 3
        target_size = (224, 224)
        _make_fake_dataset(tmp_dir, n_images=n_images)

        process_detection_dataset_v2(
            root_dirs=[tmp_dir],
            output_dir=out_dir,
            split_name="test",
            target_size=target_size,
            max_boxes=20,
            chunk_size=2000,
            grayscale=True,
        )

        chunk_files = [f for f in os.listdir(out_dir) if f.startswith("jax_detector_targets_")]
        assert len(chunk_files) == 1, f"un seul chunk attendu, trouvé {chunk_files}"
        assert not any(f.startswith("dataset_detection_") for f in os.listdir(out_dir)), \
            "le nom du chunk ne doit pas matcher le glob de nettoyage de l'outil actuel"

        data = np.load(os.path.join(out_dir, chunk_files[0]))
        assert set(data.files) >= {"images", HEATMAP_KEY, SIZE_KEY}
        assert data["images"].shape == (n_images, 224, 224, 1), data["images"].shape
        assert data[HEATMAP_KEY].shape == (n_images, 224, 224, 1), data[HEATMAP_KEY].shape
        assert data[SIZE_KEY].shape == (n_images, 224, 224, 2), data[SIZE_KEY].shape

        print("OK - test_chunk_shapes_and_keys")
        return out_dir, chunk_files[0]
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_decode_roundtrip_on_extracted_example():
    tmp_dir = tempfile.mkdtemp(prefix="test_v2_src2_")
    out_dir = tempfile.mkdtemp(prefix="test_v2_out2_")
    try:
        target_size = (224, 224)
        _make_fake_dataset(tmp_dir, n_images=2)

        process_detection_dataset_v2(
            root_dirs=[tmp_dir],
            output_dir=out_dir,
            split_name="test",
            target_size=target_size,
            max_boxes=20,
            chunk_size=2000,
            grayscale=True,
        )

        chunk_files = [f for f in os.listdir(out_dir) if f.startswith("jax_detector_targets_")]
        data = np.load(os.path.join(out_dir, chunk_files[0]))

        heatmaps_np = data[HEATMAP_KEY]
        sizes_np = data[SIZE_KEY]

        for i in range(heatmaps_np.shape[0]):
            boxes = decode_detection_targets(heatmaps_np[i], sizes_np[i])
            assert len(boxes) >= 1, f"exemple {i} : aucune boîte décodée depuis un chunk réel"
            for (x1, y1, x2, y2, score) in boxes:
                assert 0.0 <= x1 < x2 <= target_size[0] + 1
                assert 0.0 <= y1 < y2 <= target_size[1] + 1
                assert score > 0.0

        print("OK - test_decode_roundtrip_on_extracted_example")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        shutil.rmtree(out_dir, ignore_errors=True)


if __name__ == "__main__":
    out_dir, chunk_file = test_chunk_shapes_and_keys()
    shutil.rmtree(out_dir, ignore_errors=True)
    test_decode_roundtrip_on_extracted_example()
    print("Tous les tests sont passés.")
