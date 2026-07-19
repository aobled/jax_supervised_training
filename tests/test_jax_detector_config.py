"""
Test standalone pour l'entrée JAX_DETECTOR (Story 7.7).
Execution: python3 test_jax_detector_config.py
Ne nécessite pas de vrais chunks .npz (Story 7.8) - seulement la cohérence de la
config et l'instanciation des objets (modèle, stratégie).
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset_configs import get_dataset_config
from model_library import get_model
from task_strategies import CenterNetDetectionStrategy


def test_get_dataset_config_validates():
    config = get_dataset_config("JAX_DETECTOR")
    assert config["dataset_name"] == "JAX_DETECTOR"
    assert config["task_type"] == "detection_centernet"
    assert config["model_name"] == "aircraft_detector_centernet"
    assert config["num_classes"] == 1
    assert config["class_names"] == ["aircraft"]
    assert config["image_size"] == (224, 224)
    print("OK - test_get_dataset_config_validates")


def test_output_prefix_distinct_from_fighterjet_detection():
    jax_detector_config = get_dataset_config("JAX_DETECTOR")
    fighterjet_config = get_dataset_config("FIGHTERJET_DETECTION")

    jax_prefix = jax_detector_config["output_prefix"]
    fj_prefix = fighterjet_config["output_prefix"]

    assert jax_prefix != fj_prefix
    import os
    assert os.path.dirname(jax_prefix) != os.path.dirname(fj_prefix), \
        "les répertoires de sortie doivent être distincts (AD-20, pas de collision)"
    assert os.path.basename(jax_prefix) == "jax_detector_targets", \
        "doit correspondre exactement au préfixe codé en dur dans dataset_builder/jax_detector_dataset_tools.py (Story 7.4)"
    print("OK - test_output_prefix_distinct_from_fighterjet_detection")


def test_backend_hyperparams_present():
    config = get_dataset_config("JAX_DETECTOR")
    for backend in ("gpu", "tpu"):
        assert backend in config, f"clé '{backend}' manquante"
        assert "dropout_rate" in config[backend], f"'{backend}.dropout_rate' manquant (lu sans garde par main.py:98)"
        assert "micro_batch_size" in config[backend], f"'{backend}.micro_batch_size' manquant (lu sans garde par main.py:77)"
    print("OK - test_backend_hyperparams_present")


def test_no_dispatch_method_keys():
    config = get_dataset_config("JAX_DETECTOR")
    # CenterNetDetectionStrategy n'a pas de dispatch interne (Story 7.6) - ces clés
    # seraient trompeuses si présentes (suggéreraient un dispatch qui n'existe pas)
    assert "loss_method" not in config
    assert "metric_method" not in config
    assert "report_method" not in config
    assert "loss_params" in config
    # metric_threshold retire (addendum post-hoc 2026-07-18) : HeatmapActivation est
    # une moyenne continue, CenterNetDetectionStrategy.__init__ ne l'accepte plus
    assert "metric_threshold" not in config
    print("OK - test_no_dispatch_method_keys")


def test_model_instantiates():
    config = get_dataset_config("JAX_DETECTOR")
    model = get_model(config["model_name"], num_classes=config["num_classes"], dropout_rate=config["gpu"]["dropout_rate"])
    from model_library import AircraftDetectorCenterNet
    assert isinstance(model, AircraftDetectorCenterNet)
    print("OK - test_model_instantiates")


def test_strategy_instantiates_with_real_signature():
    config = get_dataset_config("JAX_DETECTOR")
    strategy = CenterNetDetectionStrategy(loss_params=config["loss_params"])
    assert strategy.primary_metric_name == "HeatmapActivation"
    assert strategy._get_export_path(config) == "best_model_jax_detector.pkl"
    print("OK - test_strategy_instantiates_with_real_signature")


def test_no_checkpoint_path_override():
    config = get_dataset_config("JAX_DETECTOR")
    # Pas de checkpoint_path/training_state_path explicite : le fallback derive de
    # dataset_name (Story 5.0/7.6) doit s'appliquer
    assert "checkpoint_path" not in config
    assert "training_state_path" not in config
    print("OK - test_no_checkpoint_path_override")


def test_fighterjet_detection_untouched():
    config = get_dataset_config("FIGHTERJET_DETECTION")
    assert config["task_type"] == "detection"
    assert config["model_name"] == "aircraft_detector_unet"
    print("OK - test_fighterjet_detection_untouched")


if __name__ == "__main__":
    test_get_dataset_config_validates()
    test_output_prefix_distinct_from_fighterjet_detection()
    test_backend_hyperparams_present()
    test_no_dispatch_method_keys()
    test_model_instantiates()
    test_strategy_instantiates_with_real_signature()
    test_no_checkpoint_path_override()
    test_fighterjet_detection_untouched()
    print("Tous les tests sont passés.")
