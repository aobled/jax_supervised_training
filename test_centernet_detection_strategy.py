"""
Test standalone pour CenterNetDetectionStrategy (Story 7.6).
Execution: python3 test_centernet_detection_strategy.py
"""
import jax
import jax.numpy as jnp

from task_strategies import CenterNetDetectionStrategy
from model_library import create_aircraft_detector_centernet
from detection_target_encoding import HEATMAP_KEY, SIZE_KEY


def _make_targets_with_object(batch=2, h=32, w=32):
    heatmap = jnp.zeros((batch, h, w, 1))
    heatmap = heatmap.at[:, 16, 16, 0].set(1.0)
    size = jnp.zeros((batch, h, w, 2))
    size = size.at[:, 16, 16, :].set(jnp.array([10.0, 14.0]))
    return {HEATMAP_KEY: heatmap, SIZE_KEY: size}


def _make_empty_targets(batch=2, h=32, w=32):
    return {HEATMAP_KEY: jnp.zeros((batch, h, w, 1)), SIZE_KEY: jnp.zeros((batch, h, w, 2))}


def test_preprocess_batch_casts_dict_to_float32():
    strategy = CenterNetDetectionStrategy()
    targets = {HEATMAP_KEY: jnp.zeros((2, 32, 32, 1), dtype=jnp.float64), SIZE_KEY: jnp.zeros((2, 32, 32, 2), dtype=jnp.float64)}
    images = jnp.zeros((2, 32, 32, 1))

    out_images, out_targets, use_onehot = strategy.preprocess_batch(images, targets, is_training=True)

    assert out_targets[HEATMAP_KEY].dtype == jnp.float32
    assert out_targets[SIZE_KEY].dtype == jnp.float32
    assert use_onehot is False
    print("OK - test_preprocess_batch_casts_dict_to_float32")


def test_compute_loss_finite_and_positive():
    strategy = CenterNetDetectionStrategy()
    model = create_aircraft_detector_centernet(dropout_rate=0.2)
    rng = jax.random.PRNGKey(0)
    x = jax.random.normal(rng, (2, 32, 32, 1))
    variables = model.init(rng, x, training=False)
    outputs = model.apply(variables, x, training=False)

    targets = _make_targets_with_object(batch=2, h=32, w=32)

    loss = strategy.compute_loss(outputs, targets)
    assert jnp.isfinite(loss)
    assert loss > 0.0
    print("OK - test_compute_loss_finite_and_positive")


def test_compute_metrics_in_unit_range():
    strategy = CenterNetDetectionStrategy()
    targets = _make_targets_with_object()

    # Prediction parfaite : activation doit valoir 1.0
    perfect_outputs = {HEATMAP_KEY: targets[HEATMAP_KEY], SIZE_KEY: targets[SIZE_KEY]}
    activation_perfect = strategy.compute_metrics(perfect_outputs, targets)
    assert jnp.isclose(activation_perfect, 1.0)

    # Prediction nulle partout : activation doit valoir 0.0
    zero_outputs = {HEATMAP_KEY: jnp.zeros_like(targets[HEATMAP_KEY]), SIZE_KEY: jnp.zeros_like(targets[SIZE_KEY])}
    activation_zero = strategy.compute_metrics(zero_outputs, targets)
    assert jnp.isclose(activation_zero, 0.0)

    assert 0.0 <= float(activation_perfect) <= 1.0
    assert 0.0 <= float(activation_zero) <= 1.0
    print("OK - test_compute_metrics_in_unit_range")


def test_compute_metrics_is_continuous_not_thresholded():
    """
    Addendum post-hoc (2026-07-18) : le point precis que ce correctif resout.
    Une prediction UNIFORMEMENT sous l'ancien seuil de 0.5 aux vrais centres doit
    remonter une valeur proportionnelle (pas 0.0 comme le ferait un seuil dur) - c'est
    exactement le scenario observe en execution reelle (Story 7.8) ou le modele
    progressait reellement sans que HeatmapRecall ne bouge de 0.0000.
    """
    strategy = CenterNetDetectionStrategy()
    targets = _make_targets_with_object()

    partial_outputs = {
        HEATMAP_KEY: jnp.full_like(targets[HEATMAP_KEY], 0.3),  # < 0.5, sous l'ancien seuil
        SIZE_KEY: targets[SIZE_KEY],
    }
    activation_partial = strategy.compute_metrics(partial_outputs, targets)
    assert jnp.isclose(activation_partial, 0.3), (
        f"attendu ~0.3 (moyenne continue), obtenu {activation_partial} - "
        f"un ancien seuil dur aurait donne 0.0 ici, invisible pour la selection de checkpoint"
    )
    print("OK - test_compute_metrics_is_continuous_not_thresholded")


def test_no_nan_on_batch_without_objects():
    strategy = CenterNetDetectionStrategy()
    model = create_aircraft_detector_centernet(dropout_rate=0.2)
    rng = jax.random.PRNGKey(1)
    x = jnp.zeros((2, 32, 32, 1))
    variables = model.init(rng, x, training=False)
    outputs = model.apply(variables, x, training=False)

    targets = _make_empty_targets()

    loss = strategy.compute_loss(outputs, targets)
    metric = strategy.compute_metrics(outputs, targets)

    assert jnp.isfinite(loss), "loss NaN/inf sur un batch sans objet réel"
    assert jnp.isfinite(metric), "metrique NaN/inf sur un batch sans objet réel"
    assert jnp.isclose(metric, 1.0), "convention : pas d'objet réel -> activation 1.0 (rien à pénaliser)"
    print("OK - test_no_nan_on_batch_without_objects")


def test_export_paths_follow_dataset_name_pattern():
    strategy = CenterNetDetectionStrategy()
    config = {"dataset_name": "JAX_DETECTOR"}
    assert strategy._get_export_path(config) == "best_model_jax_detector.pkl"
    assert strategy.get_training_state_path(config) == "best_model_training_state_jax_detector.pkl"
    print("OK - test_export_paths_follow_dataset_name_pattern")


def test_primary_metric_name_and_optimization_mode():
    strategy = CenterNetDetectionStrategy()
    assert strategy.primary_metric_name == "HeatmapActivation"
    assert strategy.optimization_mode == "max"
    print("OK - test_primary_metric_name_and_optimization_mode")


if __name__ == "__main__":
    test_preprocess_batch_casts_dict_to_float32()
    test_compute_loss_finite_and_positive()
    test_compute_metrics_in_unit_range()
    test_compute_metrics_is_continuous_not_thresholded()
    test_no_nan_on_batch_without_objects()
    test_export_paths_follow_dataset_name_pattern()
    test_primary_metric_name_and_optimization_mode()
    print("Tous les tests sont passés.")
