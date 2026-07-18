"""
Tests standalone pour AircraftDetectorCenterNet (Story 7.2, AD-9/AD-10).
Execution: python3 test_aircraft_detector_centernet.py
"""
import jax
import jax.numpy as jnp

from model_library import AircraftDetectorCenterNet, create_aircraft_detector_centernet, MODELS
from detection_target_encoding import HEATMAP_KEY, SIZE_KEY


def test_output_keys_and_shapes_eval_mode():
    model = create_aircraft_detector_centernet(dropout_rate=0.2)
    rng = jax.random.PRNGKey(0)
    x = jnp.zeros((2, 224, 224, 1))

    variables = model.init(rng, x, training=False)
    out = model.apply(variables, x, training=False)

    assert set(out.keys()) == {HEATMAP_KEY, SIZE_KEY}, f"clés inattendues: {out.keys()}"
    assert out[HEATMAP_KEY].shape == (2, 224, 224, 1), out[HEATMAP_KEY].shape
    assert out[SIZE_KEY].shape == (2, 224, 224, 2), out[SIZE_KEY].shape
    print("OK - test_output_keys_and_shapes_eval_mode")


def test_heatmap_in_unit_range():
    model = create_aircraft_detector_centernet(dropout_rate=0.2)
    rng = jax.random.PRNGKey(1)
    x = jax.random.normal(rng, (2, 224, 224, 1))

    variables = model.init(rng, x, training=False)
    out = model.apply(variables, x, training=False)

    heatmap = out[HEATMAP_KEY]
    assert jnp.all(heatmap >= 0.0) and jnp.all(heatmap <= 1.0), "heatmap hors [0,1] (sigmoid attendue)"
    print("OK - test_heatmap_in_unit_range")


def test_train_mode_with_dropout_and_batchstats():
    model = create_aircraft_detector_centernet(dropout_rate=0.2)
    rng = jax.random.PRNGKey(2)
    dropout_rng = jax.random.PRNGKey(3)
    x = jnp.ones((2, 224, 224, 1))

    variables = model.init(rng, x, training=True)
    assert 'batch_stats' in variables, "BatchNorm attendu -> batch_stats manquant"

    out, updated_state = model.apply(
        variables, x, training=True,
        rngs={'dropout': dropout_rng},
        mutable=['batch_stats'],
    )
    assert out[HEATMAP_KEY].shape == (2, 224, 224, 1)
    assert out[SIZE_KEY].shape == (2, 224, 224, 2)
    assert 'batch_stats' in updated_state
    print("OK - test_train_mode_with_dropout_and_batchstats")


def test_non_square_batch_size_one():
    model = create_aircraft_detector_centernet(dropout_rate=0.2)
    rng = jax.random.PRNGKey(4)
    x = jnp.zeros((1, 224, 224, 1))

    variables = model.init(rng, x, training=False)
    out = model.apply(variables, x, training=False)

    assert out[HEATMAP_KEY].shape == (1, 224, 224, 1)
    assert out[SIZE_KEY].shape == (1, 224, 224, 2)
    print("OK - test_non_square_batch_size_one")


def test_registered_in_models_dict():
    assert 'aircraft_detector_centernet' in MODELS
    model = MODELS['aircraft_detector_centernet'](dropout_rate=0.2)
    assert isinstance(model, AircraftDetectorCenterNet)
    print("OK - test_registered_in_models_dict")


def test_heatmap_bias_init_matches_prior():
    """
    Addendum post-hoc (2026-07-17, Story 7.2/7.8) : le biais de la tete heatmap doit
    demarrer proche de heatmap_prior, pas 0.5 - correctif d'un collapse observe en
    execution reelle (predictions quasi identiques centres/fond apres 1 epoch,
    diagnose_heatmap_predictions.py). Verifie AVANT tout entrainement (poids fraichement
    initialises).
    """
    prior = 0.0000268  # valeur reelle mesuree pour JAX_DETECTOR (dataset_configs.py)
    model = create_aircraft_detector_centernet(dropout_rate=0.0, heatmap_prior=prior)
    rng = jax.random.PRNGKey(0)
    x = jax.random.normal(rng, (2, 224, 224, 1))

    variables = model.init(rng, x, training=False)
    out = model.apply(variables, x, training=False)
    heatmap_mean = float(out[HEATMAP_KEY].mean())

    # Tolerance large (x2) : le noyau conv 1x1 (poids non-nuls) introduit une petite
    # variance autour du prior porte par le biais, ce n'est pas une egalite stricte
    assert prior * 0.5 < heatmap_mean < prior * 2.0, (
        f"heatmap_mean={heatmap_mean} attendu proche de prior={prior} (biais mal cablé ?)"
    )
    print("OK - test_heatmap_bias_init_matches_prior")


def test_heatmap_prior_default_is_backward_compatible():
    """Sans heatmap_prior explicite, le défaut (0.01, RetinaNet générique) doit s'appliquer,
    pas casser les appels existants ne connaissant pas ce paramètre."""
    model = create_aircraft_detector_centernet(dropout_rate=0.2)
    assert model.heatmap_prior == 0.01
    print("OK - test_heatmap_prior_default_is_backward_compatible")


if __name__ == "__main__":
    test_output_keys_and_shapes_eval_mode()
    test_heatmap_in_unit_range()
    test_train_mode_with_dropout_and_batchstats()
    test_non_square_batch_size_one()
    test_registered_in_models_dict()
    test_heatmap_bias_init_matches_prior()
    test_heatmap_prior_default_is_backward_compatible()
    print("Tous les tests sont passés.")
