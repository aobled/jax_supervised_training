"""
Story 8.2 : verifie que _resize_for_detector (nouveau) + load_detection_model/
build_predict_fn (existants, inference_utils.py, AD-1/AD-3 herites) fonctionnent
ensemble SANS modification de ces deux fonctions existantes, pour le detecteur
CenterNet (Story 7.2, sortie dict {HEATMAP_KEY, SIZE_KEY}).

Usage: python3 test_detector_inference_composition.py
"""
import jax.numpy as jnp

from inference_utils import load_detection_model, build_predict_fn, _resize_for_detector
from detection_target_encoding import HEATMAP_KEY, SIZE_KEY
from model_library import AircraftDetectorCenterNet

CHECKPOINT_PATH = "best_model_jax_detector.pkl"


def test_load_detection_model_loads_centernet_not_unet():
    """
    Task 2 - piege identifie en revue independante : load_detection_model retombe
    silencieusement sur 'aircraft_detector_unet' si 'model_name' est absent du
    checkpoint. Assertion EXPLICITE et SEPAREE avant tout autre test, pas une
    simple observation que "ca a l'air de marcher".
    """
    model, variables, config_model = load_detection_model(CHECKPOINT_PATH)

    assert config_model.get("model_name") == "aircraft_detector_centernet", (
        f"load_detection_model a charge '{config_model.get('model_name')}' au lieu de "
        f"'aircraft_detector_centernet' - verifier que la config a bien ete sauvegardee "
        f"dans le checkpoint (piege du fallback silencieux, inference_utils.py:123)"
    )
    assert isinstance(model, AircraftDetectorCenterNet), (
        f"model est une instance de {type(model).__name__}, pas AircraftDetectorCenterNet"
    )
    print("OK - test_load_detection_model_loads_centernet_not_unet")
    return model, variables, config_model


def test_build_predict_fn_returns_dict_output(model, variables, config_model):
    """Task 3 - build_predict_fn (generique, non modifie) doit fonctionner tel quel
    pour une sortie dict, pas seulement pour un tenseur unique."""
    predict_fn = build_predict_fn(model, variables)

    image_size = config_model["image_size"]
    grayscale = config_model.get("grayscale", True)
    channels = 1 if grayscale else 3
    dummy_batch = jnp.zeros((2, *image_size, channels))

    output = predict_fn(dummy_batch)

    assert isinstance(output, dict), f"sortie attendue dict, obtenu {type(output)}"
    assert set(output.keys()) == {HEATMAP_KEY, SIZE_KEY}, f"cles inattendues: {output.keys()}"
    assert output[HEATMAP_KEY].shape == (2, *image_size, 1), output[HEATMAP_KEY].shape
    assert output[SIZE_KEY].shape == (2, *image_size, 2), output[SIZE_KEY].shape
    print("OK - test_build_predict_fn_returns_dict_output")
    return predict_fn


def test_resize_for_detector_geometry_only_no_normalization():
    """_resize_for_detector ne doit JAMAIS normaliser (pixels bruts en entree ET en sortie)."""
    raw_image = jnp.full((1080, 1920, 1), 200.0)  # pixels bruts, PAS dans [0,1]
    resized = _resize_for_detector(raw_image, target_size=(224, 224), method="lanczos3")

    assert resized.shape == (224, 224, 1), resized.shape
    # Valeur constante -> le resize d'une image uniforme reste uniforme, proche de 200
    # (pas de division par 255 qui la ramenerait vers ~0.78)
    assert jnp.abs(resized.mean() - 200.0) < 1.0, (
        f"mean={resized.mean():.3f}, attendu ~200.0 (pixels bruts) - une normalisation "
        f"aurait fait chuter la valeur vers ~0.78"
    )
    print("OK - test_resize_for_detector_geometry_only_no_normalization")


def test_full_composition_1920x1080_to_heatmap_size(model, variables, config_model, predict_fn):
    """Task 4 - composition complete : image factice 1920x1080x1 -> heatmap+taille,
    shapes verifiees en repere resolution detecteur."""
    image_size = config_model["image_size"]  # (224, 224), source unique via config_model
    channels = 1 if config_model.get("grayscale", True) else 3  # source unique via config_model

    canonical_image = jnp.zeros((1080, 1920, channels))  # AD-12 : entree canonique 1920x1080

    resized = _resize_for_detector(canonical_image, target_size=image_size, method="lanczos3")
    assert resized.shape == (*image_size, channels), resized.shape

    batched = resized[None, ...]  # ajout explicite de l'axe batch, pas automatique
    assert batched.shape == (1, *image_size, channels)

    output = predict_fn(batched)
    assert output[HEATMAP_KEY].shape == (1, *image_size, 1), output[HEATMAP_KEY].shape
    assert output[SIZE_KEY].shape == (1, *image_size, 2), output[SIZE_KEY].shape
    print("OK - test_full_composition_1920x1080_to_heatmap_size")


if __name__ == "__main__":
    model, variables, config_model = test_load_detection_model_loads_centernet_not_unet()
    predict_fn = test_build_predict_fn_returns_dict_output(model, variables, config_model)
    test_resize_for_detector_geometry_only_no_normalization()
    test_full_composition_1920x1080_to_heatmap_size(model, variables, config_model, predict_fn)
    print("Tous les tests sont passés.")
