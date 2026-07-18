"""
Tests standalone pour les pertes CenterNet (Story 7.3).
Execution: python3 test_centernet_loss.py
"""
import jax.numpy as jnp

from loss_functions import compute_heatmap_focal_loss, compute_size_regression_loss, compute_centernet_loss
from detection_target_encoding import HEATMAP_KEY, SIZE_KEY


def _make_gt(batch=1, h=16, w=16):
    gt_heatmap = jnp.zeros((batch, h, w, 1))
    gt_heatmap = gt_heatmap.at[:, 8, 8, 0].set(1.0)
    gt_heatmap = gt_heatmap.at[:, 8, 9, 0].set(0.5)  # retombée gaussienne (négatif, pas ==1.0)

    gt_size = jnp.zeros((batch, h, w, 2))
    gt_size = gt_size.at[:, 8, 8, :].set(jnp.array([12.0, 20.0]))
    return gt_heatmap, gt_size


def test_heatmap_loss_positive_and_decreasing_towards_gt():
    gt_heatmap, _ = _make_gt()

    pred_far = jnp.full_like(gt_heatmap, 0.1)  # loin de la cible sur le positif (gt=1.0)
    pred_close = gt_heatmap.at[:, 8, 8, 0].set(0.9)  # proche de la cible sur le positif

    loss_far = compute_heatmap_focal_loss(pred_far, gt_heatmap)
    loss_close = compute_heatmap_focal_loss(pred_close, gt_heatmap)

    assert loss_far > 0.0, "la loss doit être positive"
    assert loss_close < loss_far, "la loss doit décroître quand pred se rapproche de gt"
    print("OK - test_heatmap_loss_positive_and_decreasing_towards_gt")


def test_size_loss_positive_and_decreasing_towards_gt():
    _, gt_size = _make_gt()

    pred_far = jnp.zeros_like(gt_size)
    pred_close = gt_size + 0.01

    loss_far = compute_size_regression_loss(pred_far, gt_size)
    loss_close = compute_size_regression_loss(pred_close, gt_size)

    assert loss_far > 0.0
    assert loss_close < loss_far
    print("OK - test_size_loss_positive_and_decreasing_towards_gt")


def test_size_loss_masks_background():
    # gt_size tout à zéro (aucun centre réel) -> pred loin de zéro ne doit pas être pénalisé
    gt_size = jnp.zeros((1, 16, 16, 2))
    pred_size = jnp.full((1, 16, 16, 2), 999.0)

    loss = compute_size_regression_loss(pred_size, gt_size)
    assert jnp.isfinite(loss), "pas de NaN/inf"
    assert loss < 1e-3, f"le fond (gt_size=0) ne doit pas contribuer à la loss, obtenu {loss}"
    print("OK - test_size_loss_masks_background")


def test_no_nan_on_empty_batch():
    gt_heatmap = jnp.zeros((2, 16, 16, 1))
    gt_size = jnp.zeros((2, 16, 16, 2))
    pred_heatmap = jnp.full((2, 16, 16, 1), 0.3)
    pred_size = jnp.zeros((2, 16, 16, 2))

    h_loss = compute_heatmap_focal_loss(pred_heatmap, gt_heatmap)
    s_loss = compute_size_regression_loss(pred_size, gt_size)

    assert jnp.isfinite(h_loss), "heatmap loss NaN/inf sur batch vide"
    assert jnp.isfinite(s_loss), "size loss NaN/inf sur batch vide"
    print("OK - test_no_nan_on_empty_batch")


def test_compute_centernet_loss_decreases_and_uses_dict_contract():
    gt_heatmap, gt_size = _make_gt()
    targets = {HEATMAP_KEY: gt_heatmap, SIZE_KEY: gt_size}

    outputs_far = {
        HEATMAP_KEY: jnp.full_like(gt_heatmap, 0.1),
        SIZE_KEY: jnp.zeros_like(gt_size),
    }
    outputs_close = {
        HEATMAP_KEY: gt_heatmap.at[:, 8, 8, 0].set(0.9),
        SIZE_KEY: gt_size + 0.01,
    }

    loss_far = compute_centernet_loss(outputs_far, targets)
    loss_close = compute_centernet_loss(outputs_close, targets)

    assert jnp.isfinite(loss_far) and jnp.isfinite(loss_close)
    assert loss_close < loss_far
    print("OK - test_compute_centernet_loss_decreases_and_uses_dict_contract")


def test_no_nan_on_fully_empty_batch_combined():
    gt_heatmap = jnp.zeros((1, 16, 16, 1))
    gt_size = jnp.zeros((1, 16, 16, 2))
    targets = {HEATMAP_KEY: gt_heatmap, SIZE_KEY: gt_size}
    outputs = {
        HEATMAP_KEY: jnp.full((1, 16, 16, 1), 0.05),
        SIZE_KEY: jnp.zeros((1, 16, 16, 2)),
    }

    loss = compute_centernet_loss(outputs, targets)
    assert jnp.isfinite(loss), "compute_centernet_loss doit rester fini sur un batch sans objet"
    print("OK - test_no_nan_on_fully_empty_batch_combined")


if __name__ == "__main__":
    test_heatmap_loss_positive_and_decreasing_towards_gt()
    test_size_loss_positive_and_decreasing_towards_gt()
    test_size_loss_masks_background()
    test_no_nan_on_empty_batch()
    test_compute_centernet_loss_decreases_and_uses_dict_contract()
    test_no_nan_on_fully_empty_batch_combined()
    print("Tous les tests sont passés.")
