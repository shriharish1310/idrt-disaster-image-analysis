from __future__ import annotations

import numpy as np

from idrt_analysis.pipeline import compute_change_products, conservative_cloud_mask, valid_data_mask


def test_valid_data_mask_excludes_black_border() -> None:
    image = np.ones((20, 20, 3), dtype=np.float32) * 0.5
    image[:, :4] = 0.0
    mask = valid_data_mask(image)
    assert not mask[:, :4].any()
    assert mask[:, 4:].all()


def test_identical_images_have_perfect_core_metrics() -> None:
    rng = np.random.default_rng(7)
    image = rng.uniform(0.15, 0.85, size=(96, 96, 3)).astype(np.float32)
    mask = np.ones((96, 96), dtype=bool)
    metrics, _normalized, _score, changed = compute_change_products(image, image.copy(), mask)
    assert metrics["ssim"] > 0.999
    assert metrics["rmse"] < 1e-6
    assert metrics["high_change_fraction"] == 0.0
    assert not changed.any()


def test_cloud_mask_detects_large_bright_neutral_region() -> None:
    image = np.full((180, 180, 3), 0.25, dtype=np.float32)
    image[45:135, 45:135] = 0.96
    cloud = conservative_cloud_mask(image)
    assert cloud[90, 90]
    assert cloud.mean() > 0.15

