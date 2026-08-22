import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from nlm.gpu_v4 import (
    nlm_gpu_v4,
    prepare_gpu_v4_pipeline_launch,
    validate_gpu_v4_configuration,
)


def test_v4_shape_dtype_and_range():
    image = np.random.default_rng(0).random((24, 24), dtype=np.float32)

    out = nlm_gpu_v4(
        image,
        patch_size=3,
        search_window_size=7,
        h=0.12,
        edge_search_window_size=5,
    )

    assert out.shape == image.shape
    assert out.dtype == np.float32
    assert np.isfinite(out).all()
    assert out.min() >= 0.0
    assert out.max() <= 1.0


def test_v4_constant_image_is_preserved():
    image = np.full((20, 20), 0.4, dtype=np.float32)

    out = nlm_gpu_v4(
        image,
        patch_size=3,
        search_window_size=7,
        h=0.12,
        edge_search_window_size=5,
    )

    np.testing.assert_allclose(out, image, atol=2e-5, rtol=0.0)


def test_v4_stats_are_consistent():
    image = np.random.default_rng(1).random((18, 18), dtype=np.float32)

    _, stats = nlm_gpu_v4(
        image,
        patch_size=3,
        search_window_size=7,
        h=0.12,
        edge_search_window_size=5,
        mean_threshold=0.05,
        variance_ratio_threshold=2.0,
        return_stats=True,
    )

    assert 0.0 <= stats["acceptance_ratio"] <= 1.0
    assert 0.0 <= stats["rejection_ratio"] <= 1.0
    assert stats["accepted_candidates"] <= stats["considered_candidates"]


def test_v4_no_pruning_considers_full_search():
    image = np.random.default_rng(2).random((16, 16), dtype=np.float32)

    _, stats = nlm_gpu_v4(
        image,
        patch_size=3,
        search_window_size=7,
        h=0.12,
        edge_search_window_size=7,
        texture_variance_threshold=1e9,
        mean_threshold=1e9,
        variance_ratio_threshold=1e9,
        return_stats=True,
    )

    expected = 7 * 7
    assert abs(stats["mean_considered_per_pixel"] - expected) < 1e-6
    assert abs(stats["mean_accepted_per_pixel"] - expected) < 1e-6


def test_v4_invalid_edge_window():
    with pytest.raises(ValueError):
        validate_gpu_v4_configuration(
            patch_size=3,
            search_window_size=7,
            edge_search_window_size=9,
            h=0.12,
            block_size=(16, 16),
            texture_variance_threshold=0.01,
            mean_threshold=0.10,
            variance_ratio_threshold=4.0,
        )


def test_prepare_exposes_launcher_and_stats_buffers():
    image = np.random.default_rng(3).random((16, 16), dtype=np.float32)

    prepared = prepare_gpu_v4_pipeline_launch(
        image=image,
        patch_size=3,
        search_window_size=7,
        h=0.12,
        edge_search_window_size=5,
    )

    assert callable(prepared["kernel_launcher"])
    assert prepared["output_gpu"].shape == image.shape
    assert prepared["accepted_count_gpu"].shape == image.shape
    assert prepared["considered_count_gpu"].shape == image.shape