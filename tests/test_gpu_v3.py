import numpy as np
import pytest

from src.nlm.gpu_v1 import nlm_gpu_v1
from src.nlm.gpu_v2 import nlm_gpu_v2
from src.nlm.gpu_v3 import (
    nlm_gpu_v3,
    prepare_gpu_v3_pipeline_launch,
)


def _test_image(
    size: int = 24,
) -> np.ndarray:
    rng = np.random.default_rng(42)

    return rng.random(
        (size, size),
        dtype=np.float32,
    )


def test_gpu_v3_shape_and_dtype():
    image = _test_image(24)

    output = nlm_gpu_v3(
        image=image,
        patch_size=3,
        search_window_size=7,
        h=0.12,
    )

    assert output.shape == image.shape
    assert output.dtype == np.float32


def test_gpu_v3_constant_image():
    image = np.full(
        (24, 24),
        0.4,
        dtype=np.float32,
    )

    output = nlm_gpu_v3(
        image=image,
        patch_size=5,
        search_window_size=11,
        h=0.12,
    )

    np.testing.assert_allclose(
        output,
        image,
        atol=2e-6,
        rtol=2e-6,
    )


@pytest.mark.parametrize(
    "patch_size, search_window_size",
    [
        (3, 7),
        (5, 11),
        (7, 21),
    ],
)
def test_gpu_v3_matches_gpu_v1(
    patch_size,
    search_window_size,
):
    image = _test_image(24)

    reference = nlm_gpu_v1(
        image=image,
        patch_size=patch_size,
        search_window_size=(
            search_window_size
        ),
        h=0.12,
        block_size=(16, 16),
    )

    candidate = nlm_gpu_v3(
        image=image,
        patch_size=patch_size,
        search_window_size=(
            search_window_size
        ),
        h=0.12,
        block_size=(16, 16),
        displacement_batch_size=8,
    )

    np.testing.assert_allclose(
        candidate,
        reference,
        atol=5e-5,
        rtol=5e-5,
    )


def test_gpu_v3_matches_gpu_v2_large_regime():
    image = _test_image(24)

    reference = nlm_gpu_v2(
        image=image,
        patch_size=7,
        search_window_size=21,
        h=0.12,
        block_size=(16, 16),
    )

    candidate = nlm_gpu_v3(
        image=image,
        patch_size=7,
        search_window_size=21,
        h=0.12,
        block_size=(16, 16),
        displacement_batch_size=16,
    )

    np.testing.assert_allclose(
        candidate,
        reference,
        atol=5e-5,
        rtol=5e-5,
    )


def test_gpu_v3_batch_size_does_not_change_result():
    image = _test_image(20)

    output_4 = nlm_gpu_v3(
        image=image,
        patch_size=3,
        search_window_size=7,
        h=0.12,
        displacement_batch_size=4,
    )

    output_16 = nlm_gpu_v3(
        image=image,
        patch_size=3,
        search_window_size=7,
        h=0.12,
        displacement_batch_size=16,
    )

    np.testing.assert_allclose(
        output_4,
        output_16,
        atol=2e-5,
        rtol=2e-5,
    )


def test_gpu_v3_reports_reduced_batch_count():
    image = _test_image(16)

    prepared = prepare_gpu_v3_pipeline_launch(
        image=image,
        patch_size=7,
        search_window_size=21,
        h=0.12,
        displacement_batch_size=32,
    )

    assert (
        prepared["number_of_displacements"]
        == 441
    )

    assert prepared["number_of_batches"] == 14


def test_gpu_v3_output_range():
    image = _test_image(24)

    output = nlm_gpu_v3(
        image=image,
        patch_size=7,
        search_window_size=21,
        h=0.12,
        displacement_batch_size=16,
    )

    assert float(output.min()) >= 0.0
    assert float(output.max()) <= 1.0


@pytest.mark.parametrize(
    "patch_size, search_window_size",
    [
        (4, 7),
        (3, 8),
        (0, 7),
    ],
)
def test_gpu_v3_rejects_invalid_window_sizes(
    patch_size,
    search_window_size,
):
    image = _test_image(16)

    with pytest.raises(ValueError):
        nlm_gpu_v3(
            image=image,
            patch_size=patch_size,
            search_window_size=(
                search_window_size
            ),
            h=0.12,
        )


@pytest.mark.parametrize(
    "batch_size",
    [0, -1],
)
def test_gpu_v3_rejects_invalid_batch_size(
    batch_size,
):
    image = _test_image(16)

    with pytest.raises(ValueError):
        nlm_gpu_v3(
            image=image,
            patch_size=3,
            search_window_size=7,
            h=0.12,
            displacement_batch_size=batch_size,
        )