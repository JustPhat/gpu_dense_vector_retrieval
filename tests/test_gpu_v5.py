import numpy as np
import pytest

from src.nlm.cpu import (
    nlm_cpu_naive,
)

from src.nlm.gpu_v2 import (
    nlm_gpu_v2,
)

from src.nlm.gpu_v5 import (
    nlm_gpu_v5,
    prepare_gpu_v5_kernel_launch,
)


def _test_image(
    size: int = 24,
) -> np.ndarray:
    rng = np.random.default_rng(42)

    return rng.random(
        (size, size),
        dtype=np.float32,
    )


def test_gpu_v5_shape_and_dtype():
    image = _test_image(24)

    output = nlm_gpu_v5(
        image=image,
        patch_size=7,
        search_window_size=21,
        h=0.12,
        block_size=(16, 16),
    )

    assert output.shape == image.shape
    assert output.dtype == np.float32


def test_gpu_v5_output_range():
    image = _test_image(24)

    output = nlm_gpu_v5(
        image=image,
        patch_size=7,
        search_window_size=21,
        h=0.12,
        block_size=(16, 16),
    )

    assert float(output.min()) >= 0.0
    assert float(output.max()) <= 1.0


def test_gpu_v5_constant_image():
    image = np.full(
        (24, 24),
        0.4,
        dtype=np.float32,
    )

    output = nlm_gpu_v5(
        image=image,
        patch_size=7,
        search_window_size=21,
        h=0.12,
        block_size=(16, 16),
    )

    np.testing.assert_allclose(
        output,
        image,
        atol=2e-5,
        rtol=2e-5,
    )


def test_gpu_v5_matches_gpu_v2():
    image = _test_image(32)

    reference = nlm_gpu_v2(
        image=image,
        patch_size=7,
        search_window_size=21,
        h=0.12,
        block_size=(16, 16),
    )

    candidate = nlm_gpu_v5(
        image=image,
        patch_size=7,
        search_window_size=21,
        h=0.12,
        block_size=(16, 16),
    )

    np.testing.assert_allclose(
        candidate,
        reference,
        atol=5e-5,
        rtol=5e-5,
    )


def test_gpu_v5_matches_cpu_baseline():
    # Small input because CPU baseline is intentionally naive.
    image = _test_image(16)

    reference = nlm_cpu_naive(
        image=image,
        patch_size=7,
        search_window_size=21,
        h=0.12,
    )

    candidate = nlm_gpu_v5(
        image=image,
        patch_size=7,
        search_window_size=21,
        h=0.12,
        block_size=(16, 16),
    )

    np.testing.assert_allclose(
        candidate,
        reference,
        atol=5e-5,
        rtol=5e-5,
    )


def test_gpu_v5_launcher():
    image = _test_image(32)

    prepared = (
        prepare_gpu_v5_kernel_launch(
            image=image,
            patch_size=7,
            search_window_size=21,
            h=0.12,
            block_size=(16, 16),
        )
    )

    assert callable(
        prepared["kernel_launcher"]
    )

    assert (
        prepared["output_gpu"].shape
        == image.shape
    )

    assert prepared["grid"] == (
        2,
        2,
        1,
    )

    assert prepared["block"] == (
        16,
        16,
        1,
    )


@pytest.mark.parametrize(
    (
        "patch_size",
        "search_window_size",
        "block_size",
    ),
    [
        (3, 21, (16, 16)),
        (5, 21, (16, 16)),
        (7, 11, (16, 16)),
        (7, 7, (16, 16)),
        (7, 21, (8, 8)),
    ],
)
def test_gpu_v5_rejects_non_specialized_configuration(
    patch_size,
    search_window_size,
    block_size,
):
    image = _test_image(16)

    with pytest.raises(ValueError):
        nlm_gpu_v5(
            image=image,
            patch_size=patch_size,
            search_window_size=(
                search_window_size
            ),
            h=0.12,
            block_size=block_size,
        )


def test_gpu_v5_rejects_invalid_h():
    image = _test_image(16)

    with pytest.raises(ValueError):
        nlm_gpu_v5(
            image=image,
            patch_size=7,
            search_window_size=21,
            h=0.0,
            block_size=(16, 16),
        )