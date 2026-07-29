import cupy as cp
import numpy as np
import pytest

from src.nlm.cpu import nlm_cpu_naive
from src.nlm.gpu_v1 import nlm_gpu_v1
from src.nlm.image_utils import (
    add_gaussian_noise,
    create_camera_test_image,
)


def cuda_is_available() -> bool:
    try:
        return (
            cp.cuda.runtime.getDeviceCount()
            > 0
        )
    except cp.cuda.runtime.CUDARuntimeError:
        return False


pytestmark = pytest.mark.skipif(
    not cuda_is_available(),
    reason="CUDA GPU is not available.",
)


def test_gpu_v1_output_shape_and_dtype() -> None:
    image = create_camera_test_image(
        image_size=8
    )

    output = nlm_gpu_v1(
        image=image,
        patch_size=3,
        search_window_size=7,
        h=0.12,
        block_size=(8, 8),
    )

    assert output.shape == image.shape
    assert output.dtype == np.float32


def test_gpu_v1_preserves_constant_image() -> None:
    image = np.full(
        (8, 8),
        0.5,
        dtype=np.float32,
    )

    output = nlm_gpu_v1(
        image=image,
        patch_size=3,
        search_window_size=7,
        h=0.12,
        block_size=(8, 8),
    )

    assert np.allclose(
        output,
        image,
        atol=1e-6,
        rtol=1e-6,
    )


def test_gpu_v1_matches_cpu_reference() -> None:
    clean_image = create_camera_test_image(
        image_size=16
    )

    noisy_image = add_gaussian_noise(
        image=clean_image,
        sigma=0.08,
        seed=42,
    )

    cpu_output = nlm_cpu_naive(
        image=noisy_image,
        patch_size=3,
        search_window_size=7,
        h=0.12,
    )

    gpu_output = nlm_gpu_v1(
        image=noisy_image,
        patch_size=3,
        search_window_size=7,
        h=0.12,
        block_size=(8, 8),
    )

    assert np.allclose(
        gpu_output,
        cpu_output,
        atol=1e-5,
        rtol=1e-5,
    )

    max_absolute_error = float(
        np.max(
            np.abs(
                gpu_output
                - cpu_output
            )
        )
    )

    assert max_absolute_error <= 1e-5


def test_gpu_v1_output_is_in_valid_range() -> None:
    clean_image = create_camera_test_image(
        image_size=16
    )

    noisy_image = add_gaussian_noise(
        image=clean_image,
        sigma=0.08,
        seed=42,
    )

    output = nlm_gpu_v1(
        image=noisy_image,
        patch_size=3,
        search_window_size=7,
        h=0.12,
    )

    assert float(output.min()) >= 0.0
    assert float(output.max()) <= 1.0