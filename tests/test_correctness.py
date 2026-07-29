import numpy as np

from src.nlm.cpu import nlm_cpu_naive


def test_cpu_output_shape_and_dtype() -> None:
    image = np.full(
        (8, 8),
        0.5,
        dtype=np.float32,
    )

    output = nlm_cpu_naive(
        image,
        patch_size=3,
        search_window_size=7,
        h=0.12,
    )

    assert output.shape == image.shape
    assert output.dtype == np.float32


def test_cpu_preserves_constant_image() -> None:
    image = np.full(
        (8, 8),
        0.5,
        dtype=np.float32,
    )

    output = nlm_cpu_naive(
        image,
        patch_size=3,
        search_window_size=7,
        h=0.12,
    )

    assert np.allclose(
        output,
        image,
        atol=1e-6,
        rtol=1e-6,
    )


def test_cpu_output_is_in_valid_range() -> None:
    rng = np.random.default_rng(42)

    image = rng.random(
        (8, 8),
        dtype=np.float32,
    )

    output = nlm_cpu_naive(
        image,
        patch_size=3,
        search_window_size=7,
        h=0.12,
    )

    assert float(output.min()) >= 0.0
    assert float(output.max()) <= 1.0