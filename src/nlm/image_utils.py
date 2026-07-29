from pathlib import Path

import cv2
import numpy as np
from skimage import data, transform


def create_camera_test_image(
    image_size: int,
) -> np.ndarray:
    if image_size <= 0:
        raise ValueError("image_size must be positive.")

    image = data.camera().astype(np.float32) / 255.0

    image = transform.resize(
        image,
        (image_size, image_size),
        anti_aliasing=True,
    ).astype(np.float32)

    return np.ascontiguousarray(image)


def load_grayscale_image(
    image_path: str | Path,
    image_size: int | None = None,
) -> np.ndarray:
    image_path = Path(image_path)

    image_bgr = cv2.imread(
        str(image_path),
        cv2.IMREAD_COLOR,
    )

    if image_bgr is None:
        raise FileNotFoundError(
            f"Cannot read image: {image_path}"
        )

    image_gray = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2GRAY,
    )

    if image_size is not None:
        if image_size <= 0:
            raise ValueError("image_size must be positive.")

        image_gray = cv2.resize(
            image_gray,
            (image_size, image_size),
            interpolation=cv2.INTER_AREA,
        )

    image = (
        image_gray.astype(np.float32)
        / 255.0
    )

    return np.ascontiguousarray(image)


def add_gaussian_noise(
    image: np.ndarray,
    sigma: float,
    seed: int = 42,
) -> np.ndarray:
    if image.ndim != 2:
        raise ValueError(
            "Expected a 2D grayscale image."
        )

    if sigma < 0:
        raise ValueError("sigma cannot be negative.")

    rng = np.random.default_rng(seed)

    noise = rng.normal(
        loc=0.0,
        scale=sigma,
        size=image.shape,
    ).astype(np.float32)

    noisy = np.clip(
        image.astype(np.float32) + noise,
        0.0,
        1.0,
    )

    return np.ascontiguousarray(
        noisy.astype(np.float32)
    )


def reflect_pad_for_nlm(
    image: np.ndarray,
    patch_size: int,
    search_window_size: int,
) -> tuple[np.ndarray, int, int]:
    if image.ndim != 2:
        raise ValueError(
            "Expected a 2D grayscale image."
        )

    if patch_size <= 0 or patch_size % 2 == 0:
        raise ValueError(
            "patch_size must be a positive odd integer."
        )

    if (
        search_window_size <= 0
        or search_window_size % 2 == 0
    ):
        raise ValueError(
            "search_window_size must be a positive odd integer."
        )

    if search_window_size < patch_size:
        raise ValueError(
            "search_window_size must be at least patch_size."
        )

    patch_radius = patch_size // 2
    search_radius = search_window_size // 2
    padding_radius = patch_radius + search_radius

    padded = np.pad(
        image,
        pad_width=padding_radius,
        mode="reflect",
    ).astype(np.float32)

    return (
        np.ascontiguousarray(padded),
        patch_radius,
        search_radius,
    )