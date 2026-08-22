from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Callable

import cupy as cp
import numpy as np

from .image_utils import reflect_pad_for_nlm


# ============================================================
# GPU V5 fixed configuration
# ============================================================

V5_PATCH_SIZE = 7
V5_SEARCH_WINDOW_SIZE = 21
V5_BLOCK_SIZE = (16, 16)

V5_PATCH_RADIUS = 3
V5_SEARCH_RADIUS = 10
V5_PADDING_RADIUS = 13

V5_TILE_WIDTH = 42
V5_TILE_HEIGHT = 42


_KERNEL_FILE = (
    Path(__file__).resolve().parents[2]
    / "kernels"
    / "nlm_v5_fixed.cu"
)


def validate_gpu_v5_configuration(
    patch_size: int,
    search_window_size: int,
    h: float,
    block_size: tuple[int, int],
) -> None:
    """
    GPU V5 intentionally supports only the benchmark-specialized
    configuration.

    This is deliberate rather than a limitation hidden from the
    caller: V5 investigates whether fixed-geometry specialization
    can outperform generic V2 and the reformulated V3 pipeline.
    """

    if patch_size != V5_PATCH_SIZE:
        raise ValueError(
            "GPU V5 is specialized for patch_size=7, "
            f"got {patch_size}."
        )

    if (
        search_window_size
        != V5_SEARCH_WINDOW_SIZE
    ):
        raise ValueError(
            "GPU V5 is specialized for "
            "search_window_size=21, "
            f"got {search_window_size}."
        )

    if tuple(block_size) != V5_BLOCK_SIZE:
        raise ValueError(
            "GPU V5 is specialized for "
            "block_size=(16, 16), "
            f"got {block_size}."
        )

    if h <= 0.0:
        raise ValueError(
            f"h must be positive, got {h}."
        )


@lru_cache(maxsize=1)
def get_nlm_gpu_v5_kernel() -> cp.RawKernel:
    """
    Compile and cache the specialized CUDA kernel.
    """

    source = _KERNEL_FILE.read_text(
        encoding="utf-8"
    )

    return cp.RawKernel(
        source,
        "nlm_v5_fixed_7x7_21x21",
        options=(
            "--std=c++11",
        ),
    )


def prepare_nlm_gpu_v5_input(
    image: np.ndarray,
) -> dict[str, object]:
    """
    Convert input to contiguous float32 and perform the same
    reflect padding used by the CPU baseline / previous GPU
    versions.
    """

    image = np.ascontiguousarray(
        image,
        dtype=np.float32,
    )

    if image.ndim != 2:
        raise ValueError(
            "GPU V5 expects a 2-D grayscale image, "
            f"got shape {image.shape}."
        )

    padded, patch_radius, search_radius = (
        reflect_pad_for_nlm(
            image,
            V5_PATCH_SIZE,
            V5_SEARCH_WINDOW_SIZE,
        )
    )

    # Sanity check against the fixed kernel constants.
    if patch_radius != V5_PATCH_RADIUS:
        raise RuntimeError(
            "Unexpected patch radius."
        )

    if search_radius != V5_SEARCH_RADIUS:
        raise RuntimeError(
            "Unexpected search radius."
        )

    padded = np.ascontiguousarray(
        padded,
        dtype=np.float32,
    )

    return {
        "image": image,
        "padded": padded,
        "height": int(image.shape[0]),
        "width": int(image.shape[1]),
        "padded_height": int(
            padded.shape[0]
        ),
        "padded_width": int(
            padded.shape[1]
        ),
    }


def prepare_gpu_v5_kernel_launch(
    image: np.ndarray,
    patch_size: int = 7,
    search_window_size: int = 21,
    h: float = 0.12,
    block_size: tuple[int, int] = (16, 16),
) -> dict[str, object]:
    """
    Prepare GPU buffers and return a reusable launcher.

    The returned launcher measures GPU compute only when used
    with CUDA Events:
    - padding has already happened;
    - H2D has already happened;
    - output allocation has already happened.
    """

    validate_gpu_v5_configuration(
        patch_size=patch_size,
        search_window_size=(
            search_window_size
        ),
        h=h,
        block_size=block_size,
    )

    prepared_input = (
        prepare_nlm_gpu_v5_input(
            image=image,
        )
    )

    padded_gpu = cp.asarray(
        prepared_input["padded"],
        dtype=cp.float32,
    )

    height = prepared_input["height"]
    width = prepared_input["width"]

    padded_height = (
        prepared_input["padded_height"]
    )

    padded_width = (
        prepared_input["padded_width"]
    )

    output_gpu = cp.empty(
        (height, width),
        dtype=cp.float32,
    )

    block = (
        V5_BLOCK_SIZE[0],
        V5_BLOCK_SIZE[1],
        1,
    )

    grid = (
        (
            width
            + V5_BLOCK_SIZE[0]
            - 1
        )
        // V5_BLOCK_SIZE[0],

        (
            height
            + V5_BLOCK_SIZE[1]
            - 1
        )
        // V5_BLOCK_SIZE[1],

        1,
    )

    kernel = get_nlm_gpu_v5_kernel()

    h_squared = np.float32(
        h * h
    )

    kernel_args = (
        padded_gpu,
        output_gpu,
        np.int32(padded_width),
        np.int32(padded_height),
        np.int32(width),
        np.int32(height),
        h_squared,
    )

    def kernel_launcher() -> None:
        kernel(
            grid,
            block,
            kernel_args,
        )

    return {
        "kernel_launcher": kernel_launcher,
        "output_gpu": output_gpu,
        "padded_gpu": padded_gpu,
        "grid": grid,
        "block": block,
        "height": height,
        "width": width,
        "patch_size": V5_PATCH_SIZE,
        "search_window_size": (
            V5_SEARCH_WINDOW_SIZE
        ),
        "block_size": V5_BLOCK_SIZE,
    }


def create_gpu_v5_kernel_launcher(
    image: np.ndarray,
    patch_size: int = 7,
    search_window_size: int = 21,
    h: float = 0.12,
    block_size: tuple[int, int] = (16, 16),
) -> tuple[
    Callable[[], None],
    cp.ndarray,
]:
    """
    Compatibility helper for benchmark_cuda_kernel().
    """

    prepared = (
        prepare_gpu_v5_kernel_launch(
            image=image,
            patch_size=patch_size,
            search_window_size=(
                search_window_size
            ),
            h=h,
            block_size=block_size,
        )
    )

    return (
        prepared["kernel_launcher"],
        prepared["output_gpu"],
    )


def nlm_gpu_v5_device(
    image: np.ndarray,
    patch_size: int = 7,
    search_window_size: int = 21,
    h: float = 0.12,
    block_size: tuple[int, int] = (16, 16),
) -> cp.ndarray:
    """
    Execute V5 and leave output on GPU.
    """

    prepared = (
        prepare_gpu_v5_kernel_launch(
            image=image,
            patch_size=patch_size,
            search_window_size=(
                search_window_size
            ),
            h=h,
            block_size=block_size,
        )
    )

    launcher = prepared[
        "kernel_launcher"
    ]

    launcher()

    return prepared["output_gpu"]


def nlm_gpu_v5(
    image: np.ndarray,
    patch_size: int = 7,
    search_window_size: int = 21,
    h: float = 0.12,
    block_size: tuple[int, int] = (16, 16),
) -> np.ndarray:
    """
    End-to-end public API.

    Input:
        NumPy grayscale float32 image.

    Output:
        NumPy grayscale float32 image.

    Includes:
        CPU reflect padding,
        H2D,
        CUDA kernel,
        D2H.

    Therefore this function is appropriate for the
    end-to-end benchmark.
    """

    output_gpu = nlm_gpu_v5_device(
        image=image,
        patch_size=patch_size,
        search_window_size=(
            search_window_size
        ),
        h=h,
        block_size=block_size,
    )

    return cp.asnumpy(
        output_gpu
    )