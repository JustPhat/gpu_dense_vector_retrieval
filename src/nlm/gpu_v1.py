from functools import lru_cache
from pathlib import Path

import cupy as cp
import numpy as np

from .image_utils import reflect_pad_for_nlm


PROJECT_ROOT = Path(__file__).resolve().parents[2]

GPU_V1_KERNEL_PATH = (
    PROJECT_ROOT
    / "kernels"
    / "nlm_v1_global.cu"
)


@lru_cache(maxsize=1)
def get_nlm_gpu_v1_kernel() -> cp.RawKernel:
    """
    Load and compile the GPU V1 CUDA kernel.

    The result is cached so the kernel source is compiled
    only once during the Python process.
    """
    if not GPU_V1_KERNEL_PATH.exists():
        raise FileNotFoundError(
            f"GPU V1 kernel not found: "
            f"{GPU_V1_KERNEL_PATH}"
        )

    kernel_source = GPU_V1_KERNEL_PATH.read_text(
        encoding="utf-8"
    )

    return cp.RawKernel(
        code=kernel_source,
        name="nlm_gpu_v1",
        options=("--std=c++11",),
    )


def prepare_nlm_gpu_v1_input(
    image: np.ndarray,
    patch_size: int,
    search_window_size: int,
) -> tuple[
    np.ndarray,
    int,
    int,
]:
    """
    Validate and prepare a contiguous reflect-padded
    float32 input image.
    """
    image = np.asarray(
        image,
        dtype=np.float32,
    )

    if image.ndim != 2:
        raise ValueError(
            "GPU V1 expects a 2D grayscale image."
        )

    return reflect_pad_for_nlm(
        image=image,
        patch_size=patch_size,
        search_window_size=(
            search_window_size
        ),
    )


def nlm_gpu_v1_device(
    image: np.ndarray,
    patch_size: int = 3,
    search_window_size: int = 7,
    h: float = 0.12,
    block_size: tuple[int, int] = (16, 16),
) -> cp.ndarray:
    """
    Run GPU V1 and return the output as a CuPy array.

    This function includes:
    - CPU reflect padding;
    - Host-to-Device transfer;
    - output allocation;
    - kernel execution.

    It does not copy the output back to the CPU.
    """
    if h <= 0:
        raise ValueError(
            "h must be greater than zero."
        )

    block_x, block_y = block_size

    if block_x <= 0 or block_y <= 0:
        raise ValueError(
            "CUDA block dimensions must be positive."
        )

    if block_x * block_y > 1024:
        raise ValueError(
            "CUDA block size exceeds "
            "1024 threads per block."
        )

    (
        padded_cpu,
        patch_radius,
        search_radius,
    ) = prepare_nlm_gpu_v1_input(
        image=image,
        patch_size=patch_size,
        search_window_size=(
            search_window_size
        ),
    )

    height, width = image.shape

    padded_gpu = cp.asarray(
        padded_cpu,
        dtype=cp.float32,
    )

    output_gpu = cp.empty(
        (height, width),
        dtype=cp.float32,
    )

    grid = (
        (width + block_x - 1)
        // block_x,

        (height + block_y - 1)
        // block_y,

        1,
    )

    block = (
        block_x,
        block_y,
        1,
    )

    kernel = get_nlm_gpu_v1_kernel()

    kernel(
        grid,
        block,
        (
            padded_gpu,
            output_gpu,
            np.int32(width),
            np.int32(height),
            np.int32(
                padded_cpu.shape[1]
            ),
            np.int32(patch_radius),
            np.int32(search_radius),
            np.float32(h * h),
        ),
    )

    return output_gpu


def nlm_gpu_v1(
    image: np.ndarray,
    patch_size: int = 3,
    search_window_size: int = 7,
    h: float = 0.12,
    block_size: tuple[int, int] = (16, 16),
) -> np.ndarray:
    """
    Complete GPU V1 wrapper.

    Returns a CPU NumPy float32 image in [0, 1].
    """
    output_gpu = nlm_gpu_v1_device(
        image=image,
        patch_size=patch_size,
        search_window_size=(
            search_window_size
        ),
        h=h,
        block_size=block_size,
    )

    cp.cuda.get_current_stream().synchronize()

    output_cpu = cp.asnumpy(
        output_gpu
    )

    return np.clip(
        output_cpu,
        0.0,
        1.0,
    ).astype(np.float32)
def create_gpu_v1_kernel_launcher(
    image: np.ndarray,
    patch_size: int = 3,
    search_window_size: int = 7,
    h: float = 0.12,
    block_size: tuple[int, int] = (16, 16),
):
    """
    Prepare persistent GPU buffers and return a callable
    that launches only the CUDA kernel.

    Used for kernel-only timing with CUDA Events.
    """
    if h <= 0:
        raise ValueError(
            "h must be greater than zero."
        )

    (
        padded_cpu,
        patch_radius,
        search_radius,
    ) = prepare_nlm_gpu_v1_input(
        image=image,
        patch_size=patch_size,
        search_window_size=(
            search_window_size
        ),
    )

    height, width = image.shape

    padded_gpu = cp.asarray(
        padded_cpu,
        dtype=cp.float32,
    )

    output_gpu = cp.empty(
        (height, width),
        dtype=cp.float32,
    )

    block_x, block_y = block_size

    grid = (
        (width + block_x - 1)
        // block_x,

        (height + block_y - 1)
        // block_y,

        1,
    )

    block = (
        block_x,
        block_y,
        1,
    )

    kernel = get_nlm_gpu_v1_kernel()

    kernel_arguments = (
        padded_gpu,
        output_gpu,
        np.int32(width),
        np.int32(height),
        np.int32(
            padded_cpu.shape[1]
        ),
        np.int32(patch_radius),
        np.int32(search_radius),
        np.float32(h * h),
    )

    def launch_kernel() -> None:
        kernel(
            grid,
            block,
            kernel_arguments,
        )

    return (
        launch_kernel,
        output_gpu,
    )