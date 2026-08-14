from pathlib import Path

import cupy as cp
import numpy as np

from .image_utils import reflect_pad_for_nlm


_KERNEL_PATH = (
    Path(__file__).resolve().parents[2]
    / "kernels"
    / "nlm_v2_shared.cu"
)

_KERNEL_NAME = "nlm_v2_shared"

_cached_kernel: cp.RawKernel | None = None


def _get_nlm_v2_kernel() -> cp.RawKernel:
    """
    Load và compile CUDA kernel GPU V2 một lần.

    Các lần gọi sau tái sử dụng kernel đã compile.
    """
    global _cached_kernel

    if _cached_kernel is not None:
        return _cached_kernel

    if not _KERNEL_PATH.exists():
        raise FileNotFoundError(
            f"Không tìm thấy CUDA kernel: {_KERNEL_PATH}"
        )

    kernel_source = _KERNEL_PATH.read_text(
        encoding="utf-8"
    )

    _cached_kernel = cp.RawKernel(
        code=kernel_source,
        name=_KERNEL_NAME,
        options=(
            "--std=c++11",
        ),
    )

    return _cached_kernel


def nlm_gpu_v2(
    image: np.ndarray,
    patch_size: int = 3,
    search_window_size: int = 7,
    h: float = 0.12,
    block_size: tuple[int, int] = (16, 16),
) -> np.ndarray:
    """
    GPU V2 của local-search Non-Local Means.

    Khác GPU V1:
    - V1 đọc reference patch và candidate patch
      trực tiếp từ global memory;
    - V2 cho các thread trong block hợp tác load
      tile + halo vào shared memory;
    - dữ liệu chồng lặp được tái sử dụng trong block.

    Input:
    - ảnh grayscale 2D;
    - dtype được chuyển thành float32;
    - giá trị kỳ vọng nằm trong [0, 1].

    Output:
    - NumPy float32;
    - cùng shape với input.
    """

    if image.ndim != 2:
        raise ValueError(
            "GPU V2 chỉ hỗ trợ ảnh grayscale 2D."
        )

    if h <= 0:
        raise ValueError(
            "h phải lớn hơn 0."
        )

    block_size_x, block_size_y = block_size

    if block_size_x <= 0 or block_size_y <= 0:
        raise ValueError(
            "Block size phải lớn hơn 0."
        )

    if block_size_x * block_size_y > 1024:
        raise ValueError(
            "Số thread mỗi block không được vượt quá 1024."
        )

    image = np.ascontiguousarray(
        image,
        dtype=np.float32,
    )

    (
        padded,
        patch_radius,
        search_radius,
    ) = reflect_pad_for_nlm(
        image=image,
        patch_size=patch_size,
        search_window_size=search_window_size,
    )

    padded = np.ascontiguousarray(
        padded,
        dtype=np.float32,
    )

    height, width = image.shape

    padded_height, padded_width = padded.shape

    padding_radius = (
        patch_radius + search_radius
    )

    tile_width = (
        block_size_x
        + 2 * padding_radius
    )

    tile_height = (
        block_size_y
        + 2 * padding_radius
    )

    shared_memory_bytes = (
        tile_width
        * tile_height
        * np.dtype(np.float32).itemsize
    )

    grid_size_x = (
        width + block_size_x - 1
    ) // block_size_x

    grid_size_y = (
        height + block_size_y - 1
    ) // block_size_y

    padded_gpu = cp.asarray(
        padded,
        dtype=cp.float32,
    )

    output_gpu = cp.empty(
        (height, width),
        dtype=cp.float32,
    )

    kernel = _get_nlm_v2_kernel()

    h_squared = np.float32(h * h)
    epsilon = np.float32(1e-12)

    kernel(
        (
            grid_size_x,
            grid_size_y,
            1,
        ),
        (
            block_size_x,
            block_size_y,
            1,
        ),
        (
            padded_gpu,
            output_gpu,
            np.int32(width),
            np.int32(height),
            np.int32(padded_width),
            np.int32(patch_radius),
            np.int32(search_radius),
            h_squared,
            epsilon,
        ),
        shared_mem=shared_memory_bytes,
    )

    output = cp.asnumpy(output_gpu)

    return np.ascontiguousarray(
        output,
        dtype=np.float32,
    )
def prepare_gpu_v2_kernel_launch(
    image: np.ndarray,
    patch_size: int = 3,
    search_window_size: int = 7,
    h: float = 0.12,
    block_size: tuple[int, int] = (16, 16),
) -> dict[str, object]:
    """
    Chuẩn bị dữ liệu GPU và trả về kernel launcher
    dùng cho kernel-only benchmark.
    """

    if image.ndim != 2:
        raise ValueError(
            "GPU V2 chỉ hỗ trợ ảnh grayscale 2D."
        )

    image = np.ascontiguousarray(
        image,
        dtype=np.float32,
    )

    (
        padded,
        patch_radius,
        search_radius,
    ) = reflect_pad_for_nlm(
        image,
        patch_size,
        search_window_size,
    )

    padded = np.ascontiguousarray(
        padded,
        dtype=np.float32,
    )

    height, width = image.shape
    _, padded_width = padded.shape

    block_x, block_y = block_size

    padding_radius = (
        patch_radius + search_radius
    )

    tile_width = (
        block_x + 2 * padding_radius
    )

    tile_height = (
        block_y + 2 * padding_radius
    )

    shared_memory_bytes = (
        tile_width
        * tile_height
        * np.dtype(np.float32).itemsize
    )

    grid_x = (
        width + block_x - 1
    ) // block_x

    grid_y = (
        height + block_y - 1
    ) // block_y

    padded_gpu = cp.asarray(
        padded,
        dtype=cp.float32,
    )

    output_gpu = cp.empty(
        (height, width),
        dtype=cp.float32,
    )

    kernel = _get_nlm_v2_kernel()

    arguments = (
        padded_gpu,
        output_gpu,
        np.int32(width),
        np.int32(height),
        np.int32(padded_width),
        np.int32(patch_radius),
        np.int32(search_radius),
        np.float32(h * h),
        np.float32(1e-12),
    )

    def kernel_launcher() -> None:
        kernel(
            (grid_x, grid_y, 1),
            (block_x, block_y, 1),
            arguments,
            shared_mem=shared_memory_bytes,
        )

    return {
        "kernel_launcher": kernel_launcher,
        "padded_gpu": padded_gpu,
        "output_gpu": output_gpu,
        "grid_size": (grid_x, grid_y),
        "block_size": (block_x, block_y),
        "shared_memory_bytes": (
            shared_memory_bytes
        ),
    }