from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Callable

import cupy as cp
import numpy as np


_KERNEL_FILE = (
    Path(__file__).resolve().parents[2]
    / "kernels"
    / "nlm_v3_cross_correlation.cu"
)


def _validate_odd_positive(name: str, value: int) -> None:
    if (
        not isinstance(value, int)
        or value <= 0
        or value % 2 == 0
    ):
        raise ValueError(
            f"{name} must be a positive odd integer, "
            f"got {value!r}."
        )


def validate_gpu_v3_configuration(
    patch_size: int,
    search_window_size: int,
    h: float,
    block_size: tuple[int, int],
    displacement_batch_size: int,
) -> None:
    _validate_odd_positive(
        "patch_size",
        patch_size,
    )
    _validate_odd_positive(
        "search_window_size",
        search_window_size,
    )

    if h <= 0.0:
        raise ValueError(
            f"h must be positive, got {h!r}."
        )

    if len(block_size) != 2:
        raise ValueError(
            "block_size must be a 2-tuple (x, y)."
        )

    block_x, block_y = block_size

    if block_x <= 0 or block_y <= 0:
        raise ValueError(
            "CUDA block dimensions must be positive."
        )

    if (
        not isinstance(displacement_batch_size, int)
        or displacement_batch_size <= 0
    ):
        raise ValueError(
            "displacement_batch_size must be "
            "a positive integer."
        )


@lru_cache(maxsize=1)
def get_nlm_gpu_v3_kernels() -> dict[str, cp.RawKernel]:
    source = _KERNEL_FILE.read_text(
        encoding="utf-8",
    )

    options = ("--std=c++11",)

    return {
        "patch_energy": cp.RawKernel(
            source,
            "build_patch_energy",
            options=options,
        ),
        "product_maps": cp.RawKernel(
            source,
            "build_product_maps_batched",
            options=options,
        ),
        "horizontal_sum": cp.RawKernel(
            source,
            "horizontal_box_sum_batched",
            options=options,
        ),
        "accumulate": cp.RawKernel(
            source,
            "accumulate_cross_correlation_batch",
            options=options,
        ),
        "normalize": cp.RawKernel(
            source,
            "normalize_nlm_output",
            options=options,
        ),
    }


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def prepare_nlm_gpu_v3_input(
    image: np.ndarray,
    patch_size: int,
    search_window_size: int,
) -> dict[str, object]:
    image = np.asarray(
        image,
        dtype=np.float32,
    )

    if image.ndim != 2:
        raise ValueError(
            "GPU V3 expects a 2-D grayscale image, "
            f"got shape {image.shape}."
        )

    patch_radius = patch_size // 2
    search_radius = search_window_size // 2

    # The patch centered at the farthest search candidate
    # must still be fully available.
    padding_radius = (
        patch_radius + search_radius
    )

    padded = np.pad(
        image,
        pad_width=padding_radius,
        mode="reflect",
    ).astype(
        np.float32,
        copy=False,
    )

    height, width = image.shape

    # Product maps represent all patch elements needed for
    # every output center before the horizontal/vertical sums.
    work_height = height + 2 * patch_radius
    work_width = width + 2 * patch_radius

    return {
        "image": image,
        "padded": padded,
        "height": height,
        "width": width,
        "padded_height": padded.shape[0],
        "padded_width": padded.shape[1],
        "work_height": work_height,
        "work_width": work_width,
        "patch_radius": patch_radius,
        "search_radius": search_radius,
        "padding_radius": padding_radius,
    }


def _build_offsets(
    search_radius: int,
) -> tuple[np.ndarray, np.ndarray]:
    offsets = [
        (dx, dy)
        for dy in range(
            -search_radius,
            search_radius + 1,
        )
        for dx in range(
            -search_radius,
            search_radius + 1,
        )
    ]

    offset_xs = np.asarray(
        [dx for dx, _ in offsets],
        dtype=np.int32,
    )

    offset_ys = np.asarray(
        [dy for _, dy in offsets],
        dtype=np.int32,
    )

    return offset_xs, offset_ys


def prepare_gpu_v3_pipeline_launch(
    image: np.ndarray,
    patch_size: int,
    search_window_size: int,
    h: float,
    block_size: tuple[int, int] = (16, 16),
    displacement_batch_size: int = 32,
) -> dict[str, object]:
    """
    Chuẩn bị GPU V3 cross-correlation pipeline.

    Điểm khác với V3 running-sum cũ:
    - Không launch một nhóm kernel cho từng displacement.
    - Nhiều displacement được xử lý theo batch.
    - Patch energy được tính một lần và dùng lại.
    - Cross-correlation dựa trên:
        ||P-Q||^2
        = ||P||^2 + ||Q||^2 - 2<P,Q>

    Hàm này chỉ chuẩn bị input, buffer và launcher.
    Việc benchmark compute-only nên đo kernel_launcher().
    """

    validate_gpu_v3_configuration(
        patch_size=patch_size,
        search_window_size=search_window_size,
        h=h,
        block_size=block_size,
        displacement_batch_size=(
            displacement_batch_size
        ),
    )

    prepared = prepare_nlm_gpu_v3_input(
        image=image,
        patch_size=patch_size,
        search_window_size=search_window_size,
    )

    height = int(prepared["height"])
    width = int(prepared["width"])
    padded_height = int(
        prepared["padded_height"]
    )
    padded_width = int(
        prepared["padded_width"]
    )
    work_height = int(
        prepared["work_height"]
    )
    work_width = int(
        prepared["work_width"]
    )
    patch_radius = int(
        prepared["patch_radius"]
    )
    search_radius = int(
        prepared["search_radius"]
    )
    padding_radius = int(
        prepared["padding_radius"]
    )

    offset_xs, offset_ys = _build_offsets(
        search_radius
    )

    number_of_displacements = int(
        offset_xs.size
    )

    effective_batch_size = min(
        displacement_batch_size,
        number_of_displacements,
    )

    padded_gpu = cp.asarray(
        prepared["padded"],
        dtype=cp.float32,
    )

    offset_xs_gpu = cp.asarray(
        offset_xs,
        dtype=cp.int32,
    )
    offset_ys_gpu = cp.asarray(
        offset_ys,
        dtype=cp.int32,
    )

    patch_energy_gpu = cp.empty(
        (padded_height, padded_width),
        dtype=cp.float32,
    )

    product_maps_gpu = cp.empty(
        (
            effective_batch_size,
            work_height,
            work_width,
        ),
        dtype=cp.float32,
    )

    horizontal_sums_gpu = cp.empty(
        (
            effective_batch_size,
            work_height,
            width,
        ),
        dtype=cp.float32,
    )

    weighted_sum_gpu = cp.empty(
        (height, width),
        dtype=cp.float32,
    )

    weight_sum_gpu = cp.empty(
        (height, width),
        dtype=cp.float32,
    )

    output_gpu = cp.empty(
        (height, width),
        dtype=cp.float32,
    )

    kernels = get_nlm_gpu_v3_kernels()

    block_x, block_y = block_size

    block_2d = (
        block_x,
        block_y,
        1,
    )

    patch_energy_grid = (
        _ceil_div(
            padded_width,
            block_x,
        ),
        _ceil_div(
            padded_height,
            block_y,
        ),
        1,
    )

    product_grid_xy = (
        _ceil_div(
            work_width,
            block_x,
        ),
        _ceil_div(
            work_height,
            block_y,
        ),
    )

    horizontal_grid_xy = (
        _ceil_div(
            width,
            block_x,
        ),
        _ceil_div(
            work_height,
            block_y,
        ),
    )

    accumulate_grid = (
        _ceil_div(
            width,
            block_x,
        ),
        _ceil_div(
            height,
            block_y,
        ),
        1,
    )

    normalize_block_size = 256
    normalize_block = (
        normalize_block_size,
        1,
        1,
    )
    normalize_grid = (
        _ceil_div(
            height * width,
            normalize_block_size,
        ),
        1,
        1,
    )

    inverse_patch_area = np.float32(
        1.0 / (patch_size * patch_size)
    )
    inverse_h_squared = np.float32(
        1.0 / (h * h)
    )

    padded_width_i32 = np.int32(
        padded_width
    )
    padded_height_i32 = np.int32(
        padded_height
    )
    work_width_i32 = np.int32(
        work_width
    )
    work_height_i32 = np.int32(
        work_height
    )
    width_i32 = np.int32(width)
    height_i32 = np.int32(height)
    patch_radius_i32 = np.int32(
        patch_radius
    )
    search_radius_i32 = np.int32(
        search_radius
    )
    padding_radius_i32 = np.int32(
        padding_radius
    )
    patch_size_i32 = np.int32(
        patch_size
    )
    pixel_count_i32 = np.int32(
        height * width
    )

    number_of_batches = _ceil_div(
        number_of_displacements,
        effective_batch_size,
    )

    def kernel_launcher() -> None:
        weighted_sum_gpu.fill(0.0)
        weight_sum_gpu.fill(0.0)

        kernels["patch_energy"](
            patch_energy_grid,
            block_2d,
            (
                padded_gpu,
                patch_energy_gpu,
                padded_width_i32,
                padded_height_i32,
                patch_radius_i32,
            ),
        )

        for batch_index in range(
            number_of_batches
        ):
            offset_start = (
                batch_index
                * effective_batch_size
            )

            batch_count = min(
                effective_batch_size,
                number_of_displacements
                - offset_start,
            )

            product_grid = (
                product_grid_xy[0],
                product_grid_xy[1],
                batch_count,
            )

            horizontal_grid = (
                horizontal_grid_xy[0],
                horizontal_grid_xy[1],
                batch_count,
            )

            kernels["product_maps"](
                product_grid,
                block_2d,
                (
                    padded_gpu,
                    product_maps_gpu,
                    offset_xs_gpu,
                    offset_ys_gpu,
                    np.int32(offset_start),
                    np.int32(batch_count),
                    padded_width_i32,
                    work_width_i32,
                    work_height_i32,
                    search_radius_i32,
                ),
            )

            kernels["horizontal_sum"](
                horizontal_grid,
                block_2d,
                (
                    product_maps_gpu,
                    horizontal_sums_gpu,
                    np.int32(batch_count),
                    work_width_i32,
                    work_height_i32,
                    width_i32,
                    patch_size_i32,
                ),
            )

            kernels["accumulate"](
                accumulate_grid,
                block_2d,
                (
                    padded_gpu,
                    patch_energy_gpu,
                    horizontal_sums_gpu,
                    weighted_sum_gpu,
                    weight_sum_gpu,
                    offset_xs_gpu,
                    offset_ys_gpu,
                    np.int32(offset_start),
                    np.int32(batch_count),
                    padded_width_i32,
                    width_i32,
                    height_i32,
                    work_height_i32,
                    padding_radius_i32,
                    patch_size_i32,
                    inverse_patch_area,
                    inverse_h_squared,
                ),
            )

        kernels["normalize"](
            normalize_grid,
            normalize_block,
            (
                weighted_sum_gpu,
                weight_sum_gpu,
                output_gpu,
                pixel_count_i32,
            ),
        )

    return {
        **prepared,
        "padded_gpu": padded_gpu,
        "offset_xs_gpu": offset_xs_gpu,
        "offset_ys_gpu": offset_ys_gpu,
        "patch_energy_gpu": patch_energy_gpu,
        "product_maps_gpu": product_maps_gpu,
        "horizontal_sums_gpu": (
            horizontal_sums_gpu
        ),
        "weighted_sum_gpu": weighted_sum_gpu,
        "weight_sum_gpu": weight_sum_gpu,
        "output_gpu": output_gpu,
        "kernel_launcher": kernel_launcher,
        "number_of_displacements": (
            number_of_displacements
        ),
        "displacement_batch_size": (
            effective_batch_size
        ),
        "number_of_batches": (
            number_of_batches
        ),
    }


def create_gpu_v3_kernel_launcher(
    image: np.ndarray,
    patch_size: int,
    search_window_size: int,
    h: float,
    block_size: tuple[int, int] = (16, 16),
    displacement_batch_size: int = 32,
) -> tuple[
    Callable[[], None],
    cp.ndarray,
]:
    prepared = prepare_gpu_v3_pipeline_launch(
        image=image,
        patch_size=patch_size,
        search_window_size=search_window_size,
        h=h,
        block_size=block_size,
        displacement_batch_size=(
            displacement_batch_size
        ),
    )

    return (
        prepared["kernel_launcher"],
        prepared["output_gpu"],
    )


def nlm_gpu_v3_device(
    image: np.ndarray,
    patch_size: int,
    search_window_size: int,
    h: float,
    block_size: tuple[int, int] = (16, 16),
    displacement_batch_size: int = 32,
) -> cp.ndarray:
    prepared = prepare_gpu_v3_pipeline_launch(
        image=image,
        patch_size=patch_size,
        search_window_size=search_window_size,
        h=h,
        block_size=block_size,
        displacement_batch_size=(
            displacement_batch_size
        ),
    )

    prepared["kernel_launcher"]()

    return prepared["output_gpu"]


def nlm_gpu_v3(
    image: np.ndarray,
    patch_size: int,
    search_window_size: int,
    h: float,
    block_size: tuple[int, int] = (16, 16),
    displacement_batch_size: int = 32,
) -> np.ndarray:
    output_gpu = nlm_gpu_v3_device(
        image=image,
        patch_size=patch_size,
        search_window_size=search_window_size,
        h=h,
        block_size=block_size,
        displacement_batch_size=(
            displacement_batch_size
        ),
    )

    output = cp.asnumpy(output_gpu)

    return np.clip(
        output,
        0.0,
        1.0,
    ).astype(
        np.float32,
        copy=False,
    )