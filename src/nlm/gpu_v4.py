from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Callable

import cupy as cp
import numpy as np


_KERNEL_FILE = (
    Path(__file__).resolve().parents[2]
    / "kernels"
    / "nlm_v4_candidate_preselection.cu"
)


def _validate_odd_positive(name: str, value: int) -> None:
    if (
        not isinstance(value, int)
        or value <= 0
        or value % 2 == 0
    ):
        raise ValueError(
            f"{name} must be a positive odd integer, got {value!r}."
        )


def validate_gpu_v4_configuration(
    patch_size: int,
    search_window_size: int,
    edge_search_window_size: int,
    h: float,
    block_size: tuple[int, int],
    texture_variance_threshold: float,
    mean_threshold: float,
    variance_ratio_threshold: float,
) -> None:
    _validate_odd_positive("patch_size", patch_size)
    _validate_odd_positive("search_window_size", search_window_size)
    _validate_odd_positive(
        "edge_search_window_size",
        edge_search_window_size,
    )

    if edge_search_window_size > search_window_size:
        raise ValueError(
            "edge_search_window_size must be <= search_window_size."
        )

    if h <= 0.0:
        raise ValueError(f"h must be positive, got {h!r}.")

    if len(block_size) != 2:
        raise ValueError("block_size must be a 2-tuple (x, y).")

    if block_size[0] <= 0 or block_size[1] <= 0:
        raise ValueError("CUDA block dimensions must be positive.")

    if texture_variance_threshold < 0.0:
        raise ValueError(
            "texture_variance_threshold must be non-negative."
        )

    if mean_threshold < 0.0:
        raise ValueError("mean_threshold must be non-negative.")

    if variance_ratio_threshold < 1.0:
        raise ValueError(
            "variance_ratio_threshold must be >= 1."
        )


@lru_cache(maxsize=1)
def get_nlm_gpu_v4_kernels() -> dict[str, cp.RawKernel]:
    source = _KERNEL_FILE.read_text(encoding="utf-8")
    options = ("--std=c++11",)

    return {
        "patch_statistics": cp.RawKernel(
            source,
            "build_patch_statistics",
            options=options,
        ),
        "candidate_preselection": cp.RawKernel(
            source,
            "nlm_v4_candidate_preselection",
            options=options,
        ),
    }


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def prepare_nlm_gpu_v4_input(
    image: np.ndarray,
    patch_size: int,
    search_window_size: int,
) -> dict[str, object]:
    image = np.asarray(image, dtype=np.float32)

    if image.ndim != 2:
        raise ValueError(
            "GPU V4 expects a 2-D grayscale image, "
            f"got shape {image.shape}."
        )

    patch_radius = patch_size // 2
    search_radius = search_window_size // 2
    padding_radius = patch_radius + search_radius

    padded = np.pad(
        image,
        pad_width=padding_radius,
        mode="reflect",
    ).astype(np.float32, copy=False)

    height, width = image.shape

    return {
        "image": image,
        "padded": padded,
        "height": height,
        "width": width,
        "padded_height": padded.shape[0],
        "padded_width": padded.shape[1],
        "patch_radius": patch_radius,
        "search_radius": search_radius,
        "padding_radius": padding_radius,
    }


def prepare_gpu_v4_pipeline_launch(
    image: np.ndarray,
    patch_size: int,
    search_window_size: int,
    h: float,
    block_size: tuple[int, int] = (16, 16),
    *,
    edge_search_window_size: int = 11,
    texture_variance_threshold: float = 0.01,
    mean_threshold: float = 0.10,
    variance_ratio_threshold: float = 4.0,
) -> dict[str, object]:
    """
    GPU V4: Candidate Preselection + Adaptive Search.

    Pipeline
    --------
    1. Precompute patch energy, mean and variance.
    2. One thread owns one output pixel.
    3. Choose a local search radius:
         flat region   -> full search window
         texture/edge  -> smaller edge_search_window_size
    4. Cheaply reject candidate patches using:
         |mean_ref - mean_cand| <= mean_threshold
         variance_ratio <= variance_ratio_threshold
    5. Only accepted candidates compute the expensive cross-correlation.
    6. Use V3 identity:
         ||P-Q||^2 = ||P||^2 + ||Q||^2 - 2<P,Q>
    7. Compute NLM weight and accumulate.

    Important
    ---------
    This V4 deliberately does not build dense Product Maps. Dense maps would
    spend the expensive work before per-pixel pruning can save it. V4 keeps
    the V3 mathematical reformulation but uses a sparse candidate path.
    """

    validate_gpu_v4_configuration(
        patch_size=patch_size,
        search_window_size=search_window_size,
        edge_search_window_size=edge_search_window_size,
        h=h,
        block_size=block_size,
        texture_variance_threshold=texture_variance_threshold,
        mean_threshold=mean_threshold,
        variance_ratio_threshold=variance_ratio_threshold,
    )

    prepared = prepare_nlm_gpu_v4_input(
        image=image,
        patch_size=patch_size,
        search_window_size=search_window_size,
    )

    height = int(prepared["height"])
    width = int(prepared["width"])
    padded_height = int(prepared["padded_height"])
    padded_width = int(prepared["padded_width"])
    patch_radius = int(prepared["patch_radius"])
    search_radius = int(prepared["search_radius"])
    padding_radius = int(prepared["padding_radius"])
    edge_search_radius = edge_search_window_size // 2

    padded_gpu = cp.asarray(prepared["padded"], dtype=cp.float32)

    patch_energy_gpu = cp.empty(
        (padded_height, padded_width),
        dtype=cp.float32,
    )
    patch_mean_gpu = cp.empty_like(patch_energy_gpu)
    patch_variance_gpu = cp.empty_like(patch_energy_gpu)

    output_gpu = cp.empty((height, width), dtype=cp.float32)
    accepted_count_gpu = cp.empty((height, width), dtype=cp.int32)
    considered_count_gpu = cp.empty((height, width), dtype=cp.int32)

    kernels = get_nlm_gpu_v4_kernels()

    block_x, block_y = block_size
    block = (block_x, block_y, 1)

    stats_grid = (
        _ceil_div(padded_width, block_x),
        _ceil_div(padded_height, block_y),
        1,
    )
    output_grid = (
        _ceil_div(width, block_x),
        _ceil_div(height, block_y),
        1,
    )

    inverse_patch_area = np.float32(
        1.0 / (patch_size * patch_size)
    )
    inverse_h_squared = np.float32(1.0 / (h * h))

    def kernel_launcher() -> None:
        kernels["patch_statistics"](
            stats_grid,
            block,
            (
                padded_gpu,
                patch_energy_gpu,
                patch_mean_gpu,
                patch_variance_gpu,
                np.int32(padded_width),
                np.int32(padded_height),
                np.int32(patch_radius),
                inverse_patch_area,
            ),
        )

        kernels["candidate_preselection"](
            output_grid,
            block,
            (
                padded_gpu,
                patch_energy_gpu,
                patch_mean_gpu,
                patch_variance_gpu,
                output_gpu,
                accepted_count_gpu,
                considered_count_gpu,
                np.int32(padded_width),
                np.int32(width),
                np.int32(height),
                np.int32(padding_radius),
                np.int32(patch_radius),
                np.int32(search_radius),
                np.int32(edge_search_radius),
                np.float32(texture_variance_threshold),
                np.float32(mean_threshold),
                np.float32(variance_ratio_threshold),
                inverse_patch_area,
                inverse_h_squared,
            ),
        )

    return {
        **prepared,
        "padded_gpu": padded_gpu,
        "patch_energy_gpu": patch_energy_gpu,
        "patch_mean_gpu": patch_mean_gpu,
        "patch_variance_gpu": patch_variance_gpu,
        "output_gpu": output_gpu,
        "accepted_count_gpu": accepted_count_gpu,
        "considered_count_gpu": considered_count_gpu,
        "kernel_launcher": kernel_launcher,
        "edge_search_window_size": edge_search_window_size,
        "texture_variance_threshold": texture_variance_threshold,
        "mean_threshold": mean_threshold,
        "variance_ratio_threshold": variance_ratio_threshold,
    }


def create_gpu_v4_kernel_launcher(
    image: np.ndarray,
    patch_size: int,
    search_window_size: int,
    h: float,
    block_size: tuple[int, int] = (16, 16),
    *,
    edge_search_window_size: int = 11,
    texture_variance_threshold: float = 0.01,
    mean_threshold: float = 0.10,
    variance_ratio_threshold: float = 4.0,
) -> tuple[Callable[[], None], cp.ndarray]:
    prepared = prepare_gpu_v4_pipeline_launch(
        image=image,
        patch_size=patch_size,
        search_window_size=search_window_size,
        h=h,
        block_size=block_size,
        edge_search_window_size=edge_search_window_size,
        texture_variance_threshold=texture_variance_threshold,
        mean_threshold=mean_threshold,
        variance_ratio_threshold=variance_ratio_threshold,
    )
    return prepared["kernel_launcher"], prepared["output_gpu"]


def nlm_gpu_v4_device(
    image: np.ndarray,
    patch_size: int,
    search_window_size: int,
    h: float,
    block_size: tuple[int, int] = (16, 16),
    *,
    edge_search_window_size: int = 11,
    texture_variance_threshold: float = 0.01,
    mean_threshold: float = 0.10,
    variance_ratio_threshold: float = 4.0,
) -> cp.ndarray:
    prepared = prepare_gpu_v4_pipeline_launch(
        image=image,
        patch_size=patch_size,
        search_window_size=search_window_size,
        h=h,
        block_size=block_size,
        edge_search_window_size=edge_search_window_size,
        texture_variance_threshold=texture_variance_threshold,
        mean_threshold=mean_threshold,
        variance_ratio_threshold=variance_ratio_threshold,
    )
    prepared["kernel_launcher"]()
    return prepared["output_gpu"]


def nlm_gpu_v4(
    image: np.ndarray,
    patch_size: int,
    search_window_size: int,
    h: float,
    block_size: tuple[int, int] = (16, 16),
    *,
    edge_search_window_size: int = 11,
    texture_variance_threshold: float = 0.01,
    mean_threshold: float = 0.10,
    variance_ratio_threshold: float = 4.0,
    return_stats: bool = False,
):
    prepared = prepare_gpu_v4_pipeline_launch(
        image=image,
        patch_size=patch_size,
        search_window_size=search_window_size,
        h=h,
        block_size=block_size,
        edge_search_window_size=edge_search_window_size,
        texture_variance_threshold=texture_variance_threshold,
        mean_threshold=mean_threshold,
        variance_ratio_threshold=variance_ratio_threshold,
    )

    prepared["kernel_launcher"]()
    cp.cuda.Stream.null.synchronize()

    output = cp.asnumpy(prepared["output_gpu"])
    output = np.clip(output, 0.0, 1.0).astype(np.float32, copy=False)

    if not return_stats:
        return output

    accepted = cp.asnumpy(prepared["accepted_count_gpu"])
    considered = cp.asnumpy(prepared["considered_count_gpu"])

    total_considered = int(considered.sum())
    total_accepted = int(accepted.sum())
    acceptance_ratio = (
        total_accepted / total_considered
        if total_considered > 0
        else 0.0
    )

    stats = {
        "accepted_candidates": total_accepted,
        "considered_candidates": total_considered,
        "acceptance_ratio": float(acceptance_ratio),
        "rejection_ratio": float(1.0 - acceptance_ratio),
        "mean_accepted_per_pixel": float(accepted.mean()),
        "mean_considered_per_pixel": float(considered.mean()),
    }

    return output, stats