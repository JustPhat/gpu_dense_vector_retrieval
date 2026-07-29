from collections.abc import Callable
import time

import cupy as cp
import numpy as np


def summarize_seconds(
    times_seconds: list[float],
) -> dict[str, object]:
    times = np.asarray(
        times_seconds,
        dtype=np.float64,
    )

    if times.size == 0:
        raise ValueError(
            "At least one timing sample is required."
        )

    return {
        "times_seconds": times,
        "median_seconds": float(
            np.median(times)
        ),
        "mean_seconds": float(
            np.mean(times)
        ),
        "std_seconds": float(
            np.std(times)
        ),
        "min_seconds": float(
            np.min(times)
        ),
        "max_seconds": float(
            np.max(times)
        ),
    }


def summarize_milliseconds(
    times_ms: list[float],
) -> dict[str, object]:
    times = np.asarray(
        times_ms,
        dtype=np.float64,
    )

    if times.size == 0:
        raise ValueError(
            "At least one timing sample is required."
        )

    return {
        "times_ms": times,
        "median_ms": float(
            np.median(times)
        ),
        "mean_ms": float(
            np.mean(times)
        ),
        "std_ms": float(
            np.std(times)
        ),
        "min_ms": float(
            np.min(times)
        ),
        "max_ms": float(
            np.max(times)
        ),
    }


def benchmark_cpu_function(
    function: Callable[[], object],
    warmup_runs: int = 0,
    measured_runs: int = 3,
) -> dict[str, object]:
    if warmup_runs < 0:
        raise ValueError(
            "warmup_runs cannot be negative."
        )

    if measured_runs <= 0:
        raise ValueError(
            "measured_runs must be positive."
        )

    for _ in range(warmup_runs):
        function()

    times_seconds = []

    for _ in range(measured_runs):
        start_time = time.perf_counter()

        function()

        elapsed_seconds = (
            time.perf_counter()
            - start_time
        )

        times_seconds.append(
            elapsed_seconds
        )

    return summarize_seconds(
        times_seconds
    )


def benchmark_gpu_end_to_end(
    function: Callable[[], object],
    warmup_runs: int = 2,
    measured_runs: int = 20,
) -> dict[str, object]:
    if warmup_runs < 0:
        raise ValueError(
            "warmup_runs cannot be negative."
        )

    if measured_runs <= 0:
        raise ValueError(
            "measured_runs must be positive."
        )

    for _ in range(warmup_runs):
        function()

    cp.cuda.get_current_stream().synchronize()

    times_seconds = []

    for _ in range(measured_runs):
        cp.cuda.get_current_stream().synchronize()

        start_time = time.perf_counter()

        function()

        cp.cuda.get_current_stream().synchronize()

        elapsed_seconds = (
            time.perf_counter()
            - start_time
        )

        times_seconds.append(
            elapsed_seconds
        )

    return summarize_seconds(
        times_seconds
    )


def benchmark_cuda_kernel(
    kernel_launcher: Callable[[], object],
    warmup_runs: int = 3,
    measured_runs: int = 50,
) -> dict[str, object]:
    """
    Measure kernel-only execution using CUDA Events.

    The caller should prepare GPU arrays and allocations
    before passing kernel_launcher.
    """
    if warmup_runs < 0:
        raise ValueError(
            "warmup_runs cannot be negative."
        )

    if measured_runs <= 0:
        raise ValueError(
            "measured_runs must be positive."
        )

    for _ in range(warmup_runs):
        kernel_launcher()

    cp.cuda.get_current_stream().synchronize()

    times_ms = []

    for _ in range(measured_runs):
        start_event = cp.cuda.Event()
        end_event = cp.cuda.Event()

        start_event.record()

        kernel_launcher()

        end_event.record()
        end_event.synchronize()

        elapsed_ms = float(
            cp.cuda.get_elapsed_time(
                start_event,
                end_event,
            )
        )

        times_ms.append(
            elapsed_ms
        )

    return summarize_milliseconds(
        times_ms
    )