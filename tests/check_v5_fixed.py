from __future__ import annotations

import time

import numpy as np

from src.nlm.cpu import (
    nlm_cpu_naive,
)

from src.nlm.gpu_v2 import (
    nlm_gpu_v2,
)

from src.nlm.gpu_v3 import (
    nlm_gpu_v3,
)

from src.nlm.gpu_v5 import (
    nlm_gpu_v5,
    prepare_gpu_v5_kernel_launch,
)


def main() -> None:
    rng = np.random.default_rng(42)

    # --------------------------------------------------------
    # Correctness input
    # --------------------------------------------------------

    image = rng.random(
        (32, 32),
        dtype=np.float32,
    )

    patch_size = 7
    search_window_size = 21
    h = 0.12
    block_size = (16, 16)

    print("=" * 70)
    print(
        "GPU V5 Fixed 7x7 / 21x21 "
        "Prerequisite Check"
    )
    print("=" * 70)

    print(
        "Input shape:",
        image.shape,
    )

    # --------------------------------------------------------
    # V2
    # --------------------------------------------------------

    start = time.perf_counter()

    v2 = nlm_gpu_v2(
        image=image,
        patch_size=patch_size,
        search_window_size=(
            search_window_size
        ),
        h=h,
        block_size=block_size,
    )

    print(
        "GPU V2:",
        f"{time.perf_counter() - start:.4f}s",
    )

    # --------------------------------------------------------
    # V3
    # --------------------------------------------------------

    start = time.perf_counter()

    v3 = nlm_gpu_v3(
        image=image,
        patch_size=patch_size,
        search_window_size=(
            search_window_size
        ),
        h=h,
        block_size=block_size,
        displacement_batch_size=32,
    )

    print(
        "GPU V3:",
        f"{time.perf_counter() - start:.4f}s",
    )

    # --------------------------------------------------------
    # V5
    # --------------------------------------------------------

    start = time.perf_counter()

    v5 = nlm_gpu_v5(
        image=image,
        patch_size=patch_size,
        search_window_size=(
            search_window_size
        ),
        h=h,
        block_size=block_size,
    )

    print(
        "GPU V5:",
        f"{time.perf_counter() - start:.4f}s",
    )

    # --------------------------------------------------------
    # GPU correctness
    # --------------------------------------------------------

    v5_v2_error = float(
        np.max(
            np.abs(
                v5 - v2
            )
        )
    )

    v5_v3_error = float(
        np.max(
            np.abs(
                v5 - v3
            )
        )
    )

    print()
    print(
        "V5 vs V2 max abs error:",
        v5_v2_error,
    )

    print(
        "V5 vs V3 max abs error:",
        v5_v3_error,
    )

    print(
        "V5 vs V2 allclose:",
        np.allclose(
            v5,
            v2,
            atol=5e-5,
            rtol=5e-5,
        ),
    )

    print(
        "V5 vs V3 allclose:",
        np.allclose(
            v5,
            v3,
            atol=5e-5,
            rtol=5e-5,
        ),
    )

    # --------------------------------------------------------
    # CPU correctness
    #
    # Use very small input because CPU baseline is naive.
    # --------------------------------------------------------

    cpu_image = image[:16, :16].copy()

    print()
    print(
        "Running CPU correctness check "
        "on",
        cpu_image.shape,
    )

    start = time.perf_counter()

    cpu = nlm_cpu_naive(
        image=cpu_image,
        patch_size=patch_size,
        search_window_size=(
            search_window_size
        ),
        h=h,
    )

    print(
        "CPU baseline:",
        f"{time.perf_counter() - start:.4f}s",
    )

    v5_small = nlm_gpu_v5(
        image=cpu_image,
        patch_size=patch_size,
        search_window_size=(
            search_window_size
        ),
        h=h,
        block_size=block_size,
    )

    cpu_error = float(
        np.max(
            np.abs(
                cpu - v5_small
            )
        )
    )

    print(
        "CPU vs V5 max abs error:",
        cpu_error,
    )

    print(
        "CPU vs V5 allclose:",
        np.allclose(
            cpu,
            v5_small,
            atol=5e-5,
            rtol=5e-5,
        ),
    )

    # --------------------------------------------------------
    # Launcher check
    # --------------------------------------------------------

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

    print()
    print(
        "V5 grid:",
        prepared["grid"],
    )

    print(
        "V5 block:",
        prepared["block"],
    )

    print(
        "V5 output shape:",
        prepared[
            "output_gpu"
        ].shape,
    )

    print()
    print(
        "GPU V5 prerequisite check "
        "completed."
    )


if __name__ == "__main__":
    main()