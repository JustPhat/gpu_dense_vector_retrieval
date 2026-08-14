from __future__ import annotations

import time

import numpy as np

from src.nlm.gpu_v1 import nlm_gpu_v1
from src.nlm.gpu_v2 import nlm_gpu_v2
from src.nlm.gpu_v3 import (
    nlm_gpu_v3,
    prepare_gpu_v3_pipeline_launch,
)


def main() -> None:
    rng = np.random.default_rng(42)

    image = rng.random(
        (64, 64),
        dtype=np.float32,
    )

    patch_size = 7
    search_window_size = 21
    h = 0.12

    print(
        "Checking GPU V3 cross-correlation "
        "prerequisites..."
    )

    start = time.perf_counter()
    v1 = nlm_gpu_v1(
        image=image,
        patch_size=patch_size,
        search_window_size=search_window_size,
        h=h,
        block_size=(16, 16),
    )
    print(
        "GPU V1 OK:",
        f"{time.perf_counter() - start:.3f}s",
    )

    start = time.perf_counter()
    v2 = nlm_gpu_v2(
        image=image,
        patch_size=patch_size,
        search_window_size=search_window_size,
        h=h,
        block_size=(16, 16),
    )
    print(
        "GPU V2 OK:",
        f"{time.perf_counter() - start:.3f}s",
    )

    prepared = prepare_gpu_v3_pipeline_launch(
        image=image,
        patch_size=patch_size,
        search_window_size=search_window_size,
        h=h,
        block_size=(16, 16),
        displacement_batch_size=32,
    )

    print(
        "V3 displacements:",
        prepared["number_of_displacements"],
    )
    print(
        "V3 batches:",
        prepared["number_of_batches"],
    )
    print(
        "V3 batch size:",
        prepared["displacement_batch_size"],
    )

    start = time.perf_counter()
    v3 = nlm_gpu_v3(
        image=image,
        patch_size=patch_size,
        search_window_size=search_window_size,
        h=h,
        block_size=(16, 16),
        displacement_batch_size=32,
    )
    print(
        "GPU V3 OK:",
        f"{time.perf_counter() - start:.3f}s",
    )

    print(
        "V3 vs V1 max abs error:",
        float(
            np.max(
                np.abs(v3 - v1)
            )
        ),
    )

    print(
        "V3 vs V2 max abs error:",
        float(
            np.max(
                np.abs(v3 - v2)
            )
        ),
    )

    print(
        "V3 vs V1 allclose:",
        np.allclose(
            v3,
            v1,
            atol=5e-5,
            rtol=5e-5,
        ),
    )

    print(
        "V3 vs V2 allclose:",
        np.allclose(
            v3,
            v2,
            atol=5e-5,
            rtol=5e-5,
        ),
    )


if __name__ == "__main__":
    main()