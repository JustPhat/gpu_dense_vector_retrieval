from __future__ import annotations

import numpy as np
import cupy as cp

from nlm.gpu_v4 import nlm_gpu_v4


def main() -> None:
    rng = np.random.default_rng(42)
    image = rng.random((128, 128), dtype=np.float32)

    output, stats = nlm_gpu_v4(
        image=image,
        patch_size=7,
        search_window_size=21,
        h=0.12,
        block_size=(16, 16),
        edge_search_window_size=11,
        texture_variance_threshold=0.01,
        mean_threshold=0.10,
        variance_ratio_threshold=4.0,
        return_stats=True,
    )

    print("GPU:", cp.cuda.runtime.getDeviceProperties(0)["name"])
    print("Output:", output.shape, output.dtype)
    print("Range:", float(output.min()), float(output.max()))
    print("Acceptance ratio:", stats["acceptance_ratio"])
    print("Rejection ratio:", stats["rejection_ratio"])
    print("Mean accepted / pixel:", stats["mean_accepted_per_pixel"])
    print("Mean considered / pixel:", stats["mean_considered_per_pixel"])


if __name__ == "__main__":
    main()