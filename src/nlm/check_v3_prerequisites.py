import numpy as np

from src.nlm.gpu_v1 import nlm_gpu_v1
from src.nlm.gpu_v2 import nlm_gpu_v2


rng = np.random.default_rng(42)
image = rng.random((32, 32), dtype=np.float32)

configurations = [
    (3, 7),
    (5, 11),
    (7, 21),
]

for patch_size, search_window_size in configurations:
    print(
        f"Checking patch={patch_size}, "
        f"search={search_window_size}..."
    )

    v1 = nlm_gpu_v1(
        image=image,
        patch_size=patch_size,
        search_window_size=search_window_size,
        h=0.12,
        block_size=(16, 16),
    )

    v2 = nlm_gpu_v2(
        image=image,
        patch_size=patch_size,
        search_window_size=search_window_size,
        h=0.12,
        block_size=(16, 16),
    )

    max_abs_error = float(
        np.max(
            np.abs(
                v1.astype(np.float64)
                - v2.astype(np.float64)
            )
        )
    )

    print("  V1 shape:", v1.shape)
    print("  V2 shape:", v2.shape)
    print("  max |V1-V2|:", max_abs_error)
    print()