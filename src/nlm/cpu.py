import numpy as np

from .image_utils import reflect_pad_for_nlm


def nlm_cpu_naive(
    image: np.ndarray,
    patch_size: int = 3,
    search_window_size: int = 7,
    h: float = 0.12,
) -> np.ndarray:
    """
    Naive CPU reference implementation of local-search
    Non-Local Means for a grayscale float32 image.
    """
    if image.ndim != 2:
        raise ValueError(
            "This implementation supports grayscale images only."
        )

    if h <= 0:
        raise ValueError("h must be greater than zero.")

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

    height, width = image.shape
    padding_radius = patch_radius + search_radius

    output = np.zeros(
        (height, width),
        dtype=np.float32,
    )

    h_squared = h * h
    epsilon = 1e-12

    for row in range(height):
        for col in range(width):
            center_row = row + padding_radius
            center_col = col + padding_radius

            reference_patch = padded[
                center_row - patch_radius:
                center_row + patch_radius + 1,
                center_col - patch_radius:
                center_col + patch_radius + 1,
            ]

            weighted_sum = 0.0
            weight_sum = 0.0

            for offset_row in range(
                -search_radius,
                search_radius + 1,
            ):
                for offset_col in range(
                    -search_radius,
                    search_radius + 1,
                ):
                    candidate_row = (
                        center_row + offset_row
                    )
                    candidate_col = (
                        center_col + offset_col
                    )

                    candidate_patch = padded[
                        candidate_row - patch_radius:
                        candidate_row + patch_radius + 1,
                        candidate_col - patch_radius:
                        candidate_col + patch_radius + 1,
                    ]

                    difference = (
                        reference_patch
                        - candidate_patch
                    )

                    distance = float(
                        np.mean(
                            difference * difference
                        )
                    )

                    weight = float(
                        np.exp(
                            -distance / h_squared
                        )
                    )

                    candidate_center_pixel = float(
                        padded[
                            candidate_row,
                            candidate_col,
                        ]
                    )

                    weighted_sum += (
                        weight
                        * candidate_center_pixel
                    )

                    weight_sum += weight

            output[row, col] = (
                weighted_sum
                / (weight_sum + epsilon)
            )

    return np.clip(
        output,
        0.0,
        1.0,
    ).astype(np.float32)