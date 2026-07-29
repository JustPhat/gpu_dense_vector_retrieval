import numpy as np
from skimage.metrics import (
    peak_signal_noise_ratio,
    structural_similarity,
)


def calculate_psnr(
    reference: np.ndarray,
    candidate: np.ndarray,
) -> float:
    return float(
        peak_signal_noise_ratio(
            reference,
            candidate,
            data_range=1.0,
        )
    )


def calculate_ssim(
    reference: np.ndarray,
    candidate: np.ndarray,
) -> float:
    return float(
        structural_similarity(
            reference,
            candidate,
            data_range=1.0,
        )
    )


def evaluate_denoising(
    clean: np.ndarray,
    noisy: np.ndarray,
    denoised: np.ndarray,
) -> dict[str, float]:
    noisy_psnr = calculate_psnr(
        clean,
        noisy,
    )

    denoised_psnr = calculate_psnr(
        clean,
        denoised,
    )

    return {
        "noisy_psnr": noisy_psnr,
        "denoised_psnr": denoised_psnr,
        "denoised_ssim": calculate_ssim(
            clean,
            denoised,
        ),
        "psnr_improvement": (
            denoised_psnr - noisy_psnr
        ),
    }


def compare_outputs(
    reference: np.ndarray,
    candidate: np.ndarray,
    atol: float = 1e-5,
    rtol: float = 1e-5,
) -> dict[str, float | bool]:
    if reference.shape != candidate.shape:
        raise ValueError(
            "Output shapes do not match."
        )

    error = np.abs(
        reference.astype(np.float64)
        - candidate.astype(np.float64)
    )

    return {
        "max_absolute_error": float(
            np.max(error)
        ),
        "mean_absolute_error": float(
            np.mean(error)
        ),
        "rmse": float(
            np.sqrt(
                np.mean(error * error)
            )
        ),
        "allclose": bool(
            np.allclose(
                reference,
                candidate,
                atol=atol,
                rtol=rtol,
            )
        ),
    }