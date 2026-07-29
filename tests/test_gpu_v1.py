import cupy as cp
import numpy as np
import pytest

from src.nlm.cpu import nlm_cpu_naive
from src.nlm.gpu_v1 import nlm_gpu_v1
from src.nlm.image_utils import (
    add_gaussian_noise,
    create_camera_test_image,
)


def cuda_is_available() -> bool:
    """
    Kiểm tra xem môi trường hiện tại có CUDA GPU
    mà CuPy có thể sử dụng hay không.
    """

    try:
        # getDeviceCount() trả về số lượng CUDA device
        # mà CuPy phát hiện được.
        return (
            cp.cuda.runtime.getDeviceCount()
            > 0
        )

    except cp.cuda.runtime.CUDARuntimeError:
        # Nếu driver, CUDA runtime hoặc GPU gặp lỗi,
        # coi như CUDA không khả dụng.
        return False


# Áp dụng điều kiện skip cho toàn bộ file test này.
#
# Nếu máy không có CUDA GPU:
# - các test GPU sẽ được bỏ qua;
# - pytest không báo fail.
#
# Nếu CUDA khả dụng:
# - tất cả test bên dưới sẽ được chạy.
pytestmark = pytest.mark.skipif(
    not cuda_is_available(),
    reason="CUDA GPU is not available.",
)


def test_gpu_v1_output_shape_and_dtype() -> None:
    """
    Kiểm tra GPU V1:

    - giữ nguyên shape của ảnh;
    - trả về NumPy array dạng float32.
    """

    # Tạo ảnh test grayscale 8 × 8.
    image = create_camera_test_image(
        image_size=8
    )

    # Chạy GPU V1.
    #
    # block_size = (8, 8) nghĩa là:
    # mỗi CUDA block có 64 thread.
    output = nlm_gpu_v1(
        image=image,
        patch_size=3,
        search_window_size=7,
        h=0.12,
        block_size=(8, 8),
    )

    # Output phải có cùng shape với input.
    assert output.shape == image.shape

    # Wrapper GPU V1 trả output về CPU
    # dưới dạng NumPy float32.
    assert output.dtype == np.float32


def test_gpu_v1_preserves_constant_image() -> None:
    """
    Kiểm tra GPU V1 có bảo toàn ảnh constant hay không.

    Với mọi pixel đều bằng 0.5:
    - tất cả patch giống nhau;
    - patch distance bằng 0;
    - weight bằng 1;
    - output vẫn phải gần 0.5.
    """

    # Tạo ảnh constant 8 × 8.
    image = np.full(
        (8, 8),
        0.5,
        dtype=np.float32,
    )

    # Chạy GPU V1.
    output = nlm_gpu_v1(
        image=image,
        patch_size=3,
        search_window_size=7,
        h=0.12,
        block_size=(8, 8),
    )

    # So sánh output với input trong tolerance nhỏ.
    assert np.allclose(
        output,
        image,
        atol=1e-6,
        rtol=1e-6,
    )


def test_gpu_v1_matches_cpu_reference() -> None:
    """
    Đây là test correctness quan trọng nhất.

    Mục tiêu:
    GPU V1 phải cho kết quả gần giống CPU reference
    khi sử dụng cùng:

    - input image;
    - patch size;
    - search window;
    - h;
    - padding;
    - công thức NLM.
    """

    # Tạo ảnh sạch 16 × 16.
    clean_image = create_camera_test_image(
        image_size=16
    )

    # Thêm Gaussian noise với seed cố định.
    #
    # seed = 42 giúp CPU và GPU sử dụng đúng cùng
    # một ảnh noisy trong mọi lần chạy test.
    noisy_image = add_gaussian_noise(
        image=clean_image,
        sigma=0.08,
        seed=42,
    )

    # Tính output bằng CPU reference.
    cpu_output = nlm_cpu_naive(
        image=noisy_image,
        patch_size=3,
        search_window_size=7,
        h=0.12,
    )

    # Tính output bằng GPU V1.
    gpu_output = nlm_gpu_v1(
        image=noisy_image,
        patch_size=3,
        search_window_size=7,
        h=0.12,
        block_size=(8, 8),
    )

    # Kiểm tra toàn bộ pixel của GPU output
    # có gần với CPU output hay không.
    #
    # GPU và CPU có thể khác nhau một lượng rất nhỏ
    # do thứ tự floating-point operations khác nhau.
    assert np.allclose(
        gpu_output,
        cpu_output,
        atol=1e-5,
        rtol=1e-5,
    )

    # Tính sai số tuyệt đối lớn nhất giữa hai output.
    max_absolute_error = float(
        np.max(
            np.abs(
                gpu_output
                - cpu_output
            )
        )
    )

    # Không pixel nào được khác CPU reference
    # quá tolerance 1e-5.
    assert max_absolute_error <= 1e-5


def test_gpu_v1_output_is_in_valid_range() -> None:
    """
    Kiểm tra output GPU V1 nằm trong khoảng [0, 1].

    Vì ảnh đầu vào được normalize về [0, 1],
    output denoised cũng phải nằm trong khoảng này.
    """

    # Tạo ảnh sạch 16 × 16.
    clean_image = create_camera_test_image(
        image_size=16
    )

    # Thêm Gaussian noise.
    noisy_image = add_gaussian_noise(
        image=clean_image,
        sigma=0.08,
        seed=42,
    )

    # Chạy GPU V1.
    output = nlm_gpu_v1(
        image=noisy_image,
        patch_size=3,
        search_window_size=7,
        h=0.12,
    )

    # Giá trị nhỏ nhất không được nhỏ hơn 0.
    assert float(output.min()) >= 0.0

    # Giá trị lớn nhất không được vượt quá 1.
    assert float(output.max()) <= 1.0