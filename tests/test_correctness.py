import numpy as np

from src.nlm.cpu import nlm_cpu_naive


def test_cpu_output_shape_and_dtype() -> None:
    """
    Kiểm tra rằng CPU NLM:

    - giữ nguyên kích thước ảnh đầu vào;
    - trả về dữ liệu dạng float32.
    """

    # Tạo một ảnh grayscale 8 × 8.
    # Tất cả pixel đều có giá trị 0.5.
    image = np.full(
        (8, 8),
        0.5,
        dtype=np.float32,
    )

    # Chạy CPU NLM với:
    # - patch 3 × 3;
    # - search window 7 × 7;
    # - h = 0.12.
    output = nlm_cpu_naive(
        image,
        patch_size=3,
        search_window_size=7,
        h=0.12,
    )

    # Ảnh output phải có cùng shape với ảnh input.
    assert output.shape == image.shape

    # Output phải giữ kiểu dữ liệu float32.
    assert output.dtype == np.float32


def test_cpu_preserves_constant_image() -> None:
    """
    Kiểm tra rằng CPU NLM không làm thay đổi
    một ảnh có giá trị pixel hoàn toàn đồng nhất.

    Với ảnh constant:
    - mọi patch đều giống nhau;
    - mọi candidate patch có khoảng cách bằng 0;
    - kết quả sau weighted average vẫn phải là 0.5.
    """

    # Tạo ảnh 8 × 8 với tất cả pixel bằng 0.5.
    image = np.full(
        (8, 8),
        0.5,
        dtype=np.float32,
    )

    # Chạy CPU NLM.
    output = nlm_cpu_naive(
        image,
        patch_size=3,
        search_window_size=7,
        h=0.12,
    )

    # So sánh output với input.
    #
    # atol:
    # sai số tuyệt đối tối đa được chấp nhận.
    #
    # rtol:
    # sai số tương đối tối đa được chấp nhận.
    #
    # Vì đây là tính toán floating-point,
    # không nên dùng phép so sánh bằng tuyệt đối ==.
    assert np.allclose(
        output,
        image,
        atol=1e-6,
        rtol=1e-6,
    )


def test_cpu_output_is_in_valid_range() -> None:
    """
    Kiểm tra rằng output CPU NLM luôn nằm trong
    khoảng giá trị hợp lệ [0, 1].

    Đây là khoảng giá trị của ảnh grayscale
    sau khi được normalize.
    """

    # Tạo random number generator với seed cố định.
    #
    # Dùng seed = 42 giúp test có tính reproducibility:
    # mỗi lần chạy sẽ tạo cùng một ảnh random.
    rng = np.random.default_rng(42)

    # Tạo ảnh grayscale random 8 × 8.
    # Mỗi pixel nằm trong khoảng [0, 1).
    image = rng.random(
        (8, 8),
        dtype=np.float32,
    )

    # Chạy CPU NLM.
    output = nlm_cpu_naive(
        image,
        patch_size=3,
        search_window_size=7,
        h=0.12,
    )

    # Giá trị nhỏ nhất không được nhỏ hơn 0.
    assert float(output.min()) >= 0.0

    # Giá trị lớn nhất không được vượt quá 1.
    assert float(output.max()) <= 1.0