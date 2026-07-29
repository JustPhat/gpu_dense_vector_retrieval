import numpy as np

from .image_utils import reflect_pad_for_nlm


def nlm_cpu_naive(
    image: np.ndarray,
    patch_size: int = 3,
    search_window_size: int = 7,
    h: float = 0.12,
) -> np.ndarray:
    """
    Phiên bản CPU reference đơn giản của thuật toán
    local-search Non-Local Means dành cho ảnh grayscale float32.

    Hàm này chủ yếu dùng để:
    - làm baseline về correctness;
    - so sánh với GPU V1, V2 và V3;
    - đo speedup của các phiên bản GPU.

    Đây không phải phiên bản CPU tối ưu.
    """

    # Chỉ hỗ trợ ảnh grayscale 2D.
    #
    # Ví dụ shape hợp lệ:
    # (64, 64)
    #
    # Shape không hợp lệ:
    # (64, 64, 3)
    if image.ndim != 2:
        raise ValueError(
            "Phiên bản này chỉ hỗ trợ ảnh grayscale 2D."
        )

    # h xuất hiện trong công thức:
    #
    # weight = exp(-distance / h^2)
    #
    # Vì vậy h phải lớn hơn 0.
    if h <= 0:
        raise ValueError(
            "h phải lớn hơn 0."
        )

    # Chuyển ảnh thành mảng liên tục trong bộ nhớ
    # và đảm bảo kiểu dữ liệu float32.
    #
    # contiguous array giúp truy cập dữ liệu ổn định hơn
    # và phù hợp với cách GPU xử lý dữ liệu sau này.
    image = np.ascontiguousarray(
        image,
        dtype=np.float32,
    )

    # Thực hiện reflect padding.
    #
    # Hàm trả về:
    # - padded: ảnh đã padding;
    # - patch_radius: bán kính patch;
    # - search_radius: bán kính search window.
    (
        padded,
        patch_radius,
        search_radius,
    ) = reflect_pad_for_nlm(
        image,
        patch_size,
        search_window_size,
    )

    # Lấy kích thước của ảnh gốc.
    height, width = image.shape

    # Tổng padding phải bao phủ cả:
    # - search window;
    # - patch comparison.
    #
    # Ví dụ:
    # patch_radius = 1
    # search_radius = 3
    # padding_radius = 4
    padding_radius = (
        patch_radius
        + search_radius
    )

    # Tạo ảnh output với cùng shape với ảnh input.
    #
    # Ban đầu tất cả giá trị bằng 0.
    output = np.zeros(
        (height, width),
        dtype=np.float32,
    )

    # Tính sẵn h^2 để tránh nhân lại nhiều lần.
    h_squared = h * h

    # Giá trị rất nhỏ để tránh chia cho 0.
    epsilon = 1e-12

    # Duyệt qua từng pixel trong ảnh output.
    #
    # Mỗi cặp (row, col) tương ứng với
    # một reference pixel và một reference patch.
    for row in range(height):
        for col in range(width):

            # Chuyển tọa độ trong ảnh gốc
            # sang tọa độ tương ứng trong ảnh padded.
            center_row = (
                row + padding_radius
            )

            center_col = (
                col + padding_radius
            )

            # Lấy reference patch quanh output pixel hiện tại.
            #
            # Với patch_radius = 1:
            # lấy patch 3 × 3.
            reference_patch = padded[
                center_row - patch_radius:
                center_row + patch_radius + 1,

                center_col - patch_radius:
                center_col + patch_radius + 1,
            ]

            # Tổng giá trị pixel đã nhân với weight.
            weighted_sum = 0.0

            # Tổng toàn bộ weight.
            weight_sum = 0.0

            # Duyệt qua từng candidate center
            # bên trong search window.
            #
            # Với search_radius = 3:
            # offset chạy từ -3 tới +3.
            for offset_row in range(
                -search_radius,
                search_radius + 1,
            ):
                for offset_col in range(
                    -search_radius,
                    search_radius + 1,
                ):

                    # Xác định tọa độ tâm của candidate patch.
                    candidate_row = (
                        center_row
                        + offset_row
                    )

                    candidate_col = (
                        center_col
                        + offset_col
                    )

                    # Lấy candidate patch quanh candidate center.
                    candidate_patch = padded[
                        candidate_row - patch_radius:
                        candidate_row + patch_radius + 1,

                        candidate_col - patch_radius:
                        candidate_col + patch_radius + 1,
                    ]

                    # Tính độ chênh lệch theo từng pixel
                    # giữa reference patch và candidate patch.
                    difference = (
                        reference_patch
                        - candidate_patch
                    )

                    # Tính mean squared difference:
                    #
                    # distance =
                    # mean((reference - candidate)^2)
                    #
                    # Patch càng giống nhau thì distance càng nhỏ.
                    distance = float(
                        np.mean(
                            difference
                            * difference
                        )
                    )

                    # Chuyển distance thành similarity weight.
                    #
                    # distance nhỏ:
                    # weight gần 1
                    #
                    # distance lớn:
                    # weight gần 0
                    weight = float(
                        np.exp(
                            -distance
                            / h_squared
                        )
                    )

                    # Lấy pixel trung tâm của candidate patch.
                    #
                    # NLM dùng pixel trung tâm này
                    # trong weighted average.
                    candidate_center_pixel = float(
                        padded[
                            candidate_row,
                            candidate_col,
                        ]
                    )

                    # Tích lũy:
                    #
                    # weight × candidate center pixel
                    weighted_sum += (
                        weight
                        * candidate_center_pixel
                    )

                    # Tích lũy tổng weight.
                    weight_sum += weight

            # Chuẩn hóa weighted sum
            # để tạo ra output pixel cuối cùng.
            output[row, col] = (
                weighted_sum
                / (weight_sum + epsilon)
            )

    # Giới hạn kết quả trong khoảng [0, 1]
    # và đảm bảo output là float32.
    return np.clip(
        output,
        0.0,
        1.0,
    ).astype(np.float32)