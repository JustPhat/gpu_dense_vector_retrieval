from dataclasses import dataclass


@dataclass(frozen=True)
class NLMConfig:
    """
    Lưu toàn bộ cấu hình dùng chung cho thí nghiệm NLM.

    frozen=True nghĩa là sau khi tạo object,
    các giá trị cấu hình không thể bị thay đổi trực tiếp.

    Điều này giúp tránh việc vô tình thay đổi tham số
    giữa CPU, GPU V1, GPU V2 và GPU V3.
    """

    # Kích thước ảnh vuông đầu vào.
    #
    # image_size = 64
    # tương ứng ảnh 64 × 64.
    image_size: int = 64

    # Độ lệch chuẩn của Gaussian noise.
    #
    # noise_sigma càng lớn thì ảnh càng nhiễu.
    noise_sigma: float = 0.08

    # Kích thước patch dùng để so sánh.
    #
    # patch_size = 3
    # tương ứng patch 3 × 3.
    patch_size: int = 3

    # Kích thước search window.
    #
    # search_window_size = 7
    # tương ứng vùng tìm kiếm 7 × 7.
    search_window_size: int = 7

    # Tham số điều khiển độ giảm của similarity weight.
    #
    # weight = exp(-distance / h^2)
    h: float = 0.12

    # Seed dùng để tạo noise có tính reproducibility.
    #
    # Cùng seed sẽ tạo cùng một noise pattern.
    random_seed: int = 42

    # Số thread theo chiều x trong mỗi CUDA block.
    block_size_x: int = 16

    # Số thread theo chiều y trong mỗi CUDA block.
    block_size_y: int = 16

    @property
    def patch_radius(self) -> int:
        """
        Tính bán kính patch từ patch size.

        Ví dụ:
        patch_size = 3
        patch_radius = 1
        """

        return self.patch_size // 2

    @property
    def search_radius(self) -> int:
        """
        Tính bán kính search window.

        Ví dụ:
        search_window_size = 7
        search_radius = 3
        """

        return (
            self.search_window_size
            // 2
        )

    @property
    def padding_radius(self) -> int:
        """
        Tính tổng padding cần thiết.

        Padding phải đủ để bao phủ:
        - search window;
        - patch bên trong search window.

        Ví dụ:
        patch_radius = 1
        search_radius = 3
        padding_radius = 4
        """

        return (
            self.patch_radius
            + self.search_radius
        )

    @property
    def block_size(
        self,
    ) -> tuple[int, int]:
        """
        Trả block size dưới dạng tuple.

        Ví dụ:
        block_size_x = 16
        block_size_y = 16

        kết quả:
        (16, 16)
        """

        return (
            self.block_size_x,
            self.block_size_y,
        )

    def validate(self) -> None:
        """
        Kiểm tra tính hợp lệ của toàn bộ cấu hình.

        Hàm sẽ raise ValueError nếu phát hiện
        một tham số không hợp lệ.
        """

        # Kích thước ảnh phải lớn hơn 0.
        if self.image_size <= 0:
            raise ValueError(
                "image_size phải lớn hơn 0."
            )

        # Patch size phải:
        # - lớn hơn 0;
        # - là số lẻ.
        #
        # Patch cần số lẻ để có một pixel trung tâm rõ ràng.
        if (
            self.patch_size <= 0
            or self.patch_size % 2 == 0
        ):
            raise ValueError(
                "patch_size phải là số nguyên dương lẻ."
            )

        # Search window size cũng phải:
        # - lớn hơn 0;
        # - là số lẻ.
        if (
            self.search_window_size <= 0
            or self.search_window_size % 2 == 0
        ):
            raise ValueError(
                "search_window_size phải là số nguyên dương lẻ."
            )

        # Search window không được nhỏ hơn patch.
        #
        # Vì candidate patch phải nằm trong
        # phạm vi tìm kiếm đã xác định.
        if (
            self.search_window_size
            < self.patch_size
        ):
            raise ValueError(
                "search_window_size phải lớn hơn "
                "hoặc bằng patch_size."
            )

        # Noise sigma không được âm.
        #
        # sigma = 0 vẫn hợp lệ,
        # vì tương ứng không thêm noise.
        if self.noise_sigma < 0:
            raise ValueError(
                "noise_sigma không được âm."
            )

        # h phải lớn hơn 0
        # vì xuất hiện ở mẫu số h^2.
        if self.h <= 0:
            raise ValueError(
                "h phải lớn hơn 0."
            )

        # Kích thước block theo cả hai chiều
        # phải là số dương.
        if (
            self.block_size_x <= 0
            or self.block_size_y <= 0
        ):
            raise ValueError(
                "Kích thước CUDA block phải là số dương."
            )

        # Kiểm tra tổng số thread trong một block.
        #
        # Hầu hết CUDA GPU giới hạn tối đa
        # 1024 thread cho mỗi block.
        if (
            self.block_size_x
            * self.block_size_y
            > 1024
        ):
            raise ValueError(
                "CUDA block không được vượt quá "
                "1024 thread mỗi block."
            )