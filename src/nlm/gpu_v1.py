from functools import lru_cache
from pathlib import Path

import cupy as cp
import numpy as np

from .image_utils import reflect_pad_for_nlm


# Xác định thư mục gốc của project.
#
# __file__:
# đường dẫn tới file gpu_v1.py hiện tại.
#
# resolve():
# chuyển thành đường dẫn tuyệt đối.
#
# parents[2]:
# đi ngược lên 2 cấp thư mục.
#
# Ví dụ:
# src/nlm/gpu_v1.py
# → src/nlm
# → src
# → project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]


# Tạo đường dẫn tới file CUDA kernel V1.
GPU_V1_KERNEL_PATH = (
    PROJECT_ROOT
    / "kernels"
    / "nlm_v1_global.cu"
)


@lru_cache(maxsize=1)
def get_nlm_gpu_v1_kernel() -> cp.RawKernel:
    """
    Đọc và compile CUDA kernel GPU V1.

    Kết quả được cache lại để kernel chỉ cần compile
    một lần trong suốt quá trình chạy Python hiện tại.

    Điều này giúp:
    - tránh compile lặp lại;
    - giảm overhead;
    - giúp benchmark chính xác hơn.
    """

    # Kiểm tra file CUDA kernel có tồn tại hay không.
    if not GPU_V1_KERNEL_PATH.exists():
        raise FileNotFoundError(
            f"Không tìm thấy GPU V1 kernel tại: "
            f"{GPU_V1_KERNEL_PATH}"
        )

    # Đọc toàn bộ source code CUDA từ file .cu.
    kernel_source = GPU_V1_KERNEL_PATH.read_text(
        encoding="utf-8"
    )

    # Tạo RawKernel của CuPy.
    #
    # code:
    # source code CUDA vừa đọc.
    #
    # name:
    # tên hàm kernel được định nghĩa trong file .cu.
    #
    # options:
    # yêu cầu compiler dùng chuẩn C++11.
    return cp.RawKernel(
        code=kernel_source,
        name="nlm_gpu_v1",
        options=("--std=c++11",),
    )


def prepare_nlm_gpu_v1_input(
    image: np.ndarray,
    patch_size: int,
    search_window_size: int,
) -> tuple[
    np.ndarray,
    int,
    int,
]:
    """
    Kiểm tra và chuẩn bị input cho GPU V1.

    Các bước:
    - chuyển ảnh sang NumPy float32;
    - kiểm tra ảnh grayscale 2D;
    - thực hiện reflect padding;
    - trả về patch_radius và search_radius.
    """

    # Chuyển input về NumPy array kiểu float32.
    image = np.asarray(
        image,
        dtype=np.float32,
    )

    # GPU V1 hiện chỉ hỗ trợ ảnh grayscale 2D.
    if image.ndim != 2:
        raise ValueError(
            "GPU V1 chỉ hỗ trợ ảnh grayscale 2D."
        )

    # Gọi hàm dùng chung để:
    # - reflect padding;
    # - tính patch_radius;
    # - tính search_radius.
    return reflect_pad_for_nlm(
        image=image,
        patch_size=patch_size,
        search_window_size=(
            search_window_size
        ),
    )


def nlm_gpu_v1_device(
    image: np.ndarray,
    patch_size: int = 3,
    search_window_size: int = 7,
    h: float = 0.12,
    block_size: tuple[int, int] = (16, 16),
) -> cp.ndarray:
    """
    Chạy GPU V1 và trả output dưới dạng CuPy array.

    Hàm này bao gồm:
    - reflect padding trên CPU;
    - Host-to-Device transfer;
    - cấp phát output trên GPU;
    - launch CUDA kernel.

    Hàm này chưa copy output từ GPU về CPU.
    """

    # h phải dương vì được dùng trong công thức:
    # exp(-distance / h^2)
    if h <= 0:
        raise ValueError(
            "h phải lớn hơn 0."
        )

    # Tách block size thành hai chiều.
    block_x, block_y = block_size

    # Kích thước block không được âm hoặc bằng 0.
    if block_x <= 0 or block_y <= 0:
        raise ValueError(
            "Kích thước CUDA block phải là số dương."
        )

    # CUDA thường giới hạn tối đa 1024 thread/block.
    if block_x * block_y > 1024:
        raise ValueError(
            "CUDA block vượt quá giới hạn "
            "1024 thread mỗi block."
        )

    # Chuẩn bị input:
    # - padded_cpu: ảnh đã reflect padding;
    # - patch_radius: bán kính patch;
    # - search_radius: bán kính search window.
    (
        padded_cpu,
        patch_radius,
        search_radius,
    ) = prepare_nlm_gpu_v1_input(
        image=image,
        patch_size=patch_size,
        search_window_size=(
            search_window_size
        ),
    )

    # Lấy kích thước ảnh gốc.
    height, width = image.shape

    # Copy ảnh đã padding từ CPU sang GPU.
    #
    # Đây là Host-to-Device transfer.
    padded_gpu = cp.asarray(
        padded_cpu,
        dtype=cp.float32,
    )

    # Cấp phát vùng nhớ output trên GPU.
    #
    # cp.empty chỉ cấp phát bộ nhớ,
    # chưa khởi tạo giá trị.
    output_gpu = cp.empty(
        (height, width),
        dtype=cp.float32,
    )

    # Tính số block cần thiết theo chiều x.
    #
    # Công thức làm tròn lên:
    # ceil(width / block_x)
    grid_x = (
        (width + block_x - 1)
        // block_x
    )

    # Tính số block cần thiết theo chiều y.
    grid_y = (
        (height + block_y - 1)
        // block_y
    )

    # Grid 3D theo chuẩn CUDA.
    # Ở đây chỉ dùng hai chiều x và y.
    grid = (
        grid_x,
        grid_y,
        1,
    )

    # Block 3D theo chuẩn CUDA.
    # Ở đây z = 1.
    block = (
        block_x,
        block_y,
        1,
    )

    # Lấy kernel đã compile.
    #
    # Nhờ lru_cache, lần gọi sau sẽ dùng lại kernel cũ.
    kernel = get_nlm_gpu_v1_kernel()

    # Launch CUDA kernel.
    kernel(
        grid,
        block,
        (
            # Ảnh input đã padding trên GPU.
            padded_gpu,

            # Vùng nhớ output trên GPU.
            output_gpu,

            # Chiều rộng ảnh gốc.
            np.int32(width),

            # Chiều cao ảnh gốc.
            np.int32(height),

            # Chiều rộng ảnh đã padding.
            np.int32(
                padded_cpu.shape[1]
            ),

            # Bán kính patch.
            np.int32(patch_radius),

            # Bán kính search window.
            np.int32(search_radius),

            # Truyền h^2 để kernel không phải nhân lại.
            np.float32(h * h),
        ),
    )

    # Trả output vẫn còn nằm trên GPU.
    return output_gpu


def nlm_gpu_v1(
    image: np.ndarray,
    patch_size: int = 3,
    search_window_size: int = 7,
    h: float = 0.12,
    block_size: tuple[int, int] = (16, 16),
) -> np.ndarray:
    """
    Wrapper đầy đủ cho GPU V1.

    Hàm này:
    - chạy GPU V1;
    - đồng bộ CUDA stream;
    - copy output từ GPU về CPU;
    - clip kết quả về [0, 1];
    - trả NumPy float32 array.
    """

    # Chạy phần GPU và nhận output dưới dạng CuPy array.
    output_gpu = nlm_gpu_v1_device(
        image=image,
        patch_size=patch_size,
        search_window_size=(
            search_window_size
        ),
        h=h,
        block_size=block_size,
    )

    # Đợi toàn bộ CUDA operation hiện tại hoàn thành.
    #
    # CUDA chạy bất đồng bộ,
    # nên cần synchronize trước khi copy hoặc đo thời gian.
    cp.cuda.get_current_stream().synchronize()

    # Copy output từ GPU về CPU.
    #
    # Đây là Device-to-Host transfer.
    output_cpu = cp.asnumpy(
        output_gpu
    )

    # Giới hạn output trong khoảng ảnh hợp lệ [0, 1]
    # và đảm bảo kiểu dữ liệu float32.
    return np.clip(
        output_cpu,
        0.0,
        1.0,
    ).astype(np.float32)


def create_gpu_v1_kernel_launcher(
    image: np.ndarray,
    patch_size: int = 3,
    search_window_size: int = 7,
    h: float = 0.12,
    block_size: tuple[int, int] = (16, 16),
):
    """
    Chuẩn bị toàn bộ buffer GPU trước,
    sau đó trả về một hàm chỉ thực hiện kernel launch.

    Hàm này được dùng để benchmark kernel-only.

    Mục tiêu là loại bỏ khỏi thời gian đo:
    - reflect padding trên CPU;
    - Host-to-Device transfer;
    - cấp phát GPU memory;
    - Device-to-Host transfer.
    """

    # Kiểm tra tham số h.
    if h <= 0:
        raise ValueError(
            "h phải lớn hơn 0."
        )

    # Chuẩn bị ảnh padding và các bán kính.
    (
        padded_cpu,
        patch_radius,
        search_radius,
    ) = prepare_nlm_gpu_v1_input(
        image=image,
        patch_size=patch_size,
        search_window_size=(
            search_window_size
        ),
    )

    # Kích thước ảnh gốc.
    height, width = image.shape

    # Copy input sang GPU một lần duy nhất.
    padded_gpu = cp.asarray(
        padded_cpu,
        dtype=cp.float32,
    )

    # Cấp phát output trên GPU một lần duy nhất.
    output_gpu = cp.empty(
        (height, width),
        dtype=cp.float32,
    )

    # Lấy kích thước block.
    block_x, block_y = block_size

    # Tính số block theo chiều x và y.
    grid = (
        (width + block_x - 1)
        // block_x,

        (height + block_y - 1)
        // block_y,

        1,
    )

    # Cấu hình số thread mỗi block.
    block = (
        block_x,
        block_y,
        1,
    )

    # Lấy kernel đã compile.
    kernel = get_nlm_gpu_v1_kernel()

    # Chuẩn bị toàn bộ kernel arguments trước.
    #
    # Các arguments này sẽ được tái sử dụng
    # trong mọi lần benchmark.
    kernel_arguments = (
        padded_gpu,
        output_gpu,
        np.int32(width),
        np.int32(height),
        np.int32(
            padded_cpu.shape[1]
        ),
        np.int32(patch_radius),
        np.int32(search_radius),
        np.float32(h * h),
    )

    def launch_kernel() -> None:
        """
        Chỉ launch CUDA kernel.

        Không thực hiện:
        - padding;
        - transfer;
        - allocation;
        - copy output về CPU.
        """

        kernel(
            grid,
            block,
            kernel_arguments,
        )

    # Trả về:
    # - hàm dùng để launch kernel-only;
    # - output_gpu để giữ vùng nhớ còn tồn tại.
    return (
        launch_kernel,
        output_gpu,
    )