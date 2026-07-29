from collections.abc import Callable
import time

import cupy as cp
import numpy as np


def summarize_seconds(
    times_seconds: list[float],
) -> dict[str, object]:
    """
    Tổng hợp các mẫu thời gian có đơn vị giây.

    Hàm trả về:
    - toàn bộ mẫu thời gian;
    - median;
    - mean;
    - standard deviation;
    - min;
    - max.
    """

    # Chuyển danh sách Python sang NumPy array float64.
    #
    # float64 được dùng để giảm sai số khi tính
    # các thống kê thời gian rất nhỏ.
    times = np.asarray(
        times_seconds,
        dtype=np.float64,
    )

    # Phải có ít nhất một mẫu thời gian.
    if times.size == 0:
        raise ValueError(
            "Phải có ít nhất một mẫu thời gian."
        )

    # Trả kết quả dưới dạng dictionary
    # để notebook dễ truy cập bằng key.
    return {
        # Toàn bộ mẫu thời gian gốc.
        "times_seconds": times,

        # Median ít bị ảnh hưởng bởi outlier hơn mean.
        # Đây thường là giá trị chính để báo cáo runtime.
        "median_seconds": float(
            np.median(times)
        ),

        # Thời gian trung bình của tất cả lần chạy.
        "mean_seconds": float(
            np.mean(times)
        ),

        # Độ lệch chuẩn, cho biết runtime dao động nhiều hay ít.
        "std_seconds": float(
            np.std(times)
        ),

        # Lần chạy nhanh nhất.
        "min_seconds": float(
            np.min(times)
        ),

        # Lần chạy chậm nhất.
        "max_seconds": float(
            np.max(times)
        ),
    }


def summarize_milliseconds(
    times_ms: list[float],
) -> dict[str, object]:
    """
    Tổng hợp các mẫu thời gian có đơn vị millisecond.

    Hàm này có logic giống summarize_seconds(),
    nhưng dùng cho CUDA kernel timing bằng CUDA Events.
    """

    # Chuyển danh sách thời gian sang NumPy float64 array.
    times = np.asarray(
        times_ms,
        dtype=np.float64,
    )

    # Không chấp nhận danh sách rỗng.
    if times.size == 0:
        raise ValueError(
            "Phải có ít nhất một mẫu thời gian."
        )

    return {
        # Toàn bộ mẫu thời gian kernel.
        "times_ms": times,

        # Median kernel runtime.
        "median_ms": float(
            np.median(times)
        ),

        # Mean kernel runtime.
        "mean_ms": float(
            np.mean(times)
        ),

        # Mức độ dao động runtime.
        "std_ms": float(
            np.std(times)
        ),

        # Lần kernel chạy nhanh nhất.
        "min_ms": float(
            np.min(times)
        ),

        # Lần kernel chạy chậm nhất.
        "max_ms": float(
            np.max(times)
        ),
    }


def benchmark_cpu_function(
    function: Callable[[], object],
    warmup_runs: int = 0,
    measured_runs: int = 3,
) -> dict[str, object]:
    """
    Đo runtime của một hàm CPU bằng time.perf_counter().

    function phải là một callable không cần argument.

    Ví dụ:

        benchmark_cpu_function(
            function=lambda: nlm_cpu_naive(...),
            measured_runs=3,
        )
    """

    # Số lần warm-up không được âm.
    if warmup_runs < 0:
        raise ValueError(
            "warmup_runs không được âm."
        )

    # Phải có ít nhất một lần đo thật.
    if measured_runs <= 0:
        raise ValueError(
            "measured_runs phải lớn hơn 0."
        )

    # Chạy warm-up trước khi đo.
    #
    # Với CPU NLM, warm-up thường không bắt buộc,
    # nhưng vẫn hỗ trợ để API nhất quán.
    for _ in range(warmup_runs):
        function()

    # Danh sách lưu runtime từng lần chạy.
    times_seconds = []

    # Thực hiện các lần đo chính thức.
    for _ in range(measured_runs):

        # Ghi thời điểm bắt đầu bằng timer độ phân giải cao.
        start_time = time.perf_counter()

        # Chạy hàm cần benchmark.
        function()

        # Tính thời gian đã trôi qua.
        elapsed_seconds = (
            time.perf_counter()
            - start_time
        )

        # Lưu kết quả của lần chạy hiện tại.
        times_seconds.append(
            elapsed_seconds
        )

    # Tổng hợp median, mean, std, min và max.
    return summarize_seconds(
        times_seconds
    )


def benchmark_gpu_end_to_end(
    function: Callable[[], object],
    warmup_runs: int = 2,
    measured_runs: int = 20,
) -> dict[str, object]:
    """
    Đo GPU end-to-end runtime bằng time.perf_counter().

    Phạm vi đo có thể bao gồm:
    - CPU padding;
    - Host-to-Device transfer;
    - GPU memory allocation;
    - CUDA kernel execution;
    - Device-to-Host transfer;
    - output conversion.

    Phạm vi chính xác phụ thuộc vào nội dung của function.
    """

    # Warm-up runs không được âm.
    if warmup_runs < 0:
        raise ValueError(
            "warmup_runs không được âm."
        )

    # Cần ít nhất một lần đo.
    if measured_runs <= 0:
        raise ValueError(
            "measured_runs phải lớn hơn 0."
        )

    # Warm-up giúp:
    # - compile kernel lần đầu;
    # - khởi tạo CUDA context;
    # - làm nóng memory pool;
    # - giảm ảnh hưởng của lần chạy đầu.
    for _ in range(warmup_runs):
        function()

    # Đợi toàn bộ GPU operation từ warm-up hoàn tất.
    cp.cuda.get_current_stream().synchronize()

    # Danh sách lưu thời gian end-to-end.
    times_seconds = []

    for _ in range(measured_runs):

        # Đảm bảo GPU không còn công việc cũ đang chạy
        # trước khi bắt đầu đo.
        cp.cuda.get_current_stream().synchronize()

        # Ghi thời điểm bắt đầu trên CPU.
        start_time = time.perf_counter()

        # Chạy toàn bộ pipeline GPU.
        function()

        # CUDA chạy bất đồng bộ,
        # nên phải đợi GPU hoàn thành trước khi dừng timer.
        cp.cuda.get_current_stream().synchronize()

        # Tính tổng thời gian end-to-end.
        elapsed_seconds = (
            time.perf_counter()
            - start_time
        )

        times_seconds.append(
            elapsed_seconds
        )

    return summarize_seconds(
        times_seconds
    )


def benchmark_cuda_kernel(
    kernel_launcher: Callable[[], object],
    warmup_runs: int = 3,
    measured_runs: int = 50,
) -> dict[str, object]:
    """
    Đo kernel-only runtime bằng CUDA Events.

    kernel_launcher phải chỉ thực hiện CUDA kernel launch.

    Input GPU, output GPU và memory allocation
    cần được chuẩn bị trước khi gọi hàm này.

    Nhờ đó thời gian đo không bao gồm:
    - CPU padding;
    - Host-to-Device transfer;
    - GPU allocation;
    - Device-to-Host transfer.
    """

    # Số lần warm-up không được âm.
    if warmup_runs < 0:
        raise ValueError(
            "warmup_runs không được âm."
        )

    # Phải có ít nhất một lần đo.
    if measured_runs <= 0:
        raise ValueError(
            "measured_runs phải lớn hơn 0."
        )

    # Warm-up kernel trước khi đo.
    #
    # Điều này giúp loại bỏ ảnh hưởng của:
    # - lần compile đầu;
    # - CUDA context initialization;
    # - GPU clock ramp-up.
    for _ in range(warmup_runs):
        kernel_launcher()

    # Đảm bảo toàn bộ warm-up kernel đã chạy xong.
    cp.cuda.get_current_stream().synchronize()

    # Danh sách lưu kernel time theo millisecond.
    times_ms = []

    for _ in range(measured_runs):

        # Tạo CUDA Event đánh dấu thời điểm bắt đầu.
        start_event = cp.cuda.Event()

        # Tạo CUDA Event đánh dấu thời điểm kết thúc.
        end_event = cp.cuda.Event()

        # Ghi start event vào CUDA stream hiện tại.
        start_event.record()

        # Launch kernel cần đo.
        kernel_launcher()

        # Ghi end event sau kernel.
        end_event.record()

        # Đợi cho đến khi end event hoàn thành.
        #
        # Điều này bảo đảm kernel đã thực sự chạy xong.
        end_event.synchronize()

        # Tính thời gian giữa hai CUDA Event.
        #
        # Kết quả của CuPy trả về theo millisecond.
        elapsed_ms = float(
            cp.cuda.get_elapsed_time(
                start_event,
                end_event,
            )
        )

        # Lưu thời gian kernel của lần chạy hiện tại.
        times_ms.append(
            elapsed_ms
        )

    # Tổng hợp thống kê theo millisecond.
    return summarize_milliseconds(
        times_ms
    )