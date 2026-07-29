/*
GPU V1: Moi CUDA thread tinh toan mot output pixel.

Dau vao:
- padded_input: anh grayscale dau vao sau reflect padding.
- output: anh dau ra sau khi khu nhieu.
- width, height: kich thuoc cua anh goc.
- padded_width: chieu rong cua anh sau padding.
- patch_radius: ban kinh cua moi patch dung de so sanh.
- search_radius: ban kinh cua search window.
- h_squared: binh phuong cua tham so loc h.

Thiet ke bo nho:
- Tat ca du lieu anh deu duoc doc truc tiep tu global memory.
- GPU V1 chua su dung shared memory.
*/

extern "C" __global__
void nlm_gpu_v1(
    // Con tro chi doc toi anh dau vao da duoc reflect padding.
    const float* __restrict__ padded_input,

    // Con tro toi anh dau ra ma cac CUDA thread se ghi ket qua vao.
    float* __restrict__ output,

    // Chieu rong cua anh goc, chua padding.
    const int width,

    // Chieu cao cua anh goc, chua padding.
    const int height,

    // Chieu rong cua anh da padding.
    // Dung de chuyen toa do 2D thanh chi so bo nho 1D.
    const int padded_width,

    // Vi du patch size = 3 x 3 thi patch_radius = 1.
    const int patch_radius,

    // Vi du search window = 7 x 7 thi search_radius = 3.
    const int search_radius,

    // Gia tri h * h duoc tinh san de tranh nhan lap lai trong kernel.
    const float h_squared
)
{
    /*
    Tinh toa do output pixel ma thread hien tai chiu trach nhiem xu ly.

    blockIdx xac dinh CUDA block hien tai.
    blockDim la so thread trong moi block.
    threadIdx xac dinh vi tri thread ben trong block.
    */
    const int x =
        blockIdx.x * blockDim.x
        + threadIdx.x;

    const int y =
        blockIdx.y * blockDim.y
        + threadIdx.y;

    /*
    Grid thuong duoc lam tron len de phu het anh,
    vi vay co the ton tai mot so thread nam ngoai bien anh.

    Cac thread do phai ket thuc ngay va khong duoc ghi output.
    */
    if (x >= width || y >= height)
    {
        return;
    }

    /*
    Tong padding phai du de bao phu ca hai pham vi:

    - di chuyen trong search window;
    - di chuyen ben trong patch dung de so sanh.

    Vi du:
    patch_radius = 1
    search_radius = 3
    padding_radius = 4
    */
    const int padding_radius =
        patch_radius + search_radius;

    /*
    Chuyen toa do output pixel trong anh goc
    sang toa do tam tuong ung trong anh da padding.
    */
    const int reference_center_x =
        x + padding_radius;

    const int reference_center_y =
        y + padding_radius;

    /*
    Khoi phuc kich thuoc day du cua patch tu patch_radius.

    Vi du:
    patch_radius = 1
    patch_width = 2 * 1 + 1 = 3
    patch_area = 3 * 3 = 9
    */
    const int patch_width =
        2 * patch_radius + 1;

    const int patch_area =
        patch_width * patch_width;

    /*
    weighted_sum tich luy:

        weight * gia tri pixel trung tam cua candidate patch

    weight_sum tich luy:

        weight

    Ket qua cuoi:

        weighted_sum / weight_sum
    */
    float weighted_sum = 0.0f;
    float weight_sum = 0.0f;

    /*
    Duyet qua tat ca candidate center
    nam trong local search window.

    Voi search_radius = 3:
    offset chay tu -3 toi +3,
    tao thanh search window kich thuoc 7 x 7.
    */
    for (
        int search_offset_y = -search_radius;
        search_offset_y <= search_radius;
        ++search_offset_y
    )
    {
        for (
            int search_offset_x = -search_radius;
            search_offset_x <= search_radius;
            ++search_offset_x
        )
        {
            /*
            Xac dinh toa do tam cua candidate patch
            dua tren vi tri tuong doi so voi reference patch.
            */
            const int candidate_center_x =
                reference_center_x
                + search_offset_x;

            const int candidate_center_y =
                reference_center_y
                + search_offset_y;

            /*
            Bien nay dung de tich luy tong binh phuong sai khac
            giua reference patch va candidate patch hien tai.
            */
            float squared_difference_sum = 0.0f;

            /*
            Duyet qua tung pixel nam ben trong hai patch.

            Voi patch_radius = 1:
            offset patch chay tu -1 toi +1,
            bao phu patch kich thuoc 3 x 3.
            */
            for (
                int patch_offset_y = -patch_radius;
                patch_offset_y <= patch_radius;
                ++patch_offset_y
            )
            {
                for (
                    int patch_offset_x = -patch_radius;
                    patch_offset_x <= patch_radius;
                    ++patch_offset_x
                )
                {
                    /*
                    Toa do cua mot pixel ben trong reference patch.
                    */
                    const int reference_x =
                        reference_center_x
                        + patch_offset_x;

                    const int reference_y =
                        reference_center_y
                        + patch_offset_y;

                    /*
                    Toa do cua pixel tuong ung
                    ben trong candidate patch.
                    */
                    const int candidate_x =
                        candidate_center_x
                        + patch_offset_x;

                    const int candidate_y =
                        candidate_center_y
                        + patch_offset_y;

                    /*
                    Chuyen toa do 2D sang chi so bo nho 1D.

                    Cong thuc:

                        index = row * row_width + column
                    */
                    const float reference_value =
                        padded_input[
                            reference_y
                            * padded_width
                            + reference_x
                        ];

                    const float candidate_value =
                        padded_input[
                            candidate_y
                            * padded_width
                            + candidate_x
                        ];

                    /*
                    Tinh do chenh lech giua hai pixel tuong ung.
                    */
                    const float difference =
                        reference_value
                        - candidate_value;

                    /*
                    Binh phuong do chenh lech
                    roi cong vao tong khoang cach cua patch.
                    */
                    squared_difference_sum +=
                        difference * difference;
                }
            }

            /*
            Chuyen tong binh phuong sai khac thanh
            mean squared patch distance.

                distance =
                    mean((reference - candidate)^2)
            */
            const float distance =
                squared_difference_sum
                / static_cast<float>(patch_area);

            /*
            Chuyen patch distance thanh similarity weight.

            Hai patch cang giong nhau:
                distance nho
                weight gan 1

            Hai patch cang khac nhau:
                distance lon
                weight gan 0
            */
            const float weight =
                expf(
                    -distance
                    / h_squared
                );

            /*
            NLM su dung gia tri pixel trung tam
            cua candidate patch de tinh weighted average.

            Khong dung toan bo pixel cua candidate patch
            trong buoc tich luy output.
            */
            const float candidate_center_value =
                padded_input[
                    candidate_center_y
                    * padded_width
                    + candidate_center_x
                ];

            /*
            Cong gia tri pixel trung tam da nhan voi weight.
            */
            weighted_sum +=
                weight
                * candidate_center_value;

            /*
            Cong weight vao mau so chuan hoa.
            */
            weight_sum += weight;
        }
    }

    /*
    Chuyen toa do output 2D thanh chi so output 1D.
    */
    const int output_index =
        y * width + x;

    /*
    Ghi dung mot output pixel.

    Gia tri epsilon rat nho duoc them vao
    de tranh phep chia cho 0.

    Trong thuc te, candidate patch trung voi reference patch
    luon ton tai va co weight = 1,
    nen weight_sum thong thuong se luon lon hon 0.
    */
    output[output_index] =
        weighted_sum
        / (weight_sum + 1e-12f);
}