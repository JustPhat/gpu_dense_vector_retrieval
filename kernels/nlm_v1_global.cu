extern "C" __global__
void nlm_gpu_v1(
    const float* __restrict__ padded_input,
    float* __restrict__ output,
    const int width,
    const int height,
    const int padded_width,
    const int patch_radius,
    const int search_radius,
    const float h_squared
)
{
    const int x =
        blockIdx.x * blockDim.x
        + threadIdx.x;

    const int y =
        blockIdx.y * blockDim.y
        + threadIdx.y;

    if (x >= width || y >= height)
    {
        return;
    }

    const int padding_radius =
        patch_radius + search_radius;

    const int reference_center_x =
        x + padding_radius;

    const int reference_center_y =
        y + padding_radius;

    const int patch_width =
        2 * patch_radius + 1;

    const int patch_area =
        patch_width * patch_width;

    float weighted_sum = 0.0f;
    float weight_sum = 0.0f;

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
            const int candidate_center_x =
                reference_center_x
                + search_offset_x;

            const int candidate_center_y =
                reference_center_y
                + search_offset_y;

            float squared_difference_sum = 0.0f;

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
                    const int reference_x =
                        reference_center_x
                        + patch_offset_x;

                    const int reference_y =
                        reference_center_y
                        + patch_offset_y;

                    const int candidate_x =
                        candidate_center_x
                        + patch_offset_x;

                    const int candidate_y =
                        candidate_center_y
                        + patch_offset_y;

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

                    const float difference =
                        reference_value
                        - candidate_value;

                    squared_difference_sum +=
                        difference * difference;
                }
            }

            const float distance =
                squared_difference_sum
                / static_cast<float>(patch_area);

            const float weight =
                expf(
                    -distance
                    / h_squared
                );

            const float candidate_center_value =
                padded_input[
                    candidate_center_y
                    * padded_width
                    + candidate_center_x
                ];

            weighted_sum +=
                weight
                * candidate_center_value;

            weight_sum += weight;
        }
    }

    const int output_index =
        y * width + x;

    output[output_index] =
        weighted_sum
        / (weight_sum + 1e-12f);
}