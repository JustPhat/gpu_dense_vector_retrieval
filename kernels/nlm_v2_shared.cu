extern "C" __global__
void nlm_v2_shared(
    const float* padded_input,
    float* output,
    const int width,
    const int height,
    const int padded_width,
    const int patch_radius,
    const int search_radius,
    const float h_squared,
    const float epsilon
) {
    /*
    Each block processes one output tile.

    Example:
    blockDim = 16 x 16
    patch_radius = 1
    search_radius = 3

    padding_radius = 4
    shared tile = 24 x 24
    */

    extern __shared__ float shared_tile[];

    const int padding_radius =
        patch_radius + search_radius;

    const int tile_width =
        blockDim.x + 2 * padding_radius;

    const int tile_height =
        blockDim.y + 2 * padding_radius;

    /*
    Global output coordinates handled by this thread.
    */
    const int output_x =
        blockIdx.x * blockDim.x + threadIdx.x;

    const int output_y =
        blockIdx.y * blockDim.y + threadIdx.y;

    /*
    Top-left position of this shared tile
    in the padded input image.
    */
    const int tile_global_start_x =
        blockIdx.x * blockDim.x;

    const int tile_global_start_y =
        blockIdx.y * blockDim.y;

    /*
    Threads in the block cooperatively load
    the tile and halo into shared memory.

    The shared tile can contain more pixels
    than the number of threads in the block,
    so each thread may load multiple values.
    */
    for (
        int tile_y = threadIdx.y;
        tile_y < tile_height;
        tile_y += blockDim.y
    ) {
        for (
            int tile_x = threadIdx.x;
            tile_x < tile_width;
            tile_x += blockDim.x
        ) {
            const int global_x =
                tile_global_start_x + tile_x;

            const int global_y =
                tile_global_start_y + tile_y;

            const int shared_index =
                tile_y * tile_width + tile_x;

            shared_tile[shared_index] =
                padded_input[
                    global_y * padded_width
                    + global_x
                ];
        }
    }

    /*
    Wait until the complete shared tile has been loaded.
    */
    __syncthreads();

    /*
    The grid is rounded up, so boundary blocks
    can contain threads outside the output image.
    */
    if (
        output_x >= width
        || output_y >= height
    ) {
        return;
    }

    /*
    Reference patch center inside shared memory.
    */
    const int local_center_x =
        threadIdx.x + padding_radius;

    const int local_center_y =
        threadIdx.y + padding_radius;

    float weighted_sum = 0.0f;
    float weight_sum = 0.0f;

    const int patch_size =
        2 * patch_radius + 1;

    const float patch_area =
        (float)(patch_size * patch_size);

    /*
    Traverse candidate centers in the search window.
    */
    for (
        int offset_y = -search_radius;
        offset_y <= search_radius;
        ++offset_y
    ) {
        for (
            int offset_x = -search_radius;
            offset_x <= search_radius;
            ++offset_x
        ) {
            const int candidate_center_x =
                local_center_x + offset_x;

            const int candidate_center_y =
                local_center_y + offset_y;

            float squared_difference_sum = 0.0f;

            /*
            Compare the reference patch
            with the candidate patch.
            */
            for (
                int patch_y = -patch_radius;
                patch_y <= patch_radius;
                ++patch_y
            ) {
                for (
                    int patch_x = -patch_radius;
                    patch_x <= patch_radius;
                    ++patch_x
                ) {
                    const int reference_x =
                        local_center_x + patch_x;

                    const int reference_y =
                        local_center_y + patch_y;

                    const int candidate_x =
                        candidate_center_x + patch_x;

                    const int candidate_y =
                        candidate_center_y + patch_y;

                    const float reference_value =
                        shared_tile[
                            reference_y * tile_width
                            + reference_x
                        ];

                    const float candidate_value =
                        shared_tile[
                            candidate_y * tile_width
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
                / patch_area;

            const float weight =
                expf(
                    -distance / h_squared
                );

            /*
            NLM uses the candidate center pixel
            in the weighted average.
            */
            const float candidate_center_value =
                shared_tile[
                    candidate_center_y * tile_width
                    + candidate_center_x
                ];

            weighted_sum +=
                weight
                * candidate_center_value;

            weight_sum += weight;
        }
    }

    float result =
        weighted_sum
        / (weight_sum + epsilon);

    /*
    Clamp result to the valid image range.
    */
    result = fminf(
        1.0f,
        fmaxf(0.0f, result)
    );

    output[
        output_y * width + output_x
    ] = result;
}