// GPU V3 - Batched cross-correlation accelerated NLM
//
// Core identity:
// ||P - Q||^2 = ||P||^2 + ||Q||^2 - 2 <P, Q>
//
// This implementation preserves the current project's uniform patch MSE.
// It is inspired by the FNLM cross-correlation reformulation, but it is
// not a literal port of the MATLAB weighted-kernel implementation.
//
// Main optimization over the previous running-sum V3:
// - process many search displacements in one batch;
// - reduce kernel launches from O(number_of_displacements) groups to
//   O(number_of_displacements / batch_size) groups;
// - precompute patch energies once;
// - reuse the energy map for every candidate displacement.

extern "C" __global__
void build_patch_energy(
    const float* padded,
    float* patch_energy,
    int padded_width,
    int padded_height,
    int patch_radius
) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= padded_width || y >= padded_height) {
        return;
    }

    if (
        x < patch_radius ||
        y < patch_radius ||
        x >= padded_width - patch_radius ||
        y >= padded_height - patch_radius
    ) {
        patch_energy[y * padded_width + x] = 0.0f;
        return;
    }

    float sum = 0.0f;

    for (int py = -patch_radius; py <= patch_radius; ++py) {
        const int row = (y + py) * padded_width;

        for (int px = -patch_radius; px <= patch_radius; ++px) {
            const float value = padded[row + x + px];
            sum += value * value;
        }
    }

    patch_energy[y * padded_width + x] = sum;
}


extern "C" __global__
void build_product_maps_batched(
    const float* padded,
    float* product_maps,
    const int* offset_xs,
    const int* offset_ys,
    int offset_start,
    int batch_count,
    int padded_width,
    int work_width,
    int work_height,
    int search_radius
) {
    const int wx = blockIdx.x * blockDim.x + threadIdx.x;
    const int wy = blockIdx.y * blockDim.y + threadIdx.y;
    const int b = blockIdx.z;

    if (
        wx >= work_width ||
        wy >= work_height ||
        b >= batch_count
    ) {
        return;
    }

    const int offset_index = offset_start + b;
    const int dx = offset_xs[offset_index];
    const int dy = offset_ys[offset_index];

    // work coordinate (0,0) corresponds to padded coordinate
    // (search_radius, search_radius).
    const int ref_x = search_radius + wx;
    const int ref_y = search_radius + wy;

    const int cand_x = ref_x + dx;
    const int cand_y = ref_y + dy;

    const float ref_value =
        padded[ref_y * padded_width + ref_x];

    const float cand_value =
        padded[cand_y * padded_width + cand_x];

    const int index =
        (b * work_height + wy) * work_width + wx;

    product_maps[index] = ref_value * cand_value;
}


extern "C" __global__
void horizontal_box_sum_batched(
    const float* product_maps,
    float* horizontal_sums,
    int batch_count,
    int work_width,
    int work_height,
    int output_width,
    int patch_size
) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int wy = blockIdx.y * blockDim.y + threadIdx.y;
    const int b = blockIdx.z;

    if (
        x >= output_width ||
        wy >= work_height ||
        b >= batch_count
    ) {
        return;
    }

    float sum = 0.0f;

    const int product_base =
        (b * work_height + wy) * work_width + x;

    for (int k = 0; k < patch_size; ++k) {
        sum += product_maps[product_base + k];
    }

    const int output_index =
        (b * work_height + wy) * output_width + x;

    horizontal_sums[output_index] = sum;
}


extern "C" __global__
void accumulate_cross_correlation_batch(
    const float* padded,
    const float* patch_energy,
    const float* horizontal_sums,
    float* weighted_sum,
    float* weight_sum,
    const int* offset_xs,
    const int* offset_ys,
    int offset_start,
    int batch_count,
    int padded_width,
    int output_width,
    int output_height,
    int work_height,
    int padding_radius,
    int patch_size,
    float inverse_patch_area,
    float inverse_h_squared
) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= output_width || y >= output_height) {
        return;
    }

    const int output_index = y * output_width + x;

    const int ref_center_x = x + padding_radius;
    const int ref_center_y = y + padding_radius;

    const float ref_energy =
        patch_energy[
            ref_center_y * padded_width + ref_center_x
        ];

    float local_weighted_sum = 0.0f;
    float local_weight_sum = 0.0f;

    for (int b = 0; b < batch_count; ++b) {
        const int offset_index = offset_start + b;
        const int dx = offset_xs[offset_index];
        const int dy = offset_ys[offset_index];

        float dot_product = 0.0f;

        // horizontal_sums already contains the width-wise
        // patch sum. Sum patch_size rows to complete the
        // 2-D cross-correlation / dot product.
        for (int ky = 0; ky < patch_size; ++ky) {
            const int hs_index =
                (b * work_height + (y + ky))
                * output_width
                + x;

            dot_product += horizontal_sums[hs_index];
        }

        const int cand_center_x = ref_center_x + dx;
        const int cand_center_y = ref_center_y + dy;

        const float cand_energy =
            patch_energy[
                cand_center_y * padded_width + cand_center_x
            ];

        float squared_distance =
            (
                ref_energy
                + cand_energy
                - 2.0f * dot_product
            )
            * inverse_patch_area;

        // Floating-point roundoff can make the identity produce
        // a very small negative value although the true SSD >= 0.
        squared_distance = fmaxf(squared_distance, 0.0f);

        const float weight =
            expf(-squared_distance * inverse_h_squared);

        const float candidate_value =
            padded[
                cand_center_y * padded_width + cand_center_x
            ];

        local_weighted_sum += weight * candidate_value;
        local_weight_sum += weight;
    }

    // One thread owns one output pixel, so no atomic operation
    // is needed here. Each batch adds one partial result.
    weighted_sum[output_index] += local_weighted_sum;
    weight_sum[output_index] += local_weight_sum;
}


extern "C" __global__
void normalize_nlm_output(
    const float* weighted_sum,
    const float* weight_sum,
    float* output,
    int pixel_count
) {
    const int index =
        blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= pixel_count) {
        return;
    }

    const float denominator = weight_sum[index];

    output[index] =
        denominator > 0.0f
        ? weighted_sum[index] / denominator
        : 0.0f;
}