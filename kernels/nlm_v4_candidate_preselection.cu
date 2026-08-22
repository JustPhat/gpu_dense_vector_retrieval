
// GPU V4 - Candidate Preselection + Adaptive Search
//
// Goal:
//   Keep the V3 distance identity
//       ||P-Q||^2 = ||P||^2 + ||Q||^2 - 2<P,Q>
//   but avoid computing the expensive dot product for candidates that
//   can be rejected cheaply using precomputed local statistics.
//
// Why this is intentionally NOT the dense V3 Product-Map pipeline:
//   Per-pixel pruning is sparse. If we still build dense Product Maps for
//   every displacement, most of the expensive work has already happened.
//   V4 therefore switches to a sparse/direct cross-correlation path:
//       cheap statistics -> reject -> full dot product only if accepted.
//
// One CUDA thread owns one output pixel. No atomics are required.

extern "C" __global__
void build_patch_statistics(
    const float* padded,
    float* patch_energy,
    float* patch_mean,
    float* patch_variance,
    int padded_width,
    int padded_height,
    int patch_radius,
    float inverse_patch_area
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
        const int idx = y * padded_width + x;
        patch_energy[idx] = 0.0f;
        patch_mean[idx] = 0.0f;
        patch_variance[idx] = 0.0f;
        return;
    }

    float sum = 0.0f;
    float sum_sq = 0.0f;

    for (int py = -patch_radius; py <= patch_radius; ++py) {
        const int row = (y + py) * padded_width;
        for (int px = -patch_radius; px <= patch_radius; ++px) {
            const float v = padded[row + x + px];
            sum += v;
            sum_sq += v * v;
        }
    }

    const float mean = sum * inverse_patch_area;
    float variance = sum_sq * inverse_patch_area - mean * mean;
    variance = fmaxf(variance, 0.0f);

    const int idx = y * padded_width + x;
    patch_energy[idx] = sum_sq;
    patch_mean[idx] = mean;
    patch_variance[idx] = variance;
}


extern "C" __global__
void nlm_v4_candidate_preselection(
    const float* padded,
    const float* patch_energy,
    const float* patch_mean,
    const float* patch_variance,
    float* output,
    int* accepted_count,
    int* considered_count,
    int padded_width,
    int output_width,
    int output_height,
    int padding_radius,
    int patch_radius,
    int full_search_radius,
    int edge_search_radius,
    float texture_variance_threshold,
    float mean_threshold,
    float variance_ratio_threshold,
    float inverse_patch_area,
    float inverse_h_squared
) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= output_width || y >= output_height) {
        return;
    }

    const int out_idx = y * output_width + x;

    const int ref_x = x + padding_radius;
    const int ref_y = y + padding_radius;
    const int ref_idx = ref_y * padded_width + ref_x;

    const float ref_energy = patch_energy[ref_idx];
    const float ref_mean = patch_mean[ref_idx];
    const float ref_var = patch_variance[ref_idx];

    // Adaptive search:
    // textured / edge-like regions use a smaller search radius.
    const int local_search_radius =
        (ref_var >= texture_variance_threshold)
        ? edge_search_radius
        : full_search_radius;

    float weighted_sum = 0.0f;
    float weight_sum = 0.0f;
    int local_accepted = 0;
    int local_considered = 0;

    const float eps = 1.0e-8f;

    for (int dy = -full_search_radius; dy <= full_search_radius; ++dy) {
        for (int dx = -full_search_radius; dx <= full_search_radius; ++dx) {

            // Square adaptive window. This cheap test prevents even reading
            // candidate statistics outside the local search radius.
            if (
                dx < -local_search_radius ||
                dx >  local_search_radius ||
                dy < -local_search_radius ||
                dy >  local_search_radius
            ) {
                continue;
            }

            ++local_considered;

            const int cand_x = ref_x + dx;
            const int cand_y = ref_y + dy;
            const int cand_idx = cand_y * padded_width + cand_x;

            const float cand_mean = patch_mean[cand_idx];
            const float cand_var = patch_variance[cand_idx];

            bool accept = true;

            // Always keep the reference patch itself.
            if (dx != 0 || dy != 0) {
                const float mean_diff = fabsf(ref_mean - cand_mean);
                if (mean_diff > mean_threshold) {
                    accept = false;
                }

                if (accept) {
                    const float lo = fminf(ref_var, cand_var);
                    const float hi = fmaxf(ref_var, cand_var);
                    const float variance_ratio = (hi + eps) / (lo + eps);

                    if (variance_ratio > variance_ratio_threshold) {
                        accept = false;
                    }
                }
            }

            if (!accept) {
                continue;
            }

            ++local_accepted;

            // Expensive step: only accepted candidates reach this point.
            // We keep the V3 cross-correlation identity, but compute the dot
            // product directly for this sparse candidate pair.
            float dot_product = 0.0f;

            for (int py = -patch_radius; py <= patch_radius; ++py) {
                const int ref_row = (ref_y + py) * padded_width;
                const int cand_row = (cand_y + py) * padded_width;

                for (int px = -patch_radius; px <= patch_radius; ++px) {
                    const float p = padded[ref_row + ref_x + px];
                    const float q = padded[cand_row + cand_x + px];
                    dot_product += p * q;
                }
            }

            const float cand_energy = patch_energy[cand_idx];

            float squared_distance =
                (ref_energy + cand_energy - 2.0f * dot_product)
                * inverse_patch_area;

            squared_distance = fmaxf(squared_distance, 0.0f);

            const float weight =
                expf(-squared_distance * inverse_h_squared);

            const float candidate_value = padded[cand_idx];

            weighted_sum += weight * candidate_value;
            weight_sum += weight;
        }
    }

    output[out_idx] =
        weight_sum > 0.0f
        ? weighted_sum / weight_sum
        : padded[ref_idx];

    accepted_count[out_idx] = local_accepted;
    considered_count[out_idx] = local_considered;
}
