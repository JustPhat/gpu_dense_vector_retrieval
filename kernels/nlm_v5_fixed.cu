// ============================================================
// GPU V5
// Fixed-configuration specialized NLM
//
// Fixed configuration:
//   patch       = 7 x 7
//   search      = 21 x 21
//   block       = 16 x 16
//
// Inherited from GPU V2:
//   - one CUDA thread per output pixel
//   - shared-memory tile + halo
//
// New in GPU V5:
//   - compile-time fixed geometry
//   - fully unrolled 7x7 patch distance
//   - reference patch cached in registers
//   - no runtime patch/search-radius calculations
//
// This remains DENSE NLM:
//   every output pixel still checks all 441 candidates.
//
// NVRTC note:
//   - no #include <cuda_runtime.h>
//   - no #include <math.h>
//   - use __expf directly
// ============================================================

#define BLOCK_X 16
#define BLOCK_Y 16

#define PATCH_RADIUS 3
#define PATCH_SIZE 7
#define PATCH_AREA 49

#define SEARCH_RADIUS 10
#define SEARCH_SIZE 21

// Halo must cover:
// search radius + patch radius
#define HALO_RADIUS 13

// 16 + 2*13 = 42
#define TILE_WIDTH 42
#define TILE_HEIGHT 42
#define TILE_SIZE (TILE_WIDTH * TILE_HEIGHT)

extern "C"
__global__
void nlm_v5_fixed_7x7_21x21(
    const float* __restrict__ padded,
    float* __restrict__ output,
    const int padded_width,
    const int padded_height,
    const int width,
    const int height,
    const float h_squared
)
{
    // ========================================================
    // 1. Shared-memory tile
    //
    // One 16x16 output block requires:
    //
    // 13-pixel halo
    // + 16 output pixels
    // + 13-pixel halo
    //
    // => 42 x 42 shared-memory tile
    // ========================================================

    __shared__ float tile[TILE_SIZE];

    const int tx = (int)threadIdx.x;
    const int ty = (int)threadIdx.y;

    const int tid =
        ty * BLOCK_X
        + tx;

    const int block_origin_x =
        (int)blockIdx.x * BLOCK_X;

    const int block_origin_y =
        (int)blockIdx.y * BLOCK_Y;

    // ========================================================
    // 2. Cooperative tile loading
    //
    // 256 threads cooperatively load 42x42 = 1764 values.
    //
    // padded[] already contains the full reflect padding:
    //
    // padding radius =
    // search radius + patch radius
    // = 10 + 3 = 13
    // ========================================================

    for (
        int index = tid;
        index < TILE_SIZE;
        index += BLOCK_X * BLOCK_Y
    )
    {
        const int local_y =
            index / TILE_WIDTH;

        const int local_x =
            index - local_y * TILE_WIDTH;

        const int global_x =
            block_origin_x + local_x;

        const int global_y =
            block_origin_y + local_y;

        float value = 0.0f;

        if (
            global_x >= 0
            && global_x < padded_width
            && global_y >= 0
            && global_y < padded_height
        )
        {
            value =
                padded[
                    global_y * padded_width
                    + global_x
                ];
        }

        tile[index] = value;
    }

    // Every thread in the block must finish loading
    // before shared memory is used.
    __syncthreads();

    // ========================================================
    // 3. Output coordinate
    // ========================================================

    const int x =
        block_origin_x + tx;

    const int y =
        block_origin_y + ty;

    // Important:
    // the return happens AFTER __syncthreads().
    if (
        x >= width
        || y >= height
    )
    {
        return;
    }

    // ========================================================
    // 4. Reference patch center in shared memory
    //
    // Because shared-memory tile begins at the padded
    // block origin, the center corresponding to output
    // thread (tx, ty) is offset by HALO_RADIUS.
    // ========================================================

    const int reference_center_x =
        tx + HALO_RADIUS;

    const int reference_center_y =
        ty + HALO_RADIUS;

    // ========================================================
    // 5. Cache 7x7 reference patch
    //
    // The reference patch is reused for all:
    //
    // 21 x 21 = 441 candidates.
    //
    // We therefore load its 49 values once.
    //
    // The fixed loop bounds allow the compiler to unroll
    // this code.
    // ========================================================

    float reference_patch[PATCH_AREA];

    int ref_index = 0;

    #pragma unroll
    for (
        int py = -PATCH_RADIUS;
        py <= PATCH_RADIUS;
        ++py
    )
    {
        const int reference_row =
            (
                reference_center_y
                + py
            )
            * TILE_WIDTH;

        #pragma unroll
        for (
            int px = -PATCH_RADIUS;
            px <= PATCH_RADIUS;
            ++px
        )
        {
            reference_patch[ref_index] =
                tile[
                    reference_row
                    + reference_center_x
                    + px
                ];

            ++ref_index;
        }
    }

    // ========================================================
    // 6. Dense NLM accumulation
    // ========================================================

    float weighted_sum = 0.0f;
    float weight_sum = 0.0f;

    // Search loops remain loops.
    //
    // We intentionally do NOT fully unroll 21x21 because
    // that would enormously increase kernel instruction size.
    #pragma unroll 1
    for (
        int search_y = -SEARCH_RADIUS;
        search_y <= SEARCH_RADIUS;
        ++search_y
    )
    {
        const int candidate_center_y =
            reference_center_y
            + search_y;

        #pragma unroll 1
        for (
            int search_x = -SEARCH_RADIUS;
            search_x <= SEARCH_RADIUS;
            ++search_x
        )
        {
            const int candidate_center_x =
                reference_center_x
                + search_x;

            float squared_difference_sum =
                0.0f;

            int patch_index = 0;

            // =================================================
            // 7. Fixed 7x7 patch distance
            //
            // Same exact metric as CPU/V1/V2:
            //
            // MSE(P,Q)
            // = (1/49) * sum((P-Q)^2)
            //
            // Patch loops have compile-time constant bounds.
            // =================================================

            #pragma unroll
            for (
                int py = -PATCH_RADIUS;
                py <= PATCH_RADIUS;
                ++py
            )
            {
                const int candidate_row =
                    (
                        candidate_center_y
                        + py
                    )
                    * TILE_WIDTH;

                #pragma unroll
                for (
                    int px = -PATCH_RADIUS;
                    px <= PATCH_RADIUS;
                    ++px
                )
                {
                    const float candidate_value =
                        tile[
                            candidate_row
                            + candidate_center_x
                            + px
                        ];

                    const float difference =
                        reference_patch[
                            patch_index
                        ]
                        - candidate_value;

                    squared_difference_sum +=
                        difference
                        * difference;

                    ++patch_index;
                }
            }

            // =================================================
            // 8. Convert SSD -> MSE
            // =================================================

            const float distance =
                squared_difference_sum
                * 0.02040816326530612f;
                // 1 / 49

            // =================================================
            // 9. NLM weight
            //
            // __expf is a CUDA device intrinsic and does not
            // require including math.h in this RawKernel source.
            // =================================================

            const float weight =
                __expf(
                    -distance
                    / h_squared
                );

            // Candidate center pixel
            const float candidate_center_pixel =
                tile[
                    candidate_center_y
                    * TILE_WIDTH
                    + candidate_center_x
                ];

            weighted_sum +=
                weight
                * candidate_center_pixel;

            weight_sum +=
                weight;
        }
    }

    // ========================================================
    // 10. Normalize
    // ========================================================

    const float epsilon =
        1.0e-12f;

    float result =
        weighted_sum
        / (weight_sum + epsilon);

    // ========================================================
    // 11. Clamp output to [0, 1]
    //
    // Avoid fminf/fmaxf so no math header is needed.
    // ========================================================

    if (result < 0.0f)
    {
        result = 0.0f;
    }

    if (result > 1.0f)
    {
        result = 1.0f;
    }

    // ========================================================
    // 12. Write output
    // ========================================================

    output[
        y * width
        + x
    ] = result;
}