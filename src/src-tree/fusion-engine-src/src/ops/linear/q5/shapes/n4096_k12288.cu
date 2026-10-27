#include "ops/linear/q5/q5_shapes.h"
#include "ops/linear/q5/q5_ksplit_launch.cuh"

namespace ninfer::ops::detail {

// The 9B dense MLP down projection (n=4096, k=12288).
Q5Launch select_q5_n4096_k12288(std::int32_t tokens) {
    if (tokens == 1) return launch_q5_split4_c1_k12288;
    if (tokens <= 2) return launch_q5_ksplit<12288, 2, 4>;
    if (tokens <= 3) return launch_q5_ksplit<12288, 3, 4>;
    if (tokens <= 4) return launch_q5_ksplit<12288, 4, 4>;
    if (tokens <= 5) return launch_q5_ksplit<12288, 5, 4>;
    if (tokens <= 6) return launch_q5_ksplit<12288, 6, 4>;
    if (tokens <= 7) return launch_q5_ksplit<12288, 7, 4>;
    if (tokens <= 8) return launch_q5_ksplit<12288, 8, 4>;
    if (tokens <= 16) return launch_q5_mma_r64_c16;
    if (tokens <= 64) return launch_q5_mma_r64_c32_s3;
    if (tokens <= 128) return launch_q5_mma_r64_c64;
    return launch_q5_mma_r64_c128;
}

} // namespace ninfer::ops::detail
