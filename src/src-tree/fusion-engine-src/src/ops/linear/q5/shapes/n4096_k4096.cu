#include "ops/linear/q5/q5_shapes.h"
#include "ops/linear/q5/q5_ksplit_launch.cuh"

namespace ninfer::ops::detail {

// The 9B dense attention output and GDN output (n=4096, k=4096).
Q5Launch select_q5_n4096_k4096(std::int32_t tokens) {
    if (tokens == 1) return launch_q5_split4_c1_k4096;
    if (tokens <= 2) return launch_q5_ksplit<4096, 2, 2>;
    if (tokens <= 3) return launch_q5_ksplit<4096, 3, 2>;
    if (tokens <= 4) return launch_q5_ksplit<4096, 4, 2>;
    if (tokens <= 5) return launch_q5_ksplit<4096, 5, 2>;
    if (tokens <= 6) return launch_q5_ksplit<4096, 6, 2>;
    if (tokens <= 7) return launch_q5_ksplit<4096, 7, 2>;
    if (tokens <= 8) return launch_q5_ksplit<4096, 8, 2>;
    if (tokens <= 11) return launch_q5_simt_r8_c4;
    if (tokens <= 64) return launch_q5_mma_r64_c32_s3;
    if (tokens <= 128) return launch_q5_mma_r64_c64;
    return launch_q5_mma_r64_c128;
}

} // namespace ninfer::ops::detail
