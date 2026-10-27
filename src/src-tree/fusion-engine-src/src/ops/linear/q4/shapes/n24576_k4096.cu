#include "ops/linear/q4/q4_shapes.h"
#include "ops/linear/q4/q4_ksplit_launch.cuh"
#include "ops/linear/q4/q4_mma_launch.cuh"

namespace ninfer::ops::detail {
namespace {

using MmaR32C32 = Q4RowSplitMmaGemmSchedule<32, 32, 64, 16, 16, 3, 2, Q4FragmentPipeline::Serial,
                                            Cache::cg, Cache::cg, Q4ScaleLoad::Pair32>;
using MmaR64C64 = Q4RowSplitMmaGemmSchedule<64, 64, 64, 32, 16, 2, 2, Q4FragmentPipeline::Serial,
                                            Cache::cg, Cache::cg, Q4ScaleLoad::Pair32>;

} // namespace

// The 9B dense MLP gate_up projection (n=24576, k=4096). Wide-row, k=4096: the ksplit grid grows
// with the row count and the column-tile MMA schedule is unchanged from the other k=4096 shapes.
Q4Launch select_q4_n24576_k4096(std::int32_t tokens) {
    if (tokens <= 8) return launch_q4_ksplit<24576, 4096, 8>;
    if (tokens <= 24) return launch_q4_ksplit<24576, 4096, 24>;
    if (tokens <= 64) return launch_q4_mma<MmaR32C32>;
    if (tokens <= 128) return launch_q4_mma<MmaR64C64>;
    return launch_q4_mma_r64_c128;
}

} // namespace ninfer::ops::detail
