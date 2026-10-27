#include "ops/linear/q8/q8_shapes.h"
#include "ops/linear/q8/q8_ksplit_launch.cuh"

namespace ninfer::ops::detail {
namespace {
using Geometry = Q8N1024K4096;
using Access   = Q8KSplitScaleAccess;
using Stage    = Q8KSplitActivationStage;
// The 9B MTP K/V rows projected on their own (unpacked fallback). Tile ladder follows
// n2048_k4096.
template <int KWarps, int TileTokens, int MinBlocks, Access A, Cache AC, Cache WC, Stage S>
using Tile = Q8KSplitSm8xFourWarpSchedule<KWarps, TileTokens, MinBlocks, A, AC, WC, S>;
using C4   = Tile<16, 8, 1, Access::Direct, Cache::cg, Cache::cg, Stage::RuntimeActive>;
using C8   = Tile<16, 8, 1, Access::Shared, Cache::ca, Cache::cg, Stage::RuntimeActive>;
using C16  = Tile<16, 16, 1, Access::Shared, Cache::ca, Cache::cg, Stage::ActiveOnly>;
using C24  = Tile<8, 24, 2, Access::Shared, Cache::ca, Cache::cg, Stage::RuntimeActive>;
using C32  = Tile<8, 32, 2, Access::Shared, Cache::cg, Cache::cg, Stage::ActiveOnly>;
using C48  = Tile<8, 48, 2, Access::Shared, Cache::ca, Cache::cg, Stage::RuntimeActive>;
using C64  = Tile<8, 64, 2, Access::Shared, Cache::ca, Cache::cg, Stage::RuntimeActive>;
} // namespace

Q8Launch select_q8_n1024_k4096(std::int32_t tokens) {
    if (tokens <= 4) return launch_q8_ksplit<Geometry, 4, C4>;
    if (tokens <= 8) return launch_q8_ksplit<Geometry, 8, C8>;
    if (tokens <= 16) return launch_q8_ksplit<Geometry, 16, C16>;
    if (tokens <= 24) return launch_q8_ksplit<Geometry, 24, C24>;
    if (tokens <= 32) return launch_q8_ksplit<Geometry, 32, C32>;
    if (tokens <= 48) return launch_q8_ksplit<Geometry, 48, C48>;
    if (tokens <= 64) return launch_q8_ksplit<Geometry, 64, C64>;
    if (tokens < 896) return launch_q8_mma_r32_c128;
    return launch_q8_mma_r64_c128;
}

} // namespace ninfer::ops::detail
