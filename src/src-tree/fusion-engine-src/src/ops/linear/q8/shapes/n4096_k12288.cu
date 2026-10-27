#include "ops/linear/q8/q8_shapes.h"
#include "ops/linear/q8/q8_ksplit_launch.cuh"

namespace ninfer::ops::detail {
namespace {
using Geometry = Q8N4096K12288;
using Access   = Q8KSplitScaleAccess;
using Stage    = Q8KSplitActivationStage;
// The 9B MTP down projection. Tile ladder follows n5120_k17408 (four-K-warp tiles beyond T=16).
using C4  = Q8KSplitSchedule<8, 8, 2, Access::Direct, Cache::ca, Cache::cg, Stage::RuntimeActive>;
using C8  = Q8KSplitSchedule<8, 8, 2, Access::Shared, Cache::ca, Cache::cg, Stage::RuntimeActive>;
using C16 = Q8KSplitSchedule<8, 16, 2, Access::Shared, Cache::ca, Cache::cg, Stage::ActiveOnly>;
using C24 = Q8KSplitSchedule<4, 24, 2, Access::Shared, Cache::ca, Cache::cg, Stage::ActiveOnly>;
using C32 = Q8KSplitSchedule<4, 32, 2, Access::Shared, Cache::ca, Cache::cg, Stage::ActiveOnly>;
using C40 = Q8KSplitSchedule<4, 40, 3, Access::Shared, Cache::ca, Cache::cg, Stage::RuntimeActive>;
using C48 = Q8KSplitSchedule<4, 48, 2, Access::Shared, Cache::ca, Cache::cg, Stage::RuntimeActive>;
using C56 = Q8KSplitSchedule<4, 56, 3, Access::Shared, Cache::ca, Cache::cg, Stage::ActiveOnly>;
using C64 = Q8KSplitSchedule<4, 64, 3, Access::Shared, Cache::ca, Cache::cg, Stage::ActiveOnly>;
} // namespace

Q8Launch select_q8_n4096_k12288(std::int32_t tokens) {
    if (tokens <= 4) return launch_q8_ksplit<Geometry, 4, C4>;
    if (tokens <= 8) return launch_q8_ksplit<Geometry, 8, C8>;
    if (tokens <= 16) return launch_q8_ksplit<Geometry, 16, C16>;
    if (tokens <= 24) return launch_q8_ksplit<Geometry, 24, C24>;
    if (tokens <= 32) return launch_q8_ksplit<Geometry, 32, C32>;
    if (tokens <= 40) return launch_q8_ksplit<Geometry, 40, C40>;
    if (tokens <= 48) return launch_q8_ksplit<Geometry, 48, C48>;
    if (tokens <= 64) return launch_q8_mma_r32_c64;
    if (tokens <= 128) return launch_q8_mma_r32_c128;
    if (tokens <= 192) return launch_q8_mma_r64_c96;
    return launch_q8_mma_r64_c128;
}

} // namespace ninfer::ops::detail
