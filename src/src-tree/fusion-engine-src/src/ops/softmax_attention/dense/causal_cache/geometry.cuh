#pragma once

#include "ops/softmax_attention/common/head_mapping.cuh"

namespace ninfer::ops {

template <int QHeadsValue, int KVHeadsValue, int SmallTSplitScaleValue>
struct CausalAttentionGeometry : AttentionHeadMapping<QHeadsValue, KVHeadsValue> {
    static_assert(SmallTSplitScaleValue > 0);

    static constexpr int SmallTSplitScale    = SmallTSplitScaleValue;
    static constexpr int SmallTMaximumSplits = 85 * SmallTSplitScale;
};

using CausalD256H24Kv4 = CausalAttentionGeometry<24, 4, 1>;
using CausalD256H16Kv2 = CausalAttentionGeometry<16, 2, 2>;
// The 9B dense geometry (16 Q heads, 4 KV heads, GQA group of four). The small-T split scale
// matches the other 16-head geometry so the shared q_heads-keyed split capacity bounds stay
// valid; only the wave math (KVHeads) and the GQA group differ.
using CausalD256H16Kv4 = CausalAttentionGeometry<16, 4, 2>;

} // namespace ninfer::ops
