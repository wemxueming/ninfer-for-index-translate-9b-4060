#!/usr/bin/env python3
"""Insert CausalD256H16Kv4 dispatch branches + i8 instantiation files for the 9B geometry."""
import re
from pathlib import Path

BASE = Path("/media/wen/ubuntu/ninfer-kvmem/src/src-tree/fusion-engine-src/src/ops/softmax_attention/dense/causal_cache")
SRC  = Path("/media/wen/ubuntu/ninfer-kvmem/src/src-tree/fusion-engine-src/src/ops/softmax_attention")

FILES = ["prompt.cu", "small_t.cu", "small_t_k8v4.cu", "prompt_k8v4.cu",
         "small_t_fp8.cu", "prompt_fp8.cu", "small_t_nvfp4.cu", "prompt_nvfp4.cu"]
MARK = "launch_for<CausalD256H16Kv2>("
NOTE = "// Sixteen query heads serve two geometries (16x2 and 16x4); the KV head count selects."

for name in FILES:
    p = BASE / name
    lines = p.read_text().splitlines(keepends=True)
    out, i, n = [], 0, 0
    while i < len(lines):
        line = lines[i]
        if MARK in line:
            call, j = [line], i
            while not call[-1].rstrip().endswith(");"):
                j += 1
                call.append(lines[j])
            indent = line[: len(line) - len(line.lstrip())]
            unit = indent + "    "
            view = re.search(r"scale, (\w+),", "".join(call)).group(1)
            new_call = [c.replace("CausalD256H16Kv2", "CausalD256H16Kv4") for c in call]
            out.append(f"{indent}{NOTE}\n")
            out.append(f"{indent}if (q.ne[1] == CausalD256H16Kv4::QHeads && {view}.num_kv_heads == CausalD256H16Kv4::KVHeads) {{\n")
            for k, c in enumerate(new_call):
                out.append(c if k == 0 else unit + c[4:])
            out.append(f"{unit}return;\n")
            out.append(f"{indent}}}\n")
            out.extend(call)
            n += 1
            i = j + 1
            continue
        out.append(line)
        i += 1
    assert n >= 1, f"{name}: no dispatch site found"
    p.write_text("".join(out))
    print(f"{name}: {n} site(s) patched")

# --- i8 instantiation files -------------------------------------------------
TPL = """// Width-{w} H16x4 {kind} instantiation of the INT8-family small-T partial launch; one unit per
// width, geometry and input keeps the long ptxas runs parallel.
#include "ops/softmax_attention/dense/causal_cache/small_t_i8_launch.cuh"

namespace ninfer::ops::detail {{

template void launch_small_t_i8<CausalD256H16Kv4, {w}, {cache}>(bool, bool, const Tensor&, {cache}, const Tensor&, float,
    PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, std::int32_t,
    std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);

}} // namespace ninfer::ops::detail
"""
for w in range(1, 7):
    for kind, cache in (("append", "CausalAppendInput"), ("cached", "CausalCachedInput")):
        (BASE / f"small_t_i8_w{w}_h16v4_{kind}.cu").write_text(TPL.format(w=w, kind=kind, cache=cache))
print("created 12 instantiation files")

# --- sources.cmake -----------------------------------------------------------
cm = SRC / "sources.cmake"
text = cm.read_text()
anchor = '  "${CMAKE_CURRENT_LIST_DIR}/dense/causal_cache/small_t_i8_w6_h16_cached.cu"\n'
assert anchor in text, "anchor line not found in sources.cmake"
new = anchor
for w in range(1, 7):
    new += f'  "{{CMAKE_CURRENT_LIST_DIR}}/dense/causal_cache/small_t_i8_w{w}_h16v4_append.cu"\n'
    new += f'  "{{CMAKE_CURRENT_LIST_DIR}}/dense/causal_cache/small_t_i8_w{w}_h16v4_cached.cu"\n'
cm.write_text(text.replace(anchor, new))
print("sources.cmake updated")
