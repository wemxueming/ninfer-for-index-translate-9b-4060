# Qwen3.5-9B dense: loading and port notes

Reference: ninfer-5060ti `docs/maintainer/qwen3.5-9b-model.md` / `qwen3.5-9b-artifact.md`,
model card huggingface.co/ruwwww/qwen3.5-9b-ninfer.

## Artifact

The published file is a **v2** container (`NINFER\0\2`, 6,514,199,131 B; sha256
`5e823ea5b4df7c75c630cb5ff90017cf17769c68bbf9545640cf981af5ec7bd6`, matches the HF repo).
The engine is v3-only and rejects it with the upgrade hint, so loading is two steps:

```bash
python3 tools/upgrade_ninfer_v2_to_v3.py /mnt/nas/ai/qwen3_5_9b.ninfer qwen3_5_9b_v3.ninfer
./build-9b/apps/ninfer qwen3_5_9b_v3.ninfer --prompt "..."
```

The upgrade rewrites the directory into the v3 components/bindings layout, splits the fused
parent weights into row ranges (`gdn/query_key` [4096,4096] Q4, `gdn/value_z` [8192,4096] Q5,
`attention/query_key` [5120,4096] Q4, `attention/gate_value` [5120,4096] Q4 — row-major element
ranges), and replaces `frontend/chat_template.jinja` with the maintained template
(`tools/chat_templates/qwen3_6.jinja`, `<think>` handling). Object offsets in the v3 directory are
**payload-relative** (payload starts after the header+directory chunk).

## Geometry (dense, no MoE)

hidden 4096, 32 layers, 16 query heads / 4 KV heads @ 256, GDN 32 heads
(k 128 / v 128, conv kernel 4), intermediate 12288, vocab 131072 (indexed proposal domain).

## Port pieces (this tree)

- `src/ops/wrapper/gdn_input_proj.cpp` / `attn_input_proj.cpp`: the fused q4_q5 split kernels admit
  only the 27B profile, so a generic fallback runs the two parents through the registered row-split
  linears into workspace planes, then splits each plane into its destinations. **Tensors here are
  column-major** (`set_contiguous_strides`, dim 0 fastest): a row range of a parent plane is a
  strided block (pitch = parent total rows × 2 B), not a contiguous span. The split therefore uses
  `cudaMemcpy2DAsync` with per-side pitches; plain contiguous `cudaMemcpyAsync` transposes the
  head/tail boundaries and produced a 2.6e+06 PPL (z-plane garbage). Capacity reservation for the
  unregistered profile adds the two BF16 planes (`*_split_workspace_capacity_bytes`).
- `src/ops/softmax_attention/dense/causal_cache/causal_softmax_attention.cpp`: `require_causal_geometry`
  accepts 16 query / 4 KV heads (9B) alongside 24/4 and 16/2.
- `src/ops/gdn_gating_proj/bf16/bf16_gdn_gating_proj_plan.cpp`: `k9bRoutes` — same head count and mma
  tile progression as the 35B; only weight K (4096) differs, absorbed by the geometry-templated kernel.
- `src/ops/wrapper/linear_add.cpp`: no fused Q5 linear_add kernel exists for the 4096-hidden shapes
  (12288→4096 up, 4096→4096 down); fallback projects into workspace then `residual_add`, with the
  matching capacity branch.
- `require_rowsplit` / weight checks are generalized from the hardcoded 5120 K to `weight.k`.

## MTP speculative decoding

The artifact ships the full `Qwen3_5MTP` block (12 tensors, 240 MB, all Q8_G32_FP16; the fused
`mtp/layer/attention/query_key_gate_value` [10240,4096] parent splits into q/k/gate/v row ranges
in that order). Enabling it needed:

- New Q8 dispatch shapes (legacy tables, tuned ladders inherited from the closest upstream
  shape): `N4096K4096`, `N4096K8192`, `N4096K12288`, `N10240K4096`, `N24576K4096`, `N1024K4096`
  (`src/ops/linear/q8/shapes/n4096_k*.cu`, `n1024_k4096.cu`, `n10240_k4096.cu`,
  `n24576_k4096.cu`).
- Q8 K/V pair: `linear_pair` and `q8_pair_plan` only admitted K=5120/2048. Added the K=4096
  branch with `kK4096Routes` carried over from `kK5120Routes` ({1..48 TwoSimtR8C4,
  49..448 DualMmaR32C64, 449+ DualMmaR32C128}) — not yet tuned for K=4096 on sm_86.
- `mtp_split_attn_in` was hardcoded to the 27B [6144,1024,6144,1024] row layout; the kernel now
  takes q/kv row counts at launch and the wrapper validates geometry from the operand shapes.

Usage: `--spec mtp --draft-tokens 4` (1..15; `--adaptive-mtp`, `--mtp-attention-window`,
ngram combine as usual). Measured (4060 laptop, T=4): decode **60.4 tok/s vs 43.9 baseline
(1.38x)**, acceptance 47%, 2.88 accepted tok/round, 0 fallback steps; greedy output is byte
identical to `--spec none` (verification is lossless).

## Verification (RTX 4060 laptop sm89, kv bf16, context/stride 512/256)

- Weights 5.28 GiB, load ~6.3 s; engine ready ~7 s.
- Perplexity (ppl_corpus.txt, 173 scored tokens): **PPL 4.639** (27B reference on same harness: 4.033).
  Before the pitched-copy fix: 2.62e+06.
- CLI: coherent EN/CN answers, counting, haiku; thinking mode (`--reasoning-effort medium --max-new
  700 --post-thinking`) solves arithmetic correctly. Note the chat template always opens
  `<think>` unless `enable_thinking is false`, so a thinking-mode request whose `--max-new` is spent
  inside the think block prints nothing — raise `--max-new` or pass `--no-thinking`.
- serve: `/v1/chat/completions` EN + CN requests return coherent `reasoning_content`; ~228 tok/s
  prefill, ~44 tok/s decode at these settings.
