#!/usr/bin/env bash
# start-8g.sh -- 8 GB 卡 (RTX 4060, sm_89) launcher, NEW LINE recipe
# (27B ternary / qwen3_5, 2026-10-19 配方, README「新的 8 GB 两套配方」长上下文档).
# Engine: build-89-v2 (源码 9a5a920).
#
# 与旧线 start-8g.bat 的差别(不要混用):
#   - 显式 --kv-capacity 51200 (rk2v4-e8) 取代 --kv-capacity auto + 大窗口
#   - --max-shared-prefixes 0 必带(环模式共享前缀快照会触发
#     "active KV snapshot page ownership is not ordered" fast-fail)
#   - --derive-session-keys、--prefill-chunk 1024
#   - 不再设 NINFER_HOST_PAGEABLE / NINFER_KV_RING(新线环门仍读 NINFER_KV_WINDOW)
#
# 硬边界:题面 token > 池 token ⇒ 单请求答案不可信(引擎打 unverified 告警),
# 超池题面必须走分片网关 (tools/agg_gateway.py)。
#
# 必须在本机 dsh 会话树外运行(NoNewPrivs + kernel 7.0 会破坏 libcuda):
#   systemd-run --user --unit ninfer4060 -- /bin/bash -c \
#     '/media/wen/ubuntu/ninfer-kvmem/start-8g.sh >> /media/wen/ubuntu/ninfer-kvmem/serve.log 2>&1'
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ENGINE="$HERE/build-89-v2/apps/ninfer-serve"
# Model artifact lives on the win share (the working-dir copy is gone); point at the source.
MODEL="/media/wen/win/LLM_models/Qwen3.8-27b-Ninfer/bonsai2_27b_ptq1_native_mtp_vision.ninfer"

# -- 只用 GPU0 (RTX 4060 Laptop, CC 8.9) --------------------------------------
export CUDA_VISIBLE_DEVICES=0

# -- 新线 8 GB 长上下文档 (不开 MTP):公共 env -----------------------------------
export NINFER_KV_WINDOW=10240
export NINFER_KV_RETRIEVE=8192
export NINFER_KV_REUSE_HOSTBACKED=1
export NINFER_TERNARY_PTQ1_FAST=1   # PTQ1 GEMM 快速档(性能开关, 旧配方沿用)

echo "[start-8g] engine = $ENGINE (新线, 长上下文档, 池 51,200 / rk2v4-e8)"
echo "[start-8g] model  = $MODEL"
echo "[start-8g] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES (仅 GPU0 / RTX 4060)"

exec "$ENGINE" "$MODEL" \
  --host 127.0.0.1 --port 8091 --model-id ninfer-local \
  --max-context 262144 --max-concurrency 1 \
  --derive-session-keys --max-shared-prefixes 0 \
  --prefill-chunk 1024 \
  --kv-capacity 51200 --kv-dtype rk2v4-e8 \
  --host-kv-mib 2816
