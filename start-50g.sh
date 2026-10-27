#!/usr/bin/env bash
# start-50g.sh -- RTX 50 系 (sm_120a, consumer Blackwell) launcher for the
# build-120a-v2 engine (源码 9a5a920, 新线), 16 GB 档 (RTX 5060 Ti).
# UNTESTED (2026-10-24): 按新线 8 GB 配方的显式池风格放大到 16 GB 档
# (docs/03 16 GB 行: 窗口 131,072–196,608, 推算·未验); 未在本机 5060 Ti 上运行过。
#
# Usage: systemd-run --user --unit ninfer50g -- /bin/bash -c \
#          '/media/wen/ubuntu/ninfer-kvmem/start-50g.sh >> /media/wen/ubuntu/ninfer-kvmem/serve-50g.log 2>&1'
#
# NOTE: 本机两张 5060 Ti 在 CUDA 设备 1 和 2 (设备 0 是 RTX 4060, 归 start-8g.sh)。
# 默认钉设备 1; 若设备 1 被占用, 用 GPU=2 ./start-50g.sh。
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ENGINE="$HERE/build-120a-v2/apps/ninfer-serve"
MODEL="$HERE/bonsai2_27b_ptq1_native_mtp_vision.ninfer"

# -- 只用 50 系卡 (设备 1/2), 永不碰设备 0 -------------------------------------
GPU="${GPU:-1}"
export CUDA_VISIBLE_DEVICES="$GPU"

# -- 新线 16 GB 档: 显式池 131,072 (docs/03 16 GB 行, 推算·未验) ------------------
export NINFER_KV_WINDOW=10240
export NINFER_KV_RETRIEVE=8192
export NINFER_KV_REUSE_HOSTBACKED=1
export NINFER_TERNARY_PTQ1_FAST=1

echo "[start-50g] engine = $ENGINE (新线, sm_120a, UNTESTED)"
echo "[start-50g] model  = $MODEL"
echo "[start-50g] GPU    = $CUDA_VISIBLE_DEVICES, 池 131,072 / rk2v4-e8"

exec "$ENGINE" "$MODEL" \
  --host 127.0.0.1 --port 8092 --model-id ninfer-local \
  --max-context 262144 --max-concurrency 1 \
  --derive-session-keys --max-shared-prefixes 0 \
  --prefill-chunk 1024 \
  --kv-capacity 131072 --kv-dtype rk2v4-e8 \
  --host-kv-mib 16384
