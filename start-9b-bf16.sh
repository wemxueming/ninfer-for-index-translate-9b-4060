#!/usr/bin/env bash
# start-9b.sh -- 8 GB 卡 (RTX 4060 Laptop, sm_89) launcher, Qwen3.5-9B (groupwise-int,
# v3 由 tools/upgrade_ninfer_v2_to_v3.py 从 /mnt/nas/ai/qwen3_5_9b.ninfer (v2) 升级而来).
# Engine: build-89-v2 (源码 9a5a920 + 本地 Qwen4Exp 摘除补丁).
#
# 规格 (docs/maintainer/qwen3.5-9b-model.md): 32 层 hybrid (full attn @ i%4==3 共 8 层,
#   GDN 24 层), hidden 4096, 16 Q / 4 KV / head_dim 256, GDN 32 V heads x 128,
#   intermediate 12288, vocab 248320, 单层 MTP, 27 层 vision, 位置容量 262,144.
#
# 显存: 9B 比 27B 更小 (权重 ~5.3 GiB), 沿用已验证的 131,200 档 KV 配置
#   (设备池 60,000 + 主机环 2.75 GiB, rk4v4-e8) —— 9B 每 token KV 比 27B 更省
#   (GDN 状态 4096 vs 6144/层), 余量只会更大。
#
# 第一档不开 MTP (先验证基线加载与生成); MTP 权重已在工件中, 如需快档:
#   加 --spec mtp --draft-tokens 3 并把本脚本另存一份。
#
# 运行 (dsh 会话树不能跑 CUDA, 必须 systemd-run):
#   systemctl --user stop ninfer4060-131k   # 同卡服务互斥
#   systemd-run --user --unit ninfer4060-9b -- /bin/bash -c \
#     '/media/wen/ubuntu/ninfer-kvmem/start-9b.sh >> /media/wen/ubuntu/ninfer-kvmem/serve-9b.log 2>&1'
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ENGINE="$HERE/build-89-v2/apps/ninfer-serve"
MODEL="$HERE/qwen3_5_9b_v3.ninfer"

# -- 只用 GPU0 (RTX 4060 Laptop, CC 8.9) --------------------------------------
export CUDA_VISIBLE_DEVICES=0

# -- 与 131k 档相同的 KV env --------------------------------------------------
export NINFER_KV_WINDOW=10240
export NINFER_KV_RETRIEVE=8192
export NINFER_KV_REUSE_HOSTBACKED=1

echo "[start-9b] engine = $ENGINE"
echo "[start-9b] model  = $MODEL"
echo "[start-9b] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES (仅 GPU0 / RTX 4060)"

exec "$ENGINE" "$MODEL" \
  --host 127.0.0.1 --port 8094 --model-id ninfer-local \
  --max-context 4096 --max-concurrency 1 \
  --derive-session-keys --max-shared-prefixes 0 \
  --prefill-chunk 1024 \
  --kv-capacity 4096 --kv-dtype bf16 \
  --host-kv-mib 256
