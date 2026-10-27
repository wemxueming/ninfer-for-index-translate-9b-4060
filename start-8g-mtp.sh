#!/usr/bin/env bash
# start-8g-mtp.sh -- 8 GB 卡 (RTX 4060, sm_89) launcher, NEW LINE recipe + MTP speculation
# (27B ternary / qwen3_5, 2026-10-19 配方 + --spec mtp, KV 格式 rk4v4-e8).
# Engine: build-89-v2 (源码 9a5a920 + 本地 Qwen4Exp 摘除补丁).
#
# 与 start-8g.sh (无 MTP / rk2v4-e8) 的差别:
#   - --spec mtp --draft-tokens 3  (MTP 提案头投机, 24GB 卡实测 +38% decode;
#                                   MTP 状态+权重共占 ~1 GiB, 池必须相应收小)
#   - --kv-dtype rk4v4-e8          (K 从 2bit 提到 4bit E8-lattice INT4;
#                                   每 token/KV head 216 -> 280 B, 池显存 +33%)
#   - --max-context 16384          (8GB 上 MTP 只能落快档位; 见下)
#   - --kv-capacity 12800          (MTP + rk4v4-e8 下 51,200 池装不下: 启动器报
#                                   "reservation 2.18 GiB / 可用 1.59 GiB";
#                                   非池固定成本 ~1.27 GiB [工作区 + 256 MiB 内容评分
#                                   索引(不随 max-context 缩小) + MTP 状态], 可用
#                                   1.49 GiB => 池预算 ~244 MB => 12,800)
#
#   若想要更大池(牺牲长文档检索质量): 设 NINFER_TERNARY_KVMEM_SCORE=0 关掉内容评分
#   索引, 可再省 256 MiB, 池能放到 ~24,000。本脚本默认保持内容评分开(与无 MTP 版一致)。
#   - 端口 8093 / 服务单元 ninfer4060-mtp (与 8091 的非 MTP 版共存于脚本层面;
#     但 GPU0 只有 8 GB, 两个服务不能同时跑, 切换前先停另一个)
#
# 其余与 start-8g.sh 相同:
#   - --max-shared-prefixes 0 必带, --derive-session-keys, --prefill-chunk 1024,
#     env NINFER_KV_WINDOW=10240 等
#   - 不加 --lm-head-draft (占 ~136 MiB KV 上下文), 不加 --gdn-state-fp16,
#     --max-concurrency 1
#
# 硬边界: 池只有 12,800 token, 题面 > 12.8K 超出池 ⇒ 单请求答案不可信(引擎告警),
# 超池题面必须走分片网关 (tools/agg_gateway.py); 46,315-token 长针测试在本档位
# 超 max-context, 无法直接发(需要更大池/上下文, 即 start-8g.sh 无 MTP 长上下文档).
#
# 必须在本机 dsh 会话树外运行(NoNewPrivs + kernel 7.0 会破坏 libcuda):
#   systemd-run --user --unit ninfer4060-mtp -- /bin/bash -c \
#     '/media/wen/ubuntu/ninfer-kvmem/start-8g-mtp.sh >> /media/wen/ubuntu/ninfer-kvmem/serve-mtp.log 2>&1'
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ENGINE="$HERE/build-89-v2/apps/ninfer-serve"
# Model artifact lives on the win share (the working-dir copy is gone); point at the source.
MODEL="/media/wen/win/LLM_models/Qwen3.8-27b-Ninfer/bonsai2_27b_ptq1_native_mtp_vision.ninfer"

# -- 只用 GPU0 (RTX 4060 Laptop, CC 8.9) --------------------------------------
export CUDA_VISIBLE_DEVICES=0

# -- 新线 8 GB 长上下文档 (开 MTP):公共 env ------------------------------------
export NINFER_KV_WINDOW=10240
export NINFER_KV_RETRIEVE=8192
export NINFER_KV_REUSE_HOSTBACKED=1
export NINFER_TERNARY_PTQ1_FAST=1   # PTQ1 GEMM 快速档(性能开关, 旧配方沿用)

echo "[start-8g-mtp] engine = $ENGINE (新线, MTP3, 池 12,800 / rk4v4-e8, max-context 16,384)"
echo "[start-8g-mtp] model  = $MODEL"
echo "[start-8g-mtp] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES (仅 GPU0 / RTX 4060)"

exec "$ENGINE" "$MODEL" \
  --host 127.0.0.1 --port 8093 --model-id ninfer-local \
  --max-context 16384 --max-concurrency 1 \
  --derive-session-keys --max-shared-prefixes 0 \
  --prefill-chunk 1024 \
  --kv-capacity 12800 --kv-dtype rk4v4-e8 \
  --host-kv-mib 2816 \
  --spec mtp --draft-tokens 3
