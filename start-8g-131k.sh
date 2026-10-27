#!/usr/bin/env bash
# start-8g-131k.sh -- 8 GB 卡 (RTX 4060, sm_89) launcher, NEW LINE recipe,
# 目标档: 最大上下文 131,200 token + rk4v4-e8 KV (无 MTP).
# Engine: build-89-v2 (源码 9a5a920 + 本地 Qwen4Exp 摘除补丁).
#
# 与 start-8g.sh (长档 51,200 / rk2v4-e8) 的差别:
#   - --max-context 131200   (服务端接受到 131,200 token 的题面 —— 用户目标)
#   - --kv-capacity 60000    (设备 KV 池只放 60,000 token; 启动时若把 131,200 全放设备
#                            需预留 2.66 GiB > 可用 1.90 GiB, 装不下。引擎允许 池<上下文:
#                            超出设备的 ~71,200 token 由主机环形缓存承接, 运行时淘汰。
#                            暖机需按 max-context 131,200 分配较大注意力工作区(几百 MiB):
#                            85,000 空闲 14.9 MiB、78,000 空闲 134.9 MiB 均暖机 OOM,
#                            故取 60,000 留 ~519 MiB 暖机余量)
#   - --host-kv-mib 2816     (2.75 GiB 主机环, 装下溢出的 ~0.91 GiB, 还余量缓存历史对话)
#   池预算推导: 可用 1.902 GiB − 内容评分索引 128 MiB(按 max-context 定容, 是 ring
#                基础设施, 评分器开关都分配) − 固定成本 69 MiB = 1.71 GiB → 理论最大
#                86,970; 但需给暖机留几百 MiB 工作区, 故取 60,000(留 ~443 MiB 实际空闲)
#   - --kv-dtype rk4v4-e8    (K 4bit E8-lattice, 质量高于 rk2v4-e8; 实测 ~21,142 B/token)
#   - 内容评分保持默认开: 内容评分索引(128 MiB)是 ring 基础设施, 按 max-context
#                            定容, 无论评分器开关都分配 —— 关它不省显存, 只降质量,
#                            故保持开(多针检索 6/6 而非词法 4/6)
#   - 不开 MTP: MTP 状态+权重另占 ~1 GiB, 本档放不下(MTP 请用 start-8g-mtp.sh 快档)
#
# 与 start-8g.sh 相同: --max-shared-prefixes 0 必带, --derive-session-keys,
#   --prefill-chunk 1024, --max-concurrency 1, env 同上。
#
# 硬边界: max-context 131,200 = 模型 n_ctx 262,144 之下的本档封顶。设备池 60,000 +
# 主机环承接溢出 = 131,200 总上下文。46,315-token 长针题面 < 60,000 设备池, 全程在
# 设备内, 答案可信且快; 更大题面(60K–131K)会部分淘汰到主机, 仍可用但较慢。
#
# 必须在本机 dsh 会话树外运行(NoNewPrivs + kernel 7.0 会破坏 libcuda):
#   systemd-run --user --unit ninfer4060-131k -- /bin/bash -c \
#     '/media/wen/ubuntu/ninfer-kvmem/start-8g-131k.sh >> /media/wen/ubuntu/ninfer-kvmem/serve-131k.log 2>&1'
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ENGINE="$HERE/build-89-v2/apps/ninfer-serve"
# Model artifact lives on the win share (the working-dir copy is gone); point at the source.
MODEL="/media/wen/win/LLM_models/Qwen3.8-27b-Ninfer/bonsai2_27b_ptq1_native_mtp_vision.ninfer"

# -- 只用 GPU0 (RTX 4060 Laptop, CC 8.9) --------------------------------------
export CUDA_VISIBLE_DEVICES=0

# -- 新线 8 GB 131,200 档 (rk4v4-e8, 无 MTP): 公共 env -------------------------
export NINFER_KV_WINDOW=10240
export NINFER_KV_RETRIEVE=8192
export NINFER_KV_REUSE_HOSTBACKED=1
export NINFER_TERNARY_PTQ1_FAST=1   # PTQ1 GEMM 快速档(性能开关, 旧配方沿用)
# 内容评分保持默认开(ring 配置时自动 ON): 多针检索 6/6(词法回落 4/6)。
# 注: 内容评分索引(128 MiB @ max-context 131,200)是 ring 基础设施, 无论评分器
# 开关都按 max-context 分配, 关掉评分器并不省显存, 只会降低检索质量 —— 故不关。

echo "[start-8g-131k] engine = $ENGINE (新线, max-context 131,200 / 设备池 60,000+主机环 / rk4v4-e8 / 无 MTP)"
echo "[start-8g-131k] model  = $MODEL"
echo "[start-8g-131k] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES (仅 GPU0 / RTX 4060)"

exec "$ENGINE" "$MODEL" \
  --host 127.0.0.1 --port 8091 --model-id ninfer-local \
  --max-context 131200 --max-concurrency 1 \
  --derive-session-keys --max-shared-prefixes 0 \
  --prefill-chunk 1024 \
  --kv-capacity 60000 --kv-dtype rk4v4-e8 \
  --host-kv-mib 2816
