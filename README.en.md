[中文 README](README.md) | **English README**

# ninfer-for-index-translate-9b-4060

An inference engine for running **Index-Translate-9B** on an **RTX 4060 (8 GB VRAM)**: the NInfer fusion engine with a host-backed KVMem KV ring, serving an OpenAI-compatible `/v1/chat/completions` HTTP API.

## Code provenance & modifications

- This repository is based on two upstream repositories:
  - [ruwwww/ninfer-5060ti](https://github.com/ruwwww/ninfer-5060ti)
  - [1314521gjy/ninfer-fusion-kvmem](https://github.com/1314521gjy/ninfer-fusion-kvmem) (source snapshot `9a5a920`, used as the base of this repo)
- Heavily modified ("hacked") on top of upstream using **qwen3.8-flash**. Main local changes:
  - New-shape (shape) kernels for Q4 / Q5 / Q6 / Q8 linear layers
  - `small_t` i8 `h16v4` append/cached kernels (w1/w2/w3) for causal_cache attention
  - Qwen3.5 / Qwen3.8 `flash_next` GDN (gated delta net) and MTP fixes
  - KV configuration and launcher scripts for 8 GB cards (the `start-*.sh` scripts in this repo's root)

## Model

The model artifact (`.ninfer` format, ~6.5 GB) is hosted on Hugging Face:

**https://huggingface.co/snowen/index-translate-9b-ninfer**

- Base model: [IndexTeam/Index-Translate-9B](https://huggingface.co/IndexTeam/Index-Translate-9B)
- Task: translation (EN ⇄ ZH), License: Apache-2.0
- Required file: `index_translate_9b_v3.ninfer`

Weights are not part of this repository; download them separately (see "Running" below).

## Installation / build

### Dependencies (Linux, the combination actually tested here)

| Component | Version |
|---|---|
| NVIDIA driver | ≥ 580 (hard requirement for CUDA 13.x) |
| CUDA Toolkit (nvcc) | 13.0 (tested); CUDA 13.x no longer supports Volta/sm_70 |
| CMake | ≥ 3.25 (tested with 3.28.3) |
| Ninja | any recent version |
| libcurl & ffmpeg dev headers | on Ubuntu: `libcurl-dev`, `libavcodec-dev libavformat-dev libavutil-dev libswscale-dev` (resolved via pkg-config; no vcpkg needed) |

On Windows, vcpkg is required (the manifest pins `curl`, `ffmpeg[zlib]`, `pkgconf`); the full procedure is in [`src/编译指南-怎么编.md`](src/编译指南-怎么编.md) (build guide).

### Build

```bash
# The architecture must be given explicitly:
# sm_89 = RTX 40 / Ada, sm_86 = RTX 30, sm_120a = RTX 50
cmake -S src-tree/fusion-engine-src -B build-89 -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=89 \
  -DNINFER_BUILD_APPS=ON -DBUILD_TESTING=OFF -DNINFER_BUILD_BENCHMARKS=OFF

cmake --build build-89 --target ninfer-serve -j 8
```

Output: `build-89/apps/ninfer-serve` (first full build takes ~1–2 hours; `-j 8` is the recommended parallelism because of ptxas memory usage).

### Download the model

```bash
hf download snowen/index-translate-9b-ninfer index_translate_9b_v3.ninfer --local-dir .
# or: huggingface-cli download snowen/index-translate-9b-ninfer index_translate_9b_v3.ninfer --local-dir .
```

## Running

```bash
CUDA_VISIBLE_DEVICES=0 \
NINFER_KV_WINDOW=10240 NINFER_KV_RETRIEVE=8192 NINFER_KV_REUSE_HOSTBACKED=1 \
./build-89/apps/ninfer-serve ./index_translate_9b_v3.ninfer \
  --host 127.0.0.1 --port 8091 --model-id index-translate-9b \
  --max-context 8192 --max-concurrency 1 \
  --derive-session-keys --max-shared-prefixes 0 \
  --prefill-chunk 1024 \
  --kv-capacity 10240 --kv-dtype int8 \
  --host-kv-mib 2048
```

You can also use the launcher scripts in the repo root (e.g. [`start-9b.sh`](start-9b.sh); update its `MODEL` path to the downloaded artifact filename).

Verify the server:

```bash
curl http://127.0.0.1:8091/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"index-translate-9b","messages":[{"role":"user","content":"Translate to English: 今天天气不错"}],"max_tokens":256}'
```

## Caveats

1. **One GPU architecture per clean build directory.** Mixing `.obj` files from different eras in one link produces a binary that crashes at startup (`0xC0000005 @ 0x18c729`). After switching architectures or rolling back versions, delete the `.obj` files or use a fresh directory — see [`src/根因与修法-0x18c729-20261003.md`](src/根因与修法-0x18c729-20261003.md).
2. **Do not wrap nvcc with sccache** (tested: `fatbinary` then fails to find intermediate `.ptx` files); apply the cache launcher to C/CXX only.
3. **8 GB VRAM relies on int8 KV + the host-backed ring**: `--kv-dtype int8` and `NINFER_KV_REUSE_HOSTBACKED=1` are the key settings. If VRAM is tight, lower `--kv-capacity` / `--host-kv-mib` (for a bf16 small-context profile see [`start-9b-bf16.sh`](start-9b-bf16.sh)).
4. **Only one service per GPU** — stop any previous server before starting a new one.
5. Building does not require model weights; this repo contains no weights, build artifacts, or logs (`build-*/`, `*.log`, binaries are all excluded via `.gitignore`).
6. More background and field-test notes live in `src/`: `复现白皮书-NInfer-KVMem环-20261002.md` (reproduction whitepaper), `编译指南-怎么编.md` (build guide), `8G档-配置提醒-20261008.md` (8 GB config notes), `我方-新增能力与已解决问题.md` (added capabilities & fixed issues).

## License

Upstream code keeps its original LICENSE/NOTICE — see [`src/LICENSE`](src/LICENSE) and [`src/NOTICE.md`](src/NOTICE.md); the model is Apache-2.0.
