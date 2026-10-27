---
license: apache-2.0
base_model: IndexTeam/Index-Translate-9B
pipeline_tag: translation
language:
- en
- zh
library_name: ninfer
---

# Index-Translate-9B · Quantized NInfer artifact (runs on RTX 4060 8 GB)

This repository holds the [NInfer](https://github.com/wemxueming/ninfer-for-index-translate-9b-4060) engine artifact of the **Index-Translate-9B** translation model (EN ⇄ ZH), field-tested on an **RTX 4060 Laptop (8 GB VRAM)** at roughly **70 tok/s decode** (with MTP speculative decoding enabled).

## Repository files

| File | Description |
|---|---|
| `index_translate_9b_v3.ninfer` | v3-format model artifact (~6.5 GB, groupwise-int weights, includes the single-layer MTP head and the vision tower) |
| `index_chat_template.jinja` | Chat template (passed via `--chat-template` when starting the server) |
| `test.png` | Field-test screenshot |

Base model: [IndexTeam/Index-Translate-9B](https://huggingface.co/IndexTeam/Index-Translate-9B)

## Project location

Engine and all source modifications (based on upstream [ruwwww/ninfer-5060ti](https://github.com/ruwwww/ninfer-5060ti) + [1314521gjy/ninfer-fusion-kvmem](https://github.com/1314521gjy/ninfer-fusion-kvmem), heavily modified with qwen3.8-flash):

**https://github.com/wemxueming/ninfer-for-index-translate-9b-4060**

## How to run

1. Build the engine (requires CUDA 13.x, CMake ≥ 3.25, Ninja, libcurl/ffmpeg dev headers; architecture sm_89 = RTX 40 series):

```bash
cmake -S src-tree/fusion-engine-src -B build-89 -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89 \
  -DNINFER_BUILD_APPS=ON -DBUILD_TESTING=OFF -DNINFER_BUILD_BENCHMARKS=OFF
cmake --build build-89 --target ninfer-serve -j 8
```

2. Download the artifacts from this repo:

```bash
hf download snowen/index-translate-9b-ninfer index_translate_9b_v3.ninfer index_chat_template.jinja --local-dir .
```

3. Start the server (8 GB VRAM profile: k8v4 KV + adaptive MTP speculative decoding + thinking disabled):

```bash
./build-89/apps/ninfer-serve ./index_translate_9b_v3.ninfer \
  --device 0 \
  --host 127.0.0.1 --port 8099 --chat-template index_chat_template.jinja \
  --max-context 4096 --max-concurrency 8 --no-prefix-reuse \
  --kv-dtype k8v4 \
  --spec mtp --draft-tokens 2 --adaptive-mtp \
  --no-thinking \
  --device-profile auto \
  --no-cuda-graph
```

4. Call it (OpenAI-compatible API):

```bash
curl http://127.0.0.1:8099/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"index_translate_9b_v3","messages":[{"role":"user","content":"Translate to English: 今天天气不错"}],"max_tokens":256}'
```

## Field test

RTX 4060 8 GB: ~70 tok/s decode, ~1.2k tok/s prefill, MTP acceptance 40+/45, thinking off:

![Field test](test.png)

## Caveats

- `--kv-dtype k8v4` (fp8 keys + nvfp4 values) together with `--device-profile auto` is the key combination for the 8 GB VRAM profile.
- Only one service can run per GPU at a time.
- The engine is compiled per GPU architecture (sm_89 = RTX 40 / Ada, sm_86 = RTX 30, sm_120a = RTX 50); switching architectures requires a clean build directory.

## License

Apache-2.0 (same as the base model).
