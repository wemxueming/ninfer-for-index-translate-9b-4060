---
license: apache-2.0
base_model: IndexTeam/Index-Translate-9B
pipeline_tag: translation
language:
- en
- zh
library_name: ninfer
---

# Index-Translate-9B · NInfer 量化工件（RTX 4060 8 GB 可跑）

本仓库是 **Index-Translate-9B** 翻译模型（EN ⇄ ZH）的 [NInfer](https://github.com/wemxueming/ninfer-for-index-translate-9b-4060) 引擎工件，在 **RTX 4060 Laptop（8 GB 显存）** 上实测约 **70 tok/s decode**（MTP 投机解码开启）。

## 仓库文件

| 文件 | 说明 |
|---|---|
| `index_translate_9b_v3.ninfer` | v3 格式模型工件（约 6.5 GB，groupwise int 权重，含 MTP 单层与 vision 塔） |
| `index_chat_template.jinja` | 聊天模板（serve 启动时通过 `--chat-template` 传入） |
| `test.png` | 实测演示截图 |

基座模型：[IndexTeam/Index-Translate-9B](https://huggingface.co/IndexTeam/Index-Translate-9B)

## 项目位置

引擎与全部改动代码（上游 [ruwwww/ninfer-5060ti](https://github.com/ruwwww/ninfer-5060ti) + [1314521gjy/ninfer-fusion-kvmem](https://github.com/1314521gjy/ninfer-fusion-kvmem)，使用 qwen3.8-flash 魔改）：

**https://github.com/wemxueming/ninfer-for-index-translate-9b-4060**

## 运行方法

1. 编译引擎（需 CUDA 13.x、CMake ≥ 3.25、Ninja、libcurl/ffmpeg 开发头；架构 sm_89 = RTX 40 系）：

```bash
cmake -S src-tree/fusion-engine-src -B build-89 -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89 \
  -DNINFER_BUILD_APPS=ON -DBUILD_TESTING=OFF -DNINFER_BUILD_BENCHMARKS=OFF
cmake --build build-89 --target ninfer-serve -j 8
```

2. 下载本仓库工件：

```bash
hf download snowen/index-translate-9b-ninfer index_translate_9b_v3.ninfer index_chat_template.jinja --local-dir .
```

3. 启动服务（8 GB 显存档：k8v4 KV + MTP 自适应投机解码 + 关思考）：

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

4. 调用（OpenAI 兼容 API）：

```bash
curl http://127.0.0.1:8099/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"index_translate_9b_v3","messages":[{"role":"user","content":"Translate to English: 今天天气不错"}],"max_tokens":256}'
```

## 实测演示

RTX 4060 8 GB：decode ~70 tok/s，prefill ~1.2k tok/s，MTP 接受 40+/45，thinking off：

![实测演示](test.png)

## 注意事项

- `--kv-dtype k8v4`（fp8 keys + nvfp4 values）与 `--device-profile auto` 是 8 GB 显存档的关键组合。
- 同一张卡一次只能运行一个服务。
- 引擎按 GPU 架构单独编译（sm_89 = RTX 40 / Ada，sm_86 = RTX 30，sm_120a = RTX 50）；换架构必须用干净的构建目录。

## License

Apache-2.0（与基座模型一致）。
