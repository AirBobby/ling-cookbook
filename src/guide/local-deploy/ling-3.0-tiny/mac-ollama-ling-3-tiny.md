---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.19.5
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

##### Copyright 2026 Ant Group.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

+++

# Ling-3.0-tiny on Apple Silicon Mac (Ollama INT4 / FP8 / BF16) 部署指南

<table align="left">
  <td>
    <a target="_blank" href="https://www.ant-ling.com/"><img src="https://img.shields.io/badge/Official_Website-ant--ling.com-6366f1?style=flat-square" alt="Website" /></a>
  </td>
  <td>
    <a target="_blank" href="https://github.com/inclusionAI/ling-cookbook"><img src="https://img.shields.io/badge/GitHub-Ling_Cookbook-181717?style=flat-square&logo=github" alt="GitHub" /></a>
  </td>
  <td>
    <a target="_blank" href="https://huggingface.co/inclusionAI"><img src="https://img.shields.io/badge/Hugging_Face-inclusionAI-FFD21E?style=flat-square&logo=huggingface&logoColor=black" alt="Hugging Face" /></a>
  </td>
  <td>
    <a target="_blank" href="https://www.modelscope.cn/organization/inclusionAI"><img src="https://img.shields.io/badge/ModelScope-inclusionAI-624AFF?style=flat-square" alt="ModelScope" /></a>
  </td>
  <td>
    <a target="_blank" href="https://x.com/AntLingAGI"><img src="https://img.shields.io/badge/X-@AntLingAGI-000000?style=flat-square&logo=x" alt="X" /></a>
  </td>
  <td>
    <a target="_blank" href="https://discord.com/invite/GNaQc8WC5T"><img src="https://img.shields.io/badge/Discord-Ling_Community-5865F2?style=flat-square&logo=discord&logoColor=white" alt="Discord" /></a>
  </td>
</table>

<br><br>

`Ling-3.0-tiny` 是 Ling-3.0 家族中的 **7.9B 轻量型端侧 Sparse MoE 大语言模型**，单 Token 激活参数仅 **1.3B**。模型采用 3:1 的 KDA 线性注意力与 Gated MLA 混合架构，配备 128 个路由专家，原生支持 128K 长上下文。

得益于 1.3B 的极小激活量与高效架构，`Ling-3.0-tiny` 在 **Apple Silicon Mac** (M1/M2/M3/M4 系列，8GB - 48GB+ 统一内存) 上能够实现出色的端侧吞吐与极低延迟。

本指南基于社区分支 [ollama/ollama#17643](https://github.com/ollama/ollama/pull/17643) (`bailing-moe-v3`)，在 macOS 上通过 MLX / Metal 硬件加速运行。本篇在一处完整覆盖 **INT4**、**FP8 (MXFP8)** 与 **BF16** 三种规格的下载、导入与验证。

---

### 三种部署规格与设备内存需求差异 (Memory & Context Matrix)

用户可根据自己 Mac 设备的统一内存规格，选择最适合的部署方案：

| 部署规格 | 权重体积 | 8K 上下文内存 | 64K 上下文内存 | 128K 上下文内存 | 损耗程度 |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **INT4 (MXFP4)** | ~3.95 GB | **~5.50 GiB** | **~8.20 GiB** | **~10.50 GiB** | 轻微损耗（极致轻量，8GB / 16GB 设备友好） |
| **FP8 (MXFP8)** | ~7.90 GB | **~8.34 GiB** | **~12.50 GiB** | **~16.80 GiB** | 极低损耗（默认推荐，Logits 余弦相似度 0.9992） |
| **BF16 (全精度)** | ~15.80 GB | **~14.95 GiB** | **~22.50 GiB** | **~30.00 GiB** | 无损（基准全精度，Logits 余弦相似度 0.99964） |

> [!IMPORTANT]
> **设备选型与内存要求**：
> 你需要让自己的**设备内存大于上述内存估计**（建议额外预留 2~4 GB 给 macOS 系统与日常桌面应用），以避免触发系统 Swap 内存交换导致推理速度严重下降。

> [!TIP]
> **环境准备建议**：
> - 仅支持 **Apple Silicon Mac** (M 系列芯片，不支持 Intel Mac)；
> - 推荐使用 **macOS 14 (Sonoma)** 或更高版本（已实测 macOS 15.6.1）；
> - 确保安装了完整 **Xcode** (16.0+) 及 **Metal Toolchain** 编译组件；
> - 推荐使用 **Python 3.11** 独立虚拟环境。

+++

### 步骤 1: 准备系统环境与编译工具链

首先确保配置正确的 Xcode 开发者路径并下载 Metal Toolchain 组件，随后通过 Homebrew 安装 Go、CMake、jq 工具，并安装模型下载库：

```{code-cell}
!sudo xcode-select --switch /Applications/Xcode.app/Contents/Developer
!xcodebuild -downloadComponent MetalToolchain
!brew install go cmake jq
!pip install -U modelscope huggingface_hub --quiet
```

典型运行输出：
```text
Component MetalToolchain downloaded successfully.
go 1.26.0 is already installed and up-to-date.
cmake 4.4.2 is already installed and up-to-date.
Successfully installed modelscope huggingface_hub
```

+++

### 步骤 2: 克隆 Ollama 仓库并检出 Bailing MoE V3 支持分支

Ling-3.0-tiny 的 `bailing_hybrid` 架构支持目前由社区分支 [PR #17643](https://github.com/ollama/ollama/pull/17643) 提供，需检出 `bailing-moe-v3` 分支：

```{code-cell}
!git clone https://github.com/ollama/ollama.git 2>/dev/null || (cd ollama && git fetch origin)
!cd ollama && git fetch origin refs/pull/17643/head:bailing-moe-v3 && git checkout bailing-moe-v3
```

典型运行输出：
```text
From https://github.com/ollama/ollama
 * [new ref]         refs/pull/17643/head -> bailing-moe-v3
Switched to branch 'bailing-moe-v3'
```

+++

### 步骤 3: 从源码编译构建 Ollama (开启 Metal / MLX 加速)

在 `ollama` 目录下执行 CMake 编译构建。编译完成后，请在后续步骤中统一使用当前目录生成的 `./ollama` 二进制，不要混用系统全局安装的 `ollama`：

```{code-cell}
!cd ollama && cmake -B build . && cmake --build build --parallel 8
```

典型运行输出：
```text
-- The C compiler identification is AppleClang 16.0.0
-- The CXX compiler identification is AppleClang 16.0.0
-- Found Metal toolchain
[100%] Built target ollama
```

+++

### 步骤 4: 获取模型权重并导入 Ollama

官方权重已发布至 Hugging Face 与 ModelScope。你可以根据自己的 Mac 设备内存情况，从以下 **三种规格中任选一种** 下载并导入：

- **选项 4A (默认推荐 · 均衡高吞吐)**: `FP8 (MXFP8)` -> 适合 16GB / 24GB+ 内存机型
- **选项 4B (极致轻量 · 极低显存)**: `INT4 (MXFP4)` -> 适合 8GB / 16GB 内存机型
- **选项 4C (全精度无损 · 极致效果)**: `BF16` -> 适合 24GB / 32GB / 48GB+ 内存机型

> [!IMPORTANT]
> **导入路径与工作目录注意事项**：
> 导入 FP8 / INT4 / BF16 权重时需要调用编译生成的 MLX 动态库，因此 **请务必在 `ollama` 仓库根目录下执行 `./ollama create`**，不要切换至其他目录执行。

+++

#### 选项 4A (默认推荐): 下载并导入 Ling-3.0-tiny-FP8
- ModelScope: `inclusionAI/Ling-3.0-tiny-fp8`
- Hugging Face: `inclusionAI/Ling-3.0-tiny-fp8`

```{code-cell}
# 1. 下载 FP8 权重
!modelscope download --model inclusionAI/Ling-3.0-tiny-fp8 --local_dir ~/models/Ling-3.0-tiny-fp8

# 2. 生成 Modelfile 并导入 Ollama
import os
modelfile_path = "/tmp/Modelfile.ling_tiny_fp8"
weights_path = os.path.expanduser("~/models/Ling-3.0-tiny-fp8")

with open(modelfile_path, "w", encoding="utf-8") as f:
    f.write(f"FROM {weights_path}\n")

!cd ollama && ./ollama create ling-tiny-fp8 --experimental -f /tmp/Modelfile.ling_tiny_fp8
!rm -f /tmp/Modelfile.ling_tiny_fp8
```

典型运行输出：
```text
transferring model data 
creating new layer 
creating new template 
writing manifest 
success
```

+++

#### 选项 4B: 下载并导入 Ling-3.0-tiny-INT4 (适合 8GB / 16GB 设备)
- ModelScope: `inclusionAI/Ling-3.0-tiny-int4`
- Hugging Face: `inclusionAI/Ling-3.0-tiny-int4`

```{code-cell}
# 1. 下载 INT4 权重
!modelscope download --model inclusionAI/Ling-3.0-tiny-int4 --local_dir ~/models/Ling-3.0-tiny-int4

# 2. 生成 Modelfile 并导入 Ollama
import os
modelfile_path = "/tmp/Modelfile.ling_tiny_int4"
weights_path = os.path.expanduser("~/models/Ling-3.0-tiny-int4")

with open(modelfile_path, "w", encoding="utf-8") as f:
    f.write(f"FROM {weights_path}\n")

!cd ollama && ./ollama create ling-tiny-int4 --experimental -f /tmp/Modelfile.ling_tiny_int4
!rm -f /tmp/Modelfile.ling_tiny_int4
```

典型运行输出：
```text
transferring model data 
creating new layer 
creating new template 
writing manifest 
success
```

+++

#### 选项 4C: 下载并导入 Ling-3.0-tiny-BF16 (全精度无损，适合 24GB+ 设备)
- ModelScope: `inclusionAI/Ling-3.0-tiny`
- Hugging Face: `inclusionAI/Ling-3.0-tiny`

```{code-cell}
# 1. 下载 BF16 全精度权重
!modelscope download --model inclusionAI/Ling-3.0-tiny --local_dir ~/models/Ling-3.0-tiny

# 2. 生成 Modelfile 并导入 Ollama
import os
modelfile_path = "/tmp/Modelfile.ling_tiny_bf16"
weights_path = os.path.expanduser("~/models/Ling-3.0-tiny")

with open(modelfile_path, "w", encoding="utf-8") as f:
    f.write(f"FROM {weights_path}\n")

!cd ollama && ./ollama create ling-tiny-bf16 --experimental -f /tmp/Modelfile.ling_tiny_bf16
!rm -f /tmp/Modelfile.ling_tiny_bf16
```

典型运行输出：
```text
transferring model data 
creating new layer 
creating new template 
writing manifest 
success
```

+++

#### 验证模型导入信息
执行 `ollama show` 检查模型元数据：

```{code-cell}
!cd ollama && ./ollama show ling-tiny-fp8
```

典型运行输出：
```text
  Model
    architecture        bailing_hybrid
    parameters          7.9B
    quantization        mxfp8
    format              safetensors
```

+++

### 步骤 5: 启动 Ollama 推理服务

配置上下文长度环境变量为 `8192` (8K 上下文) 并启动 Ollama 服务，默认监听 `127.0.0.1:11434`：

> [!WARNING]
> **特别注意：前台常驻与阻塞**
> `./ollama serve` 以前台常驻模式运行。如果你直接在 Notebook 中运行下方单元格，Jupyter 将阻塞而无法执行后续验证单元格。
> 建议你在**单独的终端会话**中执行启动命令：
> ```bash
> cd ollama && OLLAMA_CONTEXT_LENGTH=8192 ./ollama serve
> ```
> 启动成功后即可在 Notebook 中执行后续测试。

```{code-cell}
!cd ollama && OLLAMA_CONTEXT_LENGTH=8192 ./ollama serve
```

服务端就绪时的典型日志：
```text
time=2026-08-17T23:20:00.000+08:00 level=INFO source=server.go:120 msg="Listening on 127.0.0.1:11434 (version 0.5.12)"
time=2026-08-17T23:20:00.100+08:00 level=INFO source=runner.go:34 msg="Metal GPU accelerated runner initialized"
```

+++

### 步骤 6: 验证部署成功并使用模型服务

服务就绪后，我们进行 3 维度的验证测试：
1. **连通性与模型健康检查** - 检查 Ollama 服务接口与已导入的模型列表。
2. **流式 Reasoning 与性能测速 (TTFT & TPS)** - 使用官方原生 Role 格式进行流式生成，提取 `<think>` 思考链并测算首 Token 延迟与 Decode 速度。
3. **Function Calling 工具调用测试** - 验证 OpenAI 兼容接口的结构化 Tool Calls。

+++

#### 步骤 6.1: 连通性与模型健康检查
请求 `http://localhost:11434/api/tags` 检查服务连通性与模型镜像：

```{code-cell}
import urllib.request
import json

url = "http://localhost:11434/api/tags"
try:
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req) as response:
        status_code = response.getcode()
        body = json.loads(response.read().decode("utf-8"))
        print(f"Health Check HTTP Status: {status_code}")
        print(f"Installed Models: {[m.get('name') for m in body.get('models', [])]}")
except Exception as e:
    print(f"Health check failed: {e}")
```

典型测试结果：
```text
Health Check HTTP Status: 200
Installed Models: ['ling-tiny-fp8:latest']
```

+++

#### 步骤 6.2: 流式推理、Reasoning 与速度测量 (TTFT / TPS)

PR #17643 原生支持官方 Role 格式。我们使用 `/api/generate` 端点以 `raw: true` 格式发送确定性数学计算请求，测试流式 `<think>` 思考链解析，并精确测量 TTFT 与 Decode 吞吐：

```{code-cell}
import time
import json
import urllib.request

# 模型名称按需选择: ling-tiny-fp8 / ling-tiny-int4 / ling-tiny-bf16
model_name = "ling-tiny-fp8"
url = "http://localhost:11434/api/generate"

prompt = "<role>SYSTEM</role>detailed thinking on<|role_end|><role>HUMAN</role>请详细计算 17 × 23，并给出推导步骤。<|role_end|><role>ASSISTANT</role>\n<think>"

payload = {
    "model": model_name,
    "prompt": prompt,
    "raw": True,
    "stream": True,
    "options": {
        "temperature": 0.6,
        "top_p": 0.95,
        "seed": 1,
        "num_predict": 512
    }
}

print(f"Sending request to Ollama model '{model_name}'...")
start_time = time.time()
first_token_time = None
total_tokens = 0
full_output = ""

req = urllib.request.Request(
    url,
    data=json.dumps(payload).encode("utf-8"),
    headers={"Content-Type": "application/json"}
)

try:
    with urllib.request.urlopen(req) as resp:
        for line in resp:
            if not line:
                continue
            chunk = json.loads(line.decode("utf-8"))
            now = time.time()
            if first_token_time is None and chunk.get("response"):
                first_token_time = now

            text = chunk.get("response", "")
            if text:
                full_output += text
                total_tokens += 1
                print(text, end="", flush=True)

            if chunk.get("done", False):
                break

    end_time = time.time()
    ttft = (first_token_time - start_time) * 1000.0 if first_token_time else 0.0
    decode_duration = (end_time - first_token_time) if first_token_time else 0.001
    tps = total_tokens / decode_duration

    print("\n\n=== Latency & Throughput Metrics ===")
    print(f"TTFT (Time to First Token): {ttft:.2f} ms")
    print(f"Decode TPS (Tokens/s): {tps:.2f} tokens/s")
    print(f"Total Generated Tokens: {total_tokens}")
    print(f"Total Duration: {end_time - start_time:.2f} s")

except Exception as e:
    print(f"Inference request failed: {e}")
```

典型测试结果：
```text
我们来计算 17 × 23：
1. 将 23 拆分为 20 + 3；
2. 计算 17 × 20 = 340；
3. 计算 17 × 3 = 51；
4. 将两部分相加：340 + 51 = 391。
因此，17 × 23 = 391。<|role_end|>

=== Latency & Throughput Metrics ===
TTFT (Time to First Token): 45.20 ms
Decode TPS (Tokens/s): 88.75 tokens/s
Total Generated Tokens: 245
Total Duration: 2.81 s
```

+++

#### 步骤 6.3: 测试 Function Calling 和结构化输出

使用 OpenAI Python SDK 调用 Ollama 的 `/v1` 兼容接口，传入标准 Function Calling Schema 验证工具调用：

```{code-cell}
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)

tools_schema = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "获取指定城市的实时天气与气温信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "城市名称，例如：北京、上海、杭州"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "温度单位"
                    }
                },
                "required": ["city"]
            }
        }
    }
]

print("Testing Function Calling with OpenAI SDK on Ollama...")
try:
    response = client.chat.completions.create(
        model="ling-tiny-fp8",
        messages=[{"role": "user", "content": "请帮我查一下杭州今天的天气如何？"}],
        tools=tools_schema,
        tool_choice="auto"
    )

    message = response.choices[0].message
    if message.tool_calls:
        print("\n=== Function Call Output Detected ===")
        for tool_call in message.tool_calls:
            print(f"Tool Call ID: {tool_call.id}")
            print(f"Function Name: {tool_call.function.name}")
            print(f"Arguments JSON: {tool_call.function.arguments}")
    else:
        print(f"\n=== Direct Response ===\n{message.content}")

except Exception as e:
    print(f"Function Calling verification failed: {e}")
```

典型测试结果：
```text
Testing Function Calling with OpenAI SDK on Ollama...

=== Function Call Output Detected ===
Tool Call ID: call_01948af981a
Function Name: get_weather
Arguments JSON: {"city":"杭州"}
```

+++

### 步骤 7: 常见问题与故障排查 (Troubleshooting)

1. **`xcode-select` 或 MetalToolchain 缺失导致 CMake 报错**
   - **现象**：CMake 提示找不到 Metal 编译器或 `MetalToolchain` 缺失。
   - **解决**：确保运行了 `sudo xcode-select --switch /Applications/Xcode.app/Contents/Developer` 并执行 `xcodebuild -downloadComponent MetalToolchain`。

2. **导入 FP8 / INT4 权重时报错找不到 MLX 动态库**
   - **现象**：`./ollama create` 时报错找不到 MLX 相关 `.dylib`。
   - **解决**：务必在 Ollama 编译源码根目录（包含 `build/` 产物）下执行 `./ollama create`，不要切换到其他目录。

3. **内存压力与 Swap 颠簸 (Memory Pressure & Swap)**
   - **现象**：Mac 内存占用过高导致系统卡顿或生成速度骤降。
   - **解决**：8GB/16GB 内存机型请优先选择 **INT4 (MXFP4)** 或 **FP8** 规格；可通过设置环境变量 `OLLAMA_CONTEXT_LENGTH=4096` 降低 KV Cache 显存占用。

4. **端口 11434 冲突**
   - **现象**：提示 `bind: address already in use`。
   - **解决**：检查是否有后台或系统的 Ollama 正在运行，执行 `pkill ollama` 释放端口后再重新启动。
