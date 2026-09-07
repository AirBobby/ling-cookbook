---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.19.5
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

##### Copyright 2026 Ant Group and NVIDIA Corporation.

+++

# Ling-3.0-flash on DGX Spark：原生 W4A8 与 Humming + 在线 FP8 LM Head

> 感谢 Nvidia 团队 [@ly01325](https://github.com/ly01325) 优化和提供本部署指南。

本 Notebook 演示在 NVIDIA DGX Spark（GB10 / SM121）上从源码部署 Ling-3.0-flash MXFP4，
并对比两套互斥配置：

| | 方案 A（基线） | 方案 B（优化） |
|---|---|---|
| MoE | `flashinfer_mxfp4`（W4A8） | `humming` MXFP4 |
| Dense GEMM | `cutlass` | `cutlass` |
| LM Head | FP32（`--enable-fp32-lm-head`） | 在线量化为动态 FP8 |

两套配置除上述两项外**其余参数完全一致**，可直接做单变量对比。
本 Notebook 不涉及 MTP 投机解码；MTP 的相关改动与实测数据见 `UPSTREAM.md`。

> [!TIP]
> **环境准备：**
> - 推荐 **Python 3.11**、CUDA 13、Ubuntu / DGX OS；
> - 所有安装与启动都使用**当前 Jupyter 内核的解释器**（`sys.executable`），
>   不再另建 venv——否则 bash 单元装的包，Python 单元 import 不到；
> - 两套服务都用端口 `30000`，**启动另一套前必须先停掉当前服务**。

> [!WARNING]
> 编译缓存与 CUDA、PyTorch、FlashInfer / Humming 版本及 GPU 架构绑定，
> **不要提交到 Git**，应在目标机器上重新生成。

+++

## 步骤 0：配置

后续所有单元都依赖这里定义的变量，**请先执行本单元**。

源码目录**自动探测**——本 Notebook 就在源码树里，无需再单独克隆 SGLang。
当前使用分支为：[`ling3-flash_dgx_spark`](https://github.com/ly01325/sglang/tree/ling3-flash_dgx_spark)（此前上传到 fork 的分支）。
另含三处改动：在线 FP8 LM Head、MTP draft 共享 lm_head
模块、以及一处上游启动崩溃修复。详见 `UPSTREAM.md` 与
`patches/ling3-w4a8-humming.patch`。

运行期产物（日志、PID 文件）写到 `~/ling3-dgx-spark/`，与源码树分开，不会污染仓库。

`MEM_FRACTION_STATIC` 默认取 `0.68`。官方 cookbook 用 `0.75`，但在本机实测中
`0.75` 触发过 GPU OOM 并导致 nvidia 驱动锁死（`rmapiLockAcquire`），只能硬断电恢复。
确认稳定后可自行上调。

```{code-cell} ipython3
import os
import sys
from pathlib import Path

# 本 Notebook 就在源码树里，向上找到含 python/sglang 的仓库根目录
SOURCE_DIR = next(
    (d for d in [Path.cwd(), *Path.cwd().parents] if (d / "python" / "sglang").is_dir()),
    None,
)
if SOURCE_DIR is None:
    raise RuntimeError(
        f"未在 {Path.cwd()} 或其上级目录中找到 python/sglang。"
        "请在克隆下来的仓库内运行本 Notebook。"
    )

MODEL_DIR = Path.home() / "models" / "Ling-3.0-flash-fp4"
WORKDIR   = Path.home() / "ling3-dgx-spark"   # 运行期产物：日志、PID

# 统一使用当前 Jupyter 内核的解释器，保证 bash 单元与 Python 单元同环境
PYTHON = sys.executable

API_KEY = "sk-ling-cookbook-test"
PORT    = 30000
MEM_FRACTION_STATIC = "0.68"

for key, value in {
    "WORKDIR": WORKDIR, "SOURCE_DIR": SOURCE_DIR,
    "MODEL_DIR": MODEL_DIR, "PYTHON": PYTHON,
    "API_KEY": API_KEY, "PORT": PORT,
    "MEM_FRACTION_STATIC": MEM_FRACTION_STATIC,
}.items():
    os.environ[key] = str(value)

print(f"源码目录: {SOURCE_DIR}")
print(f"模型目录: {MODEL_DIR}")
print(f"解释器  : {PYTHON}")
```

### 两个小工具

`serve()` 启动 sglang，把输出**实时打印在 Notebook 里**（不写日志文件），
看到就绪标志就返回，服务转入后台继续运行。
`ask()` 做一次问答，验证服务确实能推理。

`COMMON` 里放两套配置的公共参数——差异项由各自的 `extra` 追加，
这样"单变量对比"是结构上保证的，不会改了一处忘了另一处。

```{code-cell} ipython3
import os, re, signal, subprocess, threading, time

COMMON = [
    "--model-path", str(MODEL_DIR), "--served-model-name", "ling-v3-flash-fp4",
    "--trust-remote-code", "--dtype", "bfloat16", "--tp-size", "1", "--ep-size", "1",
    "--host", "0.0.0.0", "--api-key", API_KEY,
    "--max-running-requests", "1", "--max-mamba-cache-size", "64",
    "--chunked-prefill-size", "8192", "--max-prefill-tokens", "16384",
    "--page-size", "64", "--context-length", "262144",
    "--cuda-graph-backend-decode", "full", "--cuda-graph-max-bs-decode", "1",
    "--cuda-graph-bs-decode", "1", "--cuda-graph-backend-prefill", "disabled",
    "--random-seed", "308534008", "--reasoning-parser", "deepseek-r1",
    "--tool-call-parser", "qwen25", "--attention-backend", "flashinfer",
    "--disable-flashinfer-autotune", "--mem-fraction-static", MEM_FRACTION_STATIC,
    "--fp8-gemm-backend", "cutlass", "--flashinfer-mxfp4-moe-precision", "default",
    "--disable-shared-experts-fusion", "--json-model-override-args",
    '{"max_position_embeddings":262144,"rope_scaling":{"rope_type":"yarn","factor":2.0,'
    '"rope_theta":6000000,"partial_rotary_factor":0.5,'
    '"original_max_position_embeddings":131072}}',
]
BASE_ENV = {"PYTHONPATH": str(SOURCE_DIR / "python"),
            "SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN": "1",
            "SGLANG_JIT_DEEPGEMM_PRECOMPILE": "0", "SGLANG_ENABLE_JIT_DEEPGEMM": "0",
            "SGLANG_DSV4_FP4_DEQUANT": "0", "SGLANG_FP8_IGNORED_LAYERS": ""}
SERVER = {"proc": None, "log": []}


def stop():
    subprocess.run(["pkill", "-f", "sglang.launch_server"], capture_output=True)
    time.sleep(5)
    SERVER["proc"] = None
    print("服务已停止。", flush=True)


def serve(extra=(), env=None, port=None, until=None, minutes=45):
    # until: 可选回调，返回 True 就提前停止等待（步骤 3 用它在 build.ninja 出现后立刻收手）
    stop()
    e = {**os.environ, **BASE_ENV, **(env or {})}
    p = subprocess.Popen([PYTHON, "-m", "sglang.launch_server", "--port",
                          str(port or PORT), *COMMON, *extra],
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         start_new_session=True, env=e)
    log, buf, end = [], "", time.monotonic() + minutes * 60
    SERVER["proc"], SERVER["log"] = p, log
    while time.monotonic() < end and not (until and until()):
        data = os.read(p.stdout.fileno(), 8192)
        if not data:
            break
        buf += data.decode("utf-8", "replace")
        *lines, buf = re.split(r"[\r\n]", buf)      # 按 \r 切分，tqdm 进度条才能实时显示
        for ln in lines:
            if ln.strip():
                log.append(ln)
                print(ln[:170], flush=True)
        if any("ready to roll" in ln for ln in log[-30:]):
            break
    # 就绪后必须继续排空管道，否则管道写满会把服务卡住
    threading.Thread(target=lambda: [log.append(x.decode("utf-8", "replace").strip())
                                     for x in iter(p.stdout.readline, b"")],
                     daemon=True).start()
    return p


def ask(question="请计算 17 × 23，并给出简洁推导。", port=None):
    import urllib.request
    from openai import OpenAI

    base = f"http://127.0.0.1:{port or PORT}/v1"
    req = urllib.request.Request(base + "/models",
                                 headers={"Authorization": f"Bearer {API_KEY}"})
    for i in range(60):
        try:
            with urllib.request.urlopen(req, timeout=5) as r:
                print("健康检查 OK:", r.read().decode()[:150], flush=True)
            break
        except Exception:
            if i == 59:
                raise RuntimeError("服务未就绪")
            time.sleep(5)

    t0, first, ntok, out = time.perf_counter(), None, None, []
    for ch in OpenAI(base_url=base, api_key=API_KEY).chat.completions.create(
            model="ling-v3-flash-fp4", messages=[{"role": "user", "content": question}],
            temperature=0.6, top_p=0.95, max_tokens=2048, stream=True,
            stream_options={"include_usage": True},
            extra_body={"top_k": 29, "chat_template_kwargs": {"enable_thinking": True}}):
        if ch.usage:
            ntok = ch.usage.completion_tokens
        if ch.choices:
            d = ch.choices[0].delta
            piece = getattr(d, "reasoning_content", None) or d.content or ""
            if piece:
                first = first or time.perf_counter()
                out.append(piece)
    t1 = time.perf_counter()
    tps = ntok / (t1 - first) if ntok and first else float("nan")
    print(f"\nTTFT {(first - t0) * 1000:.0f} ms | 总耗时 {t1 - t0:.1f} s | "
          f"{ntok} tokens | 约 {tps:.1f} token/s", flush=True)
    print("\n--- 回答 ---\n" + "".join(out), flush=True)
    return "".join(out)
```

## 步骤 1：从源码安装 SGLang

装到**当前 Jupyter 内核的环境**里（`sys.executable`），包含全量依赖 `[all]`，
其中就有 `humming-kernels`。

> 这里不另建 venv：后面的 Python 单元（步骤 5 `import humming`、步骤 9
> `from openai import OpenAI`）跑在内核里，如果装到别的 venv 就会 import 不到。
> 若你想用独立环境，请**先**为该环境注册一个 Jupyter kernel 并切换过去，再跑本 Notebook。

本步骤**不会**编译依赖模型 shape 的 FlashInfer CUTLASS MXFP4 算子——那在步骤 3。

两点说明：

- **跳过 Rust 扩展**（`SGLANG_BUILD_RUST_EXTS=none`）。`rust/` 下的
  `sglang-grpc`、`sglang-mm`、`sglang-server` 三个 crate 分别对应 gRPC 接口、
  多模态和 Rust 版 server，本部署走 Python 的 HTTP OpenAI 兼容接口、模型是纯文本，
  三个都用不到。构建它们要从 crates.io 拉整棵 Rust 依赖树，慢网络下能多花半小时。
  需要时把这个变量去掉即可。
- **并行度放开**。此时没有模型占内存，不必像步骤 3 那样保守。

```{code-cell} ipython3
%%bash
set -euo pipefail

# 本步骤没有模型占内存，并行度按核数放开（上限 8，留足余量）
J=$(( $(nproc) < 8 ? $(nproc) : 8 ))
export MAX_JOBS="$J" NINJA_NUM_JOBS="$J" CMAKE_BUILD_PARALLEL_LEVEL="$J"
# 跳过用不到的 Rust 扩展（gRPC / 多模态 / Rust server），省去 crates.io 依赖树
export SGLANG_BUILD_RUST_EXTS=none
# 网络不稳时不要死等：单次操作 30 秒无响应即重试
PIP_NET="--timeout 30 --retries 10"
echo "并行度 MAX_JOBS=$J, 跳过 Rust 扩展"

"$PYTHON" -m pip install -q --upgrade pip setuptools wheel $PIP_NET
"$PYTHON" -m pip install -e "$SOURCE_DIR/python[all]" $PIP_NET
"$PYTHON" -m pip install -q openai modelscope $PIP_NET

PYTHONPATH="$SOURCE_DIR/python" "$PYTHON" - <<'PY'
import importlib.metadata as md
import torch
for pkg in ("sglang", "torch", "flashinfer-python", "humming-kernels"):
    try:
        print(f"{pkg}={md.version(pkg)}")
    except md.PackageNotFoundError:
        print(f"{pkg}=NOT INSTALLED")
print(f"cuda={torch.version.cuda}")
print(f"gpu={torch.cuda.get_device_name(0)}")
print(f"capability={torch.cuda.get_device_capability(0)}")
PY
```

## 步骤 2：下载 Ling-3.0-flash MXFP4 权重

官方提供预量化的 MXFP4 权重（约 60.5 GB）：

- [ModelScope](https://modelscope.cn/models/inclusionAI/Ling-3.0-flash-fp4)
- [Hugging Face](https://huggingface.co/inclusionAI/Ling-3.0-flash-fp4)

已有权重可跳过本步骤（下方单元会自动检测）。

```{code-cell} ipython3
%%bash
set -euo pipefail

if [[ -f "$MODEL_DIR/config.json" ]]; then
  echo "复用已有模型: $MODEL_DIR"
else
  "$(dirname "$PYTHON")/modelscope" download \
    --model inclusionAI/Ling-3.0-flash-fp4 \
    --local-dir "$MODEL_DIR"
fi
```

## 步骤 3：预编译 FlashInfer CUTLASS MXFP4 算子

**这一步不能跳过。** 下面这个单元会自动完成整个过程：检测是否已编译 → 需要就启动一次服务生成构建规则 →
一旦生成立刻停掉服务 → 用 `ninja -j4` 独占内存编译。

> [!WARNING]
> **必须先停掉服务再编译。** 服务加载模型占 66 GB，若与编译同时进行，GB10 的
> 121 GB 统一内存会被打爆——实测触发过两次 `cudafe++ invoked oom-killer` 硬重启。
> 停掉服务后独占内存，并行度按可用内存自动选取（单个 nvcc 实测约 4.75 GB，上限 `-j8`）。
> 对比：`-j1` 实测 7 分钟才编完 1/97。

+++

### 自动流程

FlashInfer 要先探测 GPU 架构和模型 shape 才会生成 `build.ninja`，所以**必须先启动一次服务**
才有东西可编。下面的单元把这件事自动化了，你不需要手工判断：

1. `.so` 已存在 → 直接跳过；
2. 否则后台启动一次服务，盯着 `build.ninja` 出现；
3. 一出现就立刻停掉服务（**不等它用 `-j1` 慢慢编**），改用 ninja 独占内存并行编译。

第 3 步是关键：服务内联编译走的是 `-j1`，实测 7 分钟才编完 1/97，而且此时模型权重占着
66 GB，和编译叠加正是触发 OOM 的组合。停掉服务后独占内存并行编译，约 8 分钟。

**首次启动其实有两条独立的编译链路**，别把它们搞混：

| 链路 | 缓存位置 | 触发时机 |
|---|---|---|
| `sgl_kernel`（tvm-ffi JIT） | `~/.cache/tvm-ffi/sgl_kernel_jit_*` | **权重加载阶段**就开始 |
| FlashInfer CUTLASS MoE | `~/.cache/sglang/.cache/flashinfer/*/121a/cached_ops/fused_moe_120/` | CUDA Graph 捕获阶段 |

前者是一批小 kernel，编得快、影响不大，但它会让你以为"已经在编 CUTLASS 了"——
其实 `build.ninja` 还没出现。

**实测时间线**（GB10，缓存全清的干净状态）：

```
00:59:36  Load weight begin
          权重加载 24 个分片 ........................ 6 分 17 秒
01:05:56  Preparing DSv4 MXFP4 experts（逐层，共 42 层）
01:06:51  Capture target decode CUDA graph begin  ← 此时才触发 CUTLASS
          build.ninja 生成 → 自动停服务
          ninja -j8 编译 97 个目标 .................. 约 8 分钟
```

整段约 **15 分钟**。

```{code-cell} ipython3
import shutil

ROOTS = [
    Path.home() / ".cache" / "flashinfer",
    Path.home() / ".cache" / "sglang" / ".cache" / "flashinfer",
]
PATTERN = "*/121a/cached_ops/fused_moe_120"


def newest(name):
    hits = [p for r in ROOTS for p in r.glob(f"{PATTERN}/{name}")]
    return max(hits, key=lambda p: p.stat().st_mtime) if hits else None


so = newest("fused_moe_120.so")
if so:
    print(f"已编译，跳过: {so}  ({so.stat().st_size / 1048576:.1f} MB)", flush=True)
else:
    # build.ninja 一出现就停——不等服务用 -j1 慢慢编
    state = {"size": -1, "stable": 0}

    def ninja_ready():
        bn = newest("build.ninja")
        if bn is None:
            return False
        size = bn.stat().st_size
        state["stable"] = state["stable"] + 1 if size == state["size"] else 0
        state["size"] = size
        return state["stable"] >= 3      # 连续三次(约 15 秒)不再增长

    print("启动服务以触发构建规则生成（约 8 分钟）…\n", flush=True)
    serve(extra=["--moe-runner-backend", "flashinfer_mxfp4", "--enable-fp32-lm-head"],
          env={"FLASHINFER_JIT_MAX_JOBS": "1", "MAX_JOBS": "1",
               "NINJA_NUM_JOBS": "1", "CMAKE_BUILD_PARALLEL_LEVEL": "1"},
          port=30001, until=ninja_ready)

    bn = newest("build.ninja")
    if bn is None:
        raise RuntimeError("未生成 build.ninja，检查上面的服务输出")
    print(f"\n已生成构建规则: {bn}", flush=True)
    stop()

    build_dir = bn.parent
    ninja_bin = Path(sys.executable).parent / "ninja"
    if not ninja_bin.exists():
        ninja_bin = shutil.which("ninja") or "ninja"

    # 单个 nvcc 实测约 4.75 GB。除以 10 而非 4.75 是刻意留一倍余量：
    # 本机没有 swap，一旦 OOM 会连累 nvidia 驱动锁死。
    avail_gb = next(int(l.split()[1]) / 1048576
                    for l in open("/proc/meminfo") if l.startswith("MemAvailable"))
    jobs = max(1, min(os.cpu_count() or 1, int(avail_gb // 10), 8))
    print(f"编译目录: {build_dir}", flush=True)
    print(f"可用内存 {avail_gb:.0f} GB，使用 -j{jobs} 编译…", flush=True)

    # ninja 的完整 nvcc 命令行每条几百字符，这里只显示进度
    progress, shown, failures = re.compile(r"^\[(\d+)/(\d+)\]"), 0, []
    proc = subprocess.Popen([str(ninja_bin), f"-j{jobs}", "-C", str(build_dir)],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1)
    for line in proc.stdout:
        m = progress.match(line)
        if m:
            done, total = int(m.group(1)), int(m.group(2))
            if done - shown >= 5 or done == total:
                shown = done
                print(f"  [{done:>3}/{total}] {done * 100 // total:>3}%", flush=True)
        elif line.strip() and ("FAILED" in line or "error" in line.lower()):
            failures.append(line.rstrip())
            print(f"  {line.rstrip()[:170]}", flush=True)
    if proc.wait() != 0:
        raise RuntimeError("ninja 失败：\n" + "\n".join(failures[-5:]))

    for f in build_dir.glob("*.so"):
        print(f"{f.name}: {f.stat().st_size / 1048576:.1f} MB", flush=True)
```

## 步骤 4：启动方案 A —— 原生 W4A8（基线）

- MoE：`flashinfer_mxfp4`（W4A8）  ·  LM Head：FP32  ·  不启用 Humming / FP8 LM Head / MTP

输出直接打印在下面。看到 `The server is fired up and ready to roll!` 即为就绪，
服务随后转入后台。启动约 8 分钟。

```{code-cell} ipython3
serve(extra=["--moe-runner-backend", "flashinfer_mxfp4", "--enable-fp32-lm-head"],
      env={"SGLANG_ENABLE_FP8_LM_HEAD": "0"})
```

### 验证方案 A 确实能用

健康检查 + 一次真实问答，同时给出 TTFT 和近似 decode 吞吐，作为方案 B 的对照基准。

> 这只是冒烟测试。正式性能对比要用固定输入/输出长度的 benchmark，且 decode 吞吐对采样参数
> 敏感——greedy 与 `temperature=0.6 / top_p=0.95 / top_k=29` 的结果不可直接互比。

```{code-cell} ipython3
baseline = ask()
```

## 步骤 5：停止服务

两套配置都用端口 30000，**启动方案 B 前必须先停掉方案 A**。

```{code-cell} ipython3
stop()
```

## 步骤 6：准备 Humming kernel

`humming-kernels` 已随步骤 1 一并安装。这里预先构建它的通用 launcher 与 NVRTC helper；
依赖模型 shape 的 MoE CUBIN 会在首次启动方案 B 时由 NVRTC 自动编译（单个 kernel 秒级），
并缓存到 `~/.humming/cache`，**不需要像 FlashInfer 那样单独预编译**。

另外 Humming 依赖内存带宽做 roofline 选型，而 GB10 使用统一 LPDDR5X 内存、
没有独立显存时钟域，`NVML_CLOCK_MEM` 会抛 `NVML_ERROR_NOT_SUPPORTED`。
`humming-kernels` **0.1.11 起已内置**按 compute capability `(12, 1)` 判定并返回
273 GB/s 的处理；下面的单元直接**验证带宽探测的实际返回值**，只有确实探测失败时才打补丁。

```{code-cell} ipython3
from pathlib import Path

import humming.utils.device as humming_device
from humming.ops.utils import init_humming_launcher
from humming.utils.nvrtc import may_build_nvrtc_compile_binary

native = Path(humming_device.__file__).parent.parent / "_native"
prebuilt = sorted(p.name for p in native.rglob("*.so")) if native.is_dir() else []
print(f"wheel 自带的预编译库: {prebuilt or '(无)'}", flush=True)

init_humming_launcher()
may_build_nvrtc_compile_binary()
print("launcher 与 NVRTC helper 就绪（已存在则不会重新编译）", flush=True)

cache = Path.home() / ".humming" / "cache"
n = len(list(cache.rglob("kernel.cubin"))) if cache.is_dir() else 0
print(f"已缓存的 MoE CUBIN: {n} 个 —— 依赖模型 shape，"
      f"首次启动方案 B 时才由 NVRTC 现编", flush=True)

# 直接验证行为，而不是匹配某个版本的源码写法
try:
    bandwidth = humming_device.calculate_gpu_bandwidth(0)
except Exception as exc:
    bandwidth, error = None, exc
else:
    error = None

if bandwidth and bandwidth > 0:
    print(f"内存带宽探测正常: {bandwidth:.0f} GB/s —— 无需打补丁", flush=True)
else:
    print(f"带宽探测失败（{error}），尝试写入 GB10 fallback…", flush=True)
    device_file = Path(humming_device.__file__)
    source = device_file.read_text()
    anchor = "def calculate_gpu_bandwidth(gpu_index=0):"
    if anchor not in source:
        raise RuntimeError(
            f"未找到 calculate_gpu_bandwidth，humming 版本可能变化，请手工检查 {device_file}"
        )
    patch = "\n".join([
        anchor,
        "    # GB10 (DGX Spark, sm_121) 使用统一 LPDDR5X 内存，没有独立显存时钟域，",
        "    # NVML_CLOCK_MEM 会抛 NVML_ERROR_NOT_SUPPORTED。直接返回平台标称带宽。",
        "    import torch as _torch",
        "",
        "    if _torch.cuda.get_device_capability(gpu_index) == (12, 1):",
        "        return 273.0",
        "",
    ])
    device_file.write_text(source.replace(anchor, patch, 1))
    print(f"已写入 GB10 fallback: {device_file}", flush=True)
    print("请重启内核后重新执行本单元以验证。", flush=True)
```

## 步骤 7：启动方案 B —— Humming MoE + 在线 FP8 LM Head

- MoE：`humming` MXFP4  ·  LM Head：在线量化为动态 FP8
- 其余参数与方案 A **完全一致**（都来自 `COMMON`），单变量对比由结构保证

首次启动时 Humming 会用 NVRTC 现编若干 MoE kernel（依赖模型 shape，单个秒级），
缓存到 `~/.humming/cache`，之后启动直接命中。

```{code-cell} ipython3
serve(extra=["--moe-runner-backend", "humming"],
      env={"SGLANG_ENABLE_FP8_LM_HEAD": "1",
           "HUMMING_COMPILER": "nvrtc",
           "HUMMING_CACHE_DIR": str(Path.home() / ".humming" / "cache")})
```

## 步骤 8：确认优化确实生效

**静默降级是这类部署最常见的坑**：环境变量拼错、或分支缺少补丁时，SGLang 不会报错，
只是安静地跑在未优化的路径上——性能差一截却毫无提示。

下面的单元等待服务就绪，然后逐项核对方案 B 的三个标志。

```{code-cell} ipython3
log = "\n".join(SERVER["log"])

checks = {
    "Humming MoE backend": "moe_runner_backend=humming" in log,
    "Humming quant method": "Mxfp4HummingMoEMethod" in log,
    "在线 FP8 LM Head": "Online FP8 quantization enabled for lm_head" in log,
}
for name, ok in checks.items():
    print(f"{'[OK]  ' if ok else '[FAIL]'} {name}", flush=True)

if not all(checks.values()):
    raise RuntimeError(
        "有优化项未生效。若 FP8 LM Head 失败，通常是所用分支缺少对应补丁"
        "（SGLANG_ENABLE_FP8_LM_HEAD 不是上游变量，设置未知变量不会报错）。"
    )
print("\n三项检查全部通过。", flush=True)
```

### 验证方案 B 并与基线对照

```{code-cell} ipython3
optimized = ask()
```

## 步骤 9：停止服务

测完记得停，否则会一直占着 66 GB 内存和端口 30000。

```{code-cell} ipython3
stop()
```

## 步骤 10：方案 C —— 三项优化全开（Humming + FP8 LM Head + MTP3）

前面的 A/B 只对比了 Humming 和 FP8 LM Head 两项，**故意不含 MTP**，为的是保持单变量。
但完整方案的收益主要来自 MTP：

| 配置 | GSM8K-1319 实测 decode | 相对基线 |
|---|---|---|
| 方案 A（基线） | 37.8 tok/s | — |
| 方案 B（+ Humming + FP8 LM Head） | 约 41 tok/s | +9% |
| **方案 C（再 + MTP3）** | **53.9 tok/s** | **+42.5%** |

MTP 能和前两项共存，靠的是给 `BailingMoeForCausalLMNextN` 补上 `set_lm_head_from_target`
（见 `UPSTREAM.md`）。没有这个修复，draft 会拿到 FP8 权重张量却按 BF16 解释，
接受率归零，**吞吐反而掉到 24 tok/s，比不开任何优化还慢**。

下面这个单元**不复用 `serve()`**，参数全部显式写出，方便直接拷到终端里跑。
MTP 会额外加载一个 draft 模型（约 3.4 GB）并多捕获两个 CUDA graph，启动比 A/B 慢约 1 分钟。

```{code-cell} ipython3
import os, re, subprocess, threading, time

env = {
    **os.environ,
    "PYTHONPATH": str(SOURCE_DIR / "python"),
    "SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN": "1",
    "SGLANG_JIT_DEEPGEMM_PRECOMPILE": "0",
    "SGLANG_ENABLE_JIT_DEEPGEMM": "0",
    "SGLANG_DSV4_FP4_DEQUANT": "0",
    "SGLANG_FP8_IGNORED_LAYERS": "",
    "SGLANG_ENABLE_FP8_LM_HEAD": "1",              # ← 优化 2：lm_head 在线量化为 FP8
    "HUMMING_COMPILER": "nvrtc",
    "HUMMING_CACHE_DIR": str(Path.home() / ".humming" / "cache"),
}

cmd = [
    PYTHON, "-m", "sglang.launch_server",
    "--model-path", str(MODEL_DIR),
    "--served-model-name", "ling-v3-flash-fp4",
    "--trust-remote-code", "--dtype", "bfloat16",
    "--tp-size", "1", "--ep-size", "1",
    "--host", "0.0.0.0", "--port", str(PORT), "--api-key", API_KEY,
    "--max-running-requests", "1", "--max-mamba-cache-size", "64",
    "--chunked-prefill-size", "8192", "--max-prefill-tokens", "16384",
    "--page-size", "64", "--context-length", "262144",
    "--cuda-graph-backend-decode", "full",
    "--cuda-graph-max-bs-decode", "1", "--cuda-graph-bs-decode", "1",
    "--cuda-graph-backend-prefill", "disabled",
    "--random-seed", "308534008",
    "--reasoning-parser", "deepseek-r1", "--tool-call-parser", "qwen25",
    "--attention-backend", "flashinfer", "--disable-flashinfer-autotune",
    "--mem-fraction-static", MEM_FRACTION_STATIC,
    "--fp8-gemm-backend", "cutlass",
    "--moe-runner-backend", "humming",             # ← 优化 1：Humming MXFP4 MoE
    "--flashinfer-mxfp4-moe-precision", "default",
    "--disable-shared-experts-fusion",
    # ---- 优化 3：MTP / NEXTN 投机解码，3 步 ----
    "--speculative-algorithm", "NEXTN",
    "--speculative-draft-model-path", str(MODEL_DIR),
    "--speculative-num-steps", "3",
    "--speculative-eagle-topk", "1",
    "--speculative-num-draft-tokens", "4",
    "--json-model-override-args",
    '{"max_position_embeddings":262144,"rope_scaling":{"rope_type":"yarn","factor":2.0,'
    '"rope_theta":6000000,"partial_rotary_factor":0.5,'
    '"original_max_position_embeddings":131072}}',
]

subprocess.run(["pkill", "-f", "sglang.launch_server"], capture_output=True)
time.sleep(5)

proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                        start_new_session=True, env=env)
log, buf = [], ""
SERVER["proc"], SERVER["log"] = proc, log
while True:
    data = os.read(proc.stdout.fileno(), 8192)
    if not data:
        break
    buf += data.decode("utf-8", "replace")
    *lines, buf = re.split(r"[\r\n]", buf)
    for ln in lines:
        if ln.strip():
            log.append(ln)
            print(ln[:170], flush=True)
    if any("ready to roll" in ln for ln in log[-30:]):
        break
threading.Thread(target=lambda: [log.append(x.decode("utf-8", "replace").strip())
                                 for x in iter(proc.stdout.readline, b"")],
                 daemon=True).start()

print("\n=== 三项优化确认 ===", flush=True)
for key in ["moe_runner_backend=humming",
            "Online FP8 quantization enabled for lm_head",
            "BailingMoeForCausalLMNextN"]:
    print(f"{'[OK]  ' if any(key in ln for ln in log) else '[FAIL]'} {key}", flush=True)
```

### 验证方案 C

除了问答，重点看 **MTP 的接受长度**——这是判断投机解码是否真正生效的关键指标。
正常应在 **2.5 以上**（实测 2.68）；如果是 **1.00**，说明 draft 输出全错、投机完全没命中，
此时 MTP 不但不加速，反而因为每步白跑 3 次 draft 前向而**拖慢近一半**。

```{code-cell} ipython3
full = ask()

accept = [float(m) for ln in SERVER["log"]
          for m in re.findall(r"accept len: ([0-9.]+)", ln)]
if accept:
    avg = sum(accept) / len(accept)
    print(f"\nMTP 平均接受长度: {avg:.2f}  (n={len(accept)})", flush=True)
    print("  正常" if avg > 2.0 else
          "  异常——接近 1.0 说明投机没命中，检查 set_lm_head_from_target 是否生效",
          flush=True)
else:
    print("\n未采集到 accept len，可能是输出太短还没触发统计", flush=True)
```

### 停止服务

```{code-cell} ipython3
stop()
```

## 步骤 11:记录清单

对外发布结果前请补齐：

- 源码仓库 URL、分支与**完整 commit SHA**；
- `pip freeze`、CUDA Toolkit、NVIDIA driver 版本与 GPU 型号；
- 模型仓库 revision 或权重校验值；
- FlashInfer CUTLASS `.so` 与 Humming CUBIN 均在目标机器上成功生成；
- 两套配置使用**相同的** prompt、采样参数、并发度与随机种子；
- 对 FP8 LM Head 单独做真实数据集的准确率对比（本仓库的 GSM8K-1319 结果见 `UPSTREAM.md`）。

**可以提交到 Git**：Notebook、源码补丁、依赖锁文件。
**不要提交**：模型权重、`.venv`、FlashInfer 缓存、Humming CUBIN——它们与
CUDA / PyTorch / FlashInfer 版本及 GPU 架构绑定，必须在目标机器上重新生成。

```{code-cell} ipython3

```
