# swe_opd 运行环境（vanilla_opd：slime fully-async OPD）

本目录是可单独交付的源码重建包。包内不含 venv、CUDA shim、模型、数据或 `.secret/.key.sh`；重建需要内部 Git/PyPI、NVIDIA 下载权限以及 8x H800/CUDA 12.9 条件。推荐执行 `bash env/setup.sh`，它会按编号顺序运行构建和验证脚本。

本文档记录 `/path/to/swe-opd/vanilla_opd/` 下这套环境**实际是怎么装出来的**，
以及和 slime 上游构建说明（build_uv 线，随本次整理已从 `slime/` 移除）的每一处偏差。以实测为准，不以原文档为准。

**与姊妹包 `../offline_opd` 的关系**：本包不携带 `sglang/` 与 `Megatron-LM/`，构建与运行时通过
`GOLDEN_OPD_DIR`（默认 `../offline_opd`，可 `export GOLDEN_OPD_DIR=<其他位置>` 覆盖）复用它们的
源码树——两树与 offline_opd **完全一致**（同 commit），已通过树级 diff + 完整训练重跑实证（§7 3e）。
依赖清单与 "需要 `--no-build-isolation` 的包" 的安装方法见
`../offline_opd/README.md` 与 `../offline_opd/requirements.txt`；本包 `env/example` 的
20（sglang）/ 40（Megatron）阶段与之等价（分步 editable 安装，rust / helpers_cpp 现场编译）。

用法：

```bash
cd /path/to/swe-opd/vanilla_opd
bash env/setup.sh
source env/env.sh
```

三个 checkout 的 URL、分支和完整固定 commit 见 `env/repos.lock`（URL 列为内部库位置，branch/commit 是权威）。

从零重建：`env/example/` 下按编号顺序执行 `00 → 10 → 20 → 30 → 31 → 32 → 40 → 50`。
每个脚本都是 `set -ex` 且尽量幂等（`32_apex.sh` 支持 `SKIP_APEX=1` 跳过 20min 的 apex 编译）。
`60_serve.sh` 不属于构建链，它是验收/使用期拉 sglang 的运行脚本。

`env/example/` 里几个容易误认的点：torch_memory_saver 就装在 `32_apex.sh` 里，没有独立的 `32b_*.sh`；
`redist/`（NVIDIA cuda_nvcc 归档）**不入包**：`10_cuda_shim.sh` 在缺失时自动 curl 下载。

## 1. 机器 / 目录

| 项 | 值 |
| --- | --- |
| GPU | 8x H800（sm90） |
| 容器 CUDA | **只有 `/usr/local/cuda-13.0`**，没有 12.x |
| venv | `vanilla_opd/slime-venv`（Python 3.12.13，uv 创建） |
| CUDA_HOME | `vanilla_opd/cuda-12.9-shim`（264 MB，见 §3） |
| GOLDEN_OPD_DIR | `vanilla_opd/../offline_opd`（默认；`sglang/` + `Megatron-LM/` 所在） |
| venv 体积 | 17 GB |
| 最终 freeze | `env/freeze_final.txt`（401 包） |

三个代码库都是 editable 安装：`slime` 用本包 checkout，`sglang` / `Megatron-LM` 用 `$GOLDEN_OPD_DIR`
下的 checkout（commit 与 `repos.lock` 一致）：

| 库 | 路径 | 分支 | commit |
| --- | --- | --- | --- |
| slime | `vanilla_opd/slime` | - | `4b45ca146dbcc35db992104f4fb3d2bae59d502c` |
| sglang | `$GOLDEN_OPD_DIR/sglang` | offline_opd | `0e33d68d3f68bad64e1363926f3a0cfd443c016d` |
| Megatron-LM | `$GOLDEN_OPD_DIR/Megatron-LM` | offline_opd | `0c5d1c445da0ed1916c3c6c144bf0188fcf5033e` |

`offline_opd` 分支基于当时验收的 `main` 线（sglang `767c0146d`、Megatron `4bf7fca05`，
即本环境构建时的 checkout），之上叠加 offline OPD 支持。

注：安装出来的版本号（editable 的本地 git 描述）以 `env/freeze_final.txt` 为准；其中
`freeze_final.txt` 里的 `sglang==0.5.15.post2.dev37+g767c0146d`、`megatron-core==0.19.0+4bf7fca05`
是构建时（`main` 线 checkout）的实测快照，与 `offline_opd` 分支无关。

**patch 都没打**，这是刻意的：

- `docker/patch/latest/sglang.patch` —— fork（`main` 线）里已经包含相关改动，重复打会冲突
  （构建时在 `main 767c0146d` 上验证过 `git apply --check -R` 通过）。
- `docker/patch/latest/megatron.patch` —— 对不上当前 checkout（记录时 21 个文件里只有 6 个能 apply；
  `megatron/training/tokenizer/tokenizer.py` 曾在更早的记录中被认为不存在，当前树已有该文件但仍
  无法应用，如 `megatron/core/dist_checkpointing/strategies/common.py:86` 失败）。
  **后果：OPD 训练走的是 `offline_opd` 分支自带实现，不经过该 patch 线**——这条路径
  已于本次实测跑通（§7 3d，singlenode fully-async 全流程 `exit 0`），但 build_uv 打 patch 之后
  的那条线仍不算验证过。

## 2. 关键版本（实测）

```
torch                2.11.0+cu129     torch.version.cuda = 12.9
cudnn                9.17.1.4         torch.backends.cudnn.version() = 91701
transformer_engine   2.17.0           transformer_engine_cu12 / transformer_engine_torch 同为 2.17.0
flash-attn           2.7.4.post1      本地编译（85 TU，MAX_JOBS=64，18m12s）
flash-attn-4         4.0.0b15         py3-none-any，装到 flash_attn.cute 命名空间
sglang-kernel        0.4.2.post2      +cu129，来自 docs.sglang.io/whl/cu129/
sgl-deep-gemm        0.1.0            同上
triton               3.6.0
numpy                1.26.4           apex/flash-attn/megatron 都按 1.x ABI 编的
scipy                1.15.3
apex                 0.1              10417ace，--cpp_ext --cuda_ext
nvidia-resiliency-ext 0.6.0
rle-open-sdk         0.2.1            只在 pip.baidu-int.com 上有
```

> **注意：cudnn 版本号以本文件的实测为准。**实际值是 `9.17.1.4` / `91701`，
> 即 torch 2.11.0+cu129 在 `Requires-Dist` 里 pin 的那个。容器自带的系统 cudnn 会 shadow
> wheel 里的，所以 `env/env.sh` 必须把 venv 的 `nvidia/cudnn/lib` 前置到 `LD_LIBRARY_PATH`。

## 3. CUDA 12.9 shim：为什么以及怎么做的

torch 是 `+cu129`，容器只有 CUDA 13.0，flash-attn / apex / TE-torch 都要本地 nvcc 编译。
方案是**在 workspace 内合成一个自包含的 CUDA_HOME**，全程不碰 `/usr/local/cuda-13.0`。

踩到的坑：**pip 上的 `nvidia-cuda-nvcc-cu12` 不含 `nvcc`，只有 `ptxas`。**
已逐一验证 12.6.85 / 12.8.93 / 12.9.41 / 12.9.86 四个 wheel，binary 一律只有
`nvidia/cuda_nvcc/bin/ptxas` + `nvvm/lib64/libnvvm.so` + `libdevice.10.bc`；
`nvidia-cuda-nvcc`（无后缀）和 `nvidia-cuda-crt-cu12` 在 pypi 上不存在。

所以 `10_cuda_shim.sh` 的做法是：

1. nvcc 前端取自 NVIDIA redistrib 归档
   `cuda_nvcc-linux-x86_64-12.9.86-archive.tar.xz`（和 wheel 同一个 12.9.86 build，81 MB）；
2. 其余 include/lib 全部从 venv 里 `cuda-toolkit[all]==12.9.1` 的 wheel 软链过来；
3. 补 wheel 没有的**无版本号 dev 符号链接**（wheel 只给 `libcudart.so.12`，
   不给 `libcudart.so`，否则 `ld: cannot find -lcudart`）；
4. `libcuda.so` stub 链到 `/usr/local/cuda-13.0/lib64/stubs/libcuda.so`（只用于链接期）；
5. 造出 `targets/x86_64-linux/{include,lib}` 和 `lib` 这几个别名目录，很多 setup.py 会找。

自检：`nvcc --version` = 12.9.86；一个 sm_90 kernel 编译+链接+运行通过；
`<cub/cub.cuh>` + `<thrust/device_vector.h>` 编译通过。
真正的验收是 flash-attn 和 apex（`--cpp_ext --cuda_ext`）都编译成功并能 import
其 CUDA 扩展（`flash_attn_2_cuda`、`amp_C`、`fused_layer_norm_cuda`）。

## 4. head_dim=256：TE 2.17 + FA4，且必须关掉 cuDNN fused attention

Qwen3.5-9B 的 `head_dim=256`。

- TE **2.10**（上游 build_uv 线的版本）对 thd/packed + head_dim=256 找不到任何
  dot-product-attention backend，直接 abort。所以升到 **2.17.0**。
- TE 2.17 默认优先走自己的 cuDNN **FusedAttention** backend，但这条路在 cudnn 9.17.1 上
  **反向是坏的**：
  ```
  fused_attn_f16_arbitrary_seqlen.cu:934 cuDNN Error:
  set_attribute(..., CUDNN_ATTR_OPERATION_RESHAPE_MODE, ...) CUDNN_STATUS_BAD_PARAM
  ```
  （参照 venv 的 cudnn 9.17.1.4 + cudnn-frontend 1.26.0 与本环境完全一致，不是版本装错。）
- 因此 `env/env.sh` 设 **`NVTE_FUSED_ATTN=0`**，TE 就会 dispatch 到
  flash-attn-4 4.0.0b15 的 CUTE varlen kernel，`head_dim=256` thd 的 fwd+bwd 正常。
  这只影响 TE（megatron 训练侧），sglang 推理侧不经过它。
- **sglang 侧相反**：FA4 的 CUTE varlen kernel 在 head_dim=256 会 assert，
  所以 serve 用 `--attention-backend flashinfer`。

## 5. 依赖偏差清单

和上游 build_uv 构建线的**刻意**偏差：

| 项 | 上游 build_uv 线 | 本环境 | 原因 |
| --- | --- | --- | --- |
| sglang | github v0.5.12.post1 | 本地 fork offline_opd `0e33d68d3…`（基于 main `767c0146d`） | 用户指定；patch 改动已内含 |
| Megatron | `1dcf0dafa` + megatron.patch | 本地 offline_opd `0c5d1c445…`（基于 main `4bf7fca05`），不打 patch | 用户指定；patch 对不上 |
| transformer_engine | 2.10.0 | 2.17.0（cu12） | head_dim=256，见 §4 |
| CUDA_HOME | `/usr/local/cuda-12.9` | workspace 内 shim | 容器没有 12.9 |
| torch_memory_saver | `dc687690` | `6d5bce48` | 对齐参照 venv |
| Megatron-Bridge | `dev_rl` | `35b4ebfc` | 对齐参照 venv |

装的过程中踩出来、必须显式 pin 的（凡是原文档没覆盖的分歧，一律以构建时对齐的
**参照 venv**（内部另一套已验证环境，路径不再列出）为准）：

| 包 | pin | 不 pin 会怎样 |
| --- | --- | --- |
| `kernels==0.12.3` + `kernels-data==0.16.0` | 参照 venv | kernels 0.16.x 改了 `LayerRepository` 签名，transformers 5.6.0 的 `hub_kernels.py` 抛 `ValueError: Either a revision or a version must be specified` → `import sglang` 直接失败 |
| `transformer_engine_cu12==2.17.0`（先卸 `transformer-engine-cu13`） | 用户指定 | `transformer_engine[pytorch]` extra 在这台机器上解析成 cu13（233 MB），和 `+cu129` 的 torch 不是一条线 |
| `nvidia-cutlass-dsl-libs-cu13==4.5.1` | 参照 venv | 20 阶段那段 cu13 清洗（从 build_uv 线抄的）会把它删掉，而 TE 2.17 的 `dot_product_attention/backends.py` 在 import 期就 `import flash_attn.cute.interface` → 需要 `cutlass` → `ModuleNotFoundError`。这是**唯一**允许保留的 cu13 包：它只落在 `site-packages/nvidia_cutlass_dsl/`，不和 cu12 抢 `nvidia/*`，且 4.5.1 没有 cu12 flavour |
| `nvidia-cudnn-frontend==1.26.0` | 参照 venv | 默认解析到 1.28.0 |
| `nvidia-resiliency-ext==0.6.0` | Megatron 自己的 pyproject | slime 的 requirements.txt 走 git ref 装成 0.4.1，而 0.4.1 没有 `__version__` 属性；Megatron 的 `dist_checkpointing/strategies/nvrx.py` 在 `import megatron.core` 时就读它 → `AttributeError`（构建时 checkout 上的实测）。**这里 Megatron 的 pin 压过参照 venv 的 0.4.1**（那个 venv 带的是更老的 Megatron 0.16.0rc0） |
| `scipy==1.15.3` | 参照 venv | slime 的 resolve 落到 1.18.x，而 1.18 硬要求 numpy>=2，`import scipy.sparse` 报 `module 'numpy' has no attribute 'long'` |
| `numpy<2`（1.26.4） | 上游 build_uv 线 | apex / flash-attn / megatron 的扩展都按 1.x ABI 编的 |
| `fastspec==0.1.1` | 参照 venv | `ghapi==2.0.3` 要 `fastspec.oapi._build_groups`，0.2.x 已删 → `import ghapi` ImportError |
| `docker==7.2.0`、`dockerfile-parse==2.0.1`、`bs4`、`ghapi==2.0.3`、`unidiff==0.7.5`、`modal==1.5.2`、`tenacity==9.1.4` | 参照 venv | `examples/coding_agent_opd/r2egym/__init__.py` 会 eager import `.runtime` → `r2egym.agenthub.runtime.docker`，这些都是**模块级** import，即使 `SLIME_AGENT_SANDBOX_BACKEND=rle` 也躲不掉 |
| `chardet==5.2.0`、`gitpython`、`httpx-sse` | 参照 venv | 前两个被 r2egym 的 `str_replace_editor`/`file_editor` 和 vendored swebench 用到，任何 resolve 都不会自动带进来 |

有意**不装**的（参照 venv 里也没有，且会打架）：
`matplotlib`、`scikit-learn`、`seaborn`、`gpustat`、`swebench`、`sb-cli`、`google-cloud-aiplatform`
—— r2egym 的 `datasets==2.19` pin 会和 slime 的 5.x 冲突。

**r2egym 本身不 pip 装**：`examples/coding_agent_opd/r2egym/__init__.py` 是一个 `sys.path` shim，
指向 `examples/coding_agent_opd/3rdparty/R2E-Gym/src`。只装它的运行期依赖。

其余和参照 venv 的差异都是未 pin 的传递依赖小版本漂移（参照 venv 建得更早），
`env/freeze_final.txt` 是本环境的权威快照。

## 6. env.sh 里几个不显然的设置

- **`GOLDEN_OPD_DIR`**：`sglang/` / `Megatron-LM/` 的所在目录，默认 `vanilla_opd/../offline_opd`
  （swe-opd/ 布局），`export GOLDEN_OPD_DIR=<其他位置>` 可覆盖（该目录下需有这两棵树）。
  `env/env.sh` 与 `env/example/common.sh` 各自维护同样的默认推导，两处都允许环境变量优先。
- **`PYTHONPATH="$SLIME_DIR:$MEGATRON_DIR"`，slime 必须在前。**
  两棵树都有带 `__init__.py` 的 `examples/` 包，谁在前谁占用 `examples` 这个名字；
  slime 的入口 import `examples.coding_agent_opd.*`，所以 slime 得赢。
  **但注意姊妹包 `../offline_opd` 侧 `eval/run_eval.sh:119` 会把这个顺序反过来**：它自己
  `export PYTHONPATH="$MEGATRON_LM:$SLIME_DIR:$PYTHONPATH"`，Megatron 在前。
  它之所以还能跑，是因为同一脚本先 `cd "$SLIME_DIR"`（第 28 行），`sys.path[0]` 是 cwd，
  压过了 PYTHONPATH。实测：在中立 cwd 下用 Megatron-first 的顺序，
  `import examples` 直接 `ModuleNotFoundError`。所以 §7 的 3c 命令里那个 `cd slime` 不是随手写的。
- **`MEGATRON_LM`**：姊妹包 `../offline_opd` 的 `eval/run_eval.sh`、`eval/serve_ckpt.sh`
  读的是这个名字（不是 `MEGATRON_DIR`）。它们的默认值 `$SLIME_DIR/../Megatron-LM` 在 vanilla 布局下
  **不存在**（Megatron 在 `$GOLDEN_OPD_DIR`），所以 `env/env.sh` 显式导出
  `MEGATRON_LM=$MEGATRON_DIR` 是必须的——否则这些脚本会去找默认路径而失败。
- **`LD_LIBRARY_PATH` 前置 venv 的 cudnn**，否则容器系统 cudnn 会 shadow wheel 版本。
- **`NVTE_FUSED_ATTN=0`**，见 §4。
- **代理**：原构建走私有代理（地址已脱敏），复现时外网访问请先
  `export HTTP_PROXY/HTTPS_PROXY`（`env/env.sh` 默认不设代理）。
  swebench 的 test-spec builder 会对 `raw.githubusercontent.com` 发一个没有超时的
  `requests.get`，所以 rollout 阶段也必须有代理；`env/env.sh` 的 `no_proxy` 用私有
  网段（`10.0.0.0/8`、`100.64.0.0/10` 等）绕开集群内主机；RLE 的宿主 IP 已脱敏为
  `10.0.0.18` / `10.0.0.19` 占位（`scripts/*.sh` / 姊妹包 `run_eval_full_fncall.sh`
  里写的是同一对；`rle.internal.example.com` 走 `.internal.example.com` 这条通配）。
- **`ray` 要走全路径**：`$SLIME_VENV/bin/ray` 的 shebang 是绝对路径、指向本 venv，本身没问题；
  风险在于这台机器上可能还有别的 venv，PATH 里先命中哪个 `ray` 就用哪个 venv 的解释器。
  要确定性就写 `$SLIME_VENV/bin/python3 $SLIME_VENV/bin/ray`。
- **`uv pip` 的 cwd 陷阱**：工作区里若存在残缺的其他 venv（如某目录下 `pyvenv.cfg` 缺失），
  `uv pip` 在该目录内会被 uv 发现并报 `error: Broken virtual environment ...`。
  构建期一律用 `source env/example/common.sh && activate_venv`，或者至少把 cwd 放在 `swe_opd/` 里。

## 7. 验证状态

`bash env/example/50_verify.sh` → `ENV SMOKE OK`。覆盖：

- `torch 2.11.0+cu129` / cuda 12.9 / cudnn 91701 / 8 devices
- import：`sglang`、`megatron.core`、`transformer_engine.pytorch`、`flash_attn`、`apex`、
  `torch_memory_saver`、`slime`、`examples.coding_agent_opd.r2egym`、`rle_open_sdk`
- import 入口：`generate`、`opd`、`slime.rollout.sglang_rollout`、
  `slime.backends.{megatron_utils,sglang_utils}`
- cu12/cu13 命名空间体检（只允许 `nvidia-cutlass-dsl-libs-cu13`）
- 真实 GPU：bf16 matmul、cudnn conv2d、flash-attn-2 forward
- TE `head_dim=256` thd fwd+bwd，并断言选中的是 FlashAttention(FA4) 而非 FusedAttention

**Phase 3 验收：通过。**（8 卡当时被另一个 SWE-bench-verified sweep 占着，按用户「超发进程视为 idle」
的判断，改用 4-7 号卡 + `MEM_FRACTION=0.40` 跑完。）

### 3a serve

```bash
STUDENT_GPUS=4,5 TEACHER_GPUS=6,7 MEM_FRACTION=0.40 bash env/example/60_serve.sh   # 幂等，等 /health_generate
bash env/example/60_serve.sh stop
```

两个实例都 healthy，`model_type=qwen3_5` / `Qwen3_5ForConditionalGeneration`，
tp2、`--attention-backend flashinfer`、`--context-length 131072`。
权重 8.85 GB/卡；**这是个 hybrid GDN(Mamba) 模型**——日志里
`Using hybrid linear attention backend for hybrid GDN models`、
`GDN kernel dispatcher: decode=TritonGDNKernel`，另外单独分配了
`max_mamba_cache_size 437` / `ssm_state 10.27GB`；KV cache 765096 tokens（K/V 各 5.84 GB）。

日志里的 `OSError: libnppicc.so.12` 是 torchcodec 可选共享库加载失败，无害。

### 3b teacher logprob

（此验证用脚本 `gen_opd_teacher_payload.py` 原在 `scripts/test/`，已随 `test/` 移出发布体；
以下仅记录当时的命令与结论。）

```bash
python examples/coding_agent_opd/scripts/test/gen_opd_teacher_payload.py \
   --R 8 --K 4 --prompt-len 16 --vocab 240000 --url http://127.0.0.1:13141/generate
```

`http=200 time=0.377s`，`prompt_tokens=24`，`meta_info.input_token_ids_logprobs` 返回 9 行
（第 0 行 `None`，其余每行 K=4 个 `[logprob, token_id, None]`）。

### 3c rollout + RLE 打分

（smoke 数据行与结果目录不属于交付包，此处只留验收结论与踩坑记录。）

`resolved=1/2 errors=0 elapsed=298s`。四条通过标准全中：RLE pod 起来、工具循环产出 turn、
git diff 非空、`run_evaluation` 在 `summary.json` 里给出 0/1 reward。
（`submitted=0` 不是问题：两条都是耗尽 60 步预算收尾，harness 对最终 git diff 一样会打分。）

**踩到的坑：`--context-length` 必须给够，且 `MAX_TOKENS` 要留 0。**
第一次用 `CTX_LEN=32768` + `MAX_TOKENS=4096` 跑，轨迹长到 ~23k/29k token 时两条都
`status=error`：

```
aiohttp ClientResponseError: 400, message='Bad Request', url='http://127.0.0.1:13140/v1/chat/completions'
```

sglang 的 400 body 是
`Requested token count exceeds the model's maximum context length of 32768 tokens.
You requested a total of 33106 tokens: 29010 from the input messages and 4096 for the completion.`
—— 即 prompt + `max_tokens` 越界。两点要注意：

- `eval_chat.py` 的 `--max-tokens` 默认就是 **0**（不发 `max_tokens`，让 server 给
  `context_len - input_len - 2`），`--max-model-len` 默认 **131072**。把 server 开在 32768
  而 `max_model_len` 留 131072，循环不会自己刹车，只会撞 400。所以 `env/example/60_serve.sh` 的
  `CTX_LEN` 默认值已改成 131072（权重支持：`text_config.max_position_embeddings=262144`；
  KV 池不是瓶颈：0.40 的 mem-fraction 下也有 765096 tokens）。
- **slime 有个 bug 会把这个 400 的原文吞掉**：姊妹包 `eval/eval_chat.py:328` 写的是
  `raise "http 400 error"`，于是真正抛出来的是
  `TypeError: exceptions must derive from BaseException`。server 端日志也只打
  `serving_chat.py:1099 return self.create_error_response(str(e))`，不带错误文本。
  要看真因得自己复现一个超长请求去读 HTTP body。

3c 用的数据是 **扁平 sweepbench 行**（`instance_id` / `docker_image` / `test_patch` …），不是
训练用的 `prompt`/`label`/`metadata` 结构：`eval_chat._load_dataset` 读的就是前者，而
`eval/benchmarks.json` 里注册的 `data/swebench-latest.jsonl` 等文件在本 workspace 不存在。

### 3d fully-async OPD 单节点验证（train_async.py，`exit 0`）

脚本：`examples/coding_agent_opd/scripts/run-qwen3.5-9B-coding-opd-fully_async-singlenode-2tr-2te-4ro_scaleswe_topk128.sh`
—— 多节点脚本（TP8）的单机压缩版，同语义（topk_reverse_kl / OPD_ID_TOPK=128 / scale_swe /
pop_longest_first），8 卡布局：

- teacher Qwen3.6-27B sglang tp2 dp1 → GPUs 6,7（ray 外）；
- actor Qwen3.5-9B megatron TP2×DP1 → GPUs 0,1；
- rollout 2 引擎 × 2 卡 → GPUs 2-5；ray 只见 0-5。

结果：teacher healthy（one-shot prefill 打分）、rollout 引擎 cuda graph True、40 步轨迹
（~25-30k tokens）、megatron 训练 7+ 步（`train/loss` 0.1397→0.0654、`opd_topk_overlap` ~0.59、
`opd_topk_overlap_at1` ~0.93、`grad_norm` 2.95-5.08、`lr` 1e-6）、checkpoint 以 torch_dist 格式保存
（`Qwen3.5-9B_save_ckpt_scaleswe_topk128_singlenode_verify/iter_0000003/` +
`rollout/global_dataset_state_dict_3.pt`），脚本 `exit 0`。

**坑 1：TP2 下 custom allreduce 在 CUDA graph capture 必挂。**
`Exception: Capture cuda graph failed` ← `custom_all_reduce.cuh:37` `cudaIpcGetMemHandle` 报
`CUDA error: invalid argument`（IPCMemHandle 在 graph capture 中拿不到）。修法：teacher 启动加
`--disable-custom-all-reduce`；rollout 引擎（同样 TP2）加 `--sglang-disable-custom-all-reduce`
（slime 的 `sglang_utils/arguments.py` 会把 ServerArgs 字段自动包装成 `--sglang-*`）。回退 NCCL
allreduce，只影响性能。multinode 脚本的 rollout 引擎（每引擎 2 卡）也要加，teacher 是 TP1 不用。

**坑 2：轨迹长度预算。** scale_swe 无上限轨迹能到 65k-73k response tokens（100 turns × ~700 tok）。
TP2 下 `[T, V/TP]` fp32 logits = T×124160×4B：

- actor：65k → 需 **32.41 GiB**（`vocab_parallel_logprobs_on_ids`，ppo_utils.py:393），
  当时仅 32.39 GiB 空闲 → `torch.OutOfMemoryError`，差额只有 20MB；
- teacher：27B tp2 mem-fraction 0.8 只留 34.5GB 池后余量，73k 打分 prefill ≈36GB 顶到极限，
  随后 detokenizer 被 SIGKILL(-9) 整机挂掉。

修法（仅 singlenode）压轨迹预算：`MAX_CONTEXT_LEN=32768`、`MAX_TOKENS_PER_GPU=32768`、
`SLIME_AGENT_MAX_STEPS=40`（**必须**在 `scripts/utils/general_settings.sh` 被 source 前 export，否则被其默认
100 覆盖——它会把 `SLIME_AGENT_MAX_STEPS` 再导出进 sandbox runtime）。multinode（TP8）不需要：
131k 下 logits 只 ~16GB/rank。

其他观察：`OSError: libnppicc.so.12`（torchcodec 可选共享库）无害；teacher 健康检查
`curl http://127.0.0.1:13141/health_generate`；启动成功标志
`The server is fired up and ready to roll!`；teacher 强制 `--chunked-prefill-size -1`
（topk_reverse_kl 必需）+ `SGLANG_ENABLE_LOGITS_PROCESSER_CHUNK=1`。

### 3e offline_opd 复用实证（2026-11-15，`exit 0`）

本包「复用 `../offline_opd`」设计的事实依据。做法：删除旧 checkout 后，把 venv 的
`megatron-core` / `sglang` editable **重装为 offline_opd 的两棵树**（缺 `_version.py`、
egg-info、`helpers_cpp.so`、sglang-grpc `_core.so` 等构建产物时就地 rebuild，
`sglang` 的 rust 扩展须把 `CARGO_TARGET_DIR` 放本地盘避开 pfs3 mmap 问题），
`GOLDEN_OPD_DIR` 布局下重跑 §7 3d 的 singlenode 脚本：

- 结果与旧 checkout 一致：teacher 200s healthy、rollout cuda graph True、40 步轨迹
  （resp ~29k tokens）、`train/loss` 0.115→0.084（6 步）、checkpoint
  `Qwen3.5-9B_save_ckpt_scaleswe_topk128_singlenode_verify_offline_opd/iter_0000003`
  （torch_dist）、脚本 `+ exit 0`；sglang 引擎日志进程路径显示源自 offline_opd 树。
- **期间发现并修复的坑**：`general_settings.sh` 的 `build_runtime_env_json` 曾把 ray runtime env 的
  `PYTHONPATH` 硬编码为一个固定 Megatron 路径（旧 checkout 移走后悬空，ray worker
  `ModuleNotFoundError: megatron.training`）；现已改为按
  `${MEGATRON_DIR:-$MEGATRON_LM:-$GOLDEN_OPD_DIR/Megatron-LM}` 解析（请在 `source env.sh` 后运行，
  三个变量都会由它导出）。
  `megatron.training` 不靠 editable finder（finder 只映射 `megatron.core`），只靠这条 PYTHONPATH。
- 结论：offline_opd 与旧 checkout 的 sglang / Megatron-LM 源码等价可跑，复用设计成立。

## 8. 凭据

RLE 的 key 需要拷贝到 `slime/.secret/.key.sh`（gitignored；发布包不含，构建/运行前自备——
与本环境同源的参照 venv 里也有一份），`env/env.sh` 会在最后 source 它。它导出三个变量：
`RLE_ENV` / `RLE_TOOL_TYPE` / `RLE_API_KEY`。

**注意它会盖掉 `run_eval.sh` 的内置默认。**`run_eval.sh:121` 写的是
`${RLE_API_KEY:?set RLE_API_KEY}`（必填、无默认值）、tool_type 隐含 `swe-pdc`
（`kill_rle_sessions.py` 的 docstring 也还写着 swe-pdc），它们都是 `${VAR:-default}`
形式，所以 `.key.sh` 里的值优先。3c 实际生效的是
`RLE_TOOL_TYPE=<sandbox-tool-type>` + `RLE_API_KEY=<key-prefix>:...`（key 前缀已脱敏），
会话销毁走 RLE 平台的 session-destroy endpoint（`rle.internal.example.com`，API 路径已略）。
换 sandbox 池要改 `.key.sh`，改 `run_eval.sh` 是没用的。
