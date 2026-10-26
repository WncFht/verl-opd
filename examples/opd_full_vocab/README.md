# OPD shared-sync 与 Full-vocab KL MVP

基于官方 verl `263653e749306287a5032140cce3d7833c1959d0`，新增同池 FSDP2 教师评分、按微批传输有效 hidden，以及完整词表 reverse KL。默认 standalone 教师路径保持原入口；`distillation.execution_mode=shared_sync` 使用 rollout → teacher → actor 的阶段顺序。

Full-KL 在固定的学生生成前缀上优化：

$$
\frac1N\sum_{i=1}^{N}\mathrm{KL}(p_\theta(\cdot\mid s_i)\Vert T(\cdot\mid s_i)).
$$

Teacher hidden/head 无梯度；当前 student backbone/head 均更新。数值层用 token 分块在 forward 预计算一阶 dH/dW，backward 施加调用级全局归一化。支持 dense Qwen2/Qwen3、tied/untied bias-free head、同 token-ID 词表及额外 teacher 输出行；当前要求单节点 shared-sync、FSDP2、SP=1、remove-padding、无 LoRA 和任务奖励混合。

`run_e20.sh` 是唯一新增启动脚本：

| 项目 | E20 默认设置 |
| --- | --- |
| Student / teacher | R1-Distill-Qwen-1.5B / JustRL-DeepSeek-1.5B |
| 数据 / 训练长度 | DAPO；prompt=4096、response=7168 |
| Global batch / mini-batch / rollout n | 64 / 64 / 1 |
| Outer / reuse / 更新预算 | 256 / 16 / 4096 次更新尝试 |
| LR / seed | 1e-6、constant、无 warmup / 42 |
| 验证 / 保存 | 每 32 / 64 outer；不做初始验证 |
| 验证预算 | MATH-500、AMC23、AIME25，每题 16 条，response cap=31744 |
| Head chunk / hidden IO | 512 / async write、prefetch 开启 |

环境覆盖用于资源与短测，末尾 Hydra overrides 优先。数学打分复用上游 `math_dapo` verifier，评测分数不直接并入使用其他 verifier 的历史结果。完整启动步骤见下文。

## 交接与使用

先 clone 此仓库，再在已配置好 FSDP2 + vLLM 的原生 Python 环境中安装本仓源码。需要 PyTorch、Transformers、Ray、TensorDict、TransferQueue 和 FlashAttention 等上游依赖；模型和数据需另行提供。已有训练环境可用 `--no-deps` 安装，避免改动已匹配的 CUDA 依赖组合：

```bash
git clone https://github.com/WncFht/verl-opd.git
cd verl-opd
export PYTHON_BIN=/absolute/path/to/training-env/bin/python
"$PYTHON_BIN" -m pip install --no-deps -e .

export MODEL_ROOT=/absolute/path/to/models
export DATA_ROOT=/absolute/path/to/data
export CUDA_VISIBLE_DEVICES=0,1,2,3
export GPUS=4
```

PYTHON_BIN 应指向完整的训练环境；driver 和 Ray worker 使用同一个解释器，脚本会传递本仓库的绝对 PYTHONPATH。上述 pip 命令只安装源码，不负责补齐缺失的运行依赖。依赖准备沿用本上游 commit 的安装方式，相关配置见根目录 pyproject.toml。

目录需包含：

```text
MODEL_ROOT/DeepSeek-R1-Distill-Qwen-1.5B/    # HF config、tokenizer、safetensors
MODEL_ROOT/JustRL-DeepSeek-1.5B/
DATA_ROOT/math/DAPO/train.parquet
DATA_ROOT/math/MATH-500/test.parquet
DATA_ROOT/math/AMC23/test.parquet
DATA_ROOT/math/AIME25/test.parquet
```

Parquet 使用 verl 的 prompt、data_source、reward_model.ground_truth 等字段协议；这里不下载模型，也不转换原始数据。可以直接使用已有转换后的四份数据。

先检查配置，随后按需要执行两轮短测或正式 E20：

```bash
# 只解析配置，不启动 Ray 或训练。
CONFIG_ONLY=1 bash examples/opd_full_vocab/run_e20.sh

# 两轮 Full-KL 短测，保留默认 batch、reuse 和训练长度，关闭验证与保存。
TOTAL_TRAINING_STEPS=2 TEST_FREQ=-1 SAVE_FREQ=-1 \
  bash examples/opd_full_vocab/run_e20.sh

# 正式 E20：256 outer × 16 reuse；每 32 outer 验证、64 outer 保存。
bash examples/opd_full_vocab/run_e20.sh

# 共享教师与相同预算下的上游 sampled k1 OPD。
LOSS_MODE=k1 bash examples/opd_full_vocab/run_e20.sh
```

默认 GPUS=8；使用四卡时按上例显式设为 4。Global batch 仍为 64，不随卡数缩放。每次运行自动生成独立输出目录；若显式指定 RUN_DIR，它必须是尚不存在的新目录。

核对输出目录的 config.yaml、train.log 和 metrics.jsonl，检查配置、实际 outer 进度、actor loss 与梯度范数。Full-KL 通过 actor/distillation/full_kl 记录，沿用上游按 actor 调用汇总的日志方式；此分支没有引入另一个项目的逐更新指标系统。短测应完成全部计划 outer；只完成配置解析不代表真实模型训练已经通过。

## 当前验证状态

已完成 42 项 CPU 测试，涵盖独立 Full-KL 值/VJP、真实 CPU Qwen head 回调、全局分母、sampled OPD 回归、TQ 身份/异步 IO、教师清理与 E20 配置；另有 1 项 CUDA 测试跳过。生成配置一致性、ruff 和 shell 语法检查通过。GPU 验收按交接要求暂停，新分支尚未完成 CUDA/FSDP2 或真实 GPU 训练验证，不据此承诺正式 E20 已跑通。

CPU 检查命令如下；当前 GPU 队列已停止。两卡 GPU 验收入口保留在 tests/special_distributed/test_opd_full_vocab_fsdp2.py，供后续验收使用。

```bash
CUDA_VISIBLE_DEVICES='' "$PYTHON_BIN" -m pytest -q \
  tests/trainer/distillation/test_opd_*.py \
  tests/workers/test_distillation_topk_symmetry_on_cpu.py
```

若需要将功能应用到另一个 verl checkout，可在对应上游基点导入功能提交：

```bash
git fetch https://github.com/WncFht/verl-opd.git main
git cherry-pick f1c6036404025f4085a907bf0748499ad0f74ca3
```

该提交的直接父提交是 `263653e749306287a5032140cce3d7833c1959d0`。移植到其他上游版本时，需要审阅冲突并重新验证接口与运行行为。
