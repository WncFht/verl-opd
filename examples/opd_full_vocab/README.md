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

在原生训练 Python 环境中，从仓库根目录运行；模型与转换后的 parquet 数据由外部提供。默认资产根目录是相邻的 `model/`、`data/`，可用环境变量指定已有位置。

```bash
PYTHON_BIN=/path/to/native/python CONFIG_ONLY=1 bash examples/opd_full_vocab/run_e20.sh
CUDA_VISIBLE_DEVICES=0,1,2,3 GPUS=4 PYTHON_BIN=/path/to/native/python \
  MODEL_ROOT=/path/to/models DATA_ROOT=/path/to/data bash examples/opd_full_vocab/run_e20.sh
# 同一个预算与共享教师流程，切换为上游 k1 sampled OPD：
LOSS_MODE=k1 CUDA_VISIBLE_DEVICES=0,1,2,3 GPUS=4 PYTHON_BIN=/path/to/native/python \
  bash examples/opd_full_vocab/run_e20.sh
```

环境覆盖用于资源与短测；末尾 Hydra overrides 优先。模型参数、采样温度和评测预算以输出目录中的 `config.yaml` 为准。短测可设 `TOTAL_TRAINING_STEPS=2 TEST_FREQ=-1 SAVE_FREQ=-1`；正式默认仍为上表。日志为 `train.log` 和 `metrics.jsonl`。数学打分复用上游 `math_dapo` verifier，因此此 MVP 的评测分数不直接并入原项目使用其他 verifier 的历史结果。

CPU 验证：`CUDA_VISIBLE_DEVICES='' PYTHONPATH=. python -m pytest -q tests/trainer/distillation/test_opd_*.py`。CUDA BF16 数值测试位于同一测试文件，需在已分配的空闲 GPU 上单独运行。两卡生产 worker 验收使用 `torchrun --standalone --nproc-per-node=2 tests/special_distributed/test_opd_full_vocab_fsdp2.py --output outputs/fsdp2.json`；可加 `--tied` 验证 tied head。配置解析、数值单测、真实分布式更新和真实训练是不同验证层级；MVP 执行证据存放在本机忽略的 `outputs/`。
