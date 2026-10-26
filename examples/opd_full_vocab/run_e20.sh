#!/usr/bin/env bash
# Exact Full-KL on the E20 training budget; LOSS_MODE=k1 uses sampled OPD.
set -eo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)
cd "$REPO_ROOT"
PYTHON_BIN=$(command -v "${PYTHON_BIN:-python3}")
MODEL_ROOT=${MODEL_ROOT:-$(dirname -- "$REPO_ROOT")/model}
DATA_ROOT=${DATA_ROOT:-$(dirname -- "$REPO_ROOT")/data}
STUDENT_MODEL=${STUDENT_MODEL:-$MODEL_ROOT/DeepSeek-R1-Distill-Qwen-1.5B}
TEACHER_MODEL=${TEACHER_MODEL:-$MODEL_ROOT/JustRL-DeepSeek-1.5B}
GPUS=${GPUS:-8}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-64}
PPO_EPOCHS=${PPO_EPOCHS:-16}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-4096}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-7168}
MAX_VAL_RESPONSE_LENGTH=${MAX_VAL_RESPONSE_LENGTH:-31744}
ROLLOUT_RESPONSE_LIMIT=$MAX_VAL_RESPONSE_LENGTH
if (( MAX_RESPONSE_LENGTH > ROLLOUT_RESPONSE_LIMIT )); then ROLLOUT_RESPONSE_LIMIT=$MAX_RESPONSE_LENGTH; fi
LOSS_MODE=${LOSS_MODE:-full_reverse_kl}
case "$LOSS_MODE" in
    full_reverse_kl) PG=False; LOG_PROBS=False ;;
    k1) PG=True; LOG_PROBS=True ;;
    *) echo 'LOSS_MODE must be full_reverse_kl or k1' >&2; exit 2 ;;
esac
EXPERIMENT_NAME=${EXPERIMENT_NAME:-opd-e20-$LOSS_MODE-$(date +%Y%m%d-%H%M%S)-$$}
RUN_DIR=${RUN_DIR:-$REPO_ROOT/outputs/$EXPERIMENT_NAME}
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export TOKENIZERS_PARALLELISM=false
export RAY_ADDRESS=${RAY_ADDRESS:-local}
export WANDB_MODE=${WANDB_MODE:-offline}
export VERL_FILE_LOGGER_PATH="$RUN_DIR/metrics.jsonl"

ARGS=(
    "data.train_files=['$DATA_ROOT/math/DAPO/train.parquet']"
    "data.val_files=['$DATA_ROOT/math/MATH-500/test.parquet','$DATA_ROOT/math/AMC23/test.parquet','$DATA_ROOT/math/AIME25/test.parquet']"
    "data.train_batch_size=$TRAIN_BATCH_SIZE" data.train_max_samples=-1 data.val_batch_size=null
    "data.max_prompt_length=$MAX_PROMPT_LENGTH" "data.max_response_length=$MAX_RESPONSE_LENGTH"
    data.filter_overlong_prompts=True data.truncation=error data.seed=42 data.shuffle=True
    data.validation_shuffle=False data.dataloader_num_workers=0 data.continuous_token.model_family=deepseek
    "actor_rollout_ref.model.path='$STUDENT_MODEL'" actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.use_fused_kernels=False actor_rollout_ref.model.enable_gradient_checkpointing=True
    actor_rollout_ref.actor.strategy=fsdp2 actor_rollout_ref.actor.use_torch_compile=False
    actor_rollout_ref.actor.use_dynamic_bsz=True actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
    "actor_rollout_ref.actor.ppo_mini_batch_size=$TRAIN_BATCH_SIZE" "actor_rollout_ref.actor.ppo_epochs=$PPO_EPOCHS"
    "actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${ACTOR_TOKEN_BUDGET:-32768}"
    actor_rollout_ref.actor.loss_agg_mode=token-mean actor_rollout_ref.actor.optim.lr=1e-6
    'actor_rollout_ref.actor.optim.betas=[0.9,0.999]' actor_rollout_ref.actor.optim.weight_decay=0.01
    actor_rollout_ref.actor.optim.lr_scheduler_type=constant actor_rollout_ref.actor.optim.lr_warmup_steps=0
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0 actor_rollout_ref.actor.optim.clip_grad=1.0
    actor_rollout_ref.actor.entropy_coeff=0 actor_rollout_ref.actor.calculate_entropy=False
    actor_rollout_ref.actor.use_kl_loss=False actor_rollout_ref.actor.data_loader_seed=42
    actor_rollout_ref.actor.fsdp_config.seed=42 actor_rollout_ref.actor.fsdp_config.param_offload=True
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
    actor_rollout_ref.rollout.name=vllm actor_rollout_ref.rollout.tensor_model_parallel_size=1
    actor_rollout_ref.rollout.n=1 actor_rollout_ref.rollout.temperature=1.0
    actor_rollout_ref.rollout.top_p=1.0 actor_rollout_ref.rollout.top_k=-1 actor_rollout_ref.rollout.seed=42
    actor_rollout_ref.rollout.enforce_eager=False actor_rollout_ref.rollout.gpu_memory_utilization=0.65
    actor_rollout_ref.rollout.max_num_seqs=128 actor_rollout_ref.rollout.max_num_batched_tokens=8192
    "actor_rollout_ref.rollout.max_model_len=$((MAX_PROMPT_LENGTH + ROLLOUT_RESPONSE_LIMIT))"
    "actor_rollout_ref.rollout.calculate_log_probs=$LOG_PROBS"
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
    "actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${ACTOR_TOKEN_BUDGET:-32768}"
    actor_rollout_ref.rollout.val_kwargs.do_sample=True actor_rollout_ref.rollout.val_kwargs.n=16
    actor_rollout_ref.rollout.val_kwargs.temperature=0.7 actor_rollout_ref.rollout.val_kwargs.top_p=0.9
    "actor_rollout_ref.rollout.val_kwargs.max_tokens=$MAX_VAL_RESPONSE_LENGTH"
    "reward.custom_reward_function.path='$SCRIPT_DIR/reward.py'" reward.custom_reward_function.name=compute_score
    distillation.enabled=True distillation.execution_mode=shared_sync
    distillation.nnodes=0 distillation.n_gpus_per_node=0
    "distillation.teacher_models.teacher_model.model_path='$TEACHER_MODEL'"
    "distillation.shared_teacher.infer_max_token_len_per_gpu=${TEACHER_TOKEN_BUDGET:-32768}"
    "distillation.distillation_loss.loss_mode=$LOSS_MODE" "distillation.distillation_loss.use_policy_gradient=$PG"
    distillation.distillation_loss.policy_loss_mode=vanilla distillation.distillation_loss.use_task_rewards=False
    distillation.distillation_loss.loss_max_clamp=null distillation.distillation_loss.log_prob_min_clamp=null
    "distillation.distillation_loss.full_vocab.token_chunk_size=${FULL_KL_TOKEN_CHUNK:-512}"
    distillation.distillation_loss.full_vocab.async_write=True distillation.distillation_loss.full_vocab.prefetch=True
    algorithm.adv_estimator=grpo algorithm.use_kl_in_reward=False algorithm.filter_groups.enable=False
    algorithm.rollout_correction.bypass_mode=False algorithm.rollout_correction.rollout_is=null
    algorithm.rollout_correction.rollout_rs=null trainer.v1.sampler.sync_refill_failed_groups=False
    trainer.use_v1=True trainer.v1.trainer_mode=sync trainer.nnodes=1 "trainer.n_gpus_per_node=$GPUS"
    trainer.total_epochs=1 "trainer.total_training_steps=${TOTAL_TRAINING_STEPS:-256}"
    "trainer.test_freq=${TEST_FREQ:-32}" "trainer.save_freq=${SAVE_FREQ:-64}"
    trainer.val_before_train=False trainer.resume_mode=disable trainer.max_actor_ckpt_to_keep=4
    trainer.project_name=verl-opd-full-vocab "trainer.experiment_name='$EXPERIMENT_NAME'"
    'trainer.logger=[console,file]' "trainer.default_local_dir='$RUN_DIR/checkpoints'"
    "trainer.validation_data_dir='$RUN_DIR/validation'" "hydra.run.dir='$RUN_DIR/hydra'"
    "ray_kwargs.ray_init.runtime_env.py_executable='$PYTHON_BIN'"
    "++ray_kwargs.ray_init.runtime_env.env_vars.PYTHONPATH='$REPO_ROOT'"
    "++ray_kwargs.ray_init.num_gpus=$GPUS" ++ray_kwargs.ray_init.num_cpus=64
    ++ray_kwargs.ray_init.include_dashboard=False ++ray_kwargs.ray_init.object_store_memory=8589934592
    "++ray_kwargs.ray_init._temp_dir=/tmp/opd-mvp-$$"
    "++ray_kwargs.ray_init.runtime_env.env_vars.VERL_FILE_LOGGER_PATH='$VERL_FILE_LOGGER_PATH'"
)
CMD=("$PYTHON_BIN" -m verl.trainer.main_ppo)
if [[ ${DRY_RUN:-0} == 1 ]]; then
    printf '%q ' "${CMD[@]}" "${ARGS[@]}" "$@"
    printf '\n'
    exit 0
fi
if [[ ${CONFIG_ONLY:-0} == 1 ]]; then
    exec "${CMD[@]}" --cfg job --resolve "${ARGS[@]}" "$@"
fi
[[ -n ${CUDA_VISIBLE_DEVICES:-} ]] || { echo 'Set CUDA_VISIBLE_DEVICES to the assigned GPUs' >&2; exit 2; }
[[ ! -e $RUN_DIR ]] || { echo "Use a fresh RUN_DIR: $RUN_DIR" >&2; exit 2; }
mkdir -p "$RUN_DIR"
"${CMD[@]}" --cfg job --resolve "${ARGS[@]}" "$@" > "$RUN_DIR/config.yaml"
{
    "${CMD[@]}" "${ARGS[@]}" "$@" 2>&1 | tee "$RUN_DIR/train.log"
    exit
}
