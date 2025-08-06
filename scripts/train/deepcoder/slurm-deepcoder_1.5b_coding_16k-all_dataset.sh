#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=180
#SBATCH --mem=512GB
#SBATCH --gpus=8
#SBATCH --time=24:00:00
#SBATCH --partition=schmidt_sciences
#SBATCH --account=dawn_song
#SBATCH --mail-type=all
#SBATCH --mail-user=jinxiaolong1129@gmail.com
#SBATCH --output=slurm_deepcoder_1.5b_16k-all_dataset_out_%j.txt
#SBATCH --error=slurm_deepcoder_1.5b_16k-all_dataset_error_%j.txt
#SBATCH --job-name=rllm-deepcoder-1.5b-16k

echo "Starting RLLM DeepCoder 1.5B 16K training job"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"
echo "SLURM_GPUS: $SLURM_GPUS"
echo "Date: $(date)"

# Enable debug mode
set -x

# Set ulimits and environment variables
ulimit -n 1048576 2>/dev/null || echo "Warning: Could not set ulimit for open files (insufficient permissions)"
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False"
export VLLM_USE_V1=1
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_ENGINE_ITERATION_TIMEOUT_S=1000000000

# Load environment variables from .env file
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
    echo "Loaded environment variables from .env file"
else
    echo "Warning: .env file not found. Please create one with your API keys."
    exit 1
fi

# Find the directory where rllm package is located
RLLM_DIR=$(python3 -c "import rllm; import os; print(os.path.dirname(os.path.dirname(rllm.__file__)))")
echo "RLLM directory: $RLLM_DIR"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL_PATH="$2"
            shift 2
            ;;
        *)
            break
            ;;
    esac
done

# # Set default model path if not provided
# if [ -z "$MODEL_PATH" ]; then
#     MODEL_PATH="Qwen/Qwen2.5-3B"
# fi

MODEL_PATH=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B

echo "Using model: $MODEL_PATH"

# Create log directory
mkdir -p scripts/train/deepcoder/logs

echo "Starting training..."
echo "SLURM output will be written to: slurm_deepcoder_1.5b_16k_all_dataset_out_${SLURM_JOB_ID}.txt"
echo "SLURM errors will be written to: slurm_deepcoder_1.5b_16k_all_dataset_error_${SLURM_JOB_ID}.txt"

RLLM_HOME=/data/xuandong_zhao/mnt/xiaolong/MATS-rllm/
PYTHON_PATH="/data/xuandong_zhao/anaconda3/envs/rllm/bin/python"

# Run the training with error handling
$PYTHON_PATH -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$RLLM_HOME/rllm/data/deepscaler_code.parquet \
    data.val_files=$RLLM_HOME/rllm/data/test_livecodebench.parquet \
    data.train_batch_size=128 \
    data.val_batch_size=512 \
    data.max_prompt_length=2048 \
    data.max_response_length=16384 \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size=16 \
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=20000 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=2 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=0.6 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.val_kwargs.n=2 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    algorithm.mask_truncated_samples=True \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='deepscaler' \
    trainer.experiment_name="deepseek-1.5b-16k-grpo-code-all_dataset-job${SLURM_JOB_ID}" \
    trainer.val_before_train=True \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=20 \
    trainer.max_actor_ckpt_to_keep=1 \
    trainer.max_critic_ckpt_to_keep=1 \
    trainer.default_hdfs_dir=null \
    trainer.default_local_dir="checkpoints/deepscaler/deepseek-1.5b-16k-grpo-code-all_dataset-job${SLURM_JOB_ID}" \
    trainer.total_epochs=100 "${@:1}"

# Capture the exit status
EXIT_STATUS=$?

echo "Training completed with exit status: $EXIT_STATUS"
echo "End time: $(date)"

if [ $EXIT_STATUS -eq 0 ]; then
    echo "Training completed successfully!"
else
    echo "Training failed with exit status: $EXIT_STATUS"
    exit $EXIT_STATUS
fi