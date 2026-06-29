set -x
ENGINE=${1:-vllm}
# export VLLM_ATTENTION_BACKEND=XFORMERS

num_cpus_per_env_worker=0.1 # The CPU resource allocated for each environment worker. If you want to use less CPU resources, you can decrease this value.

train_data_size=16
val_data_size=32
group_size=8
mode="mean_std_norm" # "mean_norm" or "mean_std_norm"

project_name='verl_agent_alfworld'
experiment_name='gigpo_qwen2.5_32b'
mkdir -p ./logs/${project_name}/${experiment_name}
mkdir -p ./modelsave/${project_name}/${experiment_name}

model_path=/mnt/dolphinfs/ssd_pool/docker/user/hadoop-xt-ai-search/ai-search/dongchengqi/huggingface.co/Qwen/Qwen2.5-32B-Instruct
export ALFWORLD_DATA="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-xt-ai-search/ai-search/dongchengqi/huggingface.co/alfworld"
export TENSORBOARD_DIR="./logs/${project_name}/${experiment_name}"


# Set Ray num_cpus explicitly to avoid hanging in resource-limited environments (e.g., SLURM/cgroup).
# Ray defaults to null (use all CPUs), which can cause workers to hang if the detected CPU count
# exceeds the actual allocation. Here we read the real available CPUs at runtime.
echo "[DEBUG] Available CPUs (nproc): $(nproc)"
# RAY_NUM_CPUS=${RAY_NUM_CPUS:-$(nproc)}
# echo "[DEBUG] RAY_NUM_CPUS set to: $RAY_NUM_CPUS"

# We only use data preparation to indicate the modality and the data size.
python3 -m examples.data_preprocess.prepare \
    --mode 'text' \
    --train_data_size $train_data_size \
    --val_data_size $val_data_size

python3 -X faulthandler -m verl.trainer.main_ppo \
    algorithm.adv_estimator=gigpo \
    data.train_files=/mnt/dolphinfs/ssd_pool/docker/user/hadoop-xt-ai-search/ai-search/dongchengqi/verl-agent/data/verl-agent/text/train.parquet \
    data.val_files=/mnt/dolphinfs/ssd_pool/docker/user/hadoop-xt-ai-search/ai-search/dongchengqi/verl-agent/data/verl-agent/text/test.parquet \
    data.train_batch_size=$train_data_size \
    data.val_batch_size=$val_data_size \
    data.max_prompt_length=2048 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=$model_path \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
    algorithm.use_kl_in_reward=False \
    algorithm.gamma=0.95 \
    algorithm.gigpo.step_advantage_w=1.0 \
    algorithm.gigpo.mode=$mode \
    env.env_name=alfworld/AlfredTWEnv \
    env.seed=0 \
    env.max_steps=50 \
    env.rollout.n=$group_size \
    env.resources_per_worker.num_cpus=$num_cpus_per_env_worker \
    trainer.critic_warmup=0 \
    trainer.logger=['console','tensorboard'] \
    trainer.project_name=$project_name \
    trainer.experiment_name=$experiment_name \
    trainer.default_local_dir=./modelsave/${project_name}/${experiment_name} \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=2 \
    trainer.save_freq=75 \
    trainer.test_freq=5 \
    trainer.total_epochs=200 \
    trainer.val_before_train=True $@    # ray_init.num_cpus=$RAY_NUM_CPUS $@
