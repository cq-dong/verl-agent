set -x
ENGINE=${1:-vllm}
# export VLLM_ATTENTION_BACKEND=XFORMERS

num_cpus_per_env_worker=0.1 # The CPU resource allocated for each environment worker. If you want to use less CPU resources, you can decrease this value.

train_data_size=16
val_data_size=16
group_size=8


project_name='verl_agent_alfworld'
experiment_name='grpo_qwen2.5_1.5bdebug'
mkdir -p ./logs/${project_name}/${experiment_name}
mkdir -p ./modelsave/${project_name}/${experiment_name}

model_path=/mnt/dolphinfs/ssd_pool/docker/user/hadoop-xt-ai-search/ai-search/dongchengqi/huggingface.co/Qwen/Qwen2.5-1.5B-Instruct
export ALFWORLD_DATA="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-xt-ai-search/ai-search/dongchengqi/huggingface.co/alfworld"
export TENSORBOARD_DIR="./logs/${project_name}/${experiment_name}"

# Raise the per-user thread/process limit (RLIMIT_NPROC). With 8 GPUs verl
# spawns 8 colocated GPU worker processes (1 per GPU, each holding
# actor+rollout+ref+critic), and vLLM rollout processes are thread-heavy
# (hundreds of threads each). Combined with ~160 AlfWorld env worker processes,
# the per-user thread budget gets exhausted and you get:
#   "pthread_create failed: Resource temporarily unavailable"
#   "thread: Resource temporarily unavailable [system:11]"  -> SIGABRT
#   "<jemalloc>: arena 0 background thread creation failed (11)"
# This is EAGAIN on thread creation, NOT a GPU-memory OOM. It is why 4 GPUs run
# fine but 8 GPUs crash during init_workers()/ref policy init.
# `ulimit -u unlimited` only works up to the HARD limit; on this host the hard
# limit is 102400 and non-root can't raise it, so ulimit alone is insufficient —
# see the OMP/MKL/TORCH thread caps below, which is the real fix without root.
ulimit -u unlimited 2>/dev/null || true
echo "[DEBUG] ulimit -u (max user processes/threads): $(ulimit -u)"
echo "[DEBUG] ulimit -n (open files): $(ulimit -n)"

# Cap per-process CPU thread pools — THE key fix for the thread explosion.
# PyTorch/OpenMP default to one intra-op thread per CPU (here nproc=160), so
# EVERY Ray worker process (160+ AlfworldWorker + 8 GPU WorkerDict) that imports
# torch spawns ~160 threads -> 160 procs * 160 = ~25600 threads from env workers
# alone, plus vLLM's hundreds-per-process, which blows past the 102400 per-user
# NPROC limit on 8 GPUs. GPU training is GPU-bound, so large CPU thread pools
# are pure waste here. Capping to 1 cuts per-process threads ~160x.
# Override by exporting these before running (e.g. OMP_NUM_THREADS=4).
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-1}
export TORCH_NUM_THREADS=${TORCH_NUM_THREADS:-1}
# jemalloc (used by torch on this box) spawns background threads per arena;
# disable them to avoid extra pthread_create pressure at the NPROC limit.
export MALLOC_CONF=${MALLOC_CONF:-"background_thread:false,metadata_thp:auto"}
echo "[DEBUG] OMP_NUM_THREADS=$OMP_NUM_THREADS MKL_NUM_THREADS=$MKL_NUM_THREADS TORCH_NUM_THREADS=$TORCH_NUM_THREADS"

# Extend Ray worker registration timeout to handle slow AlfWorld env startup.
# Each AlfworldWorker process imports torch/torchvision/alfworld and loads
# TextWorld game files; on 8 GPUs the concurrent heavy-import process count
# doubles vs 4 GPUs (actor/rollout/ref/critic are colocated, 1 process per GPU
# per main_ppo.py resource_pool_spec), which can exceed Ray's default 60s
# registration timeout. The raylet then logs:
#   "Some workers of the worker process have not registered within the timeout.
#    The process is still alive, probably it's hanging during start."
# NOTE: the env var is in SECONDS (not milliseconds — the _milliseconds variant
# is a no-op). This is why 4 GPUs run fine but 8 GPUs hang at startup.
export RAY_worker_register_timeout_seconds=600

# Set Ray num_cpus explicitly. The default (`null` = use all CPUs) can cause
# workers to hang in resource-limited/cgrouped environments (verl's own
# ppo_trainer.yaml warns about this). Env workers demand
# (train_batch_size*group_n + val_batch_size) * 0.1 CPU-units; an explicit
# num_cpus keeps Ray's scheduling honest. Override by exporting RAY_NUM_CPUS.
if [ -z "${RAY_NUM_CPUS:-}" ]; then
    if command -v nproc >/dev/null 2>&1; then
        RAY_NUM_CPUS=$(nproc)
    else
        RAY_NUM_CPUS=$(sysctl -n hw.ncpu)   # macOS fallback
    fi
fi
echo "[DEBUG] RAY_NUM_CPUS set to: $RAY_NUM_CPUS"

# We only use data preparation to indicate the modality and the data size.
# python3 -m examples.data_preprocess.prepare \
#     --mode 'text' \
#     --train_data_size $train_data_size \
#     --val_data_size $val_data_size

python3  -X faulthandler -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
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
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
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
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=5 \
    trainer.total_epochs=150 \
    trainer.val_before_train=True \
    ray_init.num_cpus=$RAY_NUM_CPUS $@
