set -x

# =============================================================================
# 纯 OPD 消融基线运行脚本（忠实复刻 ROLL 标准 OPD）。
# 与 run_search.sh 的区别：
#   * algorithm.adv_estimator=opd（而非 gigpo）
#   * advantage = -coef·KL(pi_theta||pi_T)，无环境奖励/门控/critic
#   * actor_rollout_ref.actor.use_kl_loss=False（纯 OPD 的 KL 已折叠进 advantage，
#     不再用 ref KL loss，避免重复；对齐 ROLL use_kl_loss=False）
#   * env.rollout.n=1（纯 OPD advantage 逐轨迹独立，不依赖组内对比，n=1 合法；
#     n 只是采样数，实验时可自行改大做 batch 增广）
#   * 关闭 TGSA（tgsa_enabled=False）；教师服务仍用 algorithm.tgsa.teacher config
#     （与 TGSA 共用同一教师 sglang HTTP 服务，便于公平对比）
#
# 教师服务需单独启动，例如：
#   python -m sglang.launch_server --model-path <teacher> --port 30000
# 教师须与学生共享 tokenizer（直接发学生 input_ids）。
# =============================================================================

ENGINE=${1:-vllm}

train_data_size=256
val_data_size=512
rollout_n=1   # 纯 OPD：advantage 逐轨迹独立，n=1 合法；实验时可改大做 batch 增广

# 纯 OPD 配置（对应 algorithm.opd.*）
opd_kl_coef=1.0            # KL 蒸馏系数（ROLL opd_kl_coef，默认 1.0）
opd_kl_penalty="k3"        # KL 估计形式，与 ROLL 一致；可选 "kl"(k1)/"abs"/"mse"

# 教师服务配置（复用 algorithm.tgsa.teacher config，与 TGSA 共用同一教师）
teacher_base_url="http://localhost:30000"
teacher_max_concurrency=8
teacher_timeout=60.0

TRAIN_DATA="$HOME/data/searchR1_processed_direct/train.parquet"
VAL_DATA="$HOME/data/searchR1_processed_direct/test.parquet"

# 注意：adv_estimator=opd 触发纯 OPD 分支（ray_trainer.compute_advantage 的 OPD 分支），
# 不走 GiGPO 的 episode/step 组归一化。gigpo.* 配置在此模式下不被使用，
# 但教师 config 仍从 algorithm.tgsa.teacher 读取（OPD 与 TGSA 共用教师）。
# tgsa.enabled 保持 False（不开 TGSA 调制，只借教师前向）。
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=opd \
    data.train_files=$TRAIN_DATA \
    data.val_files=$VAL_DATA \
    data.train_batch_size=$train_data_size \
    data.val_batch_size=$val_data_size \
    data.max_prompt_length=4096 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation='left' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-7B-Instruct \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.1 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=512 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.01 \
    algorithm.use_kl_in_reward=False \
    algorithm.gamma=0.0 \
    algorithm.opd.kl_coef=$opd_kl_coef \
    algorithm.opd.kl_penalty=$opd_kl_penalty \
    algorithm.tgsa.enabled=False \
    algorithm.tgsa.teacher.base_url=$teacher_base_url \
    algorithm.tgsa.teacher.max_concurrency=$teacher_max_concurrency \
    algorithm.tgsa.teacher.timeout=$teacher_timeout \
    env.env_name=search \
    env.seed=0 \
    env.max_steps=4 \
    env.rollout.n=$rollout_n \
    env.history_length=4 \
    env.search.search_url='http://127.0.0.1:8000/retrieve' \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='verl_agent_search' \
    trainer.experiment_name='pure_opd_qwen2.5_7b_instruct' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=50 \
    trainer.total_epochs=1 \
    trainer.val_before_train=False $@
