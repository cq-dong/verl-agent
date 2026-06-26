set -x

ENGINE=${1:-vllm}

train_data_size=256
val_data_size=512
group_size=5

# GiGPO config
mode="mean_std_norm" # "mean_norm" or "mean_std_norm"
enable_similarity=True # enable similarity-based GiGPO
similarity_thresh=0.9 # similarity threshold for GiGPO
# GiGPO group-structure stats (independent of TGSA): record the per-group panel
# (group counts, size distribution, degeneration rate, all-success/all-fail
# split) so plain GiGPO can diagnose anchor-overlap sparsity without a teacher.
gigpo_log_group_stats=True
gigpo_eps_deg=0.01          # degeneration threshold on sigma^R_group (idea ~0.01)
gigpo_success_thresh=0.0    # return threshold for the all-success/all-fail split (0.0 for 0/1 rewards)

# TGSA-GRPO config (idea.md: teacher-guided step-level advantage on top of GiGPO).
# Set tgsa_enabled=True to activate. Requires a teacher sglang HTTP server at
# $teacher_base_url (launched separately; e.g. `python -m sglang.launch_server
# --model-path <teacher> --port 30000`). Teacher must share the student tokenizer.
tgsa_enabled=False
tgsa_lambda=0.3            # normal-group teacher modulation strength (idea 0.2-0.5)
tgsa_mu=0.1                # degenerate-group fallback strength (idea ~0.1)
tgsa_gamma=1.0             # tanh scale for the singleton teacher-student signal
tgsa_eps_deg=0.01          # degeneration threshold on sigma^R_group
tgsa_success_thresh=0.0    # return threshold for the all-success/all-fail degenerate-group split (idea mu+/mu-); 0.0 for 0/1 rewards
tgsa_norm_mode="minmax"    # "minmax" | "zscore" for the Case-1 group ranking
tgsa_replace_step_adv=True # True: drop GiGPO A^S; False: ADD step_advantage_w*A^S
tgsa_bounded_scaling="none" # "none" | "tanh" | "clip" on |A^E|
tgsa_delta_norm_clip=0.0     # Case-2 stability guard: clamp singleton batch-zscore to [-c,c] before tanh (0=off, ~3.0 rec.)
teacher_base_url="http://localhost:30000"
teacher_max_concurrency=8
teacher_timeout=60.0
# env-gated reverse-KL distillation regularizer (idea L245-280); 0.0 = off
tgsa_kl_coef=0.0
tgsa_kl_penalty="k3"       # "kl" (k1) | "k3" | "mse" | "abs" (single-token MC, no top-k)
tgsa_kl_gate_eta=1.0
tgsa_kl_gate_mode="hard"   # "hard" = 1[A^E>0] (sign-exact) | "soft" = sigmoid(eta*A^E)
# optional margin variant (needs teacher top-k); False = off
tgsa_margin_enabled=False
tgsa_margin_topk=2

TGSA_ARGS=""
if [ "$tgsa_enabled" = "True" ]; then
    TGSA_ARGS="algorithm.tgsa.enabled=True \
        algorithm.tgsa.lambda=$tgsa_lambda \
        algorithm.tgsa.mu=$tgsa_mu \
        algorithm.tgsa.gamma=$tgsa_gamma \
        algorithm.tgsa.eps_deg=$tgsa_eps_deg \
        algorithm.tgsa.success_thresh=$tgsa_success_thresh \
        algorithm.tgsa.normalization_mode=$tgsa_norm_mode \
        algorithm.tgsa.replace_step_advantage=$tgsa_replace_step_adv \
        algorithm.tgsa.bounded_env_scaling=$tgsa_bounded_scaling \
        algorithm.tgsa.delta_norm_clip=$tgsa_delta_norm_clip \
        algorithm.tgsa.teacher.base_url=$teacher_base_url \
        algorithm.tgsa.teacher.max_concurrency=$teacher_max_concurrency \
        algorithm.tgsa.teacher.timeout=$teacher_timeout \
        algorithm.tgsa.kl.kl_teacher_coef=$tgsa_kl_coef \
        algorithm.tgsa.kl.kl_penalty=$tgsa_kl_penalty \
        algorithm.tgsa.kl.kl_gate_eta=$tgsa_kl_gate_eta \
        algorithm.tgsa.kl.kl_gate_mode=$tgsa_kl_gate_mode \
        algorithm.tgsa.margin.enabled=$tgsa_margin_enabled \
        algorithm.tgsa.margin.topk=$tgsa_margin_topk"
fi

TRAIN_DATA="$HOME/data/searchR1_processed_direct/train.parquet"
VAL_DATA="$HOME/data/searchR1_processed_direct/test.parquet"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=gigpo \
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
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
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
    algorithm.gamma=0.95 \
    algorithm.gigpo.step_advantage_w=1.0 \
    algorithm.gigpo.mode=$mode \
    algorithm.gigpo.enable_similarity=$enable_similarity \
    algorithm.gigpo.similarity_thresh=$similarity_thresh \
    algorithm.gigpo.log_group_stats=$gigpo_log_group_stats \
    algorithm.gigpo.eps_deg=$gigpo_eps_deg \
    algorithm.gigpo.success_thresh=$gigpo_success_thresh \
    env.env_name=search \
    env.seed=0 \
    env.max_steps=4 \
    env.rollout.n=$group_size \
    env.history_length=4 \
    env.search.search_url='http://127.0.0.1:8000/retrieve' \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='verl_agent_search' \
    trainer.experiment_name='gigpo_sim0.9_qwen2.5_7b_instruct' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=50 \
    trainer.total_epochs=1 \
    trainer.val_before_train=False $TGSA_ARGS $@