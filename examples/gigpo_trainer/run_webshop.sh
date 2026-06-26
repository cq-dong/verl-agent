set -x
ENGINE=${1:-vllm}
ulimit -u 65536
# export VLLM_ATTENTION_BACKEND=XFORMERS  # disabled: xformers not built with CUDA support
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
# ---- pick a Java >= 8 (pyserini 0.17 bundles Lucene 8 = Java 8 bytecode) ----
# On this box `/usr/local/java` symlinks to jdk1.7.0_76 (Java 7) and breaks
# pyserini with `UnsupportedClassVersionError: ... version 52.0`. Scan known JDK
# dirs and pick the first whose `java -version` is >= 8. Order: inherited
# JAVA_HOME/JDK_HOME first, then system openjdk 11, /workdir JDK 11, JDK 8.
JAVA_HOME_FOUND=""
for d in "${JAVA_HOME:-}" "${JDK_HOME:-}" \
         /usr/lib/jvm/java-11-openjdk-11.0.22.0.7-1.el7_9.x86_64 \
         /workdir/mjdk-11.0.16 \
         /usr/local/jdk1.8.0_45; do
  [ -n "$d" ] && [ -x "$d/bin/java" ] || continue
  v=$("$d/bin/java" -version 2>&1 | awk -F\" '/version "/ {print $2; exit}')
  major=$(printf '%s' "$v" | awk -F. '{ if ($1==1) print $2; else print $1 }')
  if [ -n "$major" ] && [ "$major" -ge 8 ] 2>/dev/null; then
    JAVA_HOME_FOUND="$d"; break
  fi
done
export JAVA_HOME="${JAVA_HOME_FOUND:-/usr/local/java}"
export JDK_HOME="$JAVA_HOME"          # pyjnius checks JDK_HOME before JAVA_HOME
export PATH="$JAVA_HOME/bin:$PATH"
# pyjnius dlopens libjvm.so to start the JVM. Prepend the chosen JDK's libjvm
# dir (JDK11+: lib/server; JDK8: jre/lib/*/server) so dlopen doesn't fall back
# to the box's Java 7 libjvm. Propagated to workers via runtime_env.env_vars.
JVM_SERVER_DIR=$(ls -d "$JAVA_HOME"/lib/server "$JAVA_HOME"/jre/lib/*/server 2>/dev/null | head -1)
[ -n "$JVM_SERVER_DIR" ] && export LD_LIBRARY_PATH="$JVM_SERVER_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

echo "[INFO] JAVA_HOME=$JAVA_HOME, java: $($JAVA_HOME/bin/java -version 2>&1 | head -1)"

# Limit per-worker JVM heap/threads so many concurrent WebshopWorker JVMs don't
# exhaust the system thread limit (EAGAIN). pyjnius starts the JVM via
# JNI_CreateJavaVM, which ignores `_JAVA_OPTIONS` (launcher-only) but DOES read
# `JAVA_TOOL_OPTIONS` (honoured by the embedded JVM). Propagated to workers via
# runtime_env.env_vars below.
export JAVA_TOOL_OPTIONS="-Xss256k -XX:+UseSerialGC -Xmx256m -XX:-UsePerfData"

num_cpus_per_env_worker=0.1 # The CPU resource allocated for each environment worker. If you want to use less CPU resources, you can decrease this value.

train_data_size=16
val_data_size=16   # reduced from 128: 128×8=1024 workers exhausts system thread limit
group_size=8
mode="mean_norm" # "mean_norm" or "mean_std_norm"

# We only use data preparation to indicate the modality and the data size.
python3 -m examples.data_preprocess.prepare \
    --mode 'text' \
    --train_data_size $train_data_size \
    --val_data_size $((val_data_size * 2)) # evaluate 2 × val_data_size tasks during each iteration

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=gigpo \
    +ray_init.runtime_env.env_vars.JAVA_HOME=$JAVA_HOME \
    +ray_init.runtime_env.env_vars.JDK_HOME=$JDK_HOME \
    +ray_init.runtime_env.env_vars.PATH=$PATH \
    +ray_init.runtime_env.env_vars.LD_LIBRARY_PATH=$LD_LIBRARY_PATH \
    +ray_init.runtime_env.env_vars.JAVA_TOOL_OPTIONS="$JAVA_TOOL_OPTIONS" \
    data.train_files=$HOME/data/verl-agent/text/train.parquet \
    data.val_files=$HOME/data/verl-agent/text/test.parquet \
    data.train_batch_size=$train_data_size \
    data.val_batch_size=$val_data_size \
    data.max_prompt_length=4096 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-1.5B-Instruct \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
    algorithm.use_kl_in_reward=False \
    algorithm.gamma=0.95 \
    algorithm.gigpo.step_advantage_w=1.0 \
    algorithm.gigpo.mode=$mode \
    env.env_name=Webshop \
    env.seed=0 \
    env.max_steps=15 \
    env.rollout.n=$group_size \
    env.resources_per_worker.num_cpus=$num_cpus_per_env_worker \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='verl_agent_webshop' \
    trainer.experiment_name='gigpo_qwen2.5_1.5b' \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=5 \
    trainer.total_epochs=150 \
    trainer.val_before_train=True $@
