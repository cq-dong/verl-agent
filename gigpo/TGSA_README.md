# TGSA-GRPO：面向 GRPO 的教师引导步级优势（verl-agent 实现）

基于 verl-agent 的 GiGPO 实现 `Paper/idea.md`（TGSA-GRPO）。
教师模型 π_T 是 GiGPO 环境奖励框架内的**步级信用分配调节器**——
**并非** OPD 式的 KL 蒸馏。环境始终拥有最终否决权（优势符号保持不变）。

$$A_{\text{total}}(a_t) = A^E(\tau) + \lambda \cdot \tilde{L}_T(a_t) \cdot |A^E(\tau)| + \mu \cdot \mathbb{1}_{\text{deg}} \cdot \tilde{L}_T(a_t)$$

## 想法 → 代码映射

| 想法要素 | 代码位置 |
|---|---|
| 长度归一化的教师对数概率 `log π_T(a_t\|s_t)` | `gigpo/tgsa.py::length_normalize_logprob` |
| 情形 1（组，\|G_t\|≥2）排序信号（minmax/zscore） | `gigpo/tgsa.py::compute_teacher_preference_signal` |
| 情形 2（单例，\|G_t\|=1）`tanh(γ·Δ)` | 同上函数 |
| 间隔变体（`m_t = lp - runner_up`） | 同上函数（`use_margin` + `teacher_top2_logprob`） |
| 基于 `\|A^E\|` 的 `A_total` 四象限 | `gigpo/tgsa.py::compute_tgsa_advantage`（其中 `a_e.abs()` 为关键承载项） |
| 退化指示 `1_deg = 1[σ^R_group ≤ ε]` | 同上（`σ^R_group` = episode 组回报标准差） |
| 反向 KL 单 token 蒙特卡洛估计（无 top-k） | `gigpo/tgsa.py::compute_reverse_kl_token` / `reverse_kl_scalar` |
| \|G_t\| 导出（内联，无签名变更） | `gigpo/core_gigpo.py::_step_group_size_per_row` |
| σ^R_group 导出（episode 组） | `gigpo/core_gigpo.py::_episode_group_return_std` |
| GiGPO 注入点 → TGSA | `gigpo/core_gigpo.py::compute_gigpo_outcome_advantage`（TGSA 分支，向后兼容） |
| 教师前向（sglang HTTP，外部服务） | `gigpo/teacher_client.py::SGLangTeacherClient` |
| 训练循环中的教师调用（与 ref 并发） | `verl/trainer/ppo/ray_trainer.py::RayPPOTrainer.fit` |
| multi_turn mask 统一（loss_mask） | 同上文件（对 `response_mask` 的 TGSA 覆盖） |
| 环境门控的教师 KL 正则项 `β_T·σ(η·A^E)·D_KL(π_θ‖π_T)` | `verl/workers/actor/dp_actor.py::update_policy`（`use_tgsa_kl` 分支） |
| 配置 | `verl/trainer/config/ppo_trainer.yaml`（`algorithm.tgsa`） |
| 运行示例 | `examples/gigpo_trainer/run_search.sh` |

## 关键设计决策（完整理由见项目记忆）

1. **教师 = 外部 sglang HTTP 服务**，而非 verl 的 Ray worker。32B+ 的
   教师模型运行在自有 GPU 上，verl 通过 HTTP 调用。无需 `Role.Teacher`、
   无需改动 resource-pool、无需改动 `init_workers`。
2. **教师共享学生 tokenizer。** 学生 `input_ids` 直接发送至 sglang
   `/generate`，开启 `return_logprob` + `logprob_start_len=0`；
   响应（动作）窗口为最后 `response_length` 个 logprob，与
   `input_token_ids` 一一对应。不同 tokenizer 下的重分词 + 对齐为未来工作。
3. **σ^R_group 在 EPISODE（prompt）组上取值，而非步级组。**
   μ·1_deg·L_T 项是一个*兜底*机制，必须在 λ·L_T·|A^E| 项失声
   （|A^E|≈0）时触发；A^E 在 episode 组内归一化，因此 1_deg 检测的是
   episode 组回报坍缩。（若用步级组标准差，则在 A^E 仍强烈时就会触发，
   与"环境哑火时临时接管"相矛盾。）
4. **单例（\|G_t\|=1）与退化（\|G_t\|≥2 ∧ σ^R_group≤ε）是两类不同情形。**
   单例使用情形 2（`tanh(γ·Δ)`）；1_deg 仅在多成员退化组上触发。单例
   不会被 OR 入 1_deg（其 std 哨兵值为 1.0）。
5. **反向 KL = 单 token 蒙特卡洛**，无 top-k、无词表求和（与 ROLL 的 RL
   约定及 idea L256/L278 一致）。`top_logprobs_num` 仅用于可选的 margin
   runner-up。
6. **KL 门控使用 `sign(advantages)`**（= `sign(A^E)`，在 TGSA 的符号
   保持下精确）作为硬门控（默认）。软门控 `σ(η·A_total)` 是
   `σ(η·A^E)` 的近似（符号一致；幅度不同）。
7. **全部为可选启用 / 默认关闭。** `algorithm.tgsa.enabled: False` 默认
   → 教师从不被调用 → 保持原始 GiGPO 行为。所有 4 处既有
   `compute_gigpo_outcome_advantage` 调用点均受保护（新增 kwargs 默认
   `None`）。

## 配置（`algorithm.tgsa`）

```yaml
tgsa:
  enabled: False              # 总开关
  lambda: 0.3                 # 正常组的教师调节（idea 0.2-0.5）
  mu: 0.1                     # 退化组兜底（idea ~0.1）
  gamma: 1.0                  # 单例信号的 tanh 尺度
  eps_deg: 0.01               # sigma^R_group 上的退化阈值
  normalization_mode: "minmax"   # "minmax" | "zscore"
  replace_step_advantage: True   # True：丢弃 GiGPO A^S；False：ADD w*A^S（消融）
  bounded_env_scaling: "none"    # "none" | "tanh" | "clip" 作用于 |A^E|
  teacher:
    base_url: "http://localhost:30000"
    max_concurrency: 8
    timeout: 60.0
    max_retries: 3
  kl:                         # 环境门控反向 KL 正则项（idea L245-280）；默认关闭
    kl_teacher_coef: 0.0      # 0.0 = 关闭
    kl_penalty: "k3"          # "kl"|"k3"|"mse"|"abs"（单 token MC）
    kl_gate_eta: 1.0
    kl_gate_mode: "hard"      # "hard" = 1[A^E>0]（符号精确）| "soft"
  margin:                     # 可选 margin 变体；默认关闭
    enabled: False
    topk: 2
```

## 运行

1. 启动教师 sglang 服务（独立 GPU，暴露端口）：
   ```bash
   python -m sglang.launch_server --model-path <teacher-32B> --port 30000 --tp <N>
   ```
2. 编辑 `examples/gigpo_trainer/run_search.sh`：将 `tgsa_enabled=True`，
   并将 `teacher_base_url` 指向教师服务。运行：
   ```bash
   bash examples/gigpo_trainer/run_search.sh vllm
   ```

## 测试（纯 CPU，无需 Ray/sglang/GPU）

```bash
python3 tests/trainer/ppo/test_tgsa.py                 # 16：四象限、退化、单例、KL
python3 tests/trainer/ppo/test_core_gigpo_tgsa.py      #  9：GiGPO 接线 + 向后兼容
python3 tests/trainer/ppo/test_teacher_client.py       #  9：sglang 解析/对齐/margin/重试（mock HTTP）
python3 tests/trainer/ppo/test_tgsa_kl_regularizer.py  #  6：环境门控教师 KL 数学
```

## 已知限制 / 未来工作

- 不同 tokenizer 的教师（重分词 + 位置对齐）——未实现。
- 软 KL 门控使用 A_total 作为 A^E 的代理（硬门控符号精确；软门控为近似）。
- `lambda_t` 自适应（idea 中的"自适应教师权重"）、`c_t` 可信度门控、
  情形 1/情形 2 软切换、动作类型归一化、教师有害性检测——v1 未实现
  （`tgsa.py` 中已预留 hook）。
- 端到端深度验证需要在线 verl+Ray+sglang 集群（通过 `run_search.sh` 短训练）。
