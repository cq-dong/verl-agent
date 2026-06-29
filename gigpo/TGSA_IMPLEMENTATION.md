# TGSA-GRPO 实现详解 —— 算法细节与 idea 对应

> 本文档逐条说明 TGSA-GRPO(`Paper/idea.md`)在 verl-agent 中的实现细节,重点是**每一条 idea 公式如何落到具体代码**,以及实现中做出的关键工程决策与数学取舍。代码引用形如 `tgsa.py:函数名` / `core_gigpo.py:L行号`。

---

## 0. 一句话定位

教师模型 π_T 作为 **GiGPO 环境奖励框架内的步骤级信用分配调制器(modulator)**,而非 OPD 式 KL 蒸馏。环境始终拥有最终否决权(优势符号保持),教师只在环境判断基础上做"调幅不调向"。

**总公式**(idea.md L80):

$$A_{\text{total}}(a_t) = A^E(\tau) + \lambda \cdot \tilde{L}_T(a_t) \cdot |A^E(\tau)| + \mu \cdot \mathbb{1}_{\text{deg}} \cdot \tilde{L}_T(a_t)$$

---

## 1. idea 公式 → 代码逐条对应

| idea 元素 | idea 公式 / 描述 | 实现位置 | 实现关键 |
|---|---|---|---|
| 长度归一化教师对数概率 | $\log\pi_T(a_t\|s_t)=\frac{1}{\|a_t\|}\sum_k\log\pi_T(w_k\|s_t,w_{<k})$ | `tgsa.py::length_normalize_logprob` | 对 `response_mask` 标记的动作 token 跨度求均值 |
| Case1 组内排名(minmax) | $\bar L_T^{mm}=2\frac{\log\pi_T-\min}{\max-\min+\epsilon}-1$ | `compute_teacher_preference_signal` | **统计对象是教师 lp,非回报** |
| Case1 组内排名(zscore) | $\bar L_T^{z}=\frac{\log\pi_T-\mu_{group}}{\sigma_{group}+\epsilon}$ | 同上 | 单例组 σ=1.0 哨兵 |
| Case2 单例师生差值 | $\tilde L_T^{\Delta}=\tanh(\gamma\Delta_{\text{norm}})$,$\Delta_{\text{norm}}=\frac{\Delta-\mu_\Delta}{\sigma_\Delta+\epsilon}$(batch 内单例行,可 $\text{clip}$ 到 $[-c,c]$) | 同上 | 先对单例行 $\Delta$ 做 batch z-score(修正系统性偏负),`delta_norm_clip` 钳制后 `tanh`(见 §2.3) |
| **总优势** | 见上 | `tgsa.py::compute_tgsa_advantage` | **第二项用 `a_e.abs()`,load-bearing** |
| 退化指示函数 | $\mathbb{1}_{deg}=\mathbb{1}[\sigma^R_{group}\le\epsilon]$ | 同上 | σ^R_group 取 **episode 组**(见 §4) |
| 逆 KL 单 token MC | $D_{KL}(\pi_\theta\|\pi_T)\|_t \approx \log\pi_\theta(w)-\log\pi_T(w)$ | `tgsa.py::compute_reverse_kl_token` | 无 top-k、无词表求和(见 §6) |
| 环境门控 KL 正则 | $\beta_T\sigma(\eta A^E)D_{KL}(\pi_\theta\|\pi_T)$ | `dp_actor.py::update_policy`(use_tgsa_kl 分支) | 门控用 `sign(advantages)`(见 §8) |
| \|G_t\| 导出 | idea 未指定,需从 GiGPO 锚定组取出 | `core_gigpo.py::_step_group_size_per_row` | `np.unique(uids, return_counts=True)` 内联推导,不改上游签名 |
| σ^R_group 导出 | idea 未指定 | `core_gigpo.py::_episode_group_return_std` | 见 §4 的语义决策 |

---

## 2. 教师偏好信号 $\tilde{L}_T$ 的实现细节

**入口**:`compute_teacher_preference_signal`(`tgsa.py`),返回 `(bs,)` 张量,值域 `[-1,1]`。

### 2.1 前置:长度归一化

```python
teacher_lp = length_normalize_logprob(teacher_log_prob, response_mask)  # (bs,)
student_lp = length_normalize_logprob(old_log_probs, response_mask)     # (bs,)
```
对每个 row(一个 env turn = 一个动作 $a_t$),在动作 token 跨度上对 per-token 对数概率求均值。`response_mask` 标记哪些 token 属于该动作。空跨度用 `clamp(min=1.0)` + `eps` 兜底防除零。

### 2.2 Case1(有效锚定组,$|G_t|\ge 2$):组内排名

idea L39-55 明确:**排名统计对象是教师的对数概率,与回报、与学生当前策略都无关**。这是信号"稳定可靠"的来源。

实现里 `_group_stats_per_row` 按 `step_group_uids`(GiGPO `build_step_group` 产生的锚定簇 UUID)分组,在**教师 lp** 上算组内 mean/std/min/max。单例组(`len(idxs)==1`)给 std=1.0 哨兵(使 z-score→0 良定义),与 `episode_norm_reward` 的约定一致。

```python
if normalization_mode == "minmax":
    case1 = 2.0 * (ranking_value - mn) / (mx - mn).clamp(min=eps) - 1.0
elif normalization_mode == "zscore":
    case1 = (ranking_value - mean) / (std + eps)
```
minmax 严格落 $[-1,1]$;zscore 可能超界,最后统一 `clamp(-1,1)`。

**margin 可选变体**(idea "教师偏好从排名升级为 margin",L227-235):
```python
if use_margin:
    runner = teacher_top2_logprob   # 教师在该步次优 token 的对数概率(sglang top-2)
    ranking_value = teacher_lp - runner   # m_t = logπ_T(a_t) - max_{a'≠a} logπ_T(a')
```
即把排名值替换为"领先第二名多少",再做 minmax。runner-up 来自教师 top-k(sglang `top_logprobs_num=2`),**这是唯一用到 top-k 的地方**。

### 2.3 Case2(单例,$|G_t|=1$):师生差值 + batch 归一化

**为什么需要 batch 归一化（核心问题）**：

原始 $\Delta = \log\pi_T(a_t) - \log\pi_\theta(a_t)$ 存在系统性偏负。由 Jensen 不等式：

$$\mathbb{E}_{a\sim\pi_\theta}[\log\pi_T(a) - \log\pi_\theta(a)] = -D_{KL}(\pi_\theta\|\pi_T) \leq 0$$

学生是在自己采样的 token 上计算 $\log\pi_\theta$（自然获得最高似然），而教师并未针对这些具体 token 做任何优化——因此 $\Delta$ 几乎总是负数。若直接对原始 $\Delta$ 做 $\tanh$，$\tilde{L}_T^\Delta$ 会系统性偏负，导致"教师对所有单例步骤几乎都不认可"，**四象限语义完全失效**：

| 场景 | 期望 | 若未 batch-norm |
|---|---|---|
| 成功轨迹 + G=1 | 弱/强奖励 | $\tilde{L}_T < 0$ → 调制项为负 → **强行压低正向奖励** ❌ |
| 失败轨迹 + G=1 | 弱/强惩罚 | $\tilde{L}_T < 0$ → 调制项为正 → **反而减小惩罚** ❌ |

**修复方案：batch-level z-score 归一化（仅对单例行）**。

归一化后，$\Delta_{\text{norm}} = (\Delta - \mu_\Delta) / (\sigma_\Delta + \epsilon)$ 的语义变为"比本 batch 单例行的平均师生差距更大（正）/更小（负）"，与 Case1 组内排名的语义完全一致：

```python
singleton_mask = ~is_group                      # 仅单例行
singleton_deltas = delta[singleton_mask]
mu_delta  = singleton_deltas.mean()
std_delta = singleton_deltas.std(unbiased=False).clamp(min=eps)
delta_norm = (delta - mu_delta) / (std_delta + eps)   # (bs,) 广播
case2 = torch.tanh(gamma * delta_norm)
```

注意：`mu_delta`/`std_delta` 只在**单例行**上统计，但归一化后的 `delta_norm` 对所有行都计算——`torch.where(is_group, case1, case2)` 会确保有效组行最终走 `case1`，所以 `case2` 对有效组行的值没有影响。

**信号语义**（归一化后）：
- $\Delta_{\text{norm}} > 0$：教师对该动作的偏好**高于本 batch 单例行平均水平** → 正信号 ✅
- $\Delta_{\text{norm}} < 0$：教师对该动作的偏好**低于本 batch 单例行平均水平** → 负信号 ✅
- 课程效应保留：随训练 $\pi_\theta \to \pi_T$，所有 $\Delta$ 一同收缩，$\sigma_\Delta \to 0$，信号自然衰减。

**边界：batch 中仅 1 个单例行（`singleton_mask.sum()==1`）。** 此时 `std_delta=0` → `clamp(eps)` → `delta_norm=(delta-mu)/eps≈0` → `tanh(0)=0`，该行教师信号归零。这是一个**安全的退化**而非 bug：没有其他单例可作相对比较，相对信号本身无定义，行自然回落到纯 $A^E$（环境否决权保持）。真实训练 batch 通常含大量单例行（开放式 agent 任务锚定状态重合稀疏），故该边界极少触发；由 `test_singleton_single_row_zero_fallback` 锁定。

**权衡与稳定性：batch 组成依赖性。** batch z-score 把信号语义从"绝对偏好"改成"相对本 batch 单例平均的偏好"（与 Case1 组内排名哲学一致，二者可在 $[-1,1]$ 统一混用，对应 idea L213-221 "分布校准"），但带来一个副作用：**信号幅度依赖 batch 组成**。

- 当某 batch 单例的 $\Delta$ **紧密聚集** → $\sigma_\Delta$ 很小 → `delta_norm` 爆炸 → `tanh` 饱和到 $\pm1$（分辨率丢失，所有单例被极端化）。
- 当 $\Delta$ **高度分散** → $\sigma_\Delta$ 大 → `delta_norm` 被压缩 → 信号幅度偏小。

特别地，一个与紧密簇相对的**离群单例**会被 z-score 放大到 $\gg 3$ → tanh 几乎完全饱和 → 该行的教师调制强度被无意义地拉满。

**稳定性保险：`delta_norm_clip`（默认关，推荐 `~3.0`）。** 在 `tanh` 前对 `delta_norm` 做 `clamp([-c, c])`，钳制放大：

```python
if delta_norm_clip and delta_norm_clip > 0.0:
    delta_norm = delta_norm.clamp(min=-delta_norm_clip, max=delta_norm_clip)
case2 = torch.tanh(gamma * delta_norm)
```

性质（由 5 个测试锁定）：
- **只钳幅度不改极性**：`test_delta_norm_clip_does_not_flip_polarity` —— 每行 sign(clip 后)=sign(clip 前)。
- **离群行不再饱和**：`test_delta_norm_clip_bounds_signal` —— 紧密簇+离群例子里，离群行从饱和的 -0.993 钳到 tanh(-1)=-0.7616，保留分辨率。
- **只影响单例行**：`test_delta_norm_clip_only_affects_singletons` —— Case1(组)行不受 clip 影响。
- **0=完全等价无 clip**：`test_delta_norm_clip_zero_is_noop`（默认关，向后兼容）。

> 调参建议：先 `0.0`（关）跑通，观察单例 `delta_norm` 的实际分布；若常见 $|z|>3$ 的饱和，开 `3.0`。也可作为消融项（clip on/off）。

> **与标准 OPD 的关系**：标准 OPD 用 $-\log\pi_T(a_t)$ 绝对值驱动，完全不受 $D_{KL}$ 偏置影响。TGSA 的 $\Delta$ 经 batch-norm 后等价于"相对 OPD"——相对本 batch 平均水平的偏好强弱，而非绝对偏好大小，这与 Case1 组内排名的设计哲学一致。

### 2.4 统一

```python
is_group = (gsize >= 2)                  # Case1 掩码
l_tilde = torch.where(is_group, case1, case2).clamp(-1, 1)
```
**硬切换**:Case1 与 Case2 由 `|G_t|>=2` 决定,无插值(idea 的"软切换"是可选改进,未实现,见 §9)。

---

## 3. 总优势 $A_{\text{total}}$ 的实现细节

**入口**:`compute_tgsa_advantage`(`tgsa.py`)。

### 3.1 三项逐项实现

```python
a_e = episode_advantages_row                              # (bs,) GiGPO 的 A^E
l_tilde = compute_teacher_preference_signal(...)          # (bs,) 见 §2

sigma_r = episode_group_std                               # (bs,) σ^R_group,见 §4
gsize   = group_size_per_row
one_deg = ((sigma_r <= eps_deg) & (gsize >= 2)).to(a_e.dtype)   # ★ 见 §5

abs_ae = a_e.abs()                                        # ★★★ load-bearing
scale  = abs_ae   # bounded_env_scaling: "none"|"tanh"|"clip"

a_total_row = a_e + lambda_ * l_tilde * scale + mu * one_deg * l_tilde
```

### 3.2 `|A^E|` 为什么是 load-bearing(最易写错处)

idea 四象限语义(idea L94-101)的正确性**完全依赖**第二项用 $|A^E|$ 而非 $A^E$:

| 场景 | $A^E$ | $\tilde L_T$ | 用 $|A^E\|$(正确) | 若误用 $A^E$(错误) |
|---|---|---|---|---|
| 成功+认可 | >0 | >0 | $A^E+\lambda\tilde L_T\|A^E\|$,正向增大 ✅ | 同(符号相同) |
| 成功+不认可 | >0 | <0 | $A^E+\lambda(\text{负})$,正向减小 ✅ | 同 |
| 失败+认可 | <0 | >0 | $A^E+\lambda(+1)\|A^E\|$,负向减小 ✅(弱罚) | $A^E+\lambda(+1)A^E=A^E(1+\lambda)$,负向**增大** ❌(变强罚,符号错) |
| 失败+不认可 | <0 | <0 | $A^E+\lambda(-1)\|A^E\|$,负向增大 ✅(强罚) | $A^E+\lambda(-1)A^E=A^E(1-\lambda)$,负向**减小** ❌(变弱罚) |

后两象限若用 $A^E$ 会**翻转**。这一不变式由 `test_tgsa.py` 的 4 个 quadrant 测试锁定(数值精确到 1.3/0.7/-0.7/-1.3)。

### 3.3 有界环境缩放(idea "有界环境缩放项",L157-171)

```python
if bounded_env_scaling == "tanh":  scale = torch.tanh(abs_ae)
elif bounded_env_scaling == "clip": scale = abs_ae.clamp(max=1.0)
else:                               scale = abs_ae
```
把裸 $|A^E|$ 换成有界函数,防止极端成功/失败样本上教师调制项主导。默认 `"none"`。

### 3.4 step advantage 的处理

idea 的 $A_{\text{total}}$ **不含** GiGPO 的 $A^S$。实现提供两种模式:
- `replace_step_advantage=True`(默认,符合 idea):丢弃 $A^S$。
- `replace_step_advantage=False`(消融用):`a_total_row += step_advantage_w * step_advantages_row`。

> 决策:搜索任务中 $A^S$ 常因状态重合稀疏而≈0,μ 退化兜底项是有原则的替代。故默认替换。

### 3.5 广播与 mask

```python
a_total = a_total_row.unsqueeze(-1).tile([1, response_length]) * response_mask
```
每行的标量优势广播到 `(bs, response_length)`,再乘 mask。输出 shape/契约与 GiGPO 的 `scores` 完全一致 → **loss 侧零改动**(`compute_policy_loss` 只吃 advantage 张量)。

---

## 4. $\sigma^R_{group}$ 的语义决策(关键实现取舍)

idea L86-110 说 $\mathbb{1}_{deg}$ 在"组内结果一致(全成功/全失败)"时激活。但 idea **没有明确**"组"指 episode 组(prompt 组,$A^E$ 在此归一化)还是 step 组(锚定组,$A^S$ 在此归一化)。这是实现时必须决策的歧义点。

**决策:取 episode 组。** 理由(写入了 `_episode_group_return_std` 的 docstring 与项目 memory):

μ·1_deg·L_T 项是**兜底项**,它必须在 λ·L_T·|A^E| 项**哑火**时才接管。而 λ 项哑火的充要条件是 $|A^E|\approx 0$ —— $A^E$ 在 **episode 组**内归一化。因此 1_deg 必须检测 **episode 组的回报坍缩**。

**反证**:若 1_deg 取 step 组退化,会出现"step 组回报一致但 episode 组 $A^E$ 仍很大"的情况,此时 μ·L_T 会叠加在一个已经很强的环境信号上——等于教师"在环境还响着的时候接管",直接违背 idea"环境哑火时临时接管"的设计。

实现(`_episode_group_return_std`):
1. 每条轨迹的总回报 = 其各步 token 奖励之和(`token_level_rewards.sum(-1)`)。
2. 按 `traj_index` 聚合到轨迹级,再按 `index`(episode 组 uid)收集各轨迹回报。
3. episode 组内回报求 population std;单成员 episode 组(单轨迹)给 1.0 哨兵 → 1_deg=False。
4. 同组的所有 row 共享该 std。

> 与原 `step_norm_reward` 的 `id2std` 区别:后者是 step 组(锚定组)回报 std,用于 $A^S$ 归一化,**不是** 1_deg 用的量。项目 memory 里早期一笔"step_norm_reward id2std"的说法不精确,已更正。

---

## 5. 单例 vs 退化的区分(第二个易错点)

idea 有两种"组不行"的情况,语义不同,**实现必须严格区分,不能 OR**:

| 情况 | 条件 | 处理 | 为什么不混 |
|---|---|---|---|
| 单例 | $\|G_t\|=1$ | Case2 `tanh(γΔ_norm)` + batch z-score 归一化 | 单例有师生差值信号(归一化消除 D_KL 偏负后语义正确),不需要兜底 |
| 退化 | $\|G_t\|\ge 2 \land \sigma^R_{group}\le\epsilon$ | 1_deg=1,μ 兜底 | 多成员但回报一致,λ 项哑火,需 μ 接管 |

**陷阱**:单例组的 std 哨兵 = 1.0(>ε)。若 1_deg 的判定写成 `sigma_r <= eps_deg`(不带 `& gsize>=2`),单例会因为 1.0>ε 而本就不触发,看似无害;但若有人把单例 std 哨兵改成 0,单例就会被误判为退化,触发 μ 兜底——这是 bug。

实现显式写成:
```python
one_deg = ((sigma_r <= eps_deg) & (gsize >= 2)).to(a_e.dtype)
```
`gsize>=2` 把单例排除在外,**即使 σ 哨兵语义日后变化也不会误触发**。由 `test_singleton_not_treated_as_degenerate`、`test_degenerate_group_fallback` 共同锁定。

---

## 6. 逆 KL 估计:为什么单 token MC,不要 top-k

idea L256/L278:$D_{KL}(\pi_\theta\|\pi_T)$ 是**逆向 KL**(期望在 $\pi_\theta$ 下)。在 on-policy RL 里,学生采样的 token 恰好 $w\sim\pi_\theta$,所以:

$$D_{KL}(\pi_\theta\|\pi_T)\big|_t \approx \log\pi_\theta(w_t) - \log\pi_T(w_t)$$

这是 Schulman k1 估计,**单样本 Monte-Carlo,无需 top-k、无需词表求和**(期望本就在 $\pi_\theta$ 下,样本来自 $\pi_\theta$)。匹配 ROLL `compute_approx_kl` 的 `'kl'` 分支与 verl `kl_penalty` 的 `'kl'` 分支。

实现 `compute_reverse_kl_token` 提供四种变体(idea 未指定,作为消融):
```python
log_ratio = old_log_probs - teacher_log_prob
"kl"  -> log_ratio                      # k1,无偏低方差高
"abs" -> log_ratio.abs()
"mse" -> 0.5 * log_ratio.square()       # k2
"k3"  -> (exp(teacher-old) - (teacher-old) - 1).clamp(-10,10)  # 默认,低方差非负
```
**默认 k3**(低方差,匹配 ROLL actor 默认)。top-k **仅**用于 §2.2 的 margin runner-up,与 KL 估计无关。

> idea L278 数学等价性注记:$D_{KL}(\pi_\theta\|\pi_T)\|_t = -\Delta_t + \text{const}$,即 KL 贡献与 Case2 的师生差值符号相反。实现层面两者共用同一批教师 logprob,零额外前向。

---

## 7. 教师前向:sglang HTTP 与响应窗对齐

**设计**:教师是**独立 sglang HTTP 服务**(端口暴露),不是 verl Ray worker。32B+ 教师跑在自有 GPU,verl 只发 HTTP。**不新增 `Role.Teacher`、不改 `init_workers`、不改 resource pool**。

**输入契约**:每行 `input_ids = [context | response]`,response 占末尾 `response_length` 列。假设**教师与学生共享 tokenizer**(直接发学生 `input_ids`;不同 tokenizer 的重分词+对齐是未来工作)。

**sglang `/generate` 调用**(`teacher_client.py::_post_row`):
```python
payload = {
    "input_ids": row,
    "sampling_params": {"max_new_tokens": 0, "temperature": 0},  # 只打分不生成
    "return_logprob": True,
    "logprob_start_len": 0,        # 取全数组,与 input_token_ids 1:1
    "top_logprobs_num": top_logprobs_num,  # 0=关;>=2 仅 margin 用
}
```

**响应窗提取**(最脆弱处,有专门测试):
```python
resp_lp = _slice_response(lp_vals, response_length)   # 取末尾 response_length 个
```
sglang 不同 patch 版本有两种约定:返回全长数组,或丢弃首个(无条件)token 使数组长度=total-1。**两种约定下,"末尾 response_length 个"都恰好落在响应窗**,故统一取末尾。`_parse_logprobs` 还容错:logprob 条目可能是 float 或 `{"logprob":...}`;token 可能是 int/dict/str。

**并发**:`ThreadPoolExecutor`,一行一个请求,`max_concurrency` 并发,失败指数退避重试。

---

## 8. 环境门控 KL 正则(dp_actor)

idea L245-280。在 GRPO 损失上加一项:
$$\beta_T \cdot \text{gate} \cdot D_{KL}(\pi_\theta\|\pi_T),\quad \text{gate}=\sigma(\eta A^E)\text{ 或 }\mathbb{1}[A^E>0]$$

实现(`dp_actor.py::update_policy` 的 `use_tgsa_kl` 分支,默认 `coef=0` 关闭):
```python
kl_tok = compute_reverse_kl_token(
    old_log_probs=log_prob,              # ★ 当前策略 log_prob,非 old(梯度可流)
    teacher_log_prob=teacher_log_prob,   # 预计算,零额外前向
    response_mask=response_mask,
    kl_penalty=tgsa_kl_penalty)          # 默认 k3
kl_per_row = masked_mean(kl_tok, response_mask, axis=-1)
adv_per_row = masked_mean(advantages, response_mask, axis=-1)
if tgsa_kl_gate_mode == "hard":
    gate = (adv_per_row > 0).to(...)     # 1[A^E>0]
else:
    gate = torch.sigmoid(tsga_kl_gate_eta * adv_per_row)
tgsa_kl_loss = (gate * kl_per_row).mean()
policy_loss = policy_loss + tgsa_kl_loss * tgsa_kl_coef
```

**门控取 `sign(advantages)` 而非 `sign(A^E)` 的精确性**:advantages 张量在 TGSA 下保持 sign(A^E)(见 §3.2 的 sign 保持),故 `(adv_per_row>0)` 与 `1[A^E>0]` **符号精确等价**(硬门)。软门 `σ(η·A_total)` 是 `σ(η·A^E)` 的近似(符号一致,幅度不同),因为 row 级拿不到裸 $A^E$ 标量(只有广播后的 advantage)。

**为什么失败轨迹必须关闭**(idea L261-270):逆向 KL 蒸馏对失败轨迹也会把学生拉向教师,强化错误行为——这正是 OPD 的根本缺陷。门控让 KL 只在成功轨迹施压。由 `test_hard_gate_skips_fail_rows`、`test_hard_gate_all_fail_zero_loss` 锁定。

**KL 用当前策略 `log_prob`** 而非 `old_log_prob`:梯度需流经当前策略。教师 logprob 在优势计算阶段已预算好,直接复用,零额外前向(idea L280 明确)。

---

## 9. 训练循环接线(ray_trainer.fit)

时序(关键:教师 I/O 与 Ray 调用并发):

```
1. adjust_batch → response_mask = compute_response_mask
2. [TGSA] multi_turn 时: response_mask = loss_mask[:,-resp_len:]   ← mask 统一(§10)
3. balance_batch (就地重排行序)
4. [TGSA] 启动教师评分 future (balance_batch 之后,行序已定)         ← HTTP 与下面 Ray 调用并发
5. reward (Ray) / old_log_prob (Ray) / ref_log_prob (Ray)          ← 这三步与教师 HTTP 重叠
6. apply_kl_penalty / token_level_rewards
7. [TGSA] join teacher future,把 teacher_log_prob 写入 batch
8. compute_advantage → compute_gigpo_outcome_advantage(tgsa_config=...)
9. update_actor (meta_info 带 tgsa_kl 参数)
```

教师 future 必须在 `balance_batch` **之后**启动(否则行序不一致);在 `compute_advantage` **之前** join。用一个单 worker 的 `ThreadPoolExecutor` 驱动,与 Ray 调用并发隐藏 HTTP 延迟。

`compute_advantage` 的 GiGPO 分支:从 batch 读 `teacher_log_prob`/`old_log_probs`/`teacher_top2_logprob` + `tgsa_config`,转发给 `compute_gigpo_outcome_advantage`。TGSA 只调制 GiGPO,非 GiGPO 估计器跳过教师调用(避免浪费 HTTP)。

---

## 10. multi_turn mask 统一(被 TGSA 放大的既有问题)

verl 原代码:`compute_advantage` 用 attention_mask 派生的 `response_mask` 算优势,而 `dp_actor` 用 `loss_mask[:,-resp_len:]` 算策略 loss。multi_turn 下两者**选不同 token 集**——原 GiGPO 里这不致命($A^E$ 是轨迹级标量),但 TGSA 用 mask 做**教师 lp 长度归一、σ^R_group 统计、KL masked_mean**,mask 不一致会让教师信号与 loss 对不上。

实现:`tgsa_enabled and multi_turn` 时,统一 `response_mask = loss_mask[:,-resp_len:]`。这与 `dp_actor` 的 loss mask 完全对齐。

---

## 11. 配置项与默认值(`ppo_trainer.yaml::algorithm.tgsa`)

```yaml
tgsa:
  enabled: False                # 主开关;False=原 GiGPO,教师从不调用
  lambda: 0.3                   # 正常组调制强度(idea 0.2-0.5)
  mu: 0.1                       # 退化组兜底强度(idea ~0.1)
  gamma: 1.0                    # Case2 tanh 尺度
  eps_deg: 0.01                 # σ^R_group 退化阈值(idea ~0.01)
  normalization_mode: "minmax"  # Case1 排名归一化
  replace_step_advantage: True  # True=丢 A^S(idea);False=加 w*A^S(消融)
  bounded_env_scaling: "none"   # |A^E| 有界缩放
  delta_norm_clip: 0.0          # Case2 稳定性保险:tanh 前钳 delta_norm 到 [-c,c](0=关,~3.0 推荐)
  teacher: {base_url, max_concurrency, timeout, max_retries}
  kl:
    kl_teacher_coef: 0.0        # 0=关;>0 启用门控 KL 正则
    kl_penalty: "k3"
    kl_gate_eta: 1.0
    kl_gate_mode: "hard"        # hard=1[A^E>0] | soft=sigmoid(ηA^E)
  margin: {enabled: False, topk: 2}   # margin 变体,默认关
```

**向后兼容**:`enabled: False`(默认)→ 教师从不调用 → 原 GiGPO 行为。`compute_gigpo_outcome_advantage` 的新参全默认 None,4 个既有调用点(recipe/GraphGPO×2、hgpo、ray_trainer)零改动。

---

## 12. 与原 GiGPO / OPD 的对比(实现层面)

| 维度 | 原 GiGPO | OPD(ROLL) | **TGSA(本实现)** |
|---|---|---|---|
| 步骤级信号 | 锚定组对比 $A^S$,依赖状态重合 | 完全模仿教师 KL | 教师组内排名/师生差值,仅需 forward |
| 教师信任 | 无教师 | 完全信任,忽视环境 | **环境主导+教师调制**(符号保持) |
| 退化组 | 退化为 $A^E$ | 有信号但可能强化错误 | **μ 兜底**,弱但方向正确 |
| 损失侧改动 | — | KL 折叠优势(反模式) | **零**(advantage 张量契约不变),仅可选加门控 KL 正则 |
| 教师部署 | — | ROLL worker | **sglang HTTP 独立服务** |

**ROLL OPD 几乎不迁移**:其 KL 折叠优势(`advantage = -coef*KL`)正是 idea 要避免的反模式;其教师 per-token logprob 收集算子 verl 已原生有(`logprobs_from_logits`)。仅"教师前向"模式被复用,但改走 sglang HTTP。

---

## 13. 关键不变式(测试锁定)

| 不变式 | 锁定测试 |
|---|---|
| 四象限数值(用 $\|A^E\|$) | `test_four_quadrant_*`(4 个) |
| 正常组 sign(A_total)=sign(A^E)(环境否决权) | `test_sign_preservation_normal_group` |
| 退化组 $A_{total}\approx\mu\tilde L_T$ | `test_degenerate_group_fallback` |
| 正常组 μ 项关闭 | `test_nondegenerate_group_mu_term_off` |
| 单例 Case2 batch z-score 修正系统性偏负(居中),不被 μ 兜底 | `test_singleton_batch_norm_centers_systematic_bias`、`test_singleton_batch_norm_ordering_preserved`、`test_singleton_single_row_zero_fallback`、`test_singleton_not_treated_as_degenerate` |
| `delta_norm_clip` 只钳幅度不改极性、离群行不饱和、只影响单例、0=无clip | `test_delta_norm_clip_*`(5 个) |
| 逆 KL 单 token、无词表维度 | `test_reverse_kl_no_topk_no_vocab_sum` |
| KL 门控只在成功轨迹激活 | `test_hard_gate_skips_fail_rows`、`test_hard_gate_all_fail_zero_loss` |
| KL 响应当前策略 logprob | `test_kl_responds_to_current_logprob` |
| 向后兼容(None→原 GiGPO) | `test_backward_compat_no_tgsa`、`test_tgsa_disabled_config_falls_back` |
| TGSA 路径==直接调用 | `test_tgsa_path_matches_direct_call` |
| sglang 响应窗对齐(全长/首token丢弃/dict形态) | `test_extract_response_window_*`、`test_dict_form_logprob_entries` |

40 个测试全过,纯 CPU,不依赖 Ray/sglang/GPU。

---

## 14. 已知限制与未实现的 idea 改进点

**未实现(idea 可选改进,代码留有 hook)**:
- 自适应 $\lambda_t = \lambda_{\max}\min(1, D_{KL}/c)$(idea L137-145)—— `reverse_kl_scalar` 已提供标量 KL,可直接接。
- 局部可信度门 $c_t$(idea L147-155)。
- Case1/Case2 软切换(idea L197-211)—— 当前硬切换。
- 两类信号分布校准(idea L213-221)——**已实现：batch-level z-score 归一化已内置入 Case2 路径（见 §2.3）。**
- 全成功/全失败组拆分 $\mu_+/\mu_-$(idea L173-181)——**诊断侧已实现**(§15 的 `frac_degenerate_all_{success,fail}` + `degenerate_return_hist` 双峰),但**优势侧未拆 μ**($\mu$ 仍为单一系数)。
- 动作类型感知归一化(需 rollout 标注)、教师有害性检测(需分类器)。
- 理论保证(方向保持性/方差控制,idea L237-239)——数值上由 sign-preservation 测试支撑,形式化证明未做。

**实现假设**:
- 教师与学生共享 tokenizer(直接发 `input_ids`)。不同 tokenizer 的重分词+位置对齐未实现。
- 软 KL 门用 A_total 作 A^E 近似(硬门符号精确,软门幅度近似)。
- 深度端到端验证(真 verl+Ray+sglang 短训练)未跑——需集群。所有独立测试绿,源码 `py_compile` 通过。

---

## 15. tensorboard 统计量记录(迭代想法的体检面板)

TGSA 的中间量(分组结构、教师信号、优势分解、教师 I/O 健康)几乎都产生在 `gigpo/` 内部,**不在 verl 的 `metrics` dict 可见范围**(后者只从 worker `meta_info['metrics']` 汇总)。为此新增侧信道 logger `gigpo/tgsa_stats.py`:`TGSAStatsLogger` 单例自带 `SummaryWriter`,直接写到 verl 同一个 `$TENSORBOARD_DIR`(tensorboard 自动把 `tgsa_*` 曲线与 `actor/*`、`critic/*` 合并到同一视图),**不修改 verl 框架的任何 metrics/tracking 代码**。

**设计要点**:
- **分组统计已与 TGSA 解耦**。`gigpo_group/*` 面板由 `algorithm.gigpo.log_group_stats`(默认 True)控制,**纯 GiGPO(不开 TGSA、不起教师)也能看**——因为 GiGPO 的 $A^S$ 有无信号本就取决于锚定重合度,这是 GiGPO 自身的诊断量。教师信号类(`tgsa_signal/*`/`tgsa_advantage/*`/`tgsa_cfg/*`)仍需开 TGSA(本就需要教师)。
- **默认禁用 logger**。`RayPPOTrainer.fit` 在 `tgsa_enabled OR (GiGPO & log_group_stats)` 时调一次 `enable()`;单元测试不调 → `record_*`/`flush` 全 no-op、零文件。`enable()` 在缺 tensorboard 时(wandb-only 运行)优雅降级:告警一次并保持禁用,绝不崩训练。
- **全局 step**:trainer 每步在 `compute_advantage` 后调 `flush(self.global_steps)`,曲线落在真实全局步轴(resume 安全)。
- **每步全记**(标量 + 直方图),无降频。开销是 driver CPU 上若干 `.item()` + `np.histogram`,相对教师 HTTP 前向与 actor 反向可忽略(<0.001%)。
- **三方收集点**(由"数据在哪个作用域"决定,不可合并):
  - `teacher_client.py` → 教师 I/O(latency/success/retry,评分时计时计数;仅 TGSA);
  - `gigpo/tgsa.py::compute_tgsa_advantage` → per-row 信号/优势/四象限(`out_stats`/`out_hists` 可选参,默认 None 向后兼容;仅 TGSA);
  - `gigpo/core_gigpo.py` 的 `_record_and_return`(两分支公共出口)→ **分组级**(episode/step 组结构在此齐全,塌缩进 tgsa.py 后已不可得)。TGSA 与纯 GiGPO 都走此处,由 `log_group_stats` 门控;`compute_group_stats` 的 `sigma_r` 可选——纯 GiGPO 传 `None` 时组内 std 内部算(与 TGSA 的 `_episode_group_return_std` 数值一致)。

### 15.1 六个面板(`tgsa_*` / `gigpo_group/` 前缀分组)

| 面板 | 装什么 | 回答的迭代问题 | 需开 TGSA? |
|---|---|---|---|
| `gigpo_group/` | **分组级(按组去重)**:组数、组大小、退化率、全成功/全失败拆分、组大小直方图 | 分组结构健康吗?**哪类退化主导?** | 否(纯 GiGPO) |
| `tgsa_teacher/` | latency_s、success_rate、retry_count、rows | 教师活着吗?是瓶颈吗? | 是 |
| `tgsa_signal/` | delta(偏负)、std_delta、delta_norm、case1/case2、l_tilde、clip 触发率、Case1 组内教师 spread | 偏负修正了吗?clip 该开吗?信号有方向吗? | 是 |
| `tgsa_advantage/` | a_e、abs_ae、lambda/mu 项、a_total、sign_violation、四象限 frac | 四象限对吗?教师调制多大?不变式破了没? | 是 |
| `tgsa_coverage/` | frac_singleton/normal_group/degenerate(**按行**口径) | idea 主机制(λ)实际触发了多少行? | 是 |
| `tgsa_cfg/` | 数值配置回显(lambda/mu/gamma/eps_deg/delta_norm_clip/...) | 跨 run 对比 | 是 |

### 15.2 关键正确性:按组 vs 按行(§15 的核心)

**按行归约会扭曲分组真相**。例:1 个 5 成员组 + 95 单例 = 100 行、96 组:
- 按行:`frac_singleton` = 95/100 = 0.95(被大组稀释)
- 按组(`gigpo_group/step_frac_singleton`)= 95/96 ≈ 0.99(真相)

故 `gigpo_group/*` 全部**按组**(去重后)统计;`tgsa_coverage/*` 保留**按行**口径,二者交叉看大组稀释了多少。由 `test_group_stats_per_group_not_per_row` 锁定。

### 15.3 全成功 / 全失败拆分(idea μ+/μ- 的诊断侧)

退化 episode 组有两种相反语义:**全成功**(任务被平凡解决,无学习信号)vs **全失败**(没人解决,教师引导最关键)。合并的 `frac_degenerate` 对两者盲。`gigpo_group/frac_degenerate_all_{success,fail}` + `degenerate_return_hist`(双峰:≈1 全成功 / ≈0 全失败)直接暴露主导模式——回应 memory 里"search 1_deg 大面积触发"的诊断盲区。阈值 `success_thresh`(默认 0.0,0/1 奖励下正确)。由 `test_group_stats_all_success_all_fail_split` 锁定。纯 GiGPO(不开 TGSA)也能看这一面板——这正是把分组统计从 TGSA 解耦的动机。

> 注:这是**诊断侧**实现(让你看到拆分);**优势侧**仍用单一 $\mu$,未拆 $\mu_+/\mu_-$(idea L173-181 的完整实现留作后续)。

### 15.4 不变式监控(load-bearing 体检)

- `tgsa_advantage/sign_violation_frac`:正常组 `sign(A_total)≠sign(A^E)` 的行占比,应≈0。非零 = §3.2 的 `|A^E|` 四象限 bug 或 λ 过大。由 `test_sign_violation_frac_is_zero_for_default_lambda` 锁定。
- `tgsa_signal/delta_norm_frac_gt3`:单例行 `|delta_norm|>3` 占比,>0 频繁 → 开 `delta_norm_clip`。
- `tgsa_signal/case1_teacher_spread_mean`:Case1 组内 `max(lp)-min(lp)`,≈0 → minmax 排名是噪声(组内教师无差别)。

---

## 附:文件清单

```
gigpo/tgsa.py                      算法核心(L_T_tilde / A_total / 逆KL + out_stats 填充)
gigpo/tgsa_stats.py                侧信道 tensorboard logger(单例 + 分组级统计纯函数)
gigpo/core_gigpo.py                GiGPO 接线(|G_t|/σ^R_group 导出 + 签名扩展 + 分组级记录)
gigpo/teacher_client.py            sglang HTTP 教师客户端(+ 教师 I/O 健康计数)
verl/trainer/ppo/ray_trainer.py    训练循环接线(教师调用/mask统一/派发 + logger enable/flush)
verl/workers/actor/dp_actor.py     可选门控 KL 正则
verl/trainer/config/ppo_trainer.yaml   algorithm.tgsa + algorithm.gigpo + algorithm.opd 配置
examples/gigpo_trainer/run_search.sh   运行示例(TGSA 开关 + gigpo 分组统计开关 + success_thresh)
examples/gigpo_trainer/run_opd.sh      纯 OPD 消融基线运行脚本(adv_estimator=opd)
tests/trainer/ppo/test_tgsa.py                 23 测
tests/trainer/ppo/test_core_gigpo_tgsa.py       9 测
tests/trainer/ppo/test_teacher_client.py        9 测
tests/trainer/ppo/test_tgsa_kl_regularizer.py   6 测
tests/trainer/ppo/test_tgsa_stats.py           16 测
tests/trainer/ppo/test_opd.py                   5 测
```

---

## 16. 纯 OPD 消融基线(ROLL 标准实现复刻)

为消融对比 TGSA 相对标准 OPD(Online Policy Distillation)的增益,新增纯 OPD 基线分支(`algorithm.adv_estimator=opd`),**忠实复刻 ROLL 标准 OPD**。

### 16.1 ROLL 标准 OPD(已核实)

| 要素 | ROLL 实现(file:line) | 纯 OPD 复刻 |
|---|---|---|
| KL 估计 | `functionals.py:181` `compute_approx_kl`,k3=`(exp(t-s)-(t-s)-1).clamp(-10,10)` | 复用 `tgsa.py:582` `compute_reverse_kl_token` 的 k3(**逐字一致**) |
| advantage | `functionals.py:1149` `adv=-total_weighted_kld` | `gigpo/opd.py::compute_opd_advantage` 同公式 |
| mask | `functionals.py:1180` `adv*=response_mask` | 同 |
| 强制配置 | `base_config.py:888-901` gamma=0/adv_estimator=reinforce/critic_warmup=0/use_kl_loss=False | `run_opd.sh` 设 `gamma=0`/`use_kl_loss=False`;`use_critic=False`(enum OPD 进无 critic 列表) |
| **环境门控** | **无**(`functionals.py:1109-1145` 无 success/fail 判断) | **无**(纯 OPD 全轨迹无条件蒸馏) |

### 16.2 与 TGSA 的对照(消融核心)

| 维度 | 纯 OPD | TGSA(KL 开) |
|---|---|---|
| advantage | `-coef·KL`(KL 折叠进优势) | $A^E$+λ·L_T·\|A^E\|+μ·1_deg·L_T(+ 门控 KL loss) |
| 环境门控 | ❌ 无(全轨迹蒸馏,**失败轨迹也拉向教师=强化错误行为**) | ✅ `1[A^E>0]` 只蒸成功轨迹 |
| 环境奖励 | ❌ 无 | ✅ $A^E$ 主导 |
| critic | ❌ 无 | ❌ 无 |
| 教师部署 | sglang HTTP(复用 TGSA 教师 client) | sglang HTTP |

**核心对照:纯 OPD vs TGSA(KL 开)** —— 前者无门控全轨迹蒸馏(强化错误行为),后者门控只蒸成功轨迹。论证 idea 的 OPD 缺陷主张。

### 16.3 采样说明

纯 OPD advantage **逐轨迹独立,不依赖组内对比**(与 GiGPO/TGPA 不同)。故 `env.rollout.n=1` 合法(n 仅采样数,batch 增广用);为与 TGSA(`n=5`)公平对比可在 `run_opd.sh` 改大。

### 16.4 代码影响

- `gigpo/opd.py`(新增):`compute_opd_advantage` 复用 `compute_reverse_kl_token`,零新基础设施。
- `ray_trainer.py`(改 5 处):enum 加 `OPD`;`use_critic` 列表加 OPD;`__init__` 加 `opd_enabled`;`fit` 教师 client/logger/future/flush 条件从 `tgsa_enabled` 放宽为 `tgsa_enabled or opd_enabled`;`compute_advantage` 加 OPD 分支(**不调组归一化**,透传 `opd_config`)。教师 config 复用 `algorithm.tgsa.teacher`(TGSA 与 OPD 共用同一教师,公平对比)。
- 配置:`algorithm.opd{kl_coef,kl_penalty}`;`run_opd.sh`。

### 16.5 不变式(测试锁定)

| 不变式 | 测试 |
|---|---|
| adv=-coef·masked_kl 逐 token 精确 | `test_opd_advantage_equals_neg_coef_kl` |
| KL 与手算 ROLL k3 + `compute_reverse_kl_token` 三方一致 | `test_opd_kl_matches_roll_k3` |
| 无门控全轨迹蒸馏(对比 TGSA `test_hard_gate_skips_fail_rows`) | `test_opd_no_gate_distills_all_trajectories` |
| out_stats 填充 opd/* | `test_opd_out_stats_populated` |
| 向后兼容(student==teacher→adv=0) | `test_opd_backward_compat_no_teacher_args` |


