# TGSA-GRPO 消融实验配置清单

> 对应 `Paper/idea.md` L282-284 "配套消融设计"。每个消融实验说明:**目的、改哪个配置开关/代码、对应 idea 的审稿质疑**。所有消融基于已实现代码,**无需新写算法**,只改 `ppo_trainer.yaml` 的 `algorithm.tgsa` 配置块或 `run_search.sh` 的 Hydra 覆盖。

## 基线配置(消融的锚点)

```yaml
algorithm:
  adv_estimator: gigpo
  tgsa:
    enabled: True
    lambda: 0.3
    mu: 0.1
    gamma: 1.0
    eps_deg: 0.01
    normalization_mode: "minmax"
    replace_step_advantage: True
    bounded_env_scaling: "none"
    kl: {kl_teacher_coef: 0.0}        # 主实验关 KL 正则
    margin: {enabled: False}
```

下文每个消融只列**相对基线改动的那几行**。

---

## A. 教师调制主增益验证(去教师调制项)

**目的**:证明教师信号是主增益来源(idea L282 "去掉教师调制项,验证教师信号的主增益")。

**配置**:`tgsa.enabled=False`(退回原 GiGPO)。

**对照**:基线 vs A。若基线显著优于 A → 教师调制有效。

**注意**:这等价于"不开 TGSA 的纯 GiGPO",`run_search.sh` 里 `tgsa_enabled=False` 即是。无需改代码。

---

## B. 退化兜底必要性(去 μ 退化项)

**目的**:验证全成功/全失败组兜底的必要性(idea L282 "去掉退化项,验证全成功/全失败组兜底的必要性")。

**配置**:基线 + `tgsa.mu=0.0`。

```yaml
tgsa:
  mu: 0.0   # 关闭退化感知兜底
```

**对照**:基线 vs B。退化组占比高的任务(如搜索任务,非终止 reward=0 → σ^R_group≈0 → 1_deg 大面积触发,见项目 memory pitfall c)上预期 B 显著劣于基线。

**审稿质疑回应**:"μ 退化项是否真的有用?" → B 给出实证。

---

## C. 信号选择消融(全用师生差值)

**目的**:回应"为什么不统一用第二种信号(师生差值)"(idea L283 "全部使用师生差值")。

**实现**:需要一个**让 Case1 也走 Case2 公式**的开关。当前代码 Case1/Case2 由 `|G_t|>=2` 硬切换。两种做法:

- **C1(配置法,推荐)**:把 `normalization_mode` 设为某种"强制单例模式"——但当前实现无此开关,需加一个 `force_case2: True` 配置项,在 `compute_teacher_preference_signal` 里 `is_group = torch.zeros_like(gsize>=2)`(全部走 Case2)。
- **C2(临时改代码)**:在 `tgsa.py` 第 159 行附近临时把 `is_group` 置全 False。

**建议**:为消融干净,我在 `tgsa.py` 加一个 `signal_mode` 参数(`"auto"` 默认硬切换 / `"case2_only"` 全用师生差值 / `"case1_only"` 全用组内排名)。**需要我加这个参数吗?** 加了之后消融 C/D 就纯配置。

**对照**:基线(分级设计:auto)vs C(全 case2)vs D(全 case1)。证明分级设计优于单一信号。

---

## D. 信号选择消融(全用组内排名 + 单例 batch 归一化)

**目的**:验证分级设计的价值(idea L283 "全部使用组内排名并对单例做 batch 归一化")。

**实现**:同 C 的 `signal_mode`:
- `signal_mode="case1_only"`:单例也用组内排名——但单例无组内对比,需对单例的师生差值/教师 lp 做 **batch 级归一化**后塞进 Case1 通路。
- 这对应 idea L213-221 "两类教师信号的分布校准"(对单例 Δ 做 batch 标准化再 tanh)。

**对照**:基线 vs C vs D。三方对比是分级设计消融的核心,直接回应"为什么不统一用一种信号"。

**审稿质疑回应**:"组内排名与师生差值混用是否必要?" → C/D 表明单一信号各有缺陷,分级互补最优。

---

## E. 归一化方式消融(minmax vs zscore)

**目的**:idea L133 "两种归一化消融:Z-score vs Min-Max"。

**配置**:基线 + `tgsa.normalization_mode`:
- E1: `"minmax"`(基线默认)
- E2: `"zscore"`

```yaml
tgsa:
  normalization_mode: "zscore"   # E2
```

**对照**:E1 vs E2。idea 推荐 minmax(λ 语义直观),zscore 保留绝对质量差异信息。

**无需改代码**——两种归一化已实现(`tgsa.py` L175-179)。

---

## F. 锚定方式消融(精确锚定 vs 语义锚定)

**目的**:idea L283 "比较精确锚定与语义锚定",L183-185 "语义哈希/软锚定扩组"。

**配置**:用 GiGPO 已有的 `enable_similarity`(字符级 SequenceMatcher 软锚定,**非 embedding 软锚定**,用户已明确不做 embedding 版):

```yaml
algorithm:
  gigpo:
    enable_similarity: False   # F1: 精确锚定(基线)
    # enable_similarity: True  # F2: 语义软锚定
    similarity_thresh: 0.9
```

**对照**:F1 vs F2。F2 扩大有效锚定组覆盖率 → Case1(组内排名)覆盖更多步骤 → 预期教师排名信号更准。注意:这影响的是 **GiGPO 的 step group**,间接影响 TGSA 的 Case1/Case2 分布(|G_t| 变化)。

**审稿质疑回应**:"锚定状态分组对 TGSA 的影响?" → F 给出。注:用户已明确**不做 embedding 软锚定**,此处语义锚定指 GiGPO 自带的字符级 similarity。

---

## G. 退化项细分(μ+ vs μ-,全成功 vs 全失败)

**目的**:idea L173-181 "区分全成功组与全失败组",拆分 $\mu_+$/$\mu_-$,通常 $\mu_->\mu_+$。

**实现**:当前代码 1_deg 不区分全成功/全失败。需要:
1. 在 `compute_tgsa_advantage` 里,对 `one_deg=1` 的行再判 `a_e > 0`(全成功)vs `a_e <= 0`(全失败)。
2. 加 `mu_plus`/`mu_minus` 两个系数。

**需改代码**(`tgsa.py` + 配置)。**这是有价值的消融,但属于 idea 可选改进,当前未实现。需要我加吗?**

**对照**:基线(统一 μ) vs G(μ+/μ-)。

---

## H. 有界环境缩放消融

**目的**:idea L157-171 "有界环境缩放项"。

**配置**:基线 + `tgsa.bounded_env_scaling`:
- H1: `"none"`(基线默认,裸 |A^E|)
- H2: `"tanh"`
- H3: `"clip"`

```yaml
tgsa:
  bounded_env_scaling: "tanh"   # H2
```

**对照**:H1/H2/H3。验证"教师只调幅不改向"在有界约束下是否更稳。

**无需改代码**——三种已实现(`tgsa.py` L268-275)。

---

## I. 门控 KL 正则消融

**目的**:idea L245-280 "环境门控逆向 KL 蒸馏项"。

**配置**:
- I1: `kl.kl_teacher_coef=0.0`(基线,关)
- I2: `kl.kl_teacher_coef=0.01, kl.kl_gate_mode="hard"`(硬门,只成功轨迹)
- I3: `kl.kl_teacher_coef=0.01, kl.kl_gate_mode="soft"`(软门)
- I4: `kl.kl_teacher_coef=0.01`,**但去掉门控**(对照:无条件 KL,即变相 OPD)——需临时改代码把 `gate=1`。

```yaml
tgsa:
  kl:
    kl_teacher_coef: 0.01
    kl_gate_mode: "hard"   # I2
```

**对照**:I1 vs I2 vs I3 vs I4。I2>I4 证明门控必要性(失败轨迹不向教师靠拢);I2 vs I3 比较硬/软门。

**审稿质疑回应**:"门控 KL 是否真比无条件蒸馏好?" → I2 vs I4 直接回答(这正是 idea 区别于 OPD 的核心)。

**无需改代码**(I1/I2/I3);I4 需临时改 `dp_actor.py` 把 gate 置 1。

---

## J. margin 变体消融

**目的**:idea L227-235 "教师偏好从排名升级为 margin"。

**配置**:
- J1: `margin.enabled=False`(基线,纯排名)
- J2: `margin.enabled=True, margin.topk=2`(margin 变体)

```yaml
tgsa:
  margin:
    enabled: True
    topk: 2
```

**对照**:J1 vs J2。验证 margin 是否比纯排名更有判别力。

**无需改代码**——margin 已实现,但需教师 sglang 开 `top_logprobs_num=2`(`teacher_client` 会自动设)。

---

## K. 自适应 λ_t 消融(静态 vs 自适应)

**目的**:idea L137-145 "自适应教师权重 λ_t"。

**实现**:当前代码是静态 λ。自适应需要:
$$\lambda_t = \lambda_{\max}\cdot\min(1, D_{KL}(\pi_\theta\|\pi_T)/c)$$
代码 hook 已具备:`reverse_kl_scalar`(`tgsa.py`)能算 batch 级 KL 标量。需在 `compute_tgsa_advantage` 里把固定 `lambda_` 换成 `lambda_max * min(1, kl_scalar/c)`。

**需改代码**(`tgsa.py` + 配置 `lambda_max`/`lambda_c`)。**当前未实现,需要我加吗?**

**对照**:基线(静态 λ) vs K(自适应 λ_t)。验证训练后期教师调制自动退场。

---

## 消融优先级建议(论文实验顺序)

| 优先级 | 消融 | 价值 | 改动 |
|---|---|---|---|
| ★★★ | A(去教师) + B(去μ) | 主方法有效性 + 退化兜底 | 纯配置 |
| ★★★ | C(全case2) + D(全case1+单例batch归一) | 分级设计必要性(核心回应审稿) | 需加 `signal_mode` |
| ★★★ | I2 vs I4(门控 vs 无条件KL) | 与 OPD 的本质区别 | I4 临时改代码 |
| ★★ | E(minmax vs zscore) | 归一化选择 | 纯配置 |
| ★★ | H(有界缩放) | 稳定性 | 纯配置 |
| ★★ | F(精确 vs 语义锚定) | 锚定影响 | 纯配置(GiGPO自带) |
| ★ | J(margin) | 信号判别力 | 纯配置 |
| ★ | G(μ+/μ-) | 退化细分 | 需改代码 |
| ★ | K(自适应λ_t) | 课程效应 | 需改代码 |

**纯配置即可跑的消融(零代码改动)**:A、B、E、F、H、I1/I2/I3、J —— 共 8 组,建议作为论文主表。
**需少量代码的消融**:C/D(加 `signal_mode`)、G(μ+/μ-)、I4(无条件KL对照)、K(自适应λ_t)—— 4 组,作为深入分析。

---

## 需要我补的实现(待确认)

若你要跑 C/D/G/K/I4,我需要加:
1. **`signal_mode` 参数**(`tgsa.py`):`auto`/`case2_only`/`case1_only`,让 C/D 纯配置可跑。**最推荐先加这个**,因为它解锁 C/D 两个核心消融。
2. **`mu_plus`/`mu_minus`**(`tgsa.py` + 配置):解锁 G。
3. **自适应 `lambda_t`**(`tgsa.py` + 配置 `lambda_max`/`lambda_c`):解锁 K。
4. **无条件 KL 对照**(临时改 `dp_actor.py` gate=1):解锁 I4。

要我现在把 1-4 全部加上吗?还是只加最推荐的 `signal_mode`(1)?
