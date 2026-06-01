---
title: "LLM x RL"
date: 2026-05-28 17:01:00 +0800
categories: [Artificial Intelligence, Reinforcement Learning]
tags: [Tech, AI, RL, LLM]
math: True
---

> **前置知识**:本文假设读者已熟悉经典 RL(MDP、policy gradient、PPO)。LLM 架构相关(decoder-only Transformer、参数估算等)的工程基础可参考:[**LLM Architecture Speedrun**](/posts/LLM-Architecture/)。
>
> **本文目标**:从"为什么 LLM 需要 RL"开始,系统讲清现代 LLM RL 算法。
>
> 1. RLHF 怎么诞生:Christiano 2017 → InstructGPT 三阶段
> 2. RLHF 完整数学推导(SFT / Reward Model / PPO)
> 3. DPO 革命:把 RL 塌缩成监督学习
> 4. 推理时代:GRPO 与 R1-Zero
> 5. 其他 LLM RL 算法(RLOO、DAPO、REINFORCE++)
> 6. 多智能体与自博弈方向
>
> 全部带 formulation、数学推导和"为什么这么设计"的逻辑。

---
## 三、LLM 进入 RL 之前：为什么 LLM 突然需要 RL？

### 3.1 LLM 的三阶段简史（pre-RL）

要懂 RLHF，先要懂 LLM 训练原本长什么样。

**📋 记号约定（本节及全文统一使用，italic）**：

| 符号 | 含义 | 注意 |
|---|---|---|
| $u = (u_1, \ldots, u_{\lvert u \rvert})$ | **无结构文本文档**（pretraining 的训练样本，整段不分 prompt/response） | 只在 §2.1 pretraining 出现 |
| $x = (x_1, \ldots, x_{\lvert x \rvert})$ | **prompt**（指令 / 问题），token 序列 | SFT 起所有后续阶段都是这个含义 |
| $y = (y_1, \ldots, y_{\lvert y \rvert})$ | **response**（回答），token 序列 | SFT / RLHF / GRPO / RLOO 都用这个 |
| $y_t$ | $y$ 的第 $t$ 个 token | 标量 token |
| $y_{<t}$ | $y$ 的前 $t-1$ 个 token 前缀 | 用作 conditioning |
| $\pi_\theta(y \mid x)$ | 模型在 prompt $x$ 下生成完整 response $y$ 的联合概率 | 由 chain rule $\prod_t \pi_\theta(y_t \mid x, y_{<t})$ 得到 |
| $\pi_\theta(\cdot \mid x)$ | 模型在 $x$ 下的整段 response 分布（采样符号 $y \sim \pi_\theta(\cdot \mid x)$） | |
| $\beta$ | KL 正则系数 / temperature | 不同算法用法略不同（PPO-RLHF / DPO / GRPO 各自有 $\beta$） |
| $r(x, y)$ 或 $r_\phi(x, y)$ | reward（task reward 或 RM 分数） | 上下文区分 |
| $\mathrm{sigm}(z) = 1/(1+e^{-z})$ | **sigmoid（logistic）函数** | 出现在 BT 模型 / DPO loss 中。**本文不用 $\sigma$ 表示 sigmoid**（很多 ML 论文这么用），因为 $\sigma$ 在统计学里是标准差的标准记号，两个含义分用不同符号避免歧义。 |
| $\sigma, \sigma^2, \sigma_i$ | **标准差 / 方差** | 仅出现在 §6.1+ 的 variance 分析 / normalization 上下文（与统计学传统一致） |
| $V$, $\lvert V \rvert$ | **词表 (vocabulary)** 与其大小 | $V$ = token 词表集合，$\lvert V \rvert$ = 词表 size（Llama 系约 32k–128k，Qwen 系约 150k）。LLM 输出 LM head 是 $d_{\text{hidden}} \to \lvert V \rvert$ 的 linear，给每个 token 一个 logit |
| $d_{\text{hidden}}$ | **transformer 隐藏维度** | 每层 hidden state 的维数（GPT-2-XL 1600 / Llama-7B 4096 / Llama-70B 8192 等）。LM head / value head 的输入维度 |

**重要：避免一个常见 LLM 文献记号陷阱**，很多论文用同一个 $x$ 同时指"pretraining 中的整段文本"和"SFT 中的 prompt"。这两件事**结构上不同**（一个是完整文档，一个只是输入部分）。本文用**不同字母** $u$ 和 $x$ **显式区分**，避免混淆。

GRPO / 多智能体等后续章节会引入更多符号（$y^i$ for group-sampled response，$\pi_i$ for agent $i$'s policy 等），在引入处现场定义。

---

**阶段 A：自监督预训练 (Pretraining)**

- 目标：next-token prediction，最大化下一个 token 的对数似然
  $$
  \mathcal{L}\_{\text{pretrain}}(\theta) \;=\; -\sum\_{u \in \mathcal{D}\_{\text{pretrain}}} \sum\_{t=1}^{|u|} \log \pi\_\theta(u\_t \mid u\_{<t})
  $$
  其中 $u$ 是一个**完整的无结构文本文档**（一整篇文章 / 一段代码 / 一个网页），$u_t$ 是它的第 $t$ 个 token，$u_{<t}$ 是它的前 $t-1$ 个 token 前缀。
- 数据：海量互联网文本（CommonCrawl, Wikipedia, GitHub, ...），$10^{12}$–$10^{13}$ tokens
- 代表模型：GPT-2, GPT-3, LLaMA-base, ...
- 结果：模型学到了语言的"统计共现规律"，**会续写**，但**不会按指令做事**。给它 `"What is 2+2?"`，它可能续写 `"What is 2+2? What is 3+3? What is..."` 而不是回答 4。

**阶段 B：有监督微调 (SFT, Supervised Fine-Tuning)**
- 目标：在人写的 (prompt, response) 对上继续 MLE
  $$
  \mathcal{L}\_{\text{SFT}}(\theta) \;=\; -\mathbb{E}\_{(x, y) \sim \mathcal{D}\_{\text{SFT}}}\!\left[\log \pi\_\theta(y \mid x)\right]
  $$
  其中 $x$ 是 **prompt**（问题），$y$ 是对应的 **response**（回答）；$\pi_\theta(y \mid x)$ 是模型生成 $y$ 的整体序列概率，按 chain rule 拆为 $\pi_\theta(y \mid x) = \prod_{t=1}^{\lvert y \rvert} \pi_\theta(y_t \mid x, y_{<t})$。
- 数据：人工写的高质量 (问题, 答案) 配对（FLAN, T0, Alpaca, ShareGPT 风格），$10^4$–$10^6$ 对
- 结果：模型学会"听指令格式回答"，但**未必懂偏好**。

> **🤔 SFT 和 Pretraining 的区别**
>
> 两个 loss 都是 next-token cross-entropy / NLL，长得很像。先把公式拆到 token 级放一起对比，再列差别。
>
> **Pretraining**：
> $$\mathcal{L}\_{\text{pretrain}}(\theta) \;=\; -\sum\_{u \in \mathcal{D}\_{\text{pretrain}}} \sum\_{t=1}^{|u|} \log \pi\_\theta(u\_t \mid u\_{<t})$$
>
> **SFT**（用 chain rule 展开 $\log \pi_\theta(y \mid x) = \sum_{t=1}^{\lvert y \rvert} \log \pi_\theta(y_t \mid x, y_{<t})$）：
> $$\mathcal{L}\_{\text{SFT}}(\theta) \;=\; -\mathbb{E}\_{(x, y) \sim \mathcal{D}\_{\text{SFT}}}\!\left[\sum\_{t=1}^{|y|} \log \pi\_\theta(y\_t \mid x, y\_{<t})\right]$$
>
> 两者都是 next-token NLL（**这是相似的部分**），但有几处具体差别，分**公式层面**和**公式之外**两类。
>
> **差别 1（公式层面，求和范围 = loss masking）**
>
> | | Pretraining | SFT |
> |---|---|---|
> | 求和对象 | 数据集中每个文档 $u$ | 数据集中每个 (prompt, response) 对 $(x, y)$ |
> | 求和 token 范围 | 文档全部 $\lvert u \rvert$ 个 token（**每个位置都进 loss**） | 仅 response 的 $\lvert y \rvert$ 个 token（**prompt token 不进 loss**） |
> | 每个被预测 token 的 context | 同一文档自己的前缀 $u_{<t}$ | prompt $x$ + response 已生成前缀 $y_{<t}$ |
>
> 这就是 **loss masking**：SFT 把 prompt 部分 token 的 loss mask 为 0。具体例子：训练样本
> ```
> prompt:   "What is 2+2?"
> response: " 4"
> ```
> 拼接后是 `"What is 2+2? 4"`，~11 个 token。
> - **Pretraining 算 11 个 token 全部 loss**：每个位置都罚。模型学 $p(\text{text})$。
> - **SFT 只算 " 4" 那 1 个 token 的 loss**：前 10 个 prompt token 的位置 loss **mask 为 0**；只罚"看到 prompt 后，该不该说 4"。模型学 $p(\text{response} \mid \text{prompt})$。
>
> 这是 SFT 跟 pretraining 在**公式层面的核心差异**：SFT 不学怎么生成问题，只学怎么回答问题。
>
> **差别 2（公式之外，数据形态 + 量级 + curation）**
>
> | | Pretraining | SFT |
> |---|---|---|
> | 数据形态 | 无结构 raw text 流 | 结构化 (prompt, response) 对 |
> | 数据量 | $10^{12}$–$10^{13}$ tokens（TB 级） | $10^4$–$10^6$ 对（MB-GB 级） |
> | 数据来源 | 网络抓取（CommonCrawl, GitHub, Wikipedia） | 人工标注或精心 curated（FLAN, Alpaca, ShareGPT chat template） |
> | 训练成本 | $\sim 10^6$ GPU-小时（H100 计） | $\sim 10$–$10^3$ GPU-小时 |
>
> SFT 数据量比 pretraining 小 **6–9 个数量级**。这决定了 SFT 的角色不是"教模型新语言"，而是"调整输出分布"。
>
> > **🤔 "GPU-小时" 是什么？**
> >
> > **定义**：1 GPU-小时 = 1 张 GPU 工作 1 小时的算力消耗。它**不是**单卡训完的实际墙钟时间，而是**总算力 = GPU 数 × 工作小时数**。
> >
> > 例子：
> > - 8 张 H100 跑 10 小时 = $8 \times 10 = 80$ GPU-小时
> > - 1024 张 H100 跑 1 个月 (720 小时) = $1024 \times 720 \approx 7.4 \times 10^5$ GPU-小时
> >
> > **GPU 型号决定每 GPU-小时能算多少 FLOPs**。同样是 1 GPU-小时，不同卡的算力差几十倍：
> >
> > | GPU | bf16 峰值算力 (TFLOPS) | 相对 H100 | 一句话 |
> > |---|---|---|---|
> > | V100 (2017) | 125 | 0.13× | 老黄历 |
> > | A100 (2020) | 312 | 0.31× | 学术圈主力 |
> > | H100 (2022) | 989 | 1.0× | 行业基准 |
> > | H200 (2024) | 989（FLOPs 同 H100，HBM 加大） | 1.0× | 内存升级 |
> > | B200 (2024) | 2250 | 2.3× | 最新 SOTA |
> > | TPU v5p (Google) | 459 | 0.46× | 谷歌内部 |
> >
> > 所以引用"GPU-小时"数字时**必须说清是什么卡**，否则差几十倍。本文如无特别说明，**默认按 H100 bf16 算**。
> >
> > **粗略量级感**：
> > - 训 Llama-2-7B from scratch：$\approx 1.8 \times 10^5$ A100-小时（论文报告）≈ $5.6 \times 10^4$ H100-小时
> > - 训 Llama-3-70B from scratch：$\approx 6.4 \times 10^6$ H100-小时
> > - 训 GPT-4 (估计)：$\geq 5 \times 10^7$ A100-小时
> > - 7B 模型 SFT 一轮（10 万样本）：$\sim 50$–$200$ H100-小时
> > - 7B 模型 RLHF (PPO) 一轮：$\sim 500$–$2000$ H100-小时（比 SFT 贵 10× 主要因为 4 个模型同时活）
> >
> > 这就是为什么 pretraining 是 big-tech-only（千万美元到上亿美元 GPU 成本，每张 H100 \$2-3/小时 $\times 10^6$ 小时），而 SFT/RLHF 学术实验室也做得起（几万到几十万美元）。
>
> **差别 3（公式之外，起点 + 优化幅度）**
>
> | | Pretraining | SFT |
> |---|---|---|
> | 模型初始化 | 随机初始化（或从 earlier checkpoint） | **从 pretrained checkpoint 继续** |
> | 学到什么 | 语言统计规律 $p(\text{text})$ | 指令-回答格式 $p(\text{response} \mid \text{prompt})$ |
> | 输出行为 | 续写：给一段就接着写 | 应答：给问题就回答 |
> | 学习率 | 大（$\sim 10^{-4}$） | 小（$\sim 10^{-5}$ 到 $10^{-6}$） |
>
> SFT 用**小学习率 + 少量数据 + loss mask** 把 pretrained model 的输出分布**轻微 reshape 到指令格式**，又不能让它忘掉 pretraining 学到的语言能力，这就是 catastrophic forgetting 在 LLM training 里的具体表现。
>
> **总结**
>
> 三个差别共同决定了 SFT 的实际效果是"教模型应答而非续写"，把 general LM 调成 instruction-following LM。其中**差别 1（loss masking）是写在公式里的；差别 2 和 3 不在公式里、但同样关键**。
>
> 这种"loss 表达式接近，但被监督的对象 / 求和范围 / 训练设置不同 → 实际功能完全不同"的模式，后面会反复出现。例如 §4 的 DPO loss 也长得像 supervised cross-entropy，但通过精心选择被监督的对象（policy ratio 而非 token），让最优解恰好对应一个 RLHF 优化问题的解。**关键问题不是"loss 长什么样"，而是"loss 在监督哪个量、在什么数据上、训练到什么程度"**。

**阶段 B 之后的痛点**：

1. **有用性 (helpfulness)**：模型答得"正确"但没用。例：用户问 `"帮我写邮件道歉"`，模型答 `"道歉应当包含三要素..."`（正确但没写邮件）。
2. **无害性 (harmlessness)**：模型可能产出有害回答，SFT 没法显式惩罚。
3. **诚实性 / 风格**：模型会自信地胡说，SFT 没法捕捉"不要瞎编"的偏好。

**根本困境**：这三件事**没有可微的损失函数**。"有用"、"无害"、"诚实"都是模糊的人类概念，写不出 $\mathcal{L}(\theta)$。

### 3.2 关键洞察：偏好可以学，但不可微

**人类有一种比"写标准答案"容易得多的能力**：**给两个 response 选一个更好的**。这是个分类任务，可以标注。

**Christiano et al. 2017**（《Deep RL from Human Preferences》，NeurIPS）第一次把这点系统化用在 RL 上：

1. 训练 RL agent 时，让人类对若干 trajectory pairs 打偏好；
2. 用偏好数据训一个 reward model $r\_\phi$；
3. 用 RL 优化 $r\_\phi$。

这给 LLM 提供了出路：**先把"偏好"蒸馏成可微 reward model，再用 RL 优化它**。这就是 RLHF (Reinforcement Learning from Human Feedback) 的核心思想。

### 3.3 InstructGPT 的标准三阶段配方

**Ouyang et al. 2022**（InstructGPT，NeurIPS）把 Christiano 的思路落到 LLM 上，确立了 LLM RL 的事实标准：

```
Stage 1: Supervised Fine-Tuning
   π_SFT ← fine-tune base model on (prompt, demo) pairs

Stage 2: Reward Model training
   collect (prompt, y_w, y_l) preference triples
   train r_φ via Bradley-Terry loss

Stage 3: Reinforcement Learning (PPO)
   optimize π_θ to maximize r_φ(x, y), KL-regularized to π_SFT
```

后续所有 LLM RL 算法都是对这个三阶段的修改、简化、或重新解释。所以下一节我们详细拆解这三个 stage。

---

## 四、RLHF 三阶段：完整数学推导

> **RLHF = Reinforcement Learning from Human Feedback（基于人类反馈的强化学习）**；本章三阶段涉及 **SFT = Supervised Fine-Tuning（有监督微调）**、**RM = Reward Model（奖励模型）**、**PPO = Proximal Policy Optimization（近端策略优化）**。

### 4.1 Stage 1: SFT（最简单）

$$
\mathcal{L}_{\text{SFT}}(\theta) \;=\; -\mathbb{E}_{(x, y) \sim \mathcal{D}_{\text{SFT}}}\!\left[\log \pi_\theta(y \mid x)\right]
$$

展开成 token 级：

$$
\log \pi_\theta(y \mid x) \;=\; \sum_{t=1}^{T} \log \pi_\theta(y_t \mid x, y_{<t})
$$

完了。这是标准的 teacher forcing。$\pi\_\theta$ 从 pretrained base model 初始化，fine-tune 后得到 $\pi\_{\text{SFT}}$。**$\pi\_{\text{SFT}}$ 后面会作为 Stage 3 的 reference policy**。

### 4.2 Stage 2: Reward Model（Bradley-Terry）

**数据**：一批人工标注的 $(x, y\_w, y\_l)$ 三元组，$y\_w$ 是 winner，$y\_l$ 是 loser。

**Bradley-Terry 模型**（1952 年的经典模型）：假设存在一个 latent reward 函数 $r^*(x, y)$，则偏好概率为
$$
P(y_w \succ y_l \mid x) \;=\; \frac{\exp(r^*(x, y_w))}{\exp(r^*(x, y_w)) + \exp(r^*(x, y_l))}
\;=\; \mathrm{sigm}\!\left(r^*(x, y_w) - r^*(x, y_l)\right)
$$
其中 $\mathrm{sigm}(z) := \dfrac{1}{1 + e^{-z}} = \dfrac{e^z}{1 + e^z}$ 是 **sigmoid（logistic）函数**，把任意实数压到 $(0, 1)$，是概率值的标准激活。第二个等号来自分子分母同除 $\exp(r^*(x, y\_w))$ + 整理 + 用 sigmoid 定义重写。

**关键性质**：$r^*$ 只能确定到一个**加性常数**（$r^* + c$ 给出相同的偏好概率，因为 $(r^* + c)(y\_w) - (r^* + c)(y\_l) = r^*(y\_w) - r^*(y\_l)$，$c$ 抵消），所以 RM 输出的 scalar 本身没有绝对意义，**只有差值有意义**。

> **关于 sigmoid 记号**：很多 ML 论文（DPO 原文等）把 sigmoid 写作 $\sigma(\cdot)$。本文为避免与统计学里 $\sigma$ 表示标准差冲突（§6 会用到），统一用 $\mathrm{sigm}(\cdot)$ 表示 sigmoid，$\sigma$ 仅用于标准差。阅读外部论文时注意换算。

#### RM 的输入、输出、架构

Reward Model $r\_\phi$ 本质上是一个 **scalar regressor**：吃 (prompt, response) 输出一个数。

- **输入**：一对 $(x, y)$，把 prompt 和 response 拼接成单一 token 序列 $[x; y] = (x\_1, \ldots, x\_{|x|}, y\_1, \ldots, y\_{|y|})$
- **输出**：单个标量 $r\_\phi(x, y) \in \mathbb{R}$，表示"这个 $y$ 对这个 $x$ 来说有多好"。**没有绝对量纲**（见前面 BT 模型的加性常数性质），只有差值才有意义。
- **架构**（4 步）：
  1. **Transformer 主干**：通常直接**复制 $\pi\_{\text{SFT}}$ 的权重**作初始化（保留所有层），这样 RM 直接继承 SFT 学到的语言理解能力。
  2. **去掉原来的 LM head**（$d\_{\text{hidden}} \to |V|$ 的 linear 层；$|V|$ 是词表大小，对 Llama 系约 32k–128k；$d\_{\text{hidden}}$ 是 transformer 隐藏维度，对 Llama-7B 是 4096。LM head 原本用来给每个词表 token 出一个 logit，预测下一个 token 的分布）。
  3. **换上一个新的 value head**：$d\_{\text{hidden}} \to 1$ 的 linear 层，把 hidden state 投到单个 scalar。新建 + 随机初始化。
  4. **取最后一个 token 位置**（$y$ 末尾，整段序列的最后一个 token）的 hidden state $h^L\_{|x|+|y|}$，通过 value head 算 scalar：$r\_\phi(x, y) = W\_{\text{val}}\, h^L\_{|x|+|y|}$。

为什么取**最后一个 token** 的 hidden state？因为 transformer 是 causal 的，最后一个位置 attended over 了**整个 $(x, y)$ 序列**，是对整段最 informative 的表示。其他位置只看到部分序列。

示意图：

```
input tokens:   x_1   x_2   ...   x_{\lvert x \rvert}   y_1   y_2   ...   y_{\lvert y \rvert}
                 ↓     ↓            ↓         ↓     ↓            ↓
transformer:    [...  causal self-attention through all layers  ...]
                 ↓     ↓            ↓         ↓     ↓            ↓
hidden states:  h_1   h_2   ...   h_{\lvert x \rvert}   ...   ...   h_{\lvert x \rvert+\lvert y \rvert}
                                                              ↓
                                                      value head (Linear d→1)
                                                              ↓
                                                       r_φ(x, y) ∈ ℝ
```

#### RM 训练目标（最大化偏好数据的似然）

$$
\mathcal{L}_{\text{RM}}(\phi) \;=\; -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}_{\text{pref}}}\!\left[\log \mathrm{sigm}\!\left(r_\phi(x, y_w) - r_\phi(x, y_l)\right)\right]
$$

这就是 BT 模型在偏好数据上的 negative log-likelihood：让 winner 的 RM 分数高于 loser 的分数。

#### 训练流程：一个 batch 的具体步骤

```
Input: 一批 preference triples {(x^(b), y_w^(b), y_l^(b))}_{b=1}^B

# 对每个 triple 做两次 forward pass
for b = 1..B:
    score_w[b] = r_φ([x^(b); y_w^(b)])   # forward on winner，取末位 scalar
    score_l[b] = r_φ([x^(b); y_l^(b)])   # forward on loser，取末位 scalar

# 算 BT loss（per-triple sigmoid，然后 batch 平均）
loss = -(1/B) * Σ_b log sigm(score_w[b] - score_l[b])

# 标准 backprop 更新 φ（transformer 主干 + value head 一起）
loss.backward()
optimizer.step()
```

每个 batch 的核心计算：**两次 forward pass、一次 sigmoid + log、一次反传**。

注意：winner 和 loser 共享同一组参数 $\phi$，**不是两个独立模型**。是同一个 RM 在不同输入上的两次评估。

#### 完整 Stage 2 流水线

```
Stage 2 输入:
  - π_SFT   （来自 Stage 1 的 SFT 模型）
  - D_pref  （偏好数据集 {(x, y_w, y_l)}，人工或 LLM-as-judge 标注）

Stage 2 训练:
  1. 初始化 r_φ:
     - transformer 主干 ← 复制 π_SFT 的权重
     - value head      ← 新建 + 随机初始化的 Linear(d_hidden → 1)
  2. 训练 N epochs（典型 1-3 epoch）:
     for each batch (x, y_w, y_l):
         score_w = r_φ([x; y_w])
         score_l = r_φ([x; y_l])
         loss    = -mean(log sigm(score_w - score_l))
         backprop → update φ
  3. 训练结束 → r_φ 被 frozen

Stage 2 输出:
  - 训好的 r_φ : (x, y) → ℝ（scalar function，参数固定）
```

#### Stage 3（PPO）中怎么使用 RM

```
for each PPO update step:
    sample x from prompt distribution
    sample y ~ π_θ(·|x)      # current policy 生成 response
    reward = r_φ(x, y)        # 用 frozen RM 算这个 (x, y) 的 scalar reward
    ... 用这个 reward 走 PPO 更新（详见 §3.3）
```

**关键事实**：在 Stage 3 中 RM 是 **frozen** 的，只做 forward pass，**不再更新**。RM 的"知识"完全冻结在 Stage 2 训练结束时的状态。如果 RM 在某些 response distribution 上估不准，Stage 3 训练就会被这个不准带偏。这是 **reward hacking** 的来源之一（详见 §3.3 的 KL 约束讨论）。

#### 为什么这样设计：BT 偏好对比 vs 直接回归

为什么不直接监督学一个回归模型（"这个回答打 8 分"）？因为：

1. 人类对**绝对分**不稳定（同一个回答今天给 7 分，明天给 8 分；标注员之间方差大）
2. 人类对**相对偏好**稳定（A 比 B 好，基本不变；标注员一致率高）
3. BT 模型把"打分"问题等价转化为"比较"问题，绕开了人类标注的噪声
4. BT 让 RM 学到的 scalar **自动带有量级信息**：差值越大 → 偏好越强（虽然绝对值无意义，但**差值的相对大小**反映模型对偏好的置信度）

代价：RM 输出没有绝对量纲（只能比较），所以 Stage 3 的 reward 信号通常需要 normalize（PPO-RLHF 实现里用 batch-level $z$-score 把 reward 拉回 $\mathcal{O}(1)$）。

### 4.3 Stage 3: PPO-RLHF（最复杂、最关键）

**目标函数**：

$$
\max_\theta\ \mathbb{E}_{x \sim \mathcal{D},\ y \sim \pi_\theta(\cdot \mid x)}\!\left[r_\phi(x, y)\right]
\;-\; \beta \cdot \mathbb{E}_{x \sim \mathcal{D}}\!\left[\mathrm{KL}\!\left(\pi_\theta(\cdot \mid x) \,\|\, \pi_{\text{SFT}}(\cdot \mid x)\right)\right]
$$

**为什么有 KL 项**？这是 RLHF 最关键的设计：

1. **防 reward hacking**：$r\_\phi$ 不是 $r^*$（真实偏好），只是它的近似。如果 $\pi\_\theta$ 把 $r\_\phi$ 优化得太狠，模型会发现 $r\_\phi$ 的盲点并疯狂利用（典型例子：模型学会输出 `"As an AI language model..."` 因为 RM 给了高分）。KL 约束把 $\pi\_\theta$ 拴在 $\pi\_{\text{SFT}}$ 附近，避免漂太远。
2. **保持语言质量**：$\pi\_{\text{SFT}}$ 是说人话的，把 $\pi\_\theta$ 钉在它附近就保证语言不崩塌。
3. **理论解释**：这个目标其实等价于"在 KL ball 内最大化 reward"，是一种 trust region。

---

> **本节路线图**（从纯数学一路推到 PPO 工程公式，单向不回头）：
> - **4.3.1** 为什么这是 RL：把 LLM 生成显式写成 MDP
> - **4.3.2** 从目标推出 token-level policy gradient（whole-return 形式）
> - **4.3.3** 降方差：whole-return → per-token advantage（引出 value model）
> - **4.3.4** PPO 的血统：它来自 TRPO，不是 REINFORCE
> - **4.3.5** PPO 在 RLHF 的完整落地（reward / GAE / clip / 训练循环 / 成本）

#### 4.3.1 为什么这是 RL：先建立 MDP

到这里读者可能会问：**为什么这个目标必须用 RL？SFT 是监督学习、DPO 也变成监督学习，这里为什么突然就 RL 了？**

回答这个问题不能只看 $\mathbb{E}\_{y \sim \pi\_\theta}[r\_\phi(x, y)]$ 这个公式表面，必须把 **LLM 生成过程显式映射成一个 MDP**：说清楚 state / action / transition / reward 各是什么，然后才能说"这是 RL"。

**(1) LLM 生成过程 = 一个 token-level MDP**

回到 §1.1 的经典 MDP 五元组 $(\mathcal{S}, \mathcal{A}, P, r, \gamma)$。LLM 在 prompt $x$ 下生成 response $y = (y\_1, \ldots, y\_T)$ 的过程，对应一个**非常具体的 MDP**：

| MDP 要素 | LLM RLHF 里对应什么 | 备注 |
|---|---|---|
| **状态** $s\_t$ | $(x, y\_{<t})$：prompt + 已生成的 token 前缀 | 初始状态 $s\_0 = x$；状态包含完整历史，Markov 性成立 |
| **动作** $a\_t$ | 下一个 token $y\_t \in V$（从词表中选一个） | 动作空间 $\mathcal{A} = V$，大小 $\|V\| \approx 10^5$ |
| **转移** $P(s\_{t+1} \mid s\_t, a\_t)$ | **确定性**：把 $a\_t$ 拼到序列尾部，$s\_{t+1} = (x, y\_{<t}, a\_t) = (x, y\_{\leq t})$ | **无环境随机性**，唯一的随机性来自策略本身 |
| **奖励** $r(s\_t, a\_t)$ | 中间步: $r\_t = 0$；终止步 $T$（采到 EOS 或截断）: $r\_T = r\_\phi(x, y)$ | **稀疏终端 reward**：只有完整 response 写完才有 RM 评分 |
| **策略** $\pi\_\theta(a\_t \mid s\_t)$ | LLM 的 next-token 分布 $\pi\_\theta(y\_t \mid x, y\_{<t})$ | **就是 LLM 自己**，π 和模型是同一个东西 |
| **回合长度** $T$ | $\|y\|$（response 长度，到 EOS 或 max length 截断） | 一般 50–4000 |
| **折扣** $\gamma$ | RLHF 一般取 1（undiscounted） | 因为 reward 只在 episode 终端出现 |

这个 MDP 有三个特殊性质（已在 §1.4 提及）：

1. **确定性转移**：选了哪个 token 就是哪个状态，无环境噪声。整个 trajectory 的随机性全部来自策略 $\pi\_\theta$ 的采样。
2. **稀疏终端奖励**：所有中间 token 的 immediate reward = 0，只有最后一步有 $r\_\phi(x, y)$。这让它本质上接近一个 **contextual bandit**（context = prompt，action = whole response），但写成 token-level MDP 是为了让 PPO 的 per-token clip 等机制能套上去。
3. **动作空间巨大但离散**：$|V| \approx 10^5$，比经典控制问题大得多，但因为是离散的，reparameterization trick 用不了。

**(2) 为什么这就是 RL（不是 SL）**

把 §1.1 的经典 RL 目标搬过来：

$$
J_{\text{RL}}(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\Big[\sum_{t=0}^{T} \gamma^t r(s_t, a_t)\Big]
$$

在上面定义的 LLM-MDP 里（$\gamma=1$，只有终端 reward），这个目标退化为：

$$
J_{\text{RL}}(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\big[r_T\big] = \mathbb{E}_{y \sim \pi_\theta(\cdot \mid x)}\big[r_\phi(x, y)\big]
$$

这就是 §4.3 开头的 RLHF reward 项。**RLHF 的目标 = 经典 RL 期望累积回报在 LLM-MDP 上的特例**。

那为什么 SFT 不是 RL？因为 **SFT 根本没有这个 MDP**：

|  | SFT | RLHF |
|---|---|---|
| 数据来源 | 人写好的 $(x, y^*)$ 对，$y^*$ **与 $\pi\_\theta$ 无关** | rollout 自 $\pi\_\theta$ 自己：$y \sim \pi\_\theta(\cdot \mid x)$ |
| 监督信号 | 每个 token 有"正确答案" $y^*\_t$（teacher forcing） | 整段 response 完后才有一个 scalar reward，**无 token-level label** |
| 框架 | 普通 MLE：$\max\_\theta \mathbb{E}\_{(x,y^*) \sim \mathcal{D}}[\log \pi\_\theta(y^* \mid x)]$ | sequential decision-making：agent 自己生成 trajectory + credit assignment |
| 需要 MDP 吗 | **不需要**，数据集决定一切 | **需要**，agent-environment 交互产生数据 |

关键的"为什么不能用 SFT 解 RLHF"是：**RM 给的是整段 response 的 scalar 分数，没有逐 token 的 label**。要把这种 scalar 终端 reward 推回到每个 token 的更新方向，必须经过 RL 的 **credit assignment 机制**，这就是 policy gradient theorem 处理的事，SL 没有对应的工具。

**(3) MDP 形式化的数学后果：$\theta$ 出现在采样分布里**

定义了 MDP 后，立刻有一个数学结构上的后果。看两个目标的并列形式：

$$
\begin{aligned}
L_{\text{SL}}(\theta) &= \mathbb{E}_{(x,y) \sim \mathcal{D}}\big[\ell(f_\theta(x), y)\big] \quad\text{（$\theta$ 在 }\ell\text{ 内部）} \\[4pt]
J_{\text{RL}}(\theta) &= \mathbb{E}_{y \sim \pi_\theta(\cdot \mid x)}\big[r_\phi(x, y)\big] \quad\text{（$\theta$ 在采样分布 }\pi_\theta\text{ 内部）}
\end{aligned}
$$

**两者都有 $\theta$**，但出现位置不同：

|  | SL (ERM) | RL (policy optimization) |
|---|---|---|
| 采样分布 | 数据集 $\mathcal{D}$，**固定，与 $\theta$ 无关** | $\pi\_\theta$，**依赖 $\theta$**（agent 自己生成 trajectory） |
| 被积函数 | $\ell(f\_\theta(x), y)$ | $r(x, y)$（reward 是 frozen 的） |
| $\theta$ 在哪 | **integrand** 里 | **distribution** 里 |

为什么 RL 的 $\theta$ 跑到 distribution 里去了？正是因为 (a) 的 MDP 框架里 trajectory 是 agent **用自己的策略**采样出来的，**所以"采样分布"就是策略本身**，自然依赖于 $\theta$。这不是数学游戏，是 MDP 形式化的直接结果。

**关键差异在求梯度时立刻显出来**：

- **SL** 的梯度可以**直接推到积分号里面**（Leibniz 规则成立，因为 $\mathcal{D}$ 与 $\theta$ 无关）：
  $$\nabla_\theta L_{\text{SL}} = \mathbb{E}_{(x,y) \sim \mathcal{D}}\big[\nabla_\theta \ell(f_\theta(x), y)\big]$$
  这就是为什么 SL 可以用 backprop 一把搞定。

- **RL** 不能，因为 $\theta$ 在**概率密度**里。"改一次 $\theta$ → 采样分布变 → 期望重算"，必须用 **log-derivative trick** 绕一圈才能写成期望形式：见 4.3.2。

这是 RL 跟标准监督学习（ERM）在数学上的**根本分水岭**，**根源在于 (a) 的 MDP 框架要求 agent 自采样**。注意一些非主流 SL 子领域（如 VAE 的 encoder $q\_\phi(z \mid x)$）其实也有这种 "$\theta$ 在分布里" 的结构，但它们用 reparameterization trick 绕开了；LLM RL 因为**动作是离散 token**，reparameterization 用不了，只能用方差更大的 log-derivative trick。

> **🤔 为什么离散 token 上 reparameterization 用不了？**
>
> Reparameterization trick 的核心是把 $\theta$ 从采样分布移到被积函数里。**连续例子**（Gaussian policy，如 SAC）：
> $$z \sim \mathcal{N}(\mu_\theta, \sigma_\theta^2) \quad\Longleftrightarrow\quad z = \mu_\theta + \sigma_\theta \cdot \epsilon,\ \epsilon \sim \mathcal{N}(0, 1)$$
> 这样 $\theta$ 跑到 $z$ 的"表达式"里，采样分布 $\mathcal{N}(0, 1)$ 与 $\theta$ 无关，梯度可以直接 backprop 穿过 $f(z)$，方差低。
>
> **离散 token 失败原因**：LLM 采样 `y = sample(softmax(logits_θ))` 产出一个**整数索引**（或 one-hot 向量）。要做 reparameterization，需要 $y = g(\theta, \epsilon)$ 且 $g$ 关于 $\theta$ 几乎处处可微。但任何从连续 $(\theta, \epsilon)$ 映射到离散 $y$ 的函数，**必然包含阶梯/跳跃**，在大部分点是常数（梯度为 0），在边界点不连续。具体看 Gumbel-max trick：
> $$y = \arg\max_i \big\{\text{logit}_i(\theta) + g_i\big\},\quad g_i \sim \text{Gumbel}(0, 1)$$
> 形式上 $\theta$ 也跑外面了，但**致命问题在 argmax**：argmax 关于 logit 的梯度几乎处处为 0（小扰动不改变 argmax），梯度信号全丢。
>
> **Gumbel-Softmax / Concrete distribution 算是绕过去了，但 biased**：用 softmax($\cdot/\tau$) 替代 argmax 让它可微，但：
> - $\tau$ 小：接近真实 categorical，但梯度方差爆炸
> - $\tau$ 大：梯度平滑，但 $\tilde y$ 不是 one-hot，**采样分布偏离真实 categorical**
>
> 在 LLM 上这个偏差更致命：autoregressive 生成会让 soft mixture 作为下一步输入，整个 trajectory 变成"虚假的 soft vector 序列"，跟真实推理时的离散 token 序列完全不是一回事。RM 对 soft mixture 的打分与对真实 token 的打分关系未知。
>
> **所以只能退而求其次用 score function trick**：
> $$\nabla_\theta \mathbb{E}_{y \sim \pi_\theta}[r(y)] = \mathbb{E}_{y \sim \pi_\theta}\big[r(y) \cdot \nabla_\theta \log \pi_\theta(y)\big]$$
> 它对离散和连续都成立，**无偏**，但方差远高于 reparameterization。LLM RL 里所有"降方差技巧"（baseline / GAE / PPO clip / GRPO group baseline）本质上都在缓解这个副作用。
>
> **一句话**：连续 → reparameterization（低方差）；离散 → 只能 score function（高方差，必须配降方差技巧）。这是 LLM RL 工程上比连续控制 RL 更"难训"的数学根源。

#### 4.3.2 推导 token-level policy gradient

**(1) 工具：score function trick（log-derivative trick）**

经典 RL 给出关键恒等式：

$$
\nabla_\theta\, \mathbb{E}_{y \sim \pi_\theta(\cdot|x)}\!\left[f(y)\right]
\;=\; \mathbb{E}_{y \sim \pi_\theta}\!\left[f(y) \cdot \nabla_\theta \log \pi_\theta(y|x)\right]
$$

它把"对采样期望求梯度"转化为"采样后用 score function 加权"，这是 policy gradient theorem 的基础。

**(2) 应用到 RLHF 目标：reward 项 + KL 项**

reward 项：

$$
\nabla_\theta\, \mathbb{E}_{y \sim \pi_\theta}\!\left[r_\phi(x, y)\right]
\;=\; \mathbb{E}_{y \sim \pi_\theta}\!\left[r_\phi(x, y) \cdot \nabla_\theta \log \pi_\theta(y|x)\right]
$$

KL 项稍微 subtle，因为 $\theta$ 在 KL 表达式里**出现两次**：一次在采样分布 $\pi\_\theta$ 外面，一次在 $\log \pi\_\theta$ 里面。完整推导如下。

**Step 1：把 KL 展开成显式求和**（为简洁，省略 $x$）：

$$
\mathrm{KL}(\pi_\theta \,\|\, \pi_{\text{SFT}}) \;=\; \sum_y \pi_\theta(y) \cdot \big[\log \pi_\theta(y) - \log \pi_{\text{SFT}}(y)\big]
$$

**Step 2：对 $\theta$ 求梯度，用乘积法则**（$\theta$ 在 $\pi\_\theta(y)$ 和 $\log \pi\_\theta(y)$ 里都有）：

$$
\nabla_\theta\, \mathrm{KL} \;=\; \underbrace{\sum_y \nabla_\theta \pi_\theta(y) \cdot \big[\log \pi_\theta(y) - \log \pi_{\text{SFT}}(y)\big]}_{\text{(I)：$\theta$ 在分布里}} \;+\; \underbrace{\sum_y \pi_\theta(y) \cdot \nabla_\theta \log \pi_\theta(y)}_{\text{(II)：$\theta$ 在 log 里}}
$$

**Step 3a：项 (II) 直接消掉**（score function 零均值恒等式）：

$$
\sum_y \pi_\theta(y) \cdot \nabla_\theta \log \pi_\theta(y) \;=\; \sum_y \pi_\theta(y) \cdot \frac{\nabla_\theta \pi_\theta(y)}{\pi_\theta(y)} \;=\; \sum_y \nabla_\theta \pi_\theta(y) \;=\; \nabla_\theta \underbrace{\sum_y \pi_\theta(y)}_{=\,1} \;=\; 0
$$

这就是经典恒等式 $\mathbb{E}\_{y \sim \pi\_\theta}[\nabla\_\theta \log \pi\_\theta(y)] = 0$：**任何合法 policy 的 score function 都是零均值**。这也是 §1.2 里 baseline trick 不引入偏差的根本原因。

**Step 3b：项 (I) 用 log-derivative trick 转成期望形式**。关键恒等式：

$$
\nabla_\theta \pi_\theta(y) \;=\; \pi_\theta(y) \cdot \nabla_\theta \log \pi_\theta(y) \qquad\text{(等价于 } \nabla \log f = \nabla f / f \text{ 反过来用)}
$$

代入 (I)：

$$
(\text{I}) \;=\; \sum_y \pi_\theta(y) \cdot \nabla_\theta \log \pi_\theta(y) \cdot \big[\log \pi_\theta(y) - \log \pi_{\text{SFT}}(y)\big] \;=\; \mathbb{E}_{y \sim \pi_\theta}\!\left[\log\frac{\pi_\theta(y)}{\pi_{\text{SFT}}(y)} \cdot \nabla_\theta \log \pi_\theta(y)\right]
$$

**Step 4：合并** $(I) + (II) = (I) + 0$，把 $x$ 写回来：

$$
\nabla_\theta\, \mathrm{KL}(\pi_\theta \,\|\, \pi_{\text{SFT}})
\;=\; \mathbb{E}_{y \sim \pi_\theta}\!\left[\log\frac{\pi_\theta(y|x)}{\pi_{\text{SFT}}(y|x)} \cdot \nabla_\theta \log \pi_\theta(y|x)\right]
$$

> **直觉**：KL 梯度长得像"**有效 reward = $\log \frac{\pi\_\theta}{\pi\_{\text{SFT}}}$**"的 policy gradient。这就是为什么 KL 项最终能跟 RM reward 项**合并成一个单一的 effective reward**：它们俩用的是同一种 score-function 结构。
>
> **关键技巧回顾**：能把 KL 写成期望形式靠两件事。**(II) 凭 $\sum \pi = 1$ 自动消掉**（score function 零均值）；**(I) 凭 log-derivative trick 转成 $\mathbb{E}[\cdot \cdot \nabla \log \pi]$**。这两步在 RL 推导里反复出现（policy gradient theorem、baseline 不引偏差、PPO clip 推导等等），值得记牢。

合并：

$$
\nabla_\theta J(\theta)
\;=\; \mathbb{E}_{y \sim \pi_\theta}\!\left[\,\underbrace{\left(r_\phi(x, y) - \beta \log \frac{\pi_\theta(y|x)}{\pi_{\text{SFT}}(y|x)}\right)}_{=:\ \tilde r(x, y),\ \text{"有效 reward"}} \cdot \nabla_\theta \log \pi_\theta(y|x)\,\right]
$$

**这就是一个标准 policy gradient**。形式上和经典 RL 完全一致，只是"有效 reward" 由 RM 减去 KL 惩罚组成。从这里我们已经可以写出最朴素的 REINFORCE 更新。

**(3) sequence-level 拆成 token-level**

到这里我们有了 sequence-level 的 policy gradient（(2) 末尾那个 $\nabla J = \mathbb{E}[\tilde r \cdot \nabla \log \pi\_\theta(y \mid x)]$）。现在要把整段的 $\log \pi\_\theta(y \mid x)$ 拆成每个 token 的 $\sum\_t \log \pi\_\theta(y\_t \mid x, y\_{<t})$。

读者的合理疑问：**LLM 的"策略"看起来依赖自己以前输出的所有 action $y\_{<t}$，这不是 non-Markovian / history-dependent policy 吗？把它当 MDP 还合法吗？token-level 那个 $\sum\_t \nabla \log \pi$ 究竟怎么严格推出来？**

**先解决合法性疑问（state augmentation 思想）**

在 (a) 的 MDP 公式里，**状态本身就包含历史**：$s\_t = (x, y\_{<t})$。所以

$$
\pi_\theta(y_t \mid x, y_{<t}) \;=\; \pi_\theta(y_t \mid s_t)
$$

**仍然是 Markovian 的**：只依赖 $s\_t$，不显式依赖更早的状态。"history-dependent policy" 是从外部小状态空间（比如只看 "$y\_t$ 依赖前 t-1 个 token"）看的视角；MDP 里**把历史塞进状态本身**就化解了这个问题。

这是经典 trick：**state augmentation**。POMDP → MDP via belief state 也是同一个思想（把"看不见的隐状态信念"塞进状态）。代价是状态空间随 $t$ 指数增长（每步多 $\|V\|$ 种可能），但 RL 推导仍然合法。

LLM-MDP 本质上是一个 **tree-structured MDP**：从 $s\_0 = x$ 出发，每个 action 让 state 走向树的一个新分支，因为转移确定，每条 trajectory 走树上一条独立路径，**不同 trajectory 永远不汇合**。

> **🔖 提醒：$\tilde r$ 含 $\theta$，但下面当 scalar 处理**。$\tilde r(x, y) = r\_\phi(x, y) - \beta \log \frac{\pi\_\theta(y|x)}{\pi\_{\text{SFT}}(y|x)}$ 形式上含 $\log \pi\_\theta$，严格说依赖 $\theta$。但 (2) 已经证明：直接对 $J = \mathbb{E}\_{y \sim \pi\_\theta}[\tilde r(x, y; \theta)]$ 求导时，"$\theta$ 在 $\tilde r$ 里"那一项等于 $-\beta \mathbb{E}\_{y \sim \pi\_\theta}[\nabla \log \pi\_\theta] = 0$（score function 零均值，即 (2) 里 KL 推导的 Step 3a）。所以最终 policy gradient 里 $\tilde r$ 就是个 scalar，下面对它不再求 $\theta$ 梯度。

**Step 1：起点**。(2) 末尾给出的 sequence-level policy gradient，等价地写成对联合概率求和的展开式：

$$
\nabla_\theta J \;=\; \mathbb{E}_{y \sim \pi_\theta}\!\left[\tilde r \cdot \nabla_\theta \log \pi_\theta(y \mid x)\right]
\;=\; \sum_y \nabla_\theta \pi_\theta(y \mid x) \cdot \tilde r(x, y)
$$

要拆 token-level，核心是算**整段联合概率** $\pi\_\theta(y \mid x)$（$\|V\|^T$ 维空间上一个点的质量）对 $\theta$ 的导数如何按 $t$ 分解。

**Step 2：用 autoregressive chain rule 展开联合概率**

按概率论的 chain rule（与 LLM 的 autoregressive forward pass 等价）：

$$
\pi_\theta(y \mid x) \;=\; \pi_\theta(y_1, y_2, \ldots, y_T \mid x) \;=\; \prod_{t=1}^T \pi_\theta(y_t \mid x, y_{<t})
$$

**关键事实**：每一步的 $\pi\_\theta(y\_t \mid x, y\_{<t})$ 都是**同一个 $\theta$**（同一个 LLM）的 next-token 分布。这是 transformer 的 **weight sharing across time**。

**Step 3：对乘积求梯度（核心，分 4 小步）**

要算 $\nabla\_\theta \prod\_{t=1}^T \pi\_\theta(y\_t \mid x, y\_{<t})$。为简化记号，记 $g\_t(\theta) := \pi\_\theta(y\_t \mid x, y\_{<t})$，即第 $t$ 步的 next-token 条件概率（标量函数）。

**Step 3a：乘积法则展开**

经典乘积法则：

$$
\nabla_\theta \prod_{t=1}^T g_t(\theta) \;=\; \sum_{t=1}^T \Big[\prod_{\substack{s=1 \\ s \neq t}}^T g_s(\theta)\Big] \cdot \nabla_\theta g_t(\theta)
$$

意思是：**$T$ 项之和**，每一项对应"对第 $t$ 个因子求梯度 $\nabla g\_t$，其余 $T-1$ 个因子保持其当前值并相乘"。

**Step 3b：把每一项里"缺一项的乘积"补回成"完整乘积"**

对每一项乘除 $g\_t / g\_t = 1$：

$$
\Big[\prod_{\substack{s=1 \\ s \neq t}}^T g_s\Big] \cdot \nabla g_t \;=\; \frac{g_t \cdot \prod_{\substack{s=1 \\ s \neq t}}^T g_s}{g_t} \cdot \nabla g_t \;=\; \frac{\prod_{s=1}^T g_s}{g_t} \cdot \nabla g_t
$$

**关键变化**：分子的求积范围从 "$s=1$ 到 $T$ 但跳过 $t$"（$T-1$ 项）变成 "$s=1$ 到 $T$ 全部"（$T$ 项）。这个 $\prod\_{s=1}^T g\_s$ 正好是整段联合概率：

$$
\prod_{s=1}^T g_s \;=\; \prod_{s=1}^T \pi_\theta(y_s \mid x, y_{<s}) \;=\; \pi_\theta(y \mid x)
$$

代入 Step 3a：

$$
\nabla_\theta \prod_{t=1}^T g_t \;=\; \sum_{t=1}^T \frac{\pi_\theta(y \mid x)}{g_t} \cdot \nabla g_t
$$

**Step 3c：把 $\pi\_\theta(y \mid x)$ 从 $\sum\_t$ 里提出来**

这一步是读者最容易卡的地方。问题：**$\pi\_\theta(y \mid x) = \prod\_{s=1}^T g\_s$ 这个完整 T 项乘积真的能提到 $\sum\_t$ 外面吗？**

**能**。原因：$\pi\_\theta(y \mid x)$ 是"整条 trajectory $y$ 的联合概率"，**它的值只跟 trajectory $y$ 本身有关，跟外层求和指标 $t$ 完全无关**（$t$ 只是计数器，遍历哪个 token 被求梯度）。所以在 $\sum\_t$ 的每一项里，$\pi\_\theta(y \mid x)$ 都是**完全相同的标量**，可以当公因子提出来。

**用 $T = 3$ 显式写出来看最清楚**：

$$
\sum_{t=1}^{3} \frac{\pi_\theta(y \mid x)}{g_t} \cdot \nabla g_t \;=\; \underbrace{\frac{\pi_\theta(y \mid x)}{g_1} \nabla g_1}_{t=1} \;+\; \underbrace{\frac{\pi_\theta(y \mid x)}{g_2} \nabla g_2}_{t=2} \;+\; \underbrace{\frac{\pi_\theta(y \mid x)}{g_3} \nabla g_3}_{t=3}
$$

三项里的 $\pi\_\theta(y \mid x)$ **都是同一个数**（同一条 trajectory $y$ 的联合概率），提取公因子：

$$
\;=\; \pi_\theta(y \mid x) \cdot \left[\frac{\nabla g_1}{g_1} \;+\; \frac{\nabla g_2}{g_2} \;+\; \frac{\nabla g_3}{g_3}\right] \;=\; \pi_\theta(y \mid x) \cdot \sum_{t=1}^{3} \frac{\nabla g_t}{g_t}
$$

对一般 $T$ 同理：

$$
\nabla_\theta \prod_{t=1}^T g_t \;=\; \pi_\theta(y \mid x) \cdot \sum_{t=1}^{T} \frac{\nabla g_t}{g_t}
$$

**Step 3d：内层用 log-derivative trick 化简**

$\nabla \log f = \nabla f / f$ 反过来用：$\frac{\nabla g\_t}{g\_t} = \nabla \log g\_t$。代入：

$$
\boxed{\;\nabla_\theta \pi_\theta(y \mid x) \;=\; \pi_\theta(y \mid x) \cdot \sum_{t=1}^{T} \nabla_\theta \log \pi_\theta(y_t \mid x, y_{<t})\;}
$$

> **🔍 更简捷的等价证法（sanity check）**：
>
> 取 log 后乘积变求和：$\log \pi\_\theta(y \mid x) = \sum\_{t=1}^T \log \pi\_\theta(y\_t \mid x, y\_{<t})$。
>
> 用 $\nabla \log f = \nabla f / f$ 反过来用（两边乘 $\pi\_\theta(y \mid x)$）：
> $$\nabla_\theta \pi_\theta(y \mid x) \;=\; \pi_\theta(y \mid x) \cdot \nabla_\theta \log \pi_\theta(y \mid x) \;=\; \pi_\theta(y \mid x) \cdot \sum_{t=1}^T \nabla_\theta \log \pi_\theta(y_t \mid x, y_{<t})$$
>
> 跟 Step 3a-3d 的乘积法则路径结果**完全一致**。这条更短，但要求读者已经习惯 log-derivative trick。Step 3a-3d 的乘积法则路径虽然啰嗦，但每一步都是微积分基本规则，更易接受。

**Step 4：代回原式得到 token-level policy gradient**

把 Step 3 的结果代入 Step 1：

$$
\nabla_\theta J \;=\; \sum_y \pi_\theta(y \mid x) \cdot \tilde r(x, y) \cdot \sum_{t=1}^T \nabla_\theta \log \pi_\theta(y_t \mid x, y_{<t})
$$

把外层求和写回期望形式：

$$
\boxed{\;\nabla_\theta J \;=\; \mathbb{E}_{y \sim \pi_\theta(\cdot \mid x)}\!\left[\,\tilde r(x, y) \cdot \sum_{t=1}^T \nabla_\theta \log \pi_\theta(y_t \mid x, y_{<t})\,\right]\;}
$$

整段 reward 乘上"每个 token 的 score function 之和"。这就是 LLM token-level policy gradient 的严格表达。

---

> **🤔 求梯度时 $y\_{<t}$ 是常数还是变量？**
>
> 这是这个推导最容易出错的地方。求 $\nabla\_\theta \log \pi\_\theta(y\_t \mid x, y\_{<t})$ 时，conditioning 里的 $y\_{<t}$ 是**固定的、已采样的 token 前缀**，**不是 $\theta$ 的函数**。
>
> 为什么？因为我们求梯度的对象是"在给定一条具体 trajectory $y = (y\_1, \ldots, y\_T)$ 下的对数概率"。这条 trajectory 是被**外层 $\mathbb{E}\_{y \sim \pi\_\theta}$** 采样出来的：
> - 采样过程的随机性确实依赖 $\theta$（这就是为什么外层期望前需要 score function trick）
> - 但**一旦采样完成，trajectory 就是一条确定的 token 序列**，conditioning 里的 $y\_{<t}$ 就是这条 trajectory 里前 $t-1$ 个**字面常量 token**，与 $\theta$ 无关
>
> 直观看：做 1 次 rollout 得到 $x =$ `"1+1 = ?"`，$y =$ `"The answer is 2."`，然后对每个 token 的 log probability 求 $\theta$ 梯度。第 4 个 token 是 `"is"`，conditioning 是 `"1+1 = ? The answer"`，这 4 个 conditioning token 在你计算 $\nabla\_\theta \log \pi\_\theta(\text{"is"} \mid \text{"... answer"})$ 时就是字面常量。
>
> **如果错误地把 $y\_{<t}$ 也当作 $\theta$ 的函数**，会陷入无穷链：$\theta$ 影响 $y\_{<t}$ → $y\_{<t}$ 又作为 conditioning 影响 $\pi\_\theta(y\_t \mid \cdot)$ → 影响下一个 $y$ → ……
>
> **外层 $\mathbb{E}\_{y \sim \pi\_\theta}$ 已经在更高层 capture 了"$\theta$ 影响采样分布"这件事**，所以内部的 $\nabla\_\theta \log \pi\_\theta(y\_t \mid x, y\_{<t})$ 只需要对 $\pi\_\theta$ 本身求导，不需要再追究 $y\_{<t}$ 跟 $\theta$ 的关系。这一点跟经典 RL policy gradient theorem 的推导完全一致（§1.2）。

---

**小结 (4.3.2)**：上面推出的 token-level policy gradient

$$
\nabla_\theta J \;=\; \mathbb{E}_{y \sim \pi_\theta}\!\left[\tilde r(x, y) \cdot \sum_{t=1}^T \nabla_\theta \log \pi_\theta(y_t \mid x, y_{<t})\right]
$$

是**经典 policy gradient theorem 的 whole-return（REINFORCE）形式在 LLM-MDP 上的特例**：无偏，每个 token 共享同一个整段回报 $\tilde r(x, y)$。RL 不是"硬塞进 LLM"的，是 4.3.1 那个 MDP 通过 chain rule 自然导出的。

> **🧭 承上启下：接下来要解决两个正交的问题（别混成一条路）**
>
> 4.3.2 给出的 policy gradient 是一个**骨架**：$\nabla J = \mathbb{E}[\sum\_t \nabla \log \pi\_\theta(y\_t \mid x, y\_{<t}) \cdot (\textbf{权重})]$。把它真正用到 LLM 上，会撞到**两个独立的痛点**，需要两套来源不同的工具，最后在 PPO 里合并。
>
> | 痛点 | 问题 | 解决工具 | 来自哪条线 | 在哪节 |
> |---|---|---|---|---|
> | **A. 权重方差太大** | whole-return 让每个 token 共享整段 $\tilde r$，方差爆炸 | per-token advantage $A\_t$（value baseline + GAE） | **policy gradient 线**（4.3.2 的延续） | **4.3.3** |
> | **B. rollout 用一次就丢** | vanilla PG 严格 on-policy，一批样本只能走一步；LLM rollout 太贵 | importance ratio $\rho\_t$ + clip（复用 K 步、限制漂移） | **TRPO 线**（和 PG 不同的派系） | **4.3.4** |
>
> **关键：A 和 B 是正交的两件事**，不是"二选一走哪条路"。A 回答"每个 token 的梯度乘多大权重"，B 回答"怎么用一批昂贵样本安全地走多步"。**PPO 的 loss 里同时有这两个零件**：
> $$L^{\text{PPO}} = \mathbb{E}\Big[\sum_t \min\big(\underbrace{\rho_t}_{\text{零件 B}}\,\underbrace{A_t}_{\text{零件 A}},\ \mathrm{clip}(\underbrace{\rho_t}_{\text{零件 B}}, 1\pm\varepsilon)\,\underbrace{A_t}_{\text{零件 A}}\big)\Big]$$
> 4.3.3 造零件 A，4.3.4 造零件 B，4.3.5 把它们拼起来。两条线能咬合，是因为 TRPO surrogate 在第一步梯度（$\theta=\theta\_{\text{old}}$）退化成 vanilla PG，**它需要一个 advantage 来当权重，而那个 advantage 正是 4.3.3 提供的**。

#### 4.3.3 降方差：从 whole-return 到 per-token advantage（造零件 A：权重 $A\_t$）

> **本节目标**：解决痛点 A。停留在 policy gradient 框架内，把骨架里的"权重"从高方差的整段 $\tilde r$ 换成低方差的 per-token advantage $A\_t$。**本节完全不碰 TRPO / importance ratio / clip**，那些是 4.3.4 的事。

**(1) 问题：whole-return 形式方差太大**

对比 §1.2 经典 policy gradient theorem 和 4.3.2 推出的 LLM 形式：

$$
\underbrace{\nabla_\theta J = \mathbb{E}\!\Big[\textstyle\sum_t \nabla \log \pi_\theta(a_t \mid s_t) \cdot Q^\pi(s_t, a_t)\Big]}_{\text{经典：每步乘自己的 }Q^\pi}
\qquad
\underbrace{\nabla_\theta J = \mathbb{E}\!\Big[\tilde r(x, y) \cdot \textstyle\sum_t \nabla \log \pi_\theta(y_t \mid x, y_{<t})\Big]}_{\text{4.3.2：每步共享整段 }\tilde r}
$$

代入 $s\_t = (x, y\_{<t})$, $a\_t = y\_t$ 后结构一致，区别只在加权项。两者**期望相等，但不是同一个估计量**：

$$
Q^\pi(s_t, a_t) \;=\; \mathbb{E}\big[\tilde r \mid s_t, a_t\big] \;\neq\; \tilde r(x, y)
$$

$Q^\pi(s\_t, a\_t)$ 是**给定前缀后对 reward 的条件期望**（对未来 rollout 的平均），$\tilde r(x, y)$ 是**某一次采样到的整段回报**。期望相等（$\mathbb{E}[Q^\pi] = \mathbb{E}[\tilde r]$），但 whole-return 形式把整段的随机性全压在每个 token 的加权上，**方差大得多**。

> ⚠️ **不要误以为"LLM 只有终端 reward 所以 $Q$ 跟 $t$ 无关、所有 token 共享一个数"**。这只在"把全部 reward（含 KL）折到终端"这一种 reward shaping 下成立。标准 PPO 实现把 KL penalty **按 token 分摊**（见 4.3.5 token-level reward 设计），此时中间 token 的 $r\_t \neq 0$，$Q^\pi(s\_t, a\_t)$ **确实依赖 $t$**（早 token 后面还要累积更多 KL）。这正是 whole-return 形式（让所有 token 共享一个 sequence-level $\tilde r$）的根本弱点：**无法区分哪个 token 真正贡献了好坏（credit assignment），且方差极大**。

**(2) 四步降方差：把 $\tilde r$ 换成 per-token advantage $A\_t$**

把 4.3.2 的 whole-return 起点里的 $\tilde r$ 显式写成 per-token reward 之和 $\sum\_{t'=1}^T r\_{t'}$（$r\_{t'}$ 含 RM 终端项 + 每步 KL penalty，定义见 4.3.5）：

$$
\nabla_\theta J \;=\; \mathbb{E}\!\left[\Big(\textstyle\sum_{t'=1}^T r_{t'}\Big) \cdot \sum_{t=1}^T \nabla_\theta \log \pi_\theta(y_t \mid x, y_{<t})\right]
$$

**第 1 步，causality（rewards-to-go）**：token $t$ 的动作只影响 $t$ 及之后的 reward，不可能影响过去的 $r\_{t'}\ (t' < t)$。$t$ 之前的 reward 项与该 token 动作无关，其梯度贡献在期望下为 0（同 baseline 论证）。于是 whole-return $\sum\_{t'=1}^T r\_{t'}$ 可换成 **reward-to-go** $R\_t = \sum\_{t' \geq t} r\_{t'}$：

$$
\nabla_\theta J \;=\; \mathbb{E}\!\left[\sum_{t} \nabla_\theta \log \pi_\theta(y_t \mid x, y_{<t}) \cdot R_t\right], \qquad R_t = \textstyle\sum_{t' \geq t} r_{t'}
$$

无偏，方差已降（去掉无关的早期 reward 噪声）。

**第 2 步，换成条件期望 $Q^\pi$**：$R\_t$ 是 reward-to-go 的一次采样，其条件期望就是 action-value $Q^\pi(s\_t, a\_t) = \mathbb{E}[R\_t \mid s\_t, a\_t]$。policy gradient theorem 保证用 $Q^\pi$ 替代采样 $R\_t$ 仍无偏、方差更低：

$$
\nabla_\theta J \;=\; \mathbb{E}\!\left[\sum_t \nabla_\theta \log \pi_\theta(y_t \mid x, y_{<t}) \cdot Q^\pi(s_t, a_t)\right]
$$

**第 3 步，减 baseline 得 advantage**（§1.2 的 baseline trick）：减去只依赖状态的 $V^\pi(s\_t)$ 不引入偏差（score function 零均值），进一步降方差：

$$
A^\pi(s_t, a_t) = Q^\pi(s_t, a_t) - V^\pi(s_t), \qquad
\nabla_\theta J = \mathbb{E}\!\left[\sum_t \nabla_\theta \log \pi_\theta(y_t \mid x, y_{<t}) \cdot A^\pi(s_t, a_t)\right]
$$

> **🔑 这就是 PPO 里 value model $V\_\psi$ 的来源**：要算 $A\_t = Q^\pi - V^\pi$，必须估计 $V^\pi(s\_t)$，于是引入一个单独训练的 value model $V\_\psi(s\_t)$。4.3.2 的 whole-return 形式里没有 $V$，是因为它把 $Q^\pi$ 用采样回报 $\tilde r$ 顶替了（高方差版）。一旦想走 advantage 路线降方差，$V\_\psi$ 就必须出现。

**第 4 步，GAE 估计 $A\_t$**（bias-variance 折中）：实际中 $Q^\pi, V^\pi$ 都未知，用 TD residual $\delta\_t = r\_t + \gamma V\_\psi(s\_{t+1}) - V\_\psi(s\_t)$ 的指数加权和估计 advantage（具体公式见 4.3.5）。

**降方差链**：

```
whole-return (4.3.2):  ∇J = E[ (Σ_{t'} r_{t'}) · Σ_t ∇log π_t ]   ← 每 token 共享整段回报，方差大
  ↓ 第1步 causality（去无关早期 reward）
reward-to-go:      ∇J = E[ Σ_t ∇log π_t · R_t ]
  ↓ 第2步 条件期望（policy gradient theorem）
Q 形式:            ∇J = E[ Σ_t ∇log π_t · Q^π(s_t,a_t) ]
  ↓ 第3步 减 baseline → 引入 value model V_ψ
advantage 形式:    ∇J = E[ Σ_t ∇log π_t · A^π(s_t,a_t) ]       ← 每 token 自己的 advantage
  ↓ 第4步 GAE 估 A_t
```

每步无偏、逐步降方差。**至此零件 A（权重 $A\_t$）造好了，credit assignment 问题解决。** 但痛点 B 还没碰：这套 advantage-weighted PG 仍然**严格 on-policy**，一批 rollout 走完一步梯度就得扔掉重采。LLM rollout 太贵，必须能复用。这就要装零件 B（surrogate + clip），而它来自和 PG 不同的派系，即 TRPO。下一节讲清楚。

#### 4.3.4 PPO 的血统：从 TRPO 来，不是从 REINFORCE（造零件 B：更新机制）

> **本节目标**：解决痛点 B。零件 A（4.3.3 的 advantage $A\_t$）已经备好，本节造零件 B，一套能"用一批旧 rollout 安全走 K 步"的更新机制。注意这套机制的理论根基**不是** 4.3.2 的 policy gradient theorem，而是 TRPO 的 performance difference lemma，是**另一条 lineage**。它不替代零件 A，而是把零件 A 包进一个可复用样本的壳里。
>
> ⚠️ **常见误解**：很多教程把 PPO 描述成 "REINFORCE 加 patch"。这话字面没错（PPO 确实缓解 REINFORCE 的工程痛点），但**误导血统**。PPO **不是** 从 REINFORCE 改良来的，**而是从 TRPO 简化来的**，理论基础是另一条完全不同的推导线。

**(1) 两条不同的推导线**

| | REINFORCE / vanilla PG 线 | TRPO / PPO 线 |
|---|---|---|
| **优化对象** | 直接最大化 $J(\theta)$ 本身 | 最大化 **surrogate** $L\_{\pi\_{\text{old}}}(\theta) = \mathbb{E}[\rho \cdot A^{\pi\_{\text{old}}}]$ |
| **梯度公式** | $\nabla J = \mathbb{E}[\nabla \log \pi\_\theta \cdot Q]$ | 直接对 surrogate 优化（不是 $\nabla J$） |
| **数学起点** | log-derivative trick + policy gradient theorem (Sutton 2000) | performance difference lemma (Kakade & Langford 2002) |
| **保证** | 一阶随机梯度（无 monotonic improvement 保证） | TRPO 有 monotonic improvement bound；PPO 有近似保证 |
| **样本复用** | 一条 rollout 用一次就丢（严格 on-policy） | 一条 rollout 可用 K 次梯度更新（surrogate 容许小范围 off-policy） |

**两条线在 $\theta = \theta\_{\text{old}}$ 处一阶等价**：
$$
\nabla_\theta L_{\pi_{\text{old}}}(\theta)\Big|_{\theta = \theta_{\text{old}}} \;=\; \mathbb{E}\!\left[\nabla_\theta \log \pi_\theta(a \mid s) \cdot A(s, a)\right] \;=\; \text{standard PG}
$$

即 vanilla PG 是 TRPO surrogate 在 $\theta = \theta\_{\text{old}}$ 的 first-order 极限。但走多步以后（$\theta \neq \theta\_{\text{old}}$），两条线**优化的目标本质上不同**。

**(2) TRPO 的核心推导：performance difference lemma → surrogate**

TRPO 的起点不是 $\nabla J$，是 **policy improvement bound**（Kakade-Langford 2002）：

$$
J(\pi_\theta) - J(\pi_{\text{old}}) \;=\; \mathbb{E}_{s \sim d^{\pi_\theta},\ a \sim \pi_\theta}\!\big[A^{\pi_{\text{old}}}(s, a)\big]
$$

这是**等式**，不是梯度。但右边的 $d^{\pi\_\theta}$（新策略的 state visitation distribution）**我们采样不到**：要先执行 $\pi\_\theta$ 才能拿到它，但 $\pi\_\theta$ 正是我们要优化的对象，鸡生蛋问题。

**TRPO 的关键 trick**：分两层处理 $\mathbb{E}\_{s \sim d^{\pi\_\theta},\ a \sim \pi\_\theta}$。

**第一步（对 action 做 importance sampling，精确）**：固定 state $s$，对 action 换测度：

$$
\mathbb{E}_{a \sim \pi_\theta}\big[A^{\pi_{\text{old}}}(s, a)\big] \;=\; \mathbb{E}_{a \sim \pi_{\text{old}}}\!\left[\frac{\pi_\theta(a \mid s)}{\pi_{\text{old}}(a \mid s)} A^{\pi_{\text{old}}}(s, a)\right]
$$

这一步**严格成立**（标准 IS，只要 support 覆盖）。代入后：

$$
J(\pi_\theta) - J(\pi_{\text{old}}) \;=\; \mathbb{E}_{s \sim d^{\pi_\theta}}\,\mathbb{E}_{a \sim \pi_{\text{old}}}\!\left[\frac{\pi_\theta}{\pi_{\text{old}}} A^{\pi_{\text{old}}}\right] \qquad\text{（仍是等式，state 还是 }d^{\pi_\theta}\text{）}
$$

**第二步（把 state 分布也换成旧的，这一步是近似）**：

$$
L_{\pi_{\text{old}}}(\theta) \;:=\; \mathbb{E}_{s \sim d^{\pi_{\text{old}}},\ a \sim \pi_{\text{old}}}\!\left[\underbrace{\frac{\pi_\theta(a \mid s)}{\pi_{\text{old}}(a \mid s)}}_{\rho(s, a)} \cdot A^{\pi_{\text{old}}}(s, a)\right]
$$

这就是 **surrogate objective**：$L$ 表示"代理目标"，用 **importance ratio** $\rho$ 让从旧策略采的样本拿来评估新策略。

> **🤔 既然用了 importance sampling，为什么 $L$ 是近似而不是严格相等？**
>
> 关键：性能差有**两层分布**，即 state 分布 $d$ 和 action 分布 $\pi$。IS 只对其中一层精确。
>
> | 层 | surrogate 的操作 | 精确吗 |
> |---|---|---|
> | **action 分布** | $\frac{\pi\_\theta}{\pi\_{\text{old}}}$ importance sampling | ✅ 精确（第一步） |
> | **state 分布** | $d^{\pi\_\theta} \rightsquigarrow d^{\pi\_{\text{old}}}$ 直接替换 | ❌ 近似（第二步） |
>
> **为什么 state 分布不也做一次 IS 来修正？** 因为 state-visitation ratio $\frac{d^{\pi\_\theta}(s)}{d^{\pi\_{\text{old}}}(s)}$ **算不出来**。action ratio $\frac{\pi\_\theta(a|s)}{\pi\_{\text{old}}(a|s)}$ 能算，因为两个策略对给定 $(s,a)$ 直接吐概率；但
> $$d^{\pi}(s) = (1-\gamma)\sum_{t=0}^\infty \gamma^t \Pr(s_t = s \mid \pi)$$
> 依赖**整条轨迹**：初始分布 + 环境转移 $P(s'|s,a)$ + 策略在到达 $s$ 之前每一步的所有选择。要算这个 ratio 得对所有到达 $s$ 的路径积分、还要知道 dynamics model，实际中**无法逐点计算**，只能丢掉这个修正项。
>
> **所以 surrogate 的全部近似来自"偷换 state 分布"这一层**，而那一层没法 IS 修正。$\pi\_\theta$ 离 $\pi\_{\text{old}}$ 越近，$d^{\pi\_\theta} \approx d^{\pi\_{\text{old}}}$ 越准，近似误差越小。这正是下面 KL 约束的根本动机。
>
> **🔍 LLM 上这个近似在 sequence 层面消失**：contextual bandit 视角（整段 response $y$ 当一个 action）里 "state" = prompt $x$，从数据分布 $\mathcal{D}$ 采、**不依赖策略**，所以没有 state-shift，$\mathbb{E}\_{y \sim \pi\_\theta}[r] = \mathbb{E}\_{y \sim \pi\_{\text{old}}}[\frac{\pi\_\theta(y|x)}{\pi\_{\text{old}}(y|x)} r]$ **精确**。只有 token-level MDP 视角（中间状态 $(x, y\_{<t})$ 由策略生成）才重新出现 state-shift 近似。

为了让"$\pi\_\theta$ 离 $\pi\_{\text{old}}$ 不远"这个近似成立，TRPO 加 **KL 硬约束**：

$$
\max_\theta L_{\pi_{\text{old}}}(\theta) \quad \text{s.t.} \quad \mathrm{KL}(\pi_{\text{old}} \| \pi_\theta) \le \delta
$$

**这个约束有理论意义**：满足它就保证 $J(\pi\_\theta) \ge J(\pi\_{\text{old}})$（**monotonic improvement**）。vanilla PG 没有这个保证。代价是要解带约束的优化（Fisher 矩阵 + conjugate gradient + line search），工程上很麻烦。

**(3) PPO：把 KL 硬约束换成 clip**

PPO 的核心观察：**与其用 KL 硬约束限制 $\pi\_\theta$ 不远离 $\pi\_{\text{old}}$，不如直接把 importance ratio clip 在 $[1-\varepsilon, 1+\varepsilon]$ 之内**：

$$
L^{\text{CLIP}}(\theta) \;=\; \mathbb{E}\!\Big[\min\!\big(\rho_t A_t,\ \mathrm{clip}(\rho_t, 1-\varepsilon, 1+\varepsilon) A_t\big)\Big]
$$

直觉：当 $\rho\_t$ 超出 $[1-\varepsilon, 1+\varepsilon]$ 时，clip 让目标变成"不带 $\theta$ 的常数"，梯度为 0，相当于**这一步样本不再贡献更新**。这隐式地把 $\pi\_\theta$ 钉在 $\pi\_{\text{old}}$ 附近。

PPO 的理论保证比 TRPO **更弱**（不保证 monotonic improvement，只是近似），但**工程实现极简单**：first-order Adam optimizer 即可，不需要 Fisher / CG / line search。实际效果接近 TRPO，最终成为事实标准。

**(4) 把三件事归位：PPO 在 LLM RLHF 解决的三个痛点**

LLM 上朴素 REINFORCE（直接用 4.3.2 的 whole-return 形式）有三个工程痛点，PPO 用三个来源不同的工具分别对症：

| LLM 痛点 | 解决工具 | 来源 |
|---|---|---|
| **方差爆炸**（whole-return 把整段随机性压在每 token） | GAE + value baseline 替代 $\tilde r$ → per-token $A\_t$ | actor-critic 传统（**4.3.3** 已推） |
| **样本效率低**（一条 rollout 用一次就丢，LLM rollout 几秒到几十秒） | importance ratio $\rho\_t$ 让一条 rollout 做 $K$ 次更新 | **TRPO 的 surrogate**（本节 (2)） |
| **更新幅度难控**（一步太大语言崩塌） | clip $\rho\_t \in [1-\varepsilon, 1+\varepsilon]$ | **PPO 在 TRPO 上的简化**（本节 (3)） |

所以 LLM RLHF 选 PPO **不是因为它"修了 REINFORCE 的 bug"**，而是因为它的 **TRPO 血统 + actor-critic 传统** 天然凑齐了 LLM 需要的三件东西：value baseline（降方差，4.3.3）+ surrogate（样本复用）+ clip（限制偏离）。

> **🤔 那为什么 §1.3 把 PPO 跟 PG 放在一起讲？**
>
> 因为它们在**第一步梯度**（$\theta = \theta\_{\text{old}}$ 时）等价，这是 PPO 能"接住" 4.3.3 推出的 advantage / GAE 整套工具的原因。但**多步更新后**，PPO 优化的是 surrogate 而不是 $J$ 本身，理论基础完全是 TRPO 的那套。教学上把它们并列讲方便，但**别把 PPO 的来源搞错**。

**(5) TRPO 是在一般 MDP 上仔细推导的，那些结论搬到 LLM autoregressive 上还成立吗？**

这是个关键的诚实问题。答案：**TRPO 的"结构"搬得过来，但 TRPO 的"理论保证"基本搬不过来。LLM PPO 实际是当工程启发式用的。** 分三个层次看。

| TRPO 结论 | 在 LLM-MDP 还成立吗 |
|---|---|
| ① performance difference lemma（恒等式） | ✅ 严格成立 |
| ② surrogate 结构（IS ratio 复用旧样本） | ✅ 成立，sequence 级甚至更干净 |
| ③ monotonic improvement bound（理论保证） | ❌ 实际不可用 |

**① performance difference lemma：成立**。这是**任何 MDP 都成立的恒等式**，不依赖转移随机性或动作空间大小。LLM-MDP 是合法 MDP（4.3.1 已论证），lemma 照样成立。

**② surrogate 结构：成立，sequence 级更干净，但 LLM 用 token 级是务实折中**。
"用 $\rho = \pi\_\theta / \pi\_{\text{old}}$ 让旧样本评估新策略"只要 support 覆盖就成立。而且 4.3.4 (2) 已指出：**sequence 级（contextual bandit）视角下 state-shift 近似消失**，因为 state = prompt $x$ 从数据 $\mathcal{D}$ 采、不依赖策略，$d^{\pi\_\theta} = d^{\pi\_{\text{old}}} = \mathcal{D}$ 严格相等，surrogate **精确**，比一般 MDP 还好。

> ⚠️ **但 LLM PPO 实际用 token 级**（per-token $\rho\_t$、per-token $A\_t$），不是 sequence 级。一旦下沉到 token 级：(i) 中间 state $(x, y\_{<t})$ 由策略生成，**state-shift 近似重新出现**；(ii) per-token ratio $\rho\_t$ 是**务实选择**（sequence ratio $\prod\_t \rho\_t$ 连乘方差爆炸，要么炸要么塌）。所以 LLM PPO 的 token 级 surrogate 既不是纯 bandit 精确版、也不是经典 MDP 那套，是个**降方差的工程折中**。

**③ monotonic improvement bound：基本不可用**。这是 TRPO 最漂亮、也最搬不过来的结论。原 bound（Schulman et al. 2015 [TRPO]，Theorem 1）：

$$
J(\pi_\theta) \;\geq\; L_{\pi_{\text{old}}}(\theta) \;-\; \underbrace{\frac{4\epsilon\gamma}{(1-\gamma)^2}}_{C}\,\alpha^2, \qquad \epsilon = \max_{s,a}|A^{\pi_{\text{old}}}(s,a)|,\ \ \alpha = D_{\text{TV}}^{\max}(\pi_{\text{old}}, \pi_\theta)
$$

（原文 TV 版如上；论文同时给出 KL 版 $J(\pi\_\theta) \geq L\_{\pi\_{\text{old}}}(\theta) - C\,D\_{\text{KL}}^{\max}(\pi\_{\text{old}}, \pi\_\theta)$，同一常数 $C = \frac{4\epsilon\gamma}{(1-\gamma)^2}$。）三处在 LLM 上崩：

- **(a) $\gamma = 1$ 让常数 $C$ 爆炸**【本文推断，由上式直接代入】：RLHF 用 undiscounted（$\gamma = 1$），而 $C \propto \frac{\gamma}{(1-\gamma)^2} \to \infty$，bound 直接 vacuous。换 finite-horizon 版本，$\frac{1}{(1-\gamma)^2}$ 变成 $O(T^2)$，$T \sim$ 几千 token → $C \sim 10^6$ 量级，**bound 松到没有意义**。（这一步是把 TRPO 的 bound 代入 LLM 的 $\gamma=1$ 约定得到的推论，不是某篇论文的原话。）
- **(b) bound 要 $\max\_s \mathrm{KL}$，clip 控制不了它**：bound 要求对**所有 state（指数多的前缀）取 max** 的 KL $\le \delta$。PPO 的 per-token clip 只把单个 ratio 截在 $[1-\varepsilon, 1+\varepsilon]$，**既不是 KL、也不是 max、也不保证整段**。clip 是 soft 启发式，跟 bound 要的约束不是一回事。
- **(c) PPO 原文就把 clip 当启发式**【文献】：Schulman et al. 2017 [PPO] 把 clipped objective 明确描述为 unclipped 目标的 **"pessimistic estimate (i.e., lower bound)"**，是启发式下界，**并未为 clipped 版本证 monotonic improvement**（TRPO 的保证属于其 KL-penalty 版，不是 clip 版）。即在原始连续控制 MDP 上 clip 就已放弃严格保证，到 LLM 上只是雪上加霜。

**结论：LLM 用 PPO 是"借机制，不是借保证"**。真正搬过来的是 **surrogate + clip 这套机制**（用 IS ratio 复用昂贵 rollout + clip 软性限制漂移），**不是理论保证**（monotonic improvement 在 $\gamma=1$、$T\sim$ 数千、per-token clip 的 regime 下不成立）。LLM PPO 本质是工程启发式：能复用样本、经验上不崩、效果好，就用。多篇系统复现 / 剖析 RLHF-PPO 的工作（Huang et al. 2024 [N+ Details]；Zheng et al. 2023 [Secrets of RLHF I]）正是围绕这些**工程细节而非理论保证**展开。

> **🔍 佐证：RLHF 实践用三重保险**【文献事实 + 本文解读】。RLHF **额外**加了两道独立缰绳：(1) 显式 **KL-to-SFT 惩罚**（目标函数里的 $\beta$ 项，4.3 开头那个，是和 clip 独立的另一个机制；Ouyang et al. 2022 [InstructGPT] 的 PPO-ptx 目标即如此，更早见 Stiennon et al. 2020）；(2) 常配 **KL-to-$\pi\_{\text{old}}$ early stopping / adaptive-KL**（PPO 原文的 adaptive-KL 变体，及各 RLHF 实现监控 KL 超阈值即停）。**这三样并存是文献中的事实**；至于"这说明大家不信单靠 clip 的理论保证"，是本文的解读。

> **📚 本节引用（已联网核实出处）**
> - **[TRPO]** Schulman, Levine, Moritz, Jordan, Abbeel. *Trust Region Policy Optimization.* ICML 2015. arXiv:1502.05477. （Theorem 1 的 bound 与常数 $\frac{4\epsilon\gamma}{(1-\gamma)^2}$ 已核实）
> - **[PPO]** Schulman, Wolski, Dhariwal, Radford, Klimov. *Proximal Policy Optimization Algorithms.* 2017. arXiv:1707.06347. （clip = "pessimistic estimate (lower bound)" 已核实）
> - **[CPI]** Kakade, Langford. *Approximately Optimal Approximate Reinforcement Learning.* ICML 2002. （performance difference lemma / conservative policy iteration 来源）
> - **[InstructGPT]** Ouyang et al. *Training language models to follow instructions with human feedback.* 2022. arXiv:2203.02155.
> - **[N+ Details]** Huang et al. *The N+ Implementation Details of RLHF with PPO: A Case Study on TL;DR Summarization.* 2024. arXiv:2403.17031.
> - **[Secrets of RLHF I]** Zheng et al. *Secrets of RLHF in Large Language Models Part I: PPO.* 2023. arXiv:2307.04964.
> - **[ΨPO/IPO]** Gheshlaghi Azar et al. *A General Theoretical Paradigm to Understand Learning from Human Preferences.* 2023. arXiv:2310.12036.（从理论上批判 RLHF/DPO 假设，可作"RLHF 理论并不干净"的旁证）
>
> **标注约定**：上面正文里 **【文献】** = 有上述出处直接支持；**【本文推断】** = 从已核实公式推出的逻辑推论，非论文原话；**【本文解读】** = 诠释性观点。

#### 4.3.5 PPO 在 RLHF 的完整落地（零件 A ⊕ 零件 B）

到这里两个零件都齐了，PPO 就是把它们拼起来：

$$
L^{\text{PPO}}(\theta) = \mathbb{E}\Big[\sum_t \min\big(\rho_t\, A_t,\ \mathrm{clip}(\rho_t, 1-\varepsilon, 1+\varepsilon)\, A_t\big)\Big]
$$

- **$A\_t$ = 零件 A**（4.3.3）：per-token advantage，由 value model $V\_\psi$ + GAE 算出，**决定每个 token 的梯度权重**（降方差）。
- **$\rho\_t$ + clip = 零件 B**（4.3.4）：importance ratio + 截断，**决定怎么用一批旧 rollout 安全走 K 步**（样本复用 + 限制漂移）。

下面把这两个零件的具体计算逐一落地：先是 $A\_t$ 需要的 token-level reward 和 GAE（零件 A），再是 $\rho\_t$ 和 clip loss（零件 B），最后拼成训练循环。

**Token-level reward 设计**（零件 A 的输入，工程上最 subtle）：

$$
r_t \;=\;
\begin{cases}
-\beta \cdot \log \dfrac{\pi_\theta(y_t \mid x, y_{<t})}{\pi_{\text{SFT}}(y_t \mid x, y_{<t})} & t < T \\[8pt]
r_\phi(x, y) \;-\; \beta \cdot \log \dfrac{\pi_\theta(y_T \mid x, y_{<T})}{\pi_{\text{SFT}}(y_T \mid x, y_{<T})} & t = T
\end{cases}
$$

直觉：
- 中间 token 的 reward = 只有 KL penalty（每 token 一个）
- 最后 token 的 reward = RM 分数 + 最后的 KL penalty
- 整段累计 reward = $r\_\phi(x, y) - \beta \sum\_t \log\frac{\pi\_\theta}{\pi\_{\text{SFT}}}$，正好是 KL-regularized objective 的样本估计

**Advantage estimation（GAE）**：

$$
\delta_t \;=\; r_t + \gamma V_\psi(s_{t+1}) - V_\psi(s_t)
$$

$$
A_t^{\text{GAE}(\lambda)} \;=\; \sum_{l=0}^{T-t-1} (\gamma \lambda)^l\, \delta_{t+l}
$$

其中 $V\_\psi$ 是一个**单独训练的 value model**（通常 transformer 主干 + linear head）。$\lambda$ 平衡偏差和方差，$\gamma$ 通常 $\approx 1$。

**PPO clip loss**（per-token）：

$$
\rho_t(\theta) \;=\; \frac{\pi_\theta(y_t \mid x, y_{<t})}{\pi_{\text{old}}(y_t \mid x, y_{<t})}
$$

$$
L^{\text{PPO}}(\theta) \;=\; \mathbb{E}\!\left[\sum_t \min\!\big(\rho_t A_t,\ \mathrm{clip}(\rho_t, 1-\varepsilon, 1+\varepsilon) A_t\big)\right]
$$

**Value loss**:

$$
L^{V}(\psi) \;=\; \mathbb{E}\!\left[\sum_t (V_\psi(s_t) - R_t)^2\right]
$$

**完整训练循环**：

```
for each iteration:
    sample prompts x ∼ D
    rollout y ∼ π_θ(·|x)        # current policy as π_old
    compute r_t (per-token reward with KL penalty)
    compute A_t^GAE (using V_ψ)
    for K epochs:
        update θ via PPO clip loss
        update ψ via value loss
```

**工程痛点**：同时有 4 个 transformer 在显存：
- $\pi\_\theta$（policy，训练中）
- $\pi\_{\text{SFT}}$（reference，frozen，算 KL 用）
- $r\_\phi$（reward model，frozen）
- $V\_\psi$（value model，训练中）

这就是为什么 RLHF 这么贵。后面的 DPO/GRPO 革命都是从这个工程痛点出发的。

### 4.4 RLHF 的设计哲学小结

把整个三阶段串成一个故事：

1. Pretraining 给基础语言能力
2. SFT 给"听指令"能力
3. RM 把"人类偏好"蒸馏为可微 scalar
4. PPO 优化 RM scalar，KL 防漂移

**核心思想**：RL 在这里**不是为了序列决策**，而是**为了把"不可微的人类偏好"转换为可训练的损失**。LLM RL 的 MDP 是 degenerate 的（确定性转移、稀疏奖励），RL 只是借了它的 policy gradient 框架，用来反向传播一个原本不可微的信号。

---

## 五、RLHF 之后的进化：DPO 革命

> **DPO = Direct Preference Optimization（直接偏好优化）**，Rafailov et al. 2023。

### 5.1 PPO-RLHF 太贵

PPO-RLHF 痛点：
- 4 个模型同时在显存
- KL coefficient $\beta$ 难调
- 训练不稳定（典型问题：reward 早期爆涨然后 collapse）
- 工程实现复杂（RLHF 开源实现很少能直接跑通的）

**Rafailov et al. 2023（DPO，NeurIPS Outstanding Paper）** 的核心问题：**能不能跳过 reward model 和 RL，直接从偏好数据训 policy？**

答案：**可以**，而且数学上严格等价。

### 5.2 DPO 的关键推导（**这一段是 LLM RL 最优雅的几页**）

从 RLHF Stage 3 的 KL-regularized 目标出发：

$$
\max_\pi\ \mathbb{E}_{x, y \sim \pi}\!\left[r(x, y)\right]
\;-\; \beta\, \mathrm{KL}(\pi \| \pi_{\text{ref}})
$$

**第一步：求解这个目标的 closed-form 最优解。**

注意 KL 项展开：
$$
\mathrm{KL}(\pi \| \pi_{\text{ref}}) = \mathbb{E}_{y \sim \pi}\!\left[\log\frac{\pi(y|x)}{\pi_{\text{ref}}(y|x)}\right]
$$

整个目标可写为：

$$
\max_\pi\ J(\pi) \;:=\; \sum_y \pi(y \mid x)\left[r(x, y) - \beta \log\frac{\pi(y|x)}{\pi_{\text{ref}}(y|x)}\right]
$$

$$
\text{s.t.}\quad \sum_y \pi(y \mid x) = 1, \quad \pi(y \mid x) \ge 0
$$

这是一个**带约束的优化问题**，用拉格朗日法 / 变分法即可解出 closed-form 最优解。

构造 Lagrangian（用 $\lambda$ 处理归一化约束；$\pi(y|x) \ge 0$ 由 $\log\pi$ 项自动隐式保证，因为 $\pi \to 0$ 会让目标 $\to -\infty$）：

$$
\mathcal{L}(\pi, \lambda) \;=\; \sum_y \pi(y|x)\left[r(x,y) - \beta\log\pi(y|x) + \beta\log\pi_{\text{ref}}(y|x)\right] \;-\; \lambda\!\left(\sum_y \pi(y|x) - 1\right)
$$

由于 $y$ 是离散的，把每个 $\pi(y|x)$ 当独立变量，逐个求偏导（这就是"变分法"在离散情形下的样子）。**关键技巧**：注意 $\dfrac{d}{d\pi}\!\left[-\beta\,\pi\log\pi\right] = -\beta(\log\pi + 1)$，多出来的 $-\beta$ 别漏。

$$
\frac{\partial \mathcal{L}}{\partial \pi(y|x)} \;=\; r(x,y) \;\underbrace{-\beta\log\pi(y|x) - \beta}_{\text{from }-\beta\,\pi\log\pi} \;+\; \beta\log\pi_{\text{ref}}(y|x) \;-\; \lambda
$$

令偏导为零（一阶最优性条件）：

$$
r(x,y) \;-\; \beta\log\pi(y|x) \;-\; \beta \;+\; \beta\log\pi_{\text{ref}}(y|x) \;-\; \lambda \;=\; 0
$$

整理出 $\log\pi(y|x)$：

$$
\log\pi(y|x) \;=\; \log\pi_{\text{ref}}(y|x) \;+\; \frac{r(x,y)}{\beta} \;-\; \frac{\beta + \lambda}{\beta}
$$

取指数：

$$
\pi(y|x) \;=\; \pi_{\text{ref}}(y|x)\cdot\exp\!\left(\frac{r(x,y)}{\beta}\right)\cdot\underbrace{\exp\!\left(-\frac{\beta+\lambda}{\beta}\right)}_{\text{常数（不依赖 }y\text{）}}
$$

用归一化约束 $\sum\_y \pi(y|x) = 1$ 确定常数：

$$
\exp\!\left(-\frac{\beta+\lambda}{\beta}\right) \;=\; \frac{1}{Z(x)},
\quad\text{其中}\quad Z(x) \;:=\; \sum_y \pi_{\text{ref}}(y|x)\exp\!\left(\frac{r(x,y)}{\beta}\right)
$$

带回去：

$$
\boxed{\;\pi^*(y|x) \;=\; \frac{1}{Z(x)}\,\pi_{\text{ref}}(y|x)\,\exp\!\left(\frac{r(x,y)}{\beta}\right)\;}
$$

二阶条件：目标对 $\pi$ 是严格凹的（KL 项是严格凸的，加负号成严格凹），所以一阶条件给出的就是全局最大。

**关于 $\beta$ 的解读**：$\pi^* \propto \pi\_{\text{ref}} \cdot \exp(r/\beta)$ 这个形式里 $\beta$ 是温度。
- $\beta \to \infty$：$\pi^* \to \pi\_{\text{ref}}$，reward 影响消失，policy 完全跟着 reference 走；
- $\beta \to 0$：$\pi^*$ 退化为对 $\arg\max\_y r(x,y)$ 的点质量（greedy，完全忽略 reference）；
- $\beta$ 适中：在"听人话"和"追求 reward"之间取平衡。

这个 "$\pi^* \propto \pi\_{\text{ref}} \exp(r/\beta)$" 形式在统计物理（Gibbs 分布）、最大熵原理、soft Q-learning 中反复出现，是同一数学结构在不同领域的显化。

**第二步：把这个关系反过来表达 reward。**

从上式取对数：

$$
\log \pi^*(y \mid x) \;=\; \log \pi_{\text{ref}}(y \mid x) + \frac{r(x, y)}{\beta} - \log Z(x)
$$

整理：

$$
r(x, y) \;=\; \beta \log\frac{\pi^*(y \mid x)}{\pi_{\text{ref}}(y \mid x)} + \beta \log Z(x)
$$

**这个等式至关重要**：它说**任何 reward 函数都可以用 (最优 policy, reference policy) 这一对来表达**。换句话说，policy 和 reward 通过这个公式是 **dual** 的。

**第三步：把这个 reward 塞回 Bradley-Terry 偏好模型。**

BT 模型说：

$$
P(y_w \succ y_l \mid x) \;=\; \mathrm{sigm}\!\left(r(x, y_w) - r(x, y_l)\right)
$$

把第二步的 $r$ 代入：

$$
\begin{aligned}
r(x, y_w) - r(x, y_l)
&= \beta \log\frac{\pi^*(y_w|x)}{\pi_{\text{ref}}(y_w|x)} + \beta \log Z(x) \\
&\quad - \beta \log\frac{\pi^*(y_l|x)}{\pi_{\text{ref}}(y_l|x)} - \beta \log Z(x) \\
&= \beta \log\frac{\pi^*(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log\frac{\pi^*(y_l|x)}{\pi_{\text{ref}}(y_l|x)}
\end{aligned}
$$

**$\log Z(x)$ 神奇地消掉了**（因为 winner 和 loser 共享同一个 prompt，partition function 相同）。

所以：

$$
P(y_w \succ y_l \mid x) \;=\; \mathrm{sigm}\!\left(\beta \log\frac{\pi^*(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log\frac{\pi^*(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\right)
$$

**第四步：把上面的偏好概率公式做最大似然，得到 DPO loss。**

第三步得到的是**真实偏好概率**，用未知的最优 policy $\pi^*$ 表达。但 $\pi^*$ 我们不知道，它正是我们要求的未知量。从第三步到最终的 DPO loss，要走六个小步骤。

**(a) Reparametrize：用神经网络 $\pi\_\theta$ 替代 $\pi^*$**

定义"模型对偏好概率的预测"：

$$
P_\theta(y_w \succ y_l \mid x) \;:=\; \mathrm{sigm}\!\left(\beta \log\frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log\frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\right)
$$

这一步是**整个 DPO 的关键概念跳跃**：用可训练的 transformer $\pi\_\theta$ 代替未知的 $\pi^*$，把"找最优 policy" 转化为"找让偏好预测匹配数据的 $\theta$"。

**为什么合法**？这正是 policy ↔ reward 的 **dual** 关系给的合法性。$\pi\_\theta$ 一旦确定，它通过 $r = \beta\log(\pi\_\theta/\pi\_{\text{ref}})$ 自动定义了一个隐式 reward，这个 reward 又自动定义了它配合 RLHF 目标的最优 policy 就是 $\pi\_\theta$ 本身。所以以 $\pi\_\theta$ 参数化，就是在 (reward, policy) 这个 dual 对里**选了 policy 这一边**。

**(b) 用偏好数据做 MLE**

有偏好数据集 $\mathcal{D}\_{\text{pref}} = \{(x^{(i)}, y\_w^{(i)}, y\_l^{(i)})\}\_{i=1}^N$，假设 IID 采样。最大似然：找 $\theta$ 使观察到这批数据的概率最大：

$$
\max_\theta\ \prod_{i=1}^N P_\theta(y_w^{(i)} \succ y_l^{(i)} \mid x^{(i)})
$$

**(c) 取对数（单调变换，不改 argmax）**

$$
\max_\theta\ \sum_{i=1}^N \log P_\theta(y_w^{(i)} \succ y_l^{(i)} \mid x^{(i)})
$$

**(d) 改写为"最小化负对数似然"**

$$
\min_\theta\ -\sum_{i=1}^N \log P_\theta(y_w^{(i)} \succ y_l^{(i)} \mid x^{(i)})
$$

**(e) 经验期望写法（除以 $N$ 不影响 argmin）**

$$
\mathcal{L}_{\text{DPO}}(\theta) \;=\; -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}_{\text{pref}}}\!\left[\log P_\theta(y_w \succ y_l \mid x)\right]
$$

**(f) 代入 (a) 中 $P\_\theta$ 的具体形式**

$$
\boxed{\quad
\mathcal{L}_{\text{DPO}}(\theta) \;=\; -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}_{\text{pref}}}\!\left[\log \mathrm{sigm}\!\left(\beta \log\frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log\frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\right)\right]
\quad}
$$

**这就是 DPO**。

#### 整个 Step 4 的 conceptual 故事

把六小步缩成一句话：

> **(a)** 把未知的 $\pi^*$ 替换成可训练的 $\pi\_\theta$（合法，因为 dual），把第三步的真实偏好概率公式变成"模型对偏好概率的预测" $P\_\theta$。
> **(b)–(e)** 套用标准 MLE（最大化数据似然 → 等价于最小化负对数似然 → 写成期望形式）。
> **(f)** 把 $P\_\theta$ 展开成 log policy ratio 的具体形式。

整个 DPO 训练流程本质上是：**用一个 transformer 拟合人类偏好的 Bradley-Terry 模型，但偏好概率被巧妙地重写成 "log policy ratio" 形式**。所以它**不是 RL**，是**纯监督学习 / 分类任务**，只不过它的"分类器"参数化方式让它的最优解恰好对应 RLHF 目标的最优解。

### 5.3 DPO 的几个深刻含义

**(a) 没有 RM**：偏好数据直接训 policy，省掉 Stage 2 整个流程。

**(b) 没有 RL**：完全 offline supervised learning，没有 sampling，没有 PPO clip。

**(c) 没有 value model**：连 Stage 3 的 value 也省了。

**(d) $\beta$ 是温度，不是要调的 RL 超参**：固定常数（通常 0.1）。

**(e) Reward 是 policy 的 implicit reparameterization**：DPO 训练后，可以从 $\pi\_\theta$ 反推出 reward $r(x, y) = \beta \log \frac{\pi\_\theta(y|x)}{\pi\_{\text{ref}}(y|x)} + C$。所以 DPO 训出来的 $\pi\_\theta$ 内部"藏着一个 RM"。

**(f) 工程上 2 个模型够了**：$\pi\_\theta$ 训练、$\pi\_{\text{ref}}$ frozen（甚至可以用同一个 model 的 frozen 版本做 ref）。RLHF 的 4 模型 → DPO 的 2 模型。

### 5.4 DPO 梯度的直觉

DPO loss 对 $\theta$ 求梯度，设 $\hat r\_\theta(x, y) = \beta \log \frac{\pi\_\theta(y|x)}{\pi\_{\text{ref}}(y|x)}$（隐式 reward）：

$$
\nabla_\theta \mathcal{L}_{\text{DPO}} \;=\; -\beta\, \mathbb{E}\!\left[\mathrm{sigm}\!\left(\hat r_\theta(y_l) - \hat r_\theta(y_w)\right) \cdot \left(\nabla_\theta \log \pi_\theta(y_w|x) - \nabla_\theta \log \pi_\theta(y_l|x)\right)\right]
$$

直觉解读：
- 推**高** $y\_w$ 的对数概率
- 推**低** $y\_l$ 的对数概率
- 权重 $\mathrm{sigm}(\hat r\_\theta(y\_l) - \hat r\_\theta(y\_w))$ 是"当前模型对错误偏好的程度"
- 模型已经确信 $y\_w \succ y\_l$ 时，权重小 → 梯度衰减 → 避免过拟合

### 5.5 DPO 变体一览

| 变体 | 关键改动 | 解决什么问题 |
|---|---|---|
| **IPO** (Azar 2023) | 把 $\log\mathrm{sigm}$ 换成平方损失 | BT 在确定偏好时过拟合 |
| **KTO** (Ethayarajh 2024) | 用 Kahneman-Tversky 风险偏好 | 不需要配对数据，单点 like/dislike 即可 |
| **ORPO** (Hong 2024) | 加 SFT 项，去掉 reference | 不需要存 reference model |
| **SimPO** (Meng 2024) | 长度归一化的隐式 reward | 解决 DPO 偏好长输出 |
| **$\beta$-DPO** | $\beta$ 自适应 | $\beta$ 难调 |

DPO 极大降低了 LLM alignment 的成本，是 2023-2024 alignment 圈最重要的事件。但它有一个**根本局限**：完全 offline，模型从不在训练过程中"在线探索"。这给了下一段 GRPO 留出空间。

---

## 六、推理时代：GRPO 与 R1-Zero

> **GRPO = Group Relative Policy Optimization（组相对策略优化）**，Shao et al. 2024（DeepSeekMath）。

### 6.1 范式转变：可验证奖励 (RLVR)

> **RLVR = Reinforcement Learning with Verifiable Rewards（可验证奖励的强化学习）**。

2024 年起的根本性转折：人们发现**在数学、代码、形式证明上，reward 可以直接验证**，不需要 RM。

| 任务 | 可验证奖励来源 |
|---|---|
| 数学 | 与 ground-truth 答案比对 |
| 代码 | 跑 unit tests / 编译器 |
| 形式证明 | 证明检查器 (Lean, Coq) |
| 棋类 | 终局判定 |

形式上，RLVR 给的 reward 就是：

$$
r(x, y) \;=\; \mathbb{1}\!\left[\text{verify}(x, y) = \text{correct}\right] \;\in\; \{0, 1\}
$$

这导致 RLHF 时代的整套 RM pipeline 在 reasoning task 上**完全不必要**。这给了一个全新的设计空间：**既然 reward 这么干净，能不能把 PPO 也简化掉？**

### 6.2 GRPO 的核心：放弃 value model（详细推导）

> **🌳 先定位家谱：GRPO 是改 PPO，不是改 DPO（这俩常被搞混）**
>
> 一个高频误解是以为 **DPO → GRPO** 是一条演化线。**不是。** DPO 和 GRPO 是**同源反向的兄弟**：从同一个 RLHF KL-正则目标出发，往相反方向走。
>
> $$\max_\theta\ \mathbb{E}_{y \sim \pi_\theta}[r(x,y)] - \beta\,\mathrm{KL}(\pi_\theta \,\|\, \pi_{\text{ref}}) \quad\text{（共同起点）}$$
>
> ```
>             RLHF KL-正则目标
>                    │
>        ┌───────────┴───────────┐
>        ▼                        ▼
>    DPO（§五）               PPO-RLHF（§4.3）
>    解析求闭式解 → 监督 loss      保留完整 online RL
>    杀掉 RL：无 rollout            │
>    无 RM、无 value、无 clip       ▼
>                              GRPO（本节）
>                              保留 rollout + clip + IS ratio + KL
>                              只把 value model 换成 group baseline
> ```
>
> | 机制 | PPO-RLHF | **GRPO** | DPO |
> |---|---|---|---|
> | online rollout | ✅ | ✅ **照搬 PPO** | ❌ 纯 offline |
> | importance ratio + clip | ✅ | ✅ **照搬 PPO** | ❌ 无 |
> | advantage / policy gradient | ✅ | ✅ **照搬 PPO** | ❌ 是监督 loss，无梯度加权 |
> | value model | ✅ | ❌ 换 group baseline | ❌ |
>
> **GRPO 跟 PPO 共享几乎整套在线 RL 机制（rollout / clip / IS ratio / advantage），只动 baseline 一块；跟 DPO 一个机制都不共享。** 所以 **GRPO ← PPO-RLHF**（简化 PPO），**DPO ⊥ GRPO**（同源目标，反向取舍，互不继承）。一句话：DPO = 把 RL 整个删掉解析求解；GRPO = 把 PPO 这条 RL 路留着、只简化掉 value model。

> **🎯 一句话：GRPO 到底做了什么（先看这个再读推导）**
>
> 你的直觉没错：**GRPO 在算法上就是 PPO 改了 4 个地方**，没有引入新的理论框架。它和 PPO 的关系，看这张"改了啥"表最清楚：
>
> | | PPO-RLHF | GRPO | 这一改的后果 |
> |---|---|---|---|
> | **baseline 怎么来** | 学一个 value model $V\_\psi(s\_t)$ | 同一个 prompt 采 $G$ 条，用 **group 内 reward 均值**当 baseline | **干掉 value model**（少一个和 policy 同样大的网络 + 它的训练不稳定） |
> | **advantage 粒度** | per-token（GAE） | **per-sequence**（整条 $y$ 一个 $A$） | 不需要 token-level value，配合上面一条 |
> | **advantage 归一化** | 减 baseline | 减 group 均值后**再除 group 标准差** | 跨 prompt 尺度统一，训练更稳 |
> | **KL 怎么加** | 塞进 per-token reward | **直接作为 loss 加项** | sequence-level 没有 token reward 容器，只能这么加 |
>
> surrogate + clip（复用 rollout）、importance ratio、KL-to-ref，这些**全部照搬 PPO**，没动。
>
> **但"就是几个 trick"背后有一个真正承重的想法**：
> $$\textbf{用「同 prompt 多次采样的蒙特卡洛均值」替代「学出来的 value baseline」}$$
> PPO 必须学 $V\_\psi$ 来当 baseline（因为它一条 rollout 只有一个样本，没法估期望）；GRPO 直接对**同一个 prompt 采 $G$ 条**，用这 $G$ 条 reward 的均值当 baseline，于是 **baseline 从"学一个网络"变成"多采几次样平均一下"**。一旦 baseline 不用学了，value model 整个就没了，PPO 的 4 模型降到 2-3 模型。
>
> **为什么 2024 才行得通**：这招要求 reward **便宜且干净**到可以对每个 prompt 狂采 $G$ 次。RLVR（§6.1，数学/代码可验证、0/1 reward）刚好提供了这个前提：reward 是跑个 verifier，几乎不要钱。所以 GRPO 不是凭空的 trick 堆砌，是 **"RLVR 让 reward 变便宜" → "于是能用 MC group baseline 替代 value model"** 这条因果链的产物。
>
> **代价**：每个 prompt 要多跑 $G$ 倍 generation（用 inference 换掉 value model 的显存和不稳定）；失去 per-token credit assignment（但 reasoning 任务 reward 本就只在终答，per-token 信用本来也不可靠，见 6.2.11）。
>
> 下面 6.2.1–6.2.11 把这条"MC group baseline 替代 value model"逐步严格推一遍。

#### 6.2.1 动机：PPO-RLHF 的 value model 是工程痛点

PPO-RLHF（§3.3）在显存里同时有 4 个 transformer：policy、reference、reward model、value model。其中 **value model**：
- 通常和 policy 一样大（共享主干 + linear head 输出 scalar）；
- 训练目标是回归 bootstrapped return $R\_t$，比 policy gradient 更不稳；
- 在 RLVR setting 下（reward 是 0/1 binary），value function 极难学好，回归一个稀疏 0/1 信号本来就难。

在 reasoning 任务（数学/代码）上有一个新观察：**reward 可以直接验证，不再需要 RM**。这马上引出一个问题：

> 既然 RM 已经被 RLVR 替掉了，**能不能把 value model 也干掉**？

**GRPO (DeepSeekMath, Shao et al. 2024)** 给出的回答是：**可以**。具体做法是用 group sampling + group baseline 替代 value function。下面分 7 步把这个替代讲清楚。

#### 6.2.2 出发点：目标函数没变

GRPO 优化的目标和 PPO-RLHF 完全一致：

$$
J(\theta) \;=\; \mathbb{E}_{x \sim \mathcal{D},\ y \sim \pi_\theta(\cdot|x)}\!\left[r(x, y)\right] \;-\; \beta\, \mathbb{E}_{x}\!\left[\mathrm{KL}(\pi_\theta \,\|\, \pi_{\text{ref}})\right]
$$

不同之处在**视角**：GRPO 把 $y$ 当作 **atomic action**（sequence-level），不去管 token-level transition。这意味着 advantage、importance ratio 都是 per-sequence 的。

#### 6.2.3 Step 1：对目标求梯度

第一项（reward）用标准 policy gradient theorem：

$$
\nabla_\theta\, \mathbb{E}_{y \sim \pi_\theta}\!\left[r(x, y)\right] \;=\; \mathbb{E}_{y \sim \pi_\theta}\!\left[r(x, y) \cdot \nabla_\theta \log \pi_\theta(y|x)\right]
$$

第二项（KL）用 log-derivative trick。展开 KL：

$$
\mathrm{KL}(\pi_\theta \,\|\, \pi_{\text{ref}}) \;=\; \mathbb{E}_{y \sim \pi_\theta}\!\left[\log \pi_\theta(y|x) - \log \pi_{\text{ref}}(y|x)\right]
$$

对其求梯度（注意 $y$ 也依赖于 $\theta$，要用 score function trick）：

$$
\nabla_\theta\, \mathrm{KL} \;=\; \mathbb{E}_{y \sim \pi_\theta}\!\left[\log\frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)} \cdot \nabla_\theta \log \pi_\theta(y|x)\right]
$$

（其中 $\mathbb{E}\_{y \sim \pi\_\theta}[\nabla \log \pi\_\theta] = 0$ 这一项消掉了。）

合并两项：

$$
\nabla_\theta J \;=\; \mathbb{E}_{y \sim \pi_\theta}\!\left[\underbrace{\left(r(x, y) - \beta \log\frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)}\right)}_{\text{"有效 reward"}} \cdot \nabla_\theta \log \pi_\theta(y|x)\right]
$$

到这里所有 LLM RL 方法都一样。**GRPO 的设计选择**是把"有效 reward"拆开：原始 reward $r$ 用来算 advantage，KL 项作为 regularizer 独立加进 loss。这样 advantage 计算只依赖外部 reward，与 reference policy 解耦，调试更方便。

#### 6.2.4 Step 2：引入 baseline，降低方差

任何**不依赖 $y$** 的 baseline $b(x)$ 都不引入偏差：

$$
\nabla_\theta J^{\text{reward}} \;=\; \mathbb{E}_{y \sim \pi_\theta}\!\left[(r(x, y) - b(x)) \cdot \nabla_\theta \log \pi_\theta(y|x)\right]
$$

**证明 baseline 不引入偏差**：

$$
\mathbb{E}_{y \sim \pi_\theta}\!\left[b(x) \cdot \nabla_\theta \log \pi_\theta(y|x)\right] \;=\; b(x) \cdot \nabla_\theta\, \mathbb{E}_{y \sim \pi_\theta}\!\left[1\right] \;=\; b(x) \cdot \nabla_\theta\, 1 \;=\; 0
$$

**最优 baseline 是条件 reward 期望**：$b^*(x) = V^*(x) := \mathbb{E}\_{y \sim \pi\_\theta}[r(x, y)]$。PPO-RLHF 训一个神经网络 $V\_\psi(x)$ 来估这个东西，这就是要被去掉的 value model。

#### 6.2.5 Step 3：用 Monte Carlo 估计 $V^*$（group sampling 的本质）

GRPO 不训 value model，而是**直接用蒙特卡洛估计** $V^*(x)$：

对每个 prompt $x$，采样 $G$ 条 response $\{y^1, \ldots, y^G\} \sim \pi\_{\text{old}}(\cdot | x)$，得到 reward $\{r^1, \ldots, r^G\}$。用样本均值估期望：

$$
\hat V(x) \;:=\; \frac{1}{G}\sum_{i=1}^G r^i \;\approx\; V^*(x)
$$

这个估计器无需任何额外模型，只需要多采几次。

**Centered advantage**（朴素版）：

$$
A^i_{\text{centered}} \;=\; r^i - \hat V(x)
$$

**小坑：技术上有偏**

注意 $\hat V(x)$ 的求和**包含 $r^i$ 自身**，所以 $\hat V(x)$ 不严格独立于 $y^i$。这会引入 $\mathcal{O}(1/G)$ 量级的偏差：

$$
\mathbb{E}\!\left[\hat V(x) \cdot \nabla \log \pi_\theta(y^i|x)\right] \;\ne\; 0 \quad \text{(strictly)}
$$

- $G$ 大时（16-64），bias 可以忽略；
- $G$ 小时 bias 不可忽略，这是 **RLOO** 用 leave-one-out 解决的问题（详见 §6.1）。

GRPO 接受这点小偏差换取计算简单。

#### 6.2.6 Step 4：标准化—除以 std

GRPO 进一步把 advantage 除以 group 内 reward 的 std：

$$
\boxed{\quad A^i \;=\; \frac{r^i - \mathrm{mean}(r^{1..G})}{\mathrm{std}(r^{1..G}) + \epsilon_{\text{num}}} \quad}
$$

为什么标准化？

1. **尺度不变性**：reward 整体放缩（如所有 reward $\times 10$）advantage 不变，学习率等超参不必随 reward 量纲调整。
2. **跨 prompt 等权**：不同 prompt 的 reward 方差差异大（有的 prompt G 个 response 全对、有的全错），标准化让每个 prompt 对 gradient 贡献相同 "weight"，避免高方差 prompt dominates。
3. **数值稳定**：std 接近 0 时（如全对全错），$\epsilon\_{\text{num}}$ 防除零。

注意 std 也用了 $r^i$ 自身，再引入一点 bias，但同样在大 $G$ 下可忽略。

#### 6.2.7 Step 5：套用 PPO-clip 到 sequence-level ratio

到此为止用纯 REINFORCE update 已经可行：

$$
\nabla J^{\text{REINFORCE}} \;\approx\; \frac{1}{G}\sum_{i=1}^G A^i \cdot \nabla_\theta \log \pi_\theta(y^i | x)
$$

但 GRPO 想要**每条 rollout 数据多步利用**（提高 sample efficiency），所以套用 PPO-clip 框架。

定义 **sequence-level importance ratio**：

$$
\rho^i(\theta) \;=\; \frac{\pi_\theta(y^i | x)}{\pi_{\text{old}}(y^i | x)} \;=\; \prod_{t=1}^{T^i} \frac{\pi_\theta(y^i_t | x, y^i_{<t})}{\pi_{\text{old}}(y^i_t | x, y^i_{<t})}
$$

这里 $\pi\_{\text{old}}$ 是采样时的 policy 快照；$\pi\_\theta$ 是当前更新中的 policy。两者在每个 epoch 开头同步一次（"on-policy snapshot"）。

PPO-clip 风格的 surrogate loss：

$$
L^{\text{PPO-clip}}_{\text{GRPO}}(\theta) \;=\; -\frac{1}{G}\sum_{i=1}^G \min\!\left(\rho^i A^i,\ \mathrm{clip}(\rho^i, 1-\varepsilon, 1+\varepsilon) A^i\right)
$$

负号是因为我们要最小化 loss（最大化原目标）。

**clip 的作用**：当 $\pi\_\theta$ 漂离 $\pi\_{\text{old}}$ 太远时（$\rho^i$ 超出 $[1-\varepsilon, 1+\varepsilon]$），clip 限制 effective gradient，防止过大更新破坏策略。

#### 6.2.8 Step 6：KL 项怎么放？

在 PPO-RLHF（§3.3）里，KL 是放在 **token-level reward** 里的：

$$
r_t \;=\; -\beta \log \frac{\pi_\theta(y_t | x, y_{<t})}{\pi_{\text{SFT}}(y_t | x, y_{<t})}
$$

但 GRPO 是 sequence-level，没有 token reward 容器。所以 GRPO **直接把 KL 作为 loss 项加进来**：

$$
L_{\text{GRPO}}^{\text{total}}(\theta) \;=\; L^{\text{PPO-clip}}_{\text{GRPO}}(\theta) \;+\; \beta \cdot \mathbb{E}_{x}\!\left[\mathrm{KL}(\pi_\theta \,\|\, \pi_{\text{ref}})\right]
$$

**KL 的实际数值估计**：因为 $y^i \sim \pi\_{\text{old}}$（不是 $\pi\_\theta$），所以直接的样本均值 KL 估计是 biased 的。DeepSeek-R1 原文采用 **Schulman (2020) 的 K3 estimator**：

$$
\mathrm{KL}(\pi_\theta \,\|\, \pi_{\text{ref}}) \,\Big|_{y^i} \;\approx\; \frac{\pi_{\text{ref}}(y^i | x)}{\pi_\theta(y^i | x)} - 1 - \log\frac{\pi_{\text{ref}}(y^i | x)}{\pi_\theta(y^i | x)}
$$

这个估计器有两个好性质：
- **unbiased**：$\mathbb{E}\_{y \sim \pi\_\theta}$ 下等于真 KL
- **always non-negative**：来自不等式 $u - 1 - \log u \ge 0$（当且仅当 $u = 1$ 时等号）

#### 6.2.9 完整 GRPO loss

把上面所有步骤合起来：

$$
\boxed{\;
L_{\text{GRPO}}(\theta) \;=\; -\frac{1}{G}\sum_{i=1}^G \min\!\left(\rho^i A^i,\ \mathrm{clip}(\rho^i, 1-\varepsilon, 1+\varepsilon) A^i\right) \;+\; \beta \cdot \widehat{\mathrm{KL}}(\pi_\theta \,\|\, \pi_{\text{ref}})
\;}
$$

其中：
- $\rho^i = \pi\_\theta(y^i|x) / \pi\_{\text{old}}(y^i|x)$：**sequence-level** importance ratio
- $A^i = (r^i - \mathrm{mean}(r^{1..G})) / (\mathrm{std}(r^{1..G}) + \epsilon\_{\text{num}})$：group-normalized advantage
- $\widehat{\mathrm{KL}}$：K3 estimator

#### 6.2.10 完整算法

```
initialize π_θ ← base / SFT model
π_ref ← snapshot of π_θ  (frozen, for KL)

for iteration = 1, 2, ...:
    π_old ← π_θ           # snapshot for importance ratio

    # ── Rollout phase ──
    sample prompt batch {x_1, ..., x_B}
    for each x_b:
        sample G responses y^{b,1..G} ~ π_old(·|x_b)
        compute rewards r^{b,1..G}            # 通常 RLVR 0/1
        r̄_b = mean(r^{b,1..G})
        σ_b = std(r^{b,1..G})
        A^{b,i} = (r^{b,i} - r̄_b) / (σ_b + ε)  # for i = 1..G

    # ── Optimization phase: K epochs over the rollout ──
    for k = 1, ..., K:
        L = 0
        for each (b, i):
            ρ = π_θ(y^{b,i}|x_b) / π_old(y^{b,i}|x_b)
            L += -min(ρ · A^{b,i}, clip(ρ, 1-ε, 1+ε) · A^{b,i})
            L += β · K3_kl(π_θ, π_ref, y^{b,i})
        L /= (B · G)
        θ ← θ - lr · ∇L
```

注意三件事：
1. **$\pi\_{\text{old}}$ 每个 iteration 更新一次**（不是每个 mini-batch）
2. **$\pi\_{\text{ref}}$ 通常整个训练保持不变**（或定期 refresh）
3. **每个 rollout 被复用 K 次**（典型 K=1~4），这就是 PPO-clip 让我们能干的事

#### 6.2.11 核心权衡总结

**收益**：
- 去掉 value model → 显存减少、训练稳定性提升
- Sequence-level 简化 → 实现工程量减小
- 与 RLVR 完美配合（reward 是 binary 也无所谓）

**代价**：
- $G$ 倍 inference 成本（每 prompt 多跑 G 次 generation）
- 失去 per-token credit assignment
- group baseline 引入 $\mathcal{O}(1/G)$ 小偏差

**为什么在 reasoning 任务上 GRPO 特别好用**？因为 reasoning 任务 reward 极稀疏（只有终答对错），per-token credit assignment 本来就不可靠，sequence-level 反而更适合任务结构。这是 GRPO 在 reasoning 上成功的关键"结构匹配"原因。

### 6.3 GRPO 与 PPO-RLHF 的差异表

| 维度 | PPO-RLHF | GRPO |
|---|---|---|
| Value model | 需要 | **不需要** |
| Advantage 粒度 | per-token (GAE) | per-sequence |
| Reward 来源 | reward model | verifiable reward |
| Ratio | per-token | per-sequence |
| KL 注入方式 | token-level reward penalty | loss 加项 |
| 显存中模型数 | 4 | 2-3 |

### 6.4 R1-Zero：pure RL 的涌现奇迹

**DeepSeek-R1 (2025)** 最震撼的实验是 R1-Zero：

- 起点：base model（**没有任何 SFT 冷启动**！）
- 算法：GRPO
- Reward：RLVR（数学题答案对错）
- 训练时间：相对短

结果：模型在训练过程中**自发涌现**长链推理能力（chain-of-thought），开始自己说 `"Wait, let me reconsider..."`, `"Actually, the answer should be..."`, `"Let me verify by..."` 这样的元认知 pattern。

**为什么这个结果震撼**：传统观点认为 CoT 必须靠 SFT 数据示范才能学到。R1-Zero 证明：**只要给足够 reward 信号 + 模型容量足够，RL 自己会发现"想清楚再回答"是更优策略**。这意味着：

1. SFT 不是必需的（虽然实际还是要 SFT 增强可读性）
2. CoT 不是"模仿"出来的，是 RL "学"出来的
3. RLVR + GRPO 是一个**比 RLHF 更强**的 alignment 范式（至少在 reasoning 上）

### 6.5 R1 之后的复现潮

R1 发布后两个月内（2025 春），开源社区涌现大量复现：

- **Tülu 3** (AI2, 2024): 完全开源的 RLVR 配方
- **Kimi-1.5** (Moonshot, 2025): 类似配方 + long-context CoT
- **simpleRL-reason, Open-R1**: 社区复现
- **R1-Distill**: 用 R1 生成的数据蒸馏小模型

GRPO + RLVR 现在是 **reasoning-focused LLM** 的事实标准。

---

## 七、其他 LLM RL 算法（家族补全）

### 7.1 RLOO (REINFORCE Leave-One-Out) — 详细推导

> **Ahmadian et al. 2024**（ICML，*Back to Basics: Revisiting REINFORCE Style Optimization for Learning from Human Feedback in LLMs*）的核心论点：PPO 的一堆复杂性（value model、GAE、clipping、token-level reward shaping）在 LLM RLHF 上**多数是不必要的**，**REINFORCE + 留一基线**就能达到甚至超过 PPO，且实现简洁、参数少、训练快。

#### 7.1.1 动机：为什么要"回到 REINFORCE"

PPO-RLHF（§3.3）有四个工程负担：value model 训练、GAE 估计、PPO clip 调参、token-level reward+KL 摊薄。每一项都引入偏差/超参/不稳定来源。RLOO 论文逐项 ablate：
- 去掉 value model → 用 group baseline 替代 → 性能不掉
- 去掉 GAE → 用 sequence-level reward → 性能不掉
- 去掉 PPO clip → 用单步纯 REINFORCE → 性能不掉
- 把 KL 从 token-level reward 项改为 loss 项 → 性能不掉

最终留下一个极简的 estimator：**REINFORCE + leave-one-out baseline + KL-to-ref**。

#### 7.1.2 形式推导：LOO baseline 的无偏性

对每个 prompt $x$ 采样 $G$ 个 response $\{y^1, \ldots, y^G\}$，每个 reward $r^i = r(x, y^i)$。**留一基线**：

$$
b^i \;=\; \frac{1}{G - 1}\sum_{j \ne i} r^j
$$

advantage：

$$
A^i \;=\; r^i \;-\; b^i \;=\; r^i \;-\; \frac{1}{G-1}\sum_{j \ne i} r^j
$$

**无偏性证明**：标准 PG 减 baseline 不引入偏差的条件是 baseline 不依赖于被加权的 action $y^i$：

$$
\mathbb{E}_{y^i \sim \pi_\theta}\!\left[b^i \cdot \nabla_\theta \log \pi_\theta(y^i \mid x)\right]
\;=\; b^i \cdot \mathbb{E}\!\left[\nabla_\theta \log \pi_\theta(y^i \mid x)\right]
\;=\; b^i \cdot \nabla_\theta\, \mathbb{E}[1]
\;=\; 0
$$

LOO baseline $b^i = \frac{1}{G-1}\sum\_{j \ne i} r^j$ 显然不依赖 $y^i$（求和不包含 $i$），所以**严格无偏**。

对比 GRPO 的 baseline $\bar r = \frac{1}{G}\sum\_{j=1}^G r^j$ ， 它**包含** $r^i$ 自己，所以 $\bar r$ 与 $y^i$ 有依赖，引入 $\mathcal{O}(1/G)$ 偏差。这是 GRPO 的"小坑"。

#### 7.1.3 方差对比：LOO vs mean baseline

记 $\sigma^2 := \mathrm{Var}(r^i)$（同 prompt 不同采样的 reward 方差）。

LOO baseline：$\mathrm{Var}(A^i\_{\text{LOO}}) = \sigma^2 + \mathrm{Var}(b^i\_{\text{LOO}}) = \sigma^2 + \sigma^2/(G-1) = \sigma^2 \cdot \frac{G}{G-1}$

GRPO mean baseline：$\mathrm{Var}(A^i\_{\text{GRPO,raw}}) = \mathrm{Var}(r^i - \bar r)$。展开：$r^i - \bar r = \frac{G-1}{G}(r^i - \frac{1}{G-1}\sum\_{j \ne i} r^j)$，所以 $A^i\_{\text{GRPO,raw}} = \frac{G-1}{G} \cdot A^i\_{\text{LOO}}$，方差小一点点。

但 GRPO 再除以 $\mathrm{std}$ 做标准化，方差被进一步规整。**实际比较**：LOO 无偏 + 略大方差；GRPO 略偏 + 略小方差 + 标准化。两者在 $G \ge 16$ 时实验上无显著差异。

#### 7.1.4 完整 RLOO loss

加上 KL-to-ref 项（作为 loss 项，不进 reward）：

$$
\mathcal{L}_{\text{RLOO}}(\theta) \;=\; -\frac{1}{G}\sum_{i=1}^G \log \pi_\theta(y^i \mid x) \cdot A^i \;+\; \beta \cdot \widehat{\mathrm{KL}}\!\left(\pi_\theta \,\|\, \pi_{\text{ref}}\right)
$$

**没有 PPO clip**，**没有 importance ratio**，每个 epoch 完全 on-policy（这是 RLOO 的关键简化）。如果想要 off-policy 复用，可加 IS clip 退化为 GRPO 风格。

#### 7.1.5 实证结果（Ahmadian 2024）

在 OpenAssistant / TL;DR Summarization 任务上：
- RLOO ≥ PPO（在 GPT-2-XL, OPT-1.3B, LLaMA-7B 上都成立）
- RLOO 训练时间约为 PPO 的 50–70%
- RLOO 超参数空间小得多（无 GAE λ、无 value lr、无 clip ε）

---

### 7.2 REINFORCE++（Hu 2025）— 详细

> **REINFORCE++**（名字本身就是算法名，是经典 **REINFORCE** 算法的 LLM 增强版，"++" 表示加了 PPO clip 等稳定化改进，不另有缩写展开）。**Hu 2025**（*REINFORCE++: Stabilizing Critic-Free Policy Optimization with Global Advantage Normalization*，[arXiv:2501.03262](https://arxiv.org/abs/2501.03262)）：在 RLOO 之外的另一条简化路线。保留 PPO clip 框架（因为 PPO clip 让多 epoch 训练稳定），但去掉 value model 和 GAE，核心卖点是 global advantage normalization。

#### 7.2.1 核心 design choices

1. **Token-level KL penalty**（继承 PPO-RLHF）：
   $$r_t \;=\; r_\phi(x, y) \cdot \mathbb{1}[t = T] \;-\; \beta \cdot \log \frac{\pi_\theta(y_t \mid x, y_{<t})}{\pi_{\text{ref}}(y_t \mid x, y_{<t})}$$
   即末位 token 给 RM 分数，每个 token 都减 KL。
2. **Reward normalization**：在 batch 内 $z$-score：$\tilde r\_t = (r\_t - \mu\_r)/(\sigma\_r + \epsilon)$
3. **Advantage normalization**：用 batch-level $z$-score（不是 group-level，因为不分 group）：$\tilde A\_t = (A\_t - \mu\_A)/(\sigma\_A + \epsilon)$
4. **Token-level clip**（保留 PPO-clip 形式）：$L = \min(\rho\_t A\_t, \mathrm{clip}(\rho\_t, 1-\varepsilon, 1+\varepsilon) A\_t)$
5. **Advantage 计算**：累计未来 reward（无 GAE，无 value）：$A\_t = \sum\_{s=t}^T \gamma^{s-t} r\_s$

#### 7.2.2 与 PPO / RLOO / GRPO 的差异表

| 维度 | PPO-RLHF | REINFORCE++ | RLOO | GRPO |
|---|---|---|---|---|
| value model | 是 | **不要** | 不要 | 不要 |
| advantage | GAE | discounted return | LOO baseline | group mean baseline |
| clip | per-token | per-token | **无** | per-sequence |
| KL placement | token-level reward | token-level reward | loss term | loss term |
| sample efficiency | 高（K-epoch） | 高（K-epoch） | 低（1-epoch） | 高（K-epoch） |
| 实现复杂度 | 高 | 中 | **低** | 中 |

#### 7.2.3 何时选 REINFORCE++

适合**长 CoT** + **稀疏 reward** + **多 epoch off-policy 复用**的场景。如果你的任务是 RLHF（密集 RM scalar reward），REINFORCE++ 是 PPO 的更轻量替代；如果是 RLVR（稀疏 binary reward），GRPO/RLOO 更直接。

---

### 7.3 ReMax（Li 2023）— 详细

> **ReMax**（名字取自 **RE**INFORCE + 用 **Max**（贪婪 argmax 解码）的 reward 当 baseline，无标准缩写展开）。**Li et al. 2023**（*ReMax: A Simple, Effective, and Efficient Method for Aligning Large Language Models*）：把 baseline 选为"**贪婪解码**的 reward"，每个 prompt 只需 2 个 rollouts。

#### 7.3.1 算法

每个 prompt $x$：
1. 采样：$y\_{\text{sample}} \sim \pi\_\theta(\cdot \mid x)$
2. 贪婪：$y\_{\text{greedy}} = \arg\max\_y \pi\_\theta(y \mid x)$（即 temperature=0 decode）
3. Advantage：
   $$A \;=\; r(x, y_{\text{sample}}) \;-\; r(x, y_{\text{greedy}})$$
4. Update：$\nabla\_\theta J \approx \nabla\_\theta \log \pi\_\theta(y\_{\text{sample}} \mid x) \cdot A$

#### 7.3.2 直觉：贪婪基线作 control variate

把 $y\_{\text{greedy}}$ 看作"当前 policy 最自信的回答"。$A$ 衡量"采样回答比当前自信回答好多少"：
- 如果 $A > 0$：采样回答比自信回答好 → 把它推高
- 如果 $A < 0$：采样回答比自信回答差 → 把它压低
- 如果 $A = 0$：没差距 → 不动

这是一种 **structural control variate**。

#### 7.3.3 无偏性

$y\_{\text{greedy}}$ 是 $\theta$ 的确定性函数（given fixed $\theta$），独立于采样 $y\_{\text{sample}}$。因此：

$$\mathbb{E}_{y_{\text{sample}}}[r(x, y_{\text{greedy}}) \cdot \nabla \log \pi_\theta(y_{\text{sample}} \mid x)] = r(x, y_{\text{greedy}}) \cdot 0 = 0$$

→ baseline 不引入偏差。**严格无偏**。

#### 7.3.4 优势 + 局限

**优势**：
- 只 2 个 rollout vs GRPO/RLOO 的 $G$ 个 → 4-32× 便宜
- 单 prompt 内方差小（贪婪基线非常接近 mean）
- 不需要 group 调度

**局限**：
- 贪婪解码可能 mode-collapse（policy 已经 confidently 错的时候，$y\_{\text{greedy}}$ 也错；$A$ 评估失真）
- 不能多 epoch 复用（没 IS ratio）
- 早期训练 policy 还没 confident 时，$y\_{\text{greedy}}$ 是噪声

---

### 7.4 VinePPO（Kazemnejad 2024）— 详细

> **VinePPO**（名字 = "**Vine**"（取自 TRPO 论文里从中间状态分叉多次采样的 "vine" 估计法）+ **PPO**，无另外的缩写展开）。**Kazemnejad et al. 2024**（*VinePPO: Unlocking RL Potential For LLM Reasoning Through Refined Credit Assignment*）：把 PPO 的 value model 换成 **Monte Carlo rollouts** 估计中间状态的 value。

#### 7.4.1 动机

PPO-RLHF 的 GAE 依赖 value model $V\_\psi(s\_t)$ 给中间 state 的 value 估计。但在 LLM reasoning 上，value model 学得很差（reward 稀疏 + 长 CoT 让回归任务困难）。结果：GAE 估计的 advantage 充满噪声，credit assignment 失效。

VinePPO 的洞察：**反正每个中间 state 都可以 sample 多个后续完成**，直接用 MC rollouts 估 $V(s\_t)$ 比训 value network 更准。

#### 7.4.2 算法

```
for prompt x:
  rollout 完整 trajectory τ = (s_0, a_0, r_0, ..., s_T, a_T, r_T)
  for each intermediate s_t:
    重新采样 K 个 future trajectories from s_t
    V_hat(s_t) = mean(reward of K future trajectories)
  compute advantage A_t = r_t + γ V_hat(s_{t+1}) - V_hat(s_t)  # 用 MC value 算 TD
  PPO clip update on θ using A_t
```

#### 7.4.3 关键技术

- **Branching budget**：每个中间 state branch $K = 8-16$ 个 future
- **Compute cost**：$T \times K$ 倍于普通 PPO（不便宜）
- **优点**：每个 token 的 advantage 估计**更准** → credit assignment 更细致

#### 7.4.4 实验

在 GSM8K, MATH 上 VinePPO 显著超过 PPO + GAE 5-10 pts。证明在 LLM reasoning 上，**精确的 credit assignment** 比 value model 的训练效率重要。

#### 7.4.5 局限

- 计算贵（$T \times K \times$ generation cost）
- 仅适合中等长度 trajectory（$T \le 1000$）
- 不能复用 cached rollouts（每个 rerun branching 重采样）

---

### 7.5 DAPO（ByteDance 2025）— 详细

> **DAPO = Decoupled Clip and Dynamic sAmpling Policy Optimization（解耦裁剪与动态采样策略优化）**。**ByteDance Seed 2025**（[arXiv:2503.14476](https://arxiv.org/abs/2503.14476), *DAPO: An Open-Source LLM Reinforcement Learning System at Scale*）：在 GRPO 基础上加 4 个工程修正，在 AIME 2024 上用一半步数超过 R1-Zero。是 2026 年最广泛采用的 GRPO 升级。

#### 7.5.1 GRPO 暴露的具体缺陷（4 个）

1. **Length bias**：advantage 除以 std，长错答案的相对惩罚反而小 → policy 学会写越来越长的错误回答
2. **Entropy collapse**：PPO clip 上界 $1+\varepsilon$ 抑制 token 概率上升 → 高 entropy token（探索性 token）逐渐被压死 → 策略变 deterministic 太快
3. **Sample inefficiency**：group 内全 0 或全 1 reward 的 prompt（已掌握或完全不会）梯度为 0 → 白消耗 compute
4. **Long-CoT 不稳**：超长输出累积 KL 漂移大，模型容易 break

#### 7.5.2 DAPO 的 4 个修正

**(1) Clip-Higher**：上界 clip 比下界宽：

$$
L = \min\!\left(\rho A,\ \mathrm{clip}(\rho,\ 1-\varepsilon_{\text{low}},\ 1+\varepsilon_{\text{high}}) A\right), \quad \varepsilon_{\text{high}} > \varepsilon_{\text{low}}
$$

典型 $\varepsilon\_{\text{low}} = 0.2, \varepsilon\_{\text{high}} = 0.28$。让高 entropy token 有更多上升空间，**保护探索**。

**(2) Dynamic Sampling**：训练前过滤掉 group 内 reward 全 0 或全 1 的 prompt：

```
for prompt x in batch:
  sample G rollouts, get rewards r^1..G
  if all(r^i == 0) or all(r^i == 1):
    skip this prompt  # 学不到东西
  else:
    keep
```

让每个 update step 都有 effective 信号；空 group 不浪费 compute。

**(3) Token-Level Policy Gradient Loss**：长 CoT 场景下，回到 token-level loss（而非 GRPO 默认的 sequence-level）：

$$
L_{\text{token}} \;=\; \frac{1}{\sum_i T^i} \sum_i \sum_t \min(\rho^i_t A^i,\ \mathrm{clip}(\rho^i_t) A^i)
$$

其中 $\rho^i\_t$ 是 per-token IS ratio。**优点**：避免长样本主导梯度。**对应**：sequence-level GRPO 在 long-CoT 下指数级放大 ratio（我之前 §5.2.11 提到过），token-level 不会。

**(4) Overlong Reward Shaping**：对超过 max-length 的输出做软惩罚（不是直接截断）：

$$
r_{\text{shape}}(y) = \begin{cases}
r_{\text{task}}(y) & \lvert y \rvert \le L_{\text{max}} \\
r_{\text{task}}(y) - \lambda \cdot (\lvert y \rvert - L_{\text{max}}) & \lvert y \rvert > L_{\text{max}}
\end{cases}
$$

避免硬截断让 reward 信号失真。

#### 7.5.3 实验

- Qwen2.5-32B + DAPO：AIME 2024 **50 分**（vs R1-Zero-Qwen-32B 47 分），且只用 50% 训练步数
- 完全开源训练代码 + 数据 + checkpoint

---

### 7.6 其他 2025-2026 GRPO 改进算法（简略）

#### Critique-GRPO（Wu et al. 2025）

把"批评文本"作为辅助 reward 信号注入 GRPO。critic LLM 对生成结果给自然语言反馈 + 数值评分；生成 LLM 用两者联合 advantage 训练。在 Qwen 上 Pass@1 +15-21.6%。属于 actor-critic 设定的扩展。

#### TIC-GRPO（Trajectory-level Importance-Corrected GRPO）

用整段轨迹的概率比 $\rho^{\text{traj}} = \pi\_\theta(y \mid x) / \pi\_{\text{old}}(y \mid x)$ 替代 GRPO 的 token-level ratio。给出当前 policy gradient 的**无偏估计**（修正 GRPO 的小偏差），保留 critic-free 结构。

#### Lite PPO

大规模 ablation 后提出的极简 PPO 变体：critic-free + sequence-level + group baseline。声称稳定超过 GRPO 和 DAPO。本质上是 RLOO + PPO clip。

#### MGPO（Multi-Grained Policy Optimization）

引入"多粒度"importance sampling 权重：token-level + sequence-level 混合。用于多模态推理。

#### MLMT-RL（Multi-Level Multi-Turn RL）

多层级（task + subtask + step）+ 多轮（dialogue）的 RL 框架。在 reasoning task 上让 2B 模型超过 3B GRPO。

---

### 7.7 多智能体 LLM RL 算法（参考 `references/LLM通信文献精读.md`）

> 这一节简述 8 篇 2025-2026 multi-LLM RL 论文。完整精读在 `references/LLM通信文献精读.md`。

#### 7.7.1 MAGRPO（Liu et al. 2025-08，[arXiv:2508.04652](https://arxiv.org/abs/2508.04652)）

> **MAGRPO = Multi-Agent Group Relative Policy Optimization（多智能体组相对策略优化）**，即 GRPO 的多智能体扩展。

把 LLM 协作建模为 **Dec-POMDP**（Decentralized Partially Observable Markov Decision Process），用 GRPO 扩展到多智能体：
- **Centralized group-relative advantage**：用 joint-reward group 统计作 baseline，所有 agent 共享 advantage
- **Decentralized execution**：每个 agent 自己的 score function

$$
A^{(g)} = \frac{R^{(g)}_{\text{joint}} - \mathrm{mean}_g}{\mathrm{std}_g}, \qquad
\nabla_{\theta_i} J \approx \frac{1}{G}\sum_g A^{(g)} \cdot \nabla_{\theta_i} \log \pi_{\theta_i}(y^{(g)}_i \mid o^{(g)}_i)
$$

#### 7.7.2 Dr. MAS（Feng et al. 2026-02，[arXiv:2602.08847](https://arxiv.org/abs/2602.08847)）

> **MAS = Multi-Agent System（多智能体系统）**；"Dr." 为论文风格化命名（取"诊断 / 治疗"之意），全称见原文标题 *Dr. MAS: Stable Reinforcement Learning for Multi-Agent LLM Systems*。

诊断 GRPO 套到 multi-agent LLM 上的失败模式：**reward distribution heterogeneity → gradient-norm instability**。

不同 agent 做不同任务，reward 分布异质；vanilla GRPO 用 global advantage baseline 会让某些 agent 的梯度放大几十倍，触发 gradient explosion。

**Dr. MAS 修正**：per-agent normalization。每个 agent 用自己的 reward 统计：

$$
A_i = \frac{R_i - \mu_i}{\sigma_i + \epsilon}
$$

实验：Qwen2.5/Qwen3 上 math reasoning + multi-turn search 上 vs vanilla GRPO 提升 **+5.6% / +15.2% avg@16**，gradient spike 几乎消除。

#### 7.7.3 TeamTR（Xie et al. 2026-05，[arXiv:2605.15207](https://arxiv.org/abs/2605.15207), ICML 2026）

> **TeamTR = Team Trust-Region（团队信赖域）**，即把 TRPO 的 trust-region 思想用到多智能体团队协调上。原文标题 *TeamTR: Trust-Region Fine-Tuning for Multi-Agent LLM Coordination*。

形式化命名了"naive joint GRPO 发散"的具体机制：**compounding occupancy shift**。

**关键定理**：用 stale cached rollouts 评估时，certificate penalty 按 $\mathcal{O}(n^2 \sqrt{\bar\delta})$ 增长（$n$ = agent 数）。改用每次 update 后**重采样**的 intermediate-occupancy 评估只增长到 $\mathcal{O}(n \sqrt{\bar\delta})$。

**TeamTR 算法**：stage-wise trust region + 每个 component update 后重采样，给出严格的 per-update improvement lower bound。

实验：合作 reasoning 任务上比 sequential baseline 平均 +7.1%。

#### 7.7.4 Advantage Alignment for LLM（Piche et al. 2025-11，[arXiv:2511.19405](https://arxiv.org/abs/2511.19405)）

> **LOLA = Learning with Opponent-Learning Awareness（对手学习感知的学习）**，Foerster et al. 2018，是 opponent-learning-aware 这类算法的起点；**Advantage Alignment** 是它的后续简化。

把 LOLA 后续的 **Advantage Alignment** (opponent-learning-aware) 适配到 LLM scale。原始 Advantage Alignment 需要二阶 (穿过 partner gradient)；通过 group-relative baseline 简化到一阶。

新环境 **Trust-and-Split**：需自然语言通信达到 high collective welfare 的 social dilemma。

属于 **anticipatory** 风格的 opponent-learning-aware 算法（预测 partner 下一步更新），适合 mixed-motive。

#### 7.7.5 MATTRL（Hu et al. 2026-01，[arXiv:2601.09667](https://arxiv.org/abs/2601.09667)）

> **MATTRL = Multi-Agent Test-Time Reinforcement Learning（多智能体测试时强化学习）**。

承认 multi-agent RL 训练 LLM 推理 resource-intensive + co-adapting teammates 引入 non-stationarity，**绕开 weight update**，转向 test-time RL：
- 形成 expert team multi-turn 讨论
- 检索 / 构造 turn-level structured experience pool
- 把经验作为 in-context signal 注入下一轮
- 多种 credit assignment 策略

#### 7.7.6 MARFT（Liao et al. 2025-04, v4 2025-11，[arXiv:2504.16129](https://arxiv.org/abs/2504.16129)）

> **MARFT = Multi-Agent Reinforcement Fine-Tuning（多智能体强化微调）**。

提供 **LaMAS** (LLM-based Multi-Agent System) 与经典 MARL 的根本差异：**asynchronous action**（agent 不同时被调用）、**dynamic organization**、**profile-aware design**、**heterogeneous architecture**。

引入 **Flex-MG**（Flexible Markov Game）formalism，允许异步动作、动态参与、role-conditioned policy。

#### 7.7.7 ShapeLLM (Opponent Shaping in LLM Agents)（Garcia Segura et al. 2025-10，[arXiv:2510.08255](https://arxiv.org/abs/2510.08255), ICLR 2026）

第一个为 transformer-based agent 设计的 **OS（Opponent Shaping，对手塑造）** 方法：
- Model-free（不需要 differentiate through opponent）
- 用 PPO 训 shaper agent，给它一个 shaping reward signal：除自己 reward 外，鼓励它把 co-player 推向特定策略

5 个 matrix game 上验证：竞争中把对手推向 exploitable 均衡；合作中促进 coordination。

#### 7.7.8 CRAFT（Nath et al. 2026-03，[arXiv:2603.25268](https://arxiv.org/abs/2603.25268)）

Benchmark + 失败分类法（**不是算法**）：多 director agent 各持 3D 目标结构的 partial view，必须自然语言协作引导 builder 在 3D 空间中正确构建。

**三层失败分类**：spatial grounding / mind modeling / pragmatic communication

**关键发现**：stronger reasoning ability does not reliably translate to better coordination；现代 LLM 在多 agent 协调上还是 fundamentally unsolved。

---

### 7.8 算法谱系图（更新版）

```
经典 RL
  │
  ├── REINFORCE ──────────────┐
  │       │                    │
  │       └── + Actor-Critic   │ (LLM 化)
  │              │             │
  ├── TRPO ──────┴── PPO ──────┤
  │                            │
  │                            ▼
  │                       PPO-RLHF (InstructGPT)
  │                       │
  │              ┌────────┼─────────┬────────┬──────────────┐
  │              │        │         │        │              │
  │              ▼        ▼         ▼        ▼              ▼
  │            DPO      GRPO      RLOO    REINFORCE++   ReMax/VinePPO
  │            (offline)│       (无偏 LOO) (轻量+clip)   (替代基线/MC value)
  │                     │
  │                     ├── R1-Zero (RLVR)
  │                     ├── DAPO (4 修正：Clip-Higher, Dynamic Sampling, Token-Loss, Overlong Shaping)
  │                     ├── Critique-GRPO (NL+数值反馈)
  │                     ├── TIC-GRPO (trajectory ratio 修正)
  │                     ├── Lite PPO (实证最简 critic-free)
  │                     └── 多智能体扩展：
  │                          ├── MAGRPO (Dec-POMDP + centralized group adv)
  │                          ├── Dr. MAS (per-agent normalization)
  │                          ├── TeamTR (trust region for shared-context teams)
  │                          ├── MARFT (Flex-MG framework for LaMAS)
  │                          ├── Advantage Alignment for LLM (anticipatory OS)
  │                          └── ShapeLLM (model-free OS)
  │
  └── DPO 家族：IPO / KTO / ORPO / SimPO / β-DPO ...
  
（test-time 路线：MATTRL = inference-time RL 注入经验）
（评估 benchmark：CRAFT = multi-LLM partial-info 通信）
```

---

### 7.9 这些算法都用在了哪些大模型里（产业落地与影响）

> 把前面的算法对应到真实模型，看它们各自造成了什么影响。

| 算法 | 代表性落地模型 | base model | 影响 |
|---|---|---|---|
| **PPO-RLHF** | InstructGPT、ChatGPT、GPT-4、Claude、Llama-2-Chat | GPT-3 / Llama-2 等各家自有 base | **开启 alignment 时代**：第一次让 LLM "听人话、守规矩"，是 ChatGPT 现象级出圈的技术底座 |
| **DPO** | Zephyr、Tülu 2、Llama-3-Instruct（部分阶段）、大量开源 chat 模型 | Mistral-7B / Llama-2/3 等 | **把 alignment 成本打到地板**：去掉 RM + RL，开源社区做 chat 对齐的默认选择 |
| **GRPO** | **DeepSeek-R1 / R1-Zero**、DeepSeekMath | **DeepSeek-V3-Base / DeepSeekMath-Base（DeepSeek 自家）** | **开启 reasoning RL 时代**：R1-Zero 用纯 RL（无 SFT）涌现出长 CoT、自我验证、反思（"aha moment"），是 2025 reasoning 浪潮的起点 |
| **GRPO（社区复现）** | open-r1、SimpleRL、TinyZero 等 | **Qwen-2.5（base / Math / Coder）** | 证明 **Qwen base + GRPO** 在小模型上也能复现 R1 式 reasoning，是学术界能上手的入口 |
| **R1 蒸馏（SFT，非 RL）** | DeepSeek-R1-Distill-Qwen（1.5B–32B）、R1-Distill-Llama（8B/70B） | **Qwen-2.5 / Llama-3** | 把 R1 的推理能力**蒸馏**到小模型（800K 推理样本 SFT，**不含 RL**），单卡可跑，迅速普及 |
| **DAPO** | Qwen2.5-32B（ByteDance） | Qwen-2.5-32B | R1 之后**最广采用的 GRPO 升级**，AIME 2024 用一半步数超过 R1-Zero |

> **⚠️ 澄清一个高频误解：GRPO、DeepSeek、Qwen 到底什么关系？**
>
> 你问的"GRPO 是不是用在 DeepSeek 里、基于 Qwen 做的、推理变强了"，前后两段对，中间一段需要纠正：
>
> 1. **GRPO 是 DeepSeek 发明的**（DeepSeekMath, Shao et al. 2024），也确实用在 **DeepSeek-R1 / R1-Zero** 上。✅
> 2. **但 R1 / R1-Zero 本体不是基于 Qwen**，而是基于 **DeepSeek 自家的 DeepSeek-V3-Base**。❌ 常见误解
> 3. 跟 Qwen 有关的是**另外两件事**：(a) **DeepSeek-R1-Distill-Qwen** 系列，把 R1 的输出**蒸馏**（纯 SFT，无 RL）到 Qwen-2.5 base 上；(b) **学术界复现 R1**（open-r1 / SimpleRL / TinyZero 等）时，常用 **Qwen-2.5 base + GRPO**，因为 Qwen 全尺寸开源、数学预训练强（见 §2.5.4）。
> 4. **推理能力确实大幅变强**。✅ 尤其 R1-Zero 证明了：**只要 reward 可验证（RLVR）+ GRPO，纯 RL 就能让 base model 自发学会长链推理**，不需要人类示范 CoT。这是 2025 年最重要的发现之一。
>
> **一句话**：GRPO（DeepSeek 出）→ 在 DeepSeek-V3-Base 上训出 R1（推理涌现）→ 再蒸馏到 Qwen/Llama 小模型 + 社区用 Qwen base 复现。**Qwen 是"承接 R1 成果的载体"，不是"R1 本身的底座"。**

---

### 7.10 现状判断：是"都用 PPO 变体"，还是 REINFORCE 在复兴？

> 这是一个对**选 base 算法**很关键的现状问题：当前 LLM RL 是不是基本都是 PPO 的变体？纯 policy gradient（REINFORCE）这条线还有人用吗？
>
> **结论先行**：常见印象"基本都是 PPO 变体"**只对一半**。更准确的说法是：**当前主流是 critic-free 的 policy-gradient（REINFORCE）系，PPO 的完整形态（带 value model 的 actor-critic）反而在退场**。有一整条研究线明确论证"PPO 对 LLM 是 overkill"。

**(1) 两派划分（判据：有没有 PPO 标志性的 importance ratio + clip）**

| | 保留 PPO clip（名义 PPO 派） | 去掉 clip（纯 REINFORCE / policy gradient 派） |
|---|---|---|
| **代表算法** | PPO-RLHF、GRPO、DAPO、REINFORCE++ | **RLOO、ReMax、vanilla PG、RGRA** |
| **critic / value model** | PPO 有；GRPO/DAPO 已去掉 | 全部 critic-free |
| **样本复用** | clip 支持 off-policy 多步 | 多为严格 on-policy |
| **旗帜性论证** | reasoning 主流（R1 / DAPO 实绩） | **"Back to Basics" 直接证明 PPO 组件多余** |

**注意**：两派的**共同点**比差异更重要。除原版 PPO 外，**几乎所有现代算法都已经去掉了 value model（critic）**。所以真正的分水岭不是"PPO vs REINFORCE"，而是"**还要不要 critic**"（答案普遍是不要），以及"**要不要保留 clip 外壳**"。

**(2) 三个关键发现（均已联网核实出处）**

**① 有明确的"REINFORCE 比 PPO 好"研究线**。Ahmadian et al. 2024（ACL，*Back to Basics: Revisiting REINFORCE-Style Optimization for Learning from Human Feedback in LLMs*，[arXiv:2402.14740]）核心结论（原文）：

> "**many components of PPO are unnecessary in an RLHF context** and that far simpler REINFORCE-style optimization variants **outperform both PPO** and 'RL-free' methods such as DPO and RAFT"

它提出的 **RLOO**（纯 REINFORCE + leave-one-out baseline，**无 clip、无 critic**）省 50–70% 显存、快 2–3 倍，效果还更好。这是 policy-gradient 派的旗帜。

**② 连 GRPO 本身"其实是 REINFORCE"**。*Group-Relative REINFORCE Is Secretly an Off-Policy Algorithm*（[arXiv:2509.24203]）论证：GRPO **本质是 REINFORCE 的变体**（用 group 内均值当 baseline），并明确"**demystify importance sampling 和 clipping 在 GRPO 里的作用**"，即 clip/IS 并不像通常以为的那样起关键作用。多篇 2025 工作（RGRA、CoRPO）直接发现**去掉 PPO-style clip 也能学好数学推理**。换句话说：GRPO 名义戴着 PPO clip 的帽子，但在常见的 on-policy（单 inner epoch）运行下，clip 几乎不触发，**本质就是 REINFORCE + group baseline**。

**③ 整体趋势是 critic-free 化**。2025-2026 的方向是把 PPO 最重的零件（value/critic）砍掉。无论叫 GRPO、RLOO、ReMax、REINFORCE++，**共同点都是 critic-free 的 policy gradient**，区别只在 baseline 怎么估、要不要 clip 外壳。还有工作进一步质疑复杂 loss 的必要性（*Are complicated loss functions necessary for teaching LLMs to reason?* [arXiv:2603.18756]）。

**(3) 谱系定位（修正常见误解）**

```
              policy gradient theorem（一切的根）
                         │
        ┌────────────────┴────────────────┐
        ▼                                  ▼
   REINFORCE 系（critic-free）          PPO 系（actor-critic + clip）
   RLOO / ReMax / vanilla PG               PPO-RLHF（原版，带 value model）
   （无 clip，纯 score function）            │
        ▲                                  │ 去掉 critic + 加 group baseline
        │  ← 实质收敛 →                      ▼
        └──────────────────────────  GRPO / DAPO / REINFORCE++
                                      （名义有 clip，实质 ≈ REINFORCE+baseline）
```

两条线在"critic-free + group/MC baseline"处**实质收敛**了。**所以正确的现状判断不是"都用 PPO 变体"，而是"PPO 正被 REINFORCE 系简化吸收，主流是 critic-free policy gradient，只是有些保留了 clip 外壳"。**
> **📚 本节引用（已联网核实）**
> - **[Back to Basics / RLOO]** Ahmadian et al. *Back to Basics: Revisiting REINFORCE-Style Optimization for Learning from Human Feedback in LLMs.* ACL 2024. arXiv:2402.14740.（"PPO 组件多余、REINFORCE 更优"已核实原文）
> - **[GRPO=REINFORCE]** *Group-Relative REINFORCE Is Secretly an Off-Policy Algorithm.* 2025. arXiv:2509.24203.（GRPO 本质是 REINFORCE 变体、clip/IS 作用被夸大，已核实）
> - **[REINFORCE++]** Hu. *REINFORCE++: Stabilizing Critic-Free Policy Optimization with Global Advantage Normalization.* 2025. arXiv:2501.03262.
> - **[复杂 loss 之问]** *Are complicated loss functions necessary for teaching LLMs to reason?* 2026. arXiv:2603.18756.

---

## 八、奖励信号来源的分类（独立于算法）

任何 LLM RL 算法都可以配上以下三类奖励之一：

### 8.1 RLHF（人类反馈）
- 数据：人类对 $(y\_w, y\_l)$ 配对偏好
- 流程：preference → RM → RL
- 代表：InstructGPT, Claude, Llama-Chat
- 痛点：贵，且 RM 容易被 hack

### 8.2 RLAIF（AI 反馈）
- 用强 LLM 替代人类打标签
- 代表：Constitutional AI (Anthropic), RLAIF (Google)
- 优势：scale，可以让 judge 带 CoT
- 痛点：judge 自己有偏

### 8.3 RLVR（可验证奖励）

$$
r(x, y) \;=\; \mathbb{1}\!\left[\text{verify}(x, y)\right]
$$

- 代表：DeepSeek-R1, Kimi-1.5, Tülu-3
- 优势：reward 完全客观，无 reward hacking
- 痛点：仅限于 verifiable 任务（数学、代码、形式语言）

### 8.4 PRM vs ORM（奖励粒度）

正交于 reward 来源：
- **Outcome Reward Model (ORM)**：只看最终答案 $r(y\_T)$
- **Process Reward Model (PRM)**：每步打分 $r(y\_t)$

PRM 更细但训练贵（需要 step-level label）。R1 实证表明 **ORM + GRPO** 已经足够涌现 CoT，**process supervision 并非必需**。

---

## 九、多智能体与自博弈方向

### 9.1 Constitutional AI（Anthropic 2022）
用一组 written principles 让 LLM **self-critique** + **self-revise**，然后再用 RLAIF 训。是一种"内嵌 critic"的设置。

### 9.2 Self-Rewarding LLMs（Yuan 2024）
同一模型既当 actor 又当 judge：
1. Actor 生成 $K$ 个 response
2. Judge（同模型）打分
3. 用 (best, worst) 做 DPO
4. 迭代

### 9.3 SPIN / SPPO
对抗式自博弈。SPIN：当前模型 vs 上一轮模型；SPPO：在 minimax 框架下找 Nash 均衡。

### 9.4 Multi-Agent Fine-Tuning（Subramaniam 2024）
训练多个独立 LLM agent，让它们 cooperative 交互，产生 diverse 数据，各自 SFT。是多 LLM 协作训练的早期代表工作之一。

### 9.5 Debate-based training（Khan 2024 等）
两个 LLM 对一个问题做辩论，judge 决胜。理论上能 scale 到 superhuman oversight。从 information-design 视角看，**debater 在做 Bayesian persuasion**。

---

## 十、一句话总结

> **整个 RL → RLHF → LLM RL 的进化是同一件事在不同层次的展开**：
> 1. 经典 RL 解决"如何在 MDP 里做序列决策"，核心是 policy gradient theorem
> 2. Christiano 2017 发现 RL 可以把"不可微人类偏好"转成可优化损失
> 3. InstructGPT 把这个思路标准化为 SFT + RM + PPO 三段式
> 4. DPO 证明这个流程可以塌缩成一个 closed-form offline loss（policy ↔ reward dual）
> 5. GRPO + RLVR 把 reward 从"human preference"扩展到"verifiable function"，并因此让 reasoning 涌现
> 6. DAPO + Dr. MAS + TeamTR 等 2025-2026 工作把 GRPO 推到多智能体规模，处理 reward 异质性、占用漂移、长 CoT 等工程问题
>
> 每一代算法的核心进化都是**reward 信号怎么来 + reward 信号怎么传**。
