---
title: "LLM Architecture Speedrun"
date: 2026-05-28 17:00:00 +0800
categories: [Artificial Intelligence, Machine Learning Basics]
tags: [Tech, AI, LLM, Transformer]
math: True
---

> **本文目标**:速通 LLM 的架构、参数估算、激活函数、硬件需求等工程基础。
>
> 1. 三种 Transformer 架构(encoder-only / decoder-only / encoder-decoder)
> 2. Decoder-only Transformer 的 forward pass
> 3. 核心架构参数与参数量估算
> 4. 主流厂商、派系、研究界常见选择
> 5. 主流开源 LLM 的具体配置
> 6. 激活函数(GeLU vs SwiGLU)与其他工程参数
> 7. 硬件速通:GPU、显存、训练时间

---

## LLM 架构速通：参数、层数、激活、维度

> 在讲 LLM RL 之前，先把"LLM 长什么样"打底说清楚。后面所有"RM 是一个 7B 模型"、"加 value head"、"训练一步要多少显存"才有具体的尺寸感。

### 2.1 三种 Transformer 架构：encoder-only / decoder-only / encoder-decoder

#### 历史回顾：原始 Transformer 是 encoder-decoder

Transformer 由 Vaswani et al. 2017 在 *Attention Is All You Need* 中首次提出，原始设计是 **encoder-decoder 双塔结构**，专为**机器翻译**任务（"把英文句子翻成法文"）：
- **Encoder** 读源句子（英文），产出一组 context 向量
- **Decoder** **自回归地**一个 token 一个 token 生成目标句子（法文），每生成一个 token 都 cross-attend 到 encoder 的 context

> **🤔 什么是"自回归"（autoregressive）？**
>
> "自回归"（auto-regressive，**AR**）这个词来自**统计学 / 时间序列分析**：**用一个序列自己的过去值来预测当前值**。
>
> 经典 AR(1) 时间序列模型：$x\_t = \phi \cdot x\_{t-1} + \epsilon\_t$，下一个值由前一个值（加噪声）决定。AR(k)：$x\_t = f(x\_{t-1}, x\_{t-2}, \ldots, x\_{t-k})$，由前 $k$ 个值决定。
>
> **搬到 token 序列生成上**，autoregressive generation 就是
> $$p(x_1, x_2, \ldots, x_T) \;=\; \prod_{t=1}^{T} p(x_t \mid x_{<t})$$
> 即把"一段文本的联合概率"分解为"每个 token 在它之前所有 token 条件下的概率的连乘"。这正是 **chain rule of probability**。
>
> 生成时模型一次只产出**一个** token，把它附到序列末尾，然后基于"新长度的序列"再产出下一个 token，如此循环：
>
> ```
> Step 1: context = "What is 2+2?"
>         model: P(next | context) → 采样 next = " The"
>         context ← "What is 2+2? The"
>
> Step 2: context = "What is 2+2? The"
>         model: P(next | context) → 采样 next = " answer"
>         context ← "What is 2+2? The answer"
>
> Step 3: ... 继续直到生成 <EOS> 或达到 max_length
> ```
>
> 字面拆解：**"自" = 用自己的（历史）**，**"回归" = 通过函数预测**。从外部看：模型一次一个 token 顺序输出。
>
> **对立面：非自回归 (non-autoregressive, NAT)**
> - 一次性并行输出整段序列：$p(x\_1, \ldots, x\_T)$ 直接出 $T$ 个 token 的并行预测
> - 优点：生成快（一次 forward = 整段输出）
> - 缺点：每个位置独立决定，互相不 condition，文本不连贯（"the [cat/dog] [sits/runs]" 之间无 coordination）
> - 代表：早期机器翻译的 NAT 模型；2017–2019 有过研究热潮，现在通用文本生成基本不用
>
> **为什么 LLM 都用 autoregressive**：
> - 自然语言有强长程依赖（"the answer is " 后面必然接一个数 / 概念）
> - autoregressive 让每个 token 都 condition 在**全部前文**上，保证连贯性
> - 代价：生成是 **sequential** 的，生成 $T$ 个 token 需要 $T$ 次 forward pass，无法并行。这是 LLM 推理慢的根本原因（KV cache、speculative decoding 等都是在这个约束下的工程优化）
>
> **autoregressive 与 causal mask 的关系**：
> 训练时输入是一段完整文本 $(x\_1, \ldots, x\_T)$，模型一次 forward 同时算所有位置的 next-token loss（**并行训练**）。但每个位置只能 attend 它之前的 token，这就是 **causal mask** 的作用。这样训练时虽然一次性看到整段，每个位置的预测仍只用前文信息，**与生成时的 information set 完全一致**（这就是 §2.1 后面会讨论的"训练-推理一致性"）。
>
> 简言之：**autoregressive 是数学结构（链式分解概率），causal mask 是它在 transformer 上的实现机制（mask 掉未来位置）**。

之后这两半被解耦，演化出**三个家族**：

| 家族 | 代表 | Attention 模式 | 适合任务 |
|---|---|---|---|
| **Encoder-only** | BERT, RoBERTa, DeBERTa | **双向**（每个 token 既能看前也能看后） | 理解类（分类、NER、句对相似度） |
| **Decoder-only** | GPT 系, Llama, Qwen, DeepSeek | **Causal**（每个 token 只能看前） | 生成类（续写、对话、推理） |
| **Encoder-decoder** | 原 Transformer, T5, BART, Whisper | encoder 双向 + decoder causal + cross-attention | 翻译 / 摘要 / 跨模态 |

#### 为什么叫 "encoder" 和 "decoder"——命名的直觉

很多人看 transformer 时直接把 "encoder = 双向 attention"、"decoder = causal attention" 死记硬背。这是**错位的因果**——这两个名字其实来自更基础的信号处理 / 信息论传统：

| | Encoder | Decoder |
|---|---|---|
| **直译** | 编码器 | 解码器 |
| **角色** | 把原始信号"压缩"成内部表示 | 把内部表示"展开"成原始信号 |
| **职责** | 理解 / 表征 | 生成 / 产出 |
| **生活类比** | 阅读：把文字转成脑中的"意思" | 写作：把脑中的"意思"输出成文字 |
| **熟悉例子** | JPEG 编码（图片→bytes）、语音编码（声波→features） | JPEG 解码（bytes→图片）、语音解码（features→声波） |
| **更近的例子** | autoencoder 的前半（input→latent code） | autoencoder 的后半（latent code→reconstruction） |

放到原始 Transformer 的机器翻译场景：

- **Encoder 的职责**：读源句子 "I love cats"，把它**压缩成一组意义向量** $h^{\text{enc}}\_1, h^{\text{enc}}\_2, h^{\text{enc}}\_3$（每个 token 一个 $d\_{\text{hidden}}$ 维向量）。这组向量代表"这句话的意思"——是"翻译时要参考的源信息"。
- **Decoder 的职责**：以这组意义向量为参考，**一次一个 token 地生成**法文句子 "J'aime les chats"。

示意：

```
       Encoder（"理解者"）                    Decoder（"生成者"）
       
       "I"   "love"   "cats"                 一次一个 token 生成：
        ↓     ↓        ↓                     "J'" → "aime" → "les" → "chats"
       双向 self-attention                    每步：
       （每个 token 都综合整段意思）              · 看自己已生成的前缀（causal self-attention）
        ↓     ↓        ↓                      · 同时 cross-attend 到 encoder 的意义向量
       h_1   h_2      h_3   ────────────────→ （问 encoder："你说的'love'对应哪个法文词？"）
       （source 的"语义浓缩"）
```

**关键洞察**：双向 attention 和 causal attention 是 encoder/decoder 这两种角色的**实现选择**，不是定义。
- Encoder 要"理解"输入 → 输入是固定可见的整段 → 自然让每个位置看到全部上下文 → **双向 self-attention** 是最自然的实现
- Decoder 要"生成"输出 → 输出是一个 token 一个 token 顺序产生的 → 自然只能让每个位置看自己已经产生的部分 → **causal self-attention** 是最自然的实现

下面分别讲两种 attention 为什么这样设计。

#### Encoder 的双向 attention：为什么需要看两边

**直觉**：理解一个词的意思，往往需要前后**双向**的上下文。

例子（多义词消歧）：
- "The **bank** by the river" → "bank" 指**河岸**（靠后面 "river"）
- "I deposited money at the **bank**" → "bank" 指**银行**（靠前面 "deposited money"）
- "**Bank** on this idea" → "bank" 是**动词**（靠后面句法结构）

光看左边或右边任意一边，都判断不出 "bank" 的含义。**双向 self-attention** 让每个 token 的表示同时综合了**前文 + 后文**，符合"理解"任务的本质。

具体到 attention 计算：对每个位置 $i$，算 $\text{Attn}(Q\_i, K\_{1..T}, V\_{1..T})$，即 $Q\_i$ 与**所有** $K$ 做点积，**没有任何位置被 mask**。

```
Token:  x_1   x_2   x_3   x_4   x_5
         ↕     ↕     ↕     ↕     ↕      （每个 token 都与所有其他 token 双向交互）
Output: h_1   h_2   h_3   h_4   h_5     （每个 h_i 都看过整段输入）
```

#### Decoder 的 causal attention：为什么不能看未来

**直觉**：生成一段文本时，下一个词**只能**基于已经写出的部分决定——未来还没发生。

例子：你在写一篇博客，写到第三句的时候，你**根本不知道**第五句会是什么。你的第三句不能"参考"第五句。

如果模型训练时"作弊"地用了未来 token 来预测当前 token，部署时未来不存在，模型就崩——这种现象在 NLP 里叫 **exposure bias**（曝光偏差）。

具体到 attention 计算：对每个位置 $i$，只算 $Q\_i$ 与 $K\_{1..i}$（仅自己及之前）。技术上用一个 **causal mask**（下三角矩阵）把未来位置的 attention score 在 softmax 前置为 $-\infty$，softmax 后变成 0。

```
Token:  x_1   x_2   x_3   x_4   x_5
         ↑    ↑←    ↑←←   ↑←←←  ↑←←←←   （每个 token 只 attend 自己及之前）
Output: h_1   h_2   h_3   h_4   h_5     （h_i 只看过 x_1..x_i）
```

#### 训练-推理一致性：causal mask 为什么是必需的

这里有个**微妙但关键**的点。看具体的训练步骤：

训练时我们要做的事：给模型一段完整文本 $(x\_1, x\_2, x\_3, x\_4, x\_5)$，让它**在每个位置上都预测下一个 token**（这样一次 forward 同时学到 4 个位置的预测）：

| 位置 | 输入信息 | 预测目标 |
|---|---|---|
| 1 | $x\_1$ 的 hidden state | $x\_2$ |
| 2 | $x\_2$ 的 hidden state | $x\_3$ |
| 3 | $x\_3$ 的 hidden state | $x\_4$ |
| 4 | $x\_4$ 的 hidden state | $x\_5$ |

这是**并行训练**——一次 forward 同时学到所有位置上"下一个 token 是什么"。比 RNN 时代一个一个 token 训练快得多。

**问题来了**：如果用 encoder 风格的双向 attention，位置 2 的 hidden state $h\_2$ 会综合 $x\_1, x\_2, x\_3, x\_4, x\_5$ 的**全部** 信息——也就是说 **$h\_2$ 直接看见了答案 $x\_3$ 就在那里**。用它来预测 $x\_3$ 就成了 trivial 的 "copy from input"，模型学到的是个 identity 映射，没学到任何有用的语言知识。

**部署时**，模型要生成 $x\_3$ 时，$x\_3$ 根本不存在——它**正是要被预测出来的东西**。模型这时候没法"复制邻居"了，性能直接崩。

**Causal mask 的作用**：训练时强制让位置 $t$ 的 hidden state $h\_t$ **只能看 $x\_1, \ldots, x\_t$**（看不到 $x\_{t+1}, \ldots, x\_T$）。这样：
- 训练时：模型必须真的学会"用过去预测未来"
- 推理时：模型本来就只能用过去——**两者的 information set 完全一致**

> **一句话直觉**：**autoregressive 是生成时的物理约束**（你只能看过去）；**causal mask 是把这个约束在训练时显式注入 transformer**。两者必须配套，否则训练时学的能力在推理时根本用不上。

#### Encoder-Decoder 的 cross-attention：两个模块怎么 "通信"

在 encoder-decoder 架构里（如原始 Transformer / T5）：

1. **Encoder** 用双向 self-attention 把 source 句子（英文）处理成意义向量 $h^{\text{enc}}\_{1..T\_{\text{src}}}$
2. **Decoder** 在生成 target 句子（法文）时，**每一步同时做两件事**：
   - **Causal self-attention**：看自己已生成的法文前缀（保持 autoregressive）
   - **Cross-attention**：把 $Q$ 从 decoder 当前层取出来，$K, V$ 从 encoder 的 $h^{\text{enc}}$ 取出来 —— 让 decoder "提问" encoder "源句子的哪部分对当前生成最相关？"

直觉：decoder 像一个"翻译员"，一边产出译文（看自己已译的部分），一边**反复回头看源文**（看 encoder 的输出）找下一个词怎么翻。

数据流：

```
Encoder:
  source tokens → encoder self-attn (双向) → h^enc_{1..T_src}
                                              │
                                              │ (作为 K, V)
                                              ↓
Decoder:                                   ┌──────────────┐
  生成的 target prefix                      │ Cross-Attn   │
  → decoder self-attn (causal)              │ Q from dec   │ → 当前位置融合 source 信息
  → output h^dec_t            ──── Q ────→  │ K, V from enc│
                                            └──────────────┘
                                                 ↓
                                            下一层 / 下一个 token 预测
```

这就是机器翻译的核心结构。**现代 decoder-only LLM（Llama, GPT）干脆把 cross-attention 也去掉**——把 source 和 target 拼成一段，用统一的 causal self-attention 处理。这就是为什么"decoder-only"的现代 LLM 实际上比原始 Transformer 的 decoder 还要简单一点。

#### Encoder-only 代表：BERT

BERT (Devlin et al. 2018) 把 encoder 单独拎出来用：
- 双向 attention → 每个 token 表示综合上下文
- 预训练目标：**Masked Language Model (MLM)**——把输入里 15% token 换成 `[MASK]`，让模型预测被 mask 的 token（**不是**"下一个 token"，是"被遮掉的 token"）
- 还有 **Next Sentence Prediction (NSP)** 辅助目标
- 用法：fine-tune 时取 `[CLS]` token 的 final hidden state 做分类 / NER / 句对相似度

BERT 在 2018–2020 主导了 NLP benchmark（GLUE / SuperGLUE）。**但 BERT 不能直接做生成**——双向 attention 在生成时无意义（生成时未来不存在）。强行做生成（如 mask 一段然后填）效果差，远不如 GPT。

#### Decoder-only 代表：GPT 系

GPT (Radford et al. 2018) 走另一条路：
- 只用 decoder（去掉 encoder 部分 + cross-attention）
- Causal attention：每个 token 只看前面
- 预训练目标：**单纯的 next-token prediction**（最大化 $\sum\_t \log p(x\_t | x\_{<t})$）
- 用法：把任何任务都 reformulate 成"接续生成"
  - 分类 → `"this review is positive or negative?\nAnswer: [next token]"`
  - 翻译 → `"Translate to French: hello\nFrench: [next tokens]"`
  - 问答 → `"Q: ...\nA: [next tokens]"`

#### 为什么 decoder-only 赢了：6 个理由

**1. 统一的任务框架**
所有 NLP 任务都可以表达成"给一段 prompt，接着生成"。不需要为每个任务设计专门的 head / fine-tune 程序。这就是 GPT-3 论文标题 *Language Models are Few-Shot Learners* 的核心论点：**in-context learning** 在 decoder-only 上自然涌现。

**2. 训练目标更简单、信号更密**
- BERT MLM：每个 batch 只对 15% 位置算 loss → 大部分 forward 计算 wasted
- GPT next-token：**每个位置都算 loss** → 信号密度 ~6× MLM
- 同样的 compute 下 decoder-only 学得更快

**3. 推理时的信息一致性**
推理时模型必然 autoregressive 生成（一次出一个 token）。decoder-only 训练时也是 causal，**训练与推理 information set 完全一致**。BERT 推理时要么不能生成，要么需要 trick；T5 等 encoder-decoder 也需要 teacher-forcing 之类的小动作。

**4. 自然涌现 in-context learning**
当 decoder-only 模型够大、训练数据够多时，它**自发**学会从 prompt 中的少量例子推断任务（无需 fine-tune）。这在 BERT 上不会发生（BERT 不知道"生成"是什么意思）。

**5. 部署简化**
- BERT：每个下游任务都要 fine-tune 一个 checkpoint，部署 N 个任务 = N 个模型
- GPT：一个 base model 服务所有任务，只需要不同的 prompt
- 这导致 OpenAI / Anthropic / DeepSeek 等 API-as-product 模式——单一大模型 + prompt 工程

**6. 实证 scaling law 更陡**
Kaplan et al. 2020 和 Hoffmann et al. (Chinchilla) 2022 的 scaling law 是在 decoder-only 上做的，结果显示 loss vs compute 是清晰的幂律。Encoder-only / encoder-decoder 的同等 scaling 研究少，且没有体现出同样的可预测扩展规律（部分原因是 MLM 训练信号稀疏导致 sample efficiency 低）。

#### Encoder-Decoder 的现状：T5 / BART / Whisper

T5 (Raffel et al. 2019) 和 BART (Lewis et al. 2019) 是 encoder-decoder 的代表。在生成式任务上有过一段辉煌（T5 在 SuperGLUE / SQuAD 都很强）。但**到 GPT-3 之后基本被 decoder-only 取代**——T5-11B 跟 GPT-3-175B 比效果差很多，而且部署复杂。

Encoder-decoder 仍然活在两类任务上：
- **跨模态任务**：input 与 output 在不同 modality 时，encoder 处理 input modality、decoder 生成 output modality。例如 Whisper（音频编码器 + 文本解码器）、CLIP-style 模型、image captioning。
- **结构性翻译类**：明显有"source → target"双向映射的场景，如机器翻译（虽然现在也用 decoder-only 做了）。

#### "decoder-only" 名字的历史包袱

> 现代所谓 "decoder-only Transformer" 其实就是**"causal-attention 的 Transformer"**。"decoder" 这个名字是历史包袱——原始 Transformer 的 decoder 才有 causal mask + cross-attention，现代 LLM **去掉了 cross-attention**（没有 encoder 可 cross），只剩 causal self-attention。但叫了这么多年，名字就保留了。

理解了这点，后面看到 "decoder-only" 别再望文生义——它就是个 causal self-attention transformer，处理任何 token 序列。

### 2.2 LLM 的本体：decoder-only Transformer 的 forward pass

99% 的现代 LLM (GPT 系 / Llama / Qwen / DeepSeek 等) 都是 **decoder-only Transformer**（上面 §2.1 解释了为什么）：
- **Causal self-attention**：每个 token 只能 attend 它之前的 token，看不到未来。这是 autoregressive 的硬件实现。
- 输入 = token ID 序列；输出 = 每个位置上"下一个 token"的 logits over vocab。

forward pass 的整体数据流：

```
Token IDs (整数序列，长度 T)
   ↓
Token Embedding lookup: int → vector of dim d_hidden
   ↓ (加上 RoPE / ALiBi 等 position encoding)
┌──────────────────────────────────────────────────────┐
│  Transformer Block × L 层                            │
│  ┌────────────────────────────────────────────────┐  │
│  │ Pre-Norm: RMSNorm (Llama 系) 或 LayerNorm     │  │
│  │ ↓                                              │  │
│  │ Multi-Head Self-Attention (causal masked)      │  │
│  │ - Q, K, V 各一个 linear: d_hidden → d_hidden   │  │
│  │ - Scaled-Dot-Product Attention                 │  │
│  │ - Output projection: d_hidden → d_hidden       │  │
│  │ ↓ + residual                                   │  │
│  │ Pre-Norm                                       │  │
│  │ ↓                                              │  │
│  │ Feed-Forward Network (FFN / MLP)               │  │
│  │ - SwiGLU: 3 个 linear (gate, up, down)         │  │
│  │   维度: d_hidden ↔ d_ff ↔ d_hidden            │  │
│  │ ↓ + residual                                   │  │
│  └────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────┘
   ↓
Final RMSNorm
   ↓
LM head (Linear d_hidden → |V|)
   ↓
Logits over vocab (T × |V|)
   ↓
softmax → token 分布 → 采样 / argmax
```

### 2.3 几个核心架构参数

| 参数 | 含义 | 典型值（Llama-2 系）|
|---|---|---|
| $L$ | Transformer 层数（depth） | 7B: 32 / 13B: 40 / 70B: 80 |
| $d\_{\text{hidden}}$ | 隐藏维度（每个 token vector 的宽度） | 7B: 4096 / 13B: 5120 / 70B: 8192 |
| $n\_{\text{heads}}$ | attention head 数 | 7B: 32 / 70B: 64 |
| $d\_{\text{head}}$ | 每个 head 的维度 $= d\_{\text{hidden}} / n\_{\text{heads}}$ | 7B/70B: 128 |
| $d\_{\text{ff}}$ | FFN 中间维度 | 通常 $\approx 4 \cdot d\_{\text{hidden}}$（SwiGLU 用 $\approx 8/3 \cdot d\_{\text{hidden}}$） |
| $\|V\|$ | vocab size | Llama-2: 32k / Llama-3: 128k / Qwen-2.5: 152k |
| context length | 最大输入 token 数 | 4k → 32k → 128k → 1M（演进中） |

### 2.4 参数量怎么估

#### 2.4.1 "X B" 是什么意思

LLM 名字里的 **B** = **Billion (10 亿)**。其它常见单位：
- **K** = thousand (千) = $10^3$
- **M** = million (百万) = $10^6$
- **B** = billion (十亿) = $10^9$
- **T** = trillion (万亿) = $10^{12}$

所以：
- **Qwen-7B** = 7 × 10⁹ ≈ **70 亿个参数**
- **Llama-70B** = 700 亿个参数
- **GPT-3 175B** = 1750 亿个参数
- **GPT-4 估计 1.8T** = 1.8 万亿个参数（未公开，估计值）
- **DeepSeek-V3 671B** = 6710 亿总参（MoE：活跃 37B）

**"参数"** 是什么？就是神经网络里所有可训练的浮点数 weight。比如一个 linear layer $W \in \mathbb{R}^{4096 \times 4096}$ 有 $4096^2 \approx 16.8\text{M}$ 个参数；一个 LLM 由几百个这种 linear / embedding / norm 拼起来，参数加总就是 "X B"。

注意：**这个数字只算 weight 本身**，不算梯度、optimizer state、activation 等训练时才有的额外数据。所以"7B 模型"和"训 7B 模型需要的总显存"是两件不同的事（差 4-6 倍，见 §2.9.3）。

#### 2.4.2 参数量怎么算出来

每层 transformer block 的参数主要在两块：
- **Self-attention**：4 个 $d\_{\text{hidden}} \times d\_{\text{hidden}}$ linear (Q, K, V, output) = $4 d\_{\text{hidden}}^2$
- **FFN (SwiGLU)**：3 个 linear (gate, up, down)，总参数 = $3 d\_{\text{hidden}} \cdot d\_{\text{ff}} \approx 8 d\_{\text{hidden}}^2$（when $d\_{\text{ff}} \approx 8/3 \cdot d\_{\text{hidden}}$）

每层总参数 $\approx 12 d\_{\text{hidden}}^2$。
总参数 $\approx L \cdot 12 d\_{\text{hidden}}^2 + \text{embedding overhead}$（embedding 表是 $|V| \cdot d\_{\text{hidden}}$）。

**例：Llama-7B**（$L=32, d\_{\text{hidden}}=4096$）：
- 每层 $\approx 12 \times 4096^2 \approx 201\text{M}$
- 32 层 $\approx 6.4\text{B}$
- + embedding ($32k \times 4096 \approx 0.13\text{B}$，输入输出共享则只算一次)
- 总共 $\approx 6.7\text{B}$ ✓

**记忆口诀**：$L \cdot 12 d^2$ ≈ 总参数量。

#### 2.4.3 X B 模型占多少存储

存储占用 = **参数数量 × 每个参数的字节数**。每个 weight 占几个字节，由数值精度决定：

| 精度 | 每参数字节 | 用途 |
|---|---|---|
| fp32 (单精度) | 4 bytes | 训练时 master weight |
| **bf16 / fp16** (半精度) | **2 bytes** | **推理 / 训练标准** |
| fp8 | 1 byte | 推理优化，部分新模型支持 |
| int8 量化 | 1 byte | 推理量化 |
| int4 / nf4 量化 | 0.5 byte | 极限推理量化（笔记本跑大模型） |

**口诀**：
- **bf16 推理显存（GB） ≈ 模型参数（B） × 2**
- **int4 推理显存（GB） ≈ 模型参数（B） × 0.5**

具体例子：

| 模型 | 参数量 | fp32 | bf16 | int8 | int4 |
|---|---|---|---|---|---|
| Qwen-0.5B | 0.5 B | 2 GB | 1 GB | 0.5 GB | 0.25 GB |
| Qwen-1.5B | 1.5 B | 6 GB | 3 GB | 1.5 GB | 0.75 GB |
| Qwen-3B | 3 B | 12 GB | 6 GB | 3 GB | 1.5 GB |
| **Qwen/Llama-7B** | 7 B | 28 GB | **14 GB** | 7 GB | **3.5 GB** |
| Llama-13B | 13 B | 52 GB | 26 GB | 13 GB | 6.5 GB |
| Qwen-32B | 32 B | 128 GB | 64 GB | 32 GB | 16 GB |
| Llama-70B | 70 B | 280 GB | **140 GB** | 70 GB | **35 GB** |
| GPT-3 | 175 B | 700 GB | 350 GB | 175 GB | 87 GB |
| DeepSeek-V3 (全部) | 671 B | 2.7 TB | 1.3 TB | 671 GB | 335 GB |

**外存（磁盘）= 同 bf16 显存大小**，因为 HuggingFace 下载的 `.safetensors` / `.bin` 文件一般就是 bf16/fp16 存的。例如 `Qwen2.5-7B-Instruct` HuggingFace 下载约 **15 GB**（14 GB 权重 + 几百 MB tokenizer / config）。**Llama-70B 下载约 140 GB**，你要预留够磁盘。

**关键映射**（必须背下来）：
- **7B 模型 ≈ 14 GB（bf16 推理 / 下载大小）**
- **13B 模型 ≈ 26 GB**
- **70B 模型 ≈ 140 GB**

#### 2.4.4 推理 vs 训练显存差几倍

刚才算的"X B × 2 GB"**只够推理**。训练还需要：
- gradient: + 1× weight (bf16)
- Adam optimizer state: + 2× weight (一阶矩 + 二阶矩，fp32 一般)
- activation memory: 取决于 batch / seq len

| 用途 | 显存倍数 | 7B 实际占用 |
|---|---|---|
| **推理** (bf16) | 2× B | **14 GB** |
| **LoRA 训练** (~1% 参数) | ~3× B | ~20 GB |
| **全参 SFT** (bf16 + Adam) | ~10-12× B | ~70-85 GB |
| **PPO-RLHF** (4 模型) | ~16× B | ~112 GB |

详见 §2.9.3 / §2.9.4。这里先记一个粗粒度直觉：
- **推理**显存 ≈ 参数量 × 2 (GB)
- **全参训练**显存 ≈ 参数量 × 10-12 (GB)
- **PPO-RLHF**显存 ≈ 参数量 × 16 (GB)

所以同一张 80 GB 的 H100：
- 推理：可以跑到 ~30B 模型
- 全参 SFT：最多 7B
- PPO-RLHF：连 7B 都要拆到 2 张

### 2.5 主流厂商、派系、研究界常见选择

> 在直接看下一节的"配置表"之前，先把"是谁做的、属于哪个派系、研究界一般用哪个"讲清楚。

#### 2.5.1 业界主要派系

按开源程度从最闭到最开列出：

**全闭源派**（API-only，只能调接口）：

| 厂商 | 模型系列 | 特色 |
|---|---|---|
| **OpenAI** | GPT-3.5 / GPT-4 / GPT-4o / o1 / o3 / GPT-5 | 行业标杆；API-first；引领 reasoning model 潮流（o1 系） |
| **Anthropic** | Claude 1 / 2 / 3 (Haiku/Sonnet/Opus) / Claude 4 | 创始团队来自 OpenAI；Constitutional AI 路线；强 coding + 长 context |
| **Google** | Gemini 1.5 / 2.0 / 2.5 | 超长 context (1M+ tokens)、原生多模态 |
| **xAI** | Grok 系（部分开源） | Musk 旗下；规模激进 |

**开源 / 可商用派**（学术研究主战场）：

| 厂商 | 模型系列 | 特色 | 研究界用法 |
|---|---|---|---|
| **Meta** | LLaMA 1 (2023, 学术) / Llama 2 (2023, 商用) / Llama 3 (2024) / Llama 4 (2025) | 开源 SOTA base model；模型权重 + 训练 recipe 公开；社区生态丰富 | 2023–2024 通用 base 首选；现仍是历史对照标准 |
| **阿里 (通义千问)** | Qwen 1.5 / 2 / 2.5 / 3 | **全尺寸覆盖**（0.5B–72B）；中英双强；Qwen-Math / Qwen-Coder / Qwen-VL 等专用变体；版本迭代极快 | **🌟 2024 后学术界最常用**，SFT/RLHF/reasoning RL 默认 base |
| **DeepSeek** | DeepSeek-V2 / V3 (MoE) / R1 (reasoning) | **MoE 节省推理算力**（V3: 671B 总 / 37B 活）；**GRPO 出自他们**；R1 引领开源 reasoning 浪潮 | reasoning RL / MoE / R1 distill 研究必引 |
| **Mistral AI** (法国) | Mistral 7B / Mixtral 8×7B / Mistral Large | 欧洲代表；MoE 早期开源样板 | 欧洲学术圈 + 部分小模型实验 |
| **智谱 AI** | ChatGLM / GLM-4 | 中文社区起家 | 国内学术 |
| **Moonshot (Kimi)** | Kimi K1.5 (RL 路线代表) | 闭源主营 + 少量开源 | reasoning RL 复现常对照 |
| **Google** | Gemma 2 / 3（开源小弟） | 与 Gemini 同基础，但 license 限制多 | 用得不多 |
| **MiniMax** | MiniMax 系 | 国内 | 国内产品 |
| **百度** | ERNIE 系 | 闭源 | 国内产品 |

**专门化派系**（非通用 LLM）：
- **多模态**：LLaVA (UWisc, 学术) / CogVLM / InternVL (上海 AI Lab) / Qwen-VL
- **代码**：Code Llama (Meta) / DeepSeek-Coder / Qwen-Coder / StarCoder (BigCode)
- **数学**：DeepSeek-Math / Qwen-Math / InternLM-Math

#### 2.5.2 "Llama 系"、"Qwen 系" 到底是什么意思

这种"X 系"的说法在 LLM 圈里很常见，含义有两层：

1. **"由 X 公司发布的模型"**：例如"Llama 系" = Meta 发布的所有 Llama 1/2/3/4 base models。
2. **"基于 X 的 base model fine-tune 出的所有衍生模型"**：例如"Llama-2 衍生" 包括 Vicuna (LMSYS) / Alpaca (Stanford) / WizardLM (微软) / CodeLlama / Open-Llama-Chat 等。一个 base 出来几百个 fine-tune 后代。

类比生物学的"科 / 属 / 种"：
- 派系 = 同 base 演化出的所有模型
- 大家共享同一个 transformer 主干、tokenizer、训练数据底层

**举一个具体例子**：`DeepSeek-R1-Distill-Qwen-7B` 这个名字告诉你三件事：

- **R1-Distill**：用 DeepSeek-R1 蒸馏过
- **Qwen-7B**：底座是 Qwen-7B（不是 Llama-7B）
- 因此它**隶属 Qwen 派 + DeepSeek 蒸馏 lineage** 

模型名字里通常能读出血统，看到陌生模型时先 parse 名字。

#### 2.5.3 研究界常见 base 选择

| 研究方向 | 常用 base model | 为什么 |
|---|---|---|
| **通用 SFT / RLHF** | Qwen-2.5-7B / Llama-3-8B | 中等规模、单卡 80GB 能放、效果好 |
| **数学 reasoning RL**（R1 风格） | **Qwen-2.5-Math-7B** | 数学预训练加强；R1 setup 就用它；社区复现都用 |
| **代码 RL** | **Qwen-2.5-Coder-7B** / DeepSeek-Coder | 代码预训练加强 |
| **DPO / 偏好优化对照** | Llama-2-7B-base + Anthropic HH-RLHF | 历史标准 benchmark |
| **小模型 / 快速 ablation** | Qwen-2.5-0.5B / 1.5B / 3B | 单卡 24GB 就能训，迭代快 |
| **MoE 研究** | DeepSeek-V2/V3 / Mixtral 8×7B | 开源 MoE 主力 |
| **长 context** | Llama-3 (128k) / Qwen-2.5 (32k–1M) | 各自有 long-context 变体 |
| **多模态** | Qwen-VL / LLaVA-NeXT / InternVL | 视觉+语言 SOTA 开源 |
| **复现 R1 / R1-Zero** | Qwen-2.5-Base 系列 | R1 原文实验就用 Qwen base |

#### 2.5.4 为什么 2024 后 Qwen 在研究圈这么火

学术 RL / SFT 论文 **2024 起约 60–70% 用 Qwen 系作 base**（粗略观察 NeurIPS 2025 / ICML 2026 接受论文）。原因：

1. **全尺寸覆盖**：0.5B / 1.5B / 3B / 7B / 14B / 32B / 72B 全有，研究者能按 compute 预算精选最匹配尺寸做 ablation
2. **专门化变体齐全**：Qwen-Math（数学预训练强化）/ Qwen-Coder（代码预训练强化）/ Qwen-VL（多模态），每个研究方向都有对口 base
3. **中英双强**：跟英文-only 的 Llama 比，对中文社区更友好
4. **License 友好**：商用许可宽松（Llama 2/3 商用 license 复杂，部分用途要申请审批）
5. **R1 选用 Qwen 做 base**：DeepSeek-R1 原论文实验用 Qwen-2.5-Math，整个 R1 复现 / 改进的 social proof 都流向 Qwen
6. **迭代极快**：阿里 Qwen 团队几乎每 3-6 个月放新版本，base 持续涨点
7. **生态成熟**：HuggingFace 下载量大、第三方 inference framework 已优化

**例外**：
- 论文要跟 InstructGPT / Anthropic HH 历史 benchmark 对照 → 还是 Llama-2
- 做工业级超大规模复现 → DeepSeek-V3 (MoE)
- 多模态 → LLaVA 系仍流行

#### 2.5.5 一句话总结

> **2024 之前**学术界 base 默认 **Llama-2-7B**；**2024 之后**换成 **Qwen-2.5 系**（特别是 Qwen-2.5-7B / Qwen-2.5-Math-7B / Qwen-2.5-Coder-7B）。reasoning RL 复现普遍用 **Qwen base + DeepSeek-R1 风格的 GRPO**。**Llama / Qwen / DeepSeek** 是当前研究界最重要的三个开源 base 派系，分别在通用 / 全尺寸 / MoE 上有差异化定位。

### 2.6 主流开源 LLM 的具体配置

| 模型 | $L$ | $d\_{\text{hidden}}$ | $n\_{\text{heads}}$ | $\|V\|$ | 参数量 | FFN 激活 | 备注 |
|---|---|---|---|---|---|---|---|
| GPT-2 small | 12 | 768 | 12 | 50k | 124M | GeLU | 远古 |
| GPT-2 XL | 48 | 1600 | 25 | 50k | 1.5B | GeLU | |
| GPT-3 | 96 | 12288 | 96 | 50k | 175B | GeLU | 闭源 |
| **LLaMA-1-7B** | 32 | 4096 | 32 | 32k | 7B | **SwiGLU** | 现代设计起点 |
| Llama-2-7B | 32 | 4096 | 32 | 32k | 7B | SwiGLU | + RLHF |
| Llama-2-13B | 40 | 5120 | 40 | 32k | 13B | SwiGLU | |
| Llama-2-70B | 80 | 8192 | 64 (GQA-8) | 32k | 70B | SwiGLU | GQA 起步 |
| Llama-3-8B | 32 | 4096 | 32 (GQA-8) | **128k** | 8B | SwiGLU | 大 vocab |
| Llama-3-70B | 80 | 8192 | 64 (GQA-8) | 128k | 70B | SwiGLU | |
| **Qwen-2.5-7B** | 28 | 3584 | 28 (GQA-4) | 152k | 7.6B | SwiGLU | R1 base 之一 |
| Qwen-2.5-32B | 64 | 5120 | 40 (GQA-8) | 152k | 32.5B | SwiGLU | |
| Qwen-2.5-72B | 80 | 8192 | 64 (GQA-8) | 152k | 72.7B | SwiGLU | |
| DeepSeek-V2 (MoE) | 60 | 5120 | 128 (MLA) | 102k | **236B 总 / 21B 活** | SwiGLU | MLA + MoE |
| DeepSeek-V3 (MoE) | 61 | 7168 | 128 (MLA) | 129k | **671B 总 / 37B 活** | SwiGLU | DeepSeek-R1 base |

**缩写解释**：
- **GQA (Grouped-Query Attention)**：Q head 多、K/V head 少（多个 Q head 共享一组 K/V），减小 KV cache 显存
- **MoE (Mixture-of-Experts)**：每个 token 只激活一小部分专家 FFN，"总参数大但激活参数小"
- **MLA (Multi-head Latent Attention)**：DeepSeek 的 KV 压缩技巧

### 2.7 激活函数：GeLU vs SwiGLU

- **GeLU** (Gaussian Error Linear Unit，GPT-2/3, BERT 时代)：
  $$\text{GeLU}(x) = x \cdot \Phi(x) \approx 0.5 x \left[1 + \tanh\!\left(\sqrt{2/\pi}(x + 0.044715 x^3)\right)\right]$$
  其中 $\Phi$ 是标准正态累积分布。直觉：smooth 版的 ReLU。FFN 结构是 `Linear → GeLU → Linear`。

- **SwiGLU** (Llama / Qwen / DeepSeek 现代主流)：
  $$\text{SwiGLU}(x) = \text{Swish}(W_{\text{gate}} x) \odot (W_{\text{up}} x), \quad \text{Swish}(z) = z \cdot \mathrm{sigm}(z)$$
  然后 $W\_{\text{down}}$ 投回 $d\_{\text{hidden}}$。**Gated**（门控）+ **Swish**（smooth ReLU）的组合：两个 linear，一个走 swish 当 gate，一个当 value，element-wise 乘后再投回。

  → SwiGLU 的 FFN 是 3 个 linear（gate, up, down），所以参数比 GeLU 的 2-linear FFN 多 50%。为了保持总参数量，SwiGLU 的 $d\_{\text{ff}}$ 通常缩到 $\approx 8/3 \cdot d\_{\text{hidden}}$（GeLU 用 $4 \cdot d\_{\text{hidden}}$）。

- **ReLU**：早期模型偶有使用，现代基本绝迹。

SwiGLU 的优势：实证上略好；gating 允许某些 hidden unit 选择性 "关闭"，增加表达力。

### 2.8 其他重要工程参数

- **Position encoding**：**RoPE** (Rotary Position Embedding) 已成事实标准；旧的 learned/sinusoidal 几乎绝迹。RoPE 在 attention 里以旋转方式注入位置信息，支持外推到训练时未见过的长度（配合 NTK-aware scaling）。
- **Normalization**：**RMSNorm** 替代 LayerNorm（Llama 之后主流），无 bias 项，计算更便宜。
- **Attention 实现**：FlashAttention v2/v3、Memory-efficient SDPA 已是必备
- **KV cache**：生成时缓存历史 attention 的 K/V tensor，避免重复计算。是部署成本的核心之一。
- **量化 / 精度**：bf16 weight + fp32 accumulate 是训练标配；推理常用 int8/int4 量化

### 2.9 硬件速通：GPU、显存、训练时间

> 这一节是给"不知道训练一个模型要多少卡 / 多大显存 / 跑多久"的读者打底。理解了这些数字后，后面 RL 章节里"训一次 PPO-RLHF 几百 GPU-小时"才有具体的尺寸感。

#### 2.9.1 显卡厂商与产品线脉络

LLM 训练 95%+ 用 **NVIDIA GPU**。原因有三：CUDA + cuDNN 生态最成熟（PyTorch / JAX 都是 CUDA-first）；Tensor Core 硬件加速矩阵乘（FP16/BF16/FP8）；NVLink 高速多卡互联。竞争对手（AMD Instinct MI300 / Intel Gaudi / Google TPU / 华为昇腾）在追，但生态差距明显。**默认假设训练用 NVIDIA**。

NVIDIA 的产品线分成**两条独立的线**，名字、license、价格都不一样，这是初学者最容易混的地方。

**消费级（GeForce RTX 系）**：原本给游戏 / 个人渲染用。命名规律 `RTX <代号><档位>`，例如 RTX **30**90 / RTX **40**90 / RTX **50**90。前两位是代号（每代换一次硬件架构），后两位是同代档位（60/70/80/90 由低到高）。

| 代号 | 架构 | 发布 | 旗舰 |
|---|---|---|---|
| 30 | Ampere | 2020 | RTX 3090 |
| 40 | Ada Lovelace | 2022 | RTX 4090 |
| 50 | Blackwell | 2024 | RTX 5090 |

特征：单卡 VRAM 24–32 GB，价格 1–2K USD，适合学生 / 个人 / 推理 / LoRA。但 NVIDIA 在 driver 层面**限制消费卡的多卡训练带宽**（不能用 NVLink，显存不能 P2P），所以 4 张 RTX 4090 的通信带宽 ≪ 2 张 A100。

**数据中心 / 训练专用线**：给云厂、研究机构、企业用，命名规律是**单字母 + 数字**，字母代表代号：

| 字母 | 架构 | 发布 | 旗舰 | 含义 |
|---|---|---|---|---|
| **V** | Volta | 2017 | **V100** | 第一代真正实用的 ML GPU；Tensor Core 引入；至今很多老 cluster 仍在用 |
| **A** | Ampere | 2020 | **A100** (40/80 GB) | 学术界 2020–2023 的事实主力 |
| **H** | Hopper | 2022 | **H100** / H200 | 当前事实标准；H200 是 H100 的高显存版（141 GB） |
| **B** | Blackwell | 2024 | **B100** / B200 / GB200 | 最新一代；B200 算力 ≈ H100 的 2.3 倍 |
| **R** | Rubin | 2026 规划 | R100 | 下一代，仅披露路线图 |

代号字母按字母表顺序推进（V → A → H → B → R），**记忆口诀**：`Volta-Ampere-Hopper-Blackwell-Rubin`。每代名字都来自数学家或科学家（**A**mpere 安培 / **H**opper 计算机先驱 / **B**lackwell 统计学家 / **R**ubin 天文学家）。

**专业工作站线（RTX A / Ada / L 系）**：介于消费级和数据中心之间，常见于工作站。例如 RTX A6000 (Ampere, 48 GB) / RTX 6000 Ada (Ada, 48 GB) / L40S (Ada, 数据中心 inference 卡)。这些卡 license 允许在数据中心用，但训练算力不如 A100/H100。

**Mac 不在这条线**：苹果 M 系列芯片有统一内存（最高 192 GB）能放下大模型，但训练框架（PyTorch MPS）和 CUDA 生态差距大，只适合本地推理和实验，不适合训练。

**实用辨认法**：看到 `RTX <两位数>90` 一律是消费级 / 工作站；看到 `<单字母>100/200` 一律是数据中心。例如读到 "A100" 立刻知道是 Ampere 数据中心卡（不是 Apple 也不是别的），读到 "RTX 4090" 立刻知道是 Ada 消费级旗舰。

#### 2.9.2 主流 GPU 规格速查表

**消费级 / 工作站**：

| GPU | 发布 | VRAM | bf16 TFLOPS | 价格 (USD) | 用途 |
|---|---|---|---|---|---|
| RTX 3090 | 2020 | 24 GB | 71 | ~$1000（二手） | 7B 推理 / LoRA |
| RTX 4090 | 2022 | 24 GB | 165 | ~$1.5K | 个人实验主力 |
| RTX 5090 | 2024 | 32 GB | ~300 | ~$2K | 最新消费级旗舰 |
| RTX A6000 / 6000 Ada | 2021 / 2022 | 48 GB | 91 / 91 | $5–7K | 工作站旗舰 |
| L40 / L40S | 2023 | 48 GB | 181 / 362 | $7–10K | 数据中心 inference |

**数据中心训练旗舰**（学术 / 公司 / 云）：

| GPU | 发布 | VRAM | bf16 TFLOPS | 价格 (USD) | 云租 (USD/hr) | 备注 |
|---|---|---|---|---|---|---|
| V100 | 2017 | 16/32 GB HBM2 | 125 | $5–8K（二手）| ~$0.5 | 第一代实用 ML 卡 |
| A100 40GB | 2020 | 40 GB HBM2 | 312 | ~$8K | ~$1 | 学术老主力 |
| A100 80GB | 2021 | 80 GB HBM2e | 312 | ~$15K | ~$1.5 | 80GB 版价值大 |
| **H100 80GB** | 2022 | 80 GB HBM3 | **989** | ~$30K | $2–3 | **当前事实标准** |
| H200 | 2024 | **141 GB** HBM3e | 989 | ~$35K | $3–4 | FLOPS 同 H100，显存 +75% |
| B100 | 2024 | 192 GB HBM3e | ~1800 | ~$40K | - | Blackwell 新架构 |
| **B200** | 2024 | 192 GB | **2250** | ~$50K | $4–6 | 当前 SOTA |
| GB200 (2 GPU + 1 CPU) | 2025 | 384 GB | 5000 | $80K+ | - | 数据中心顶级 |

**关键算力数字感**：
- H100 算力 $\approx$ A100 的 **3×**
- B200 算力 $\approx$ H100 的 **2.3×**（即 A100 的 ~7×）
- 同样 1 GPU-小时，B200 干的活 ≈ 7 张 A100 ≈ 21 张 V100

#### 2.9.3 训练一个 Qwen-7B SFT 需要多少显存

step by step 估算（**全参数训练**，bf16 + Adam）：

| 项目 | 占用 | 推导 |
|---|---|---|
| 模型权重 (bf16) | 14 GB | $7\text{B} \times 2$ bytes |
| 梯度 (bf16) | 14 GB | 每个 weight 一个 gradient |
| Adam 优化器状态 | 28 GB | 一阶矩 + 二阶矩，各 14 GB |
| Activation memory | 10–30 GB | 取决于 batch size + seq len + gradient checkpointing |
| **合计** | **~65–85 GB** | |

**单 H100 80GB 紧巴巴**：理论刚好够，实际会 OOM。常见解决：
- **2× H100** 用 ZeRO-1（split optimizer state），每卡只占 ~50 GB
- **2× A100 80GB** 同样能跑
- **8× A100 40GB** 用 ZeRO-3（split everything），最经济的学术配置

**LoRA 训练**（只训 ~1% 参数）：
- 不需要全参数 gradient + optimizer state
- 显存需求降到 ~20 GB
- **单卡 RTX 4090 24GB 就能训 7B LoRA**，学生级硬件可用

#### 2.9.4 PPO-RLHF / GRPO 显存需求

PPO-RLHF **4 个模型同时存活**：
- policy $\pi_\theta$（训练，14 + 28 = 42 GB）
- reference $\pi_{\text{SFT}}$（frozen，14 GB）
- reward model $r_\phi$（frozen，14 GB）
- value model $V_\psi$（训练，14 + 28 = 42 GB）
- **总计 ≈ 112 GB** → 单卡不够，**至少 2× H100 80GB 用 ZeRO**

GRPO（**去掉 value model**）：
- policy + ref + RM ≈ 70 GB → **单 H100 80GB 刚好**
- 或 2× A100 40GB 用 ZeRO

**这就是为什么 GRPO 在 LLM RL 圈这么受欢迎**：硬件门槛从"至少 2× H100"降到"1× H100 或 2× A100"，单卡 80GB 用户也能玩 RL。

#### 2.9.5 训练时间估算

以 **Qwen-7B SFT** 一轮（10万条样本，3 epoch，平均 1k token / 样本）为例：

| 硬件 | 时长（LoRA） | 时长（全参） |
|---|---|---|
| 1× RTX 4090 24GB | 1–3 天 | ❌ 显存不够 |
| 1× A100 80GB | 12–24 小时 | 5–10 天 |
| 1× H100 80GB | 5–10 小时 | 2–4 天 |
| 8× A100 80GB | 1.5–3 小时 | 12–24 小时 |
| 8× H100 80GB | 30–60 分钟 | 4–8 小时 |

**Qwen-7B GRPO RL run**（1000 GRPO step，G=16，每 step 16 个 rollout）：

| 硬件 | 时长 |
|---|---|
| 8× A100 80GB | 3–7 天 |
| 8× H100 80GB | 1–3 天 |
| 32× H100 80GB | 6–24 小时 |

**Pretraining 7B from scratch**（1.5T tokens）：

| 硬件 | 时长 |
|---|---|
| 256× H100 | ~3 周 |
| 1024× H100 | ~1 周 |
| 256× A100 | ~2 月 |

#### 2.9.6 学术研究的最低配置 & 典型配置

**最低配置（学生 / 单人）**：
- **1× RTX 4090 24GB** （~$1.5K）：跑 7B 推理 + LoRA 微调够用，做不了全参 SFT
- **1× A100 80GB 云租**（~$1.5/h）：可以做 7B 全参 SFT（慢但可行）+ LoRA RL

**典型学术配置**：
- **8× A100 80GB 节点**（系组共享 + 学校 cluster）：能做 7B 全参 SFT + GRPO + 13B LoRA
- **8× H100 80GB 节点**（云租 ~$30/h，或大学 cluster）：能做 7B PPO-RLHF + 13B 全参 + 70B LoRA

**工业研究 / 大厂**：
- **64× H100 / 128× H100 cluster**：做 70B 训练、长 context、大 RL
- **千卡以上**：pretraining from scratch、超大 RL run

#### 2.9.7 几个工程上必备的省显存技巧

实际训练时几乎都会用：

1. **混合精度 (bf16 / fp16)**：weight 用 16 位而非 32 位 → 显存减半
2. **Gradient checkpointing**：不保存中间 activation，反传时重新前向算 → 显存换计算时间，~30-50% 显存节省
3. **ZeRO-1/2/3 (DeepSpeed) / FSDP (PyTorch)**：跨多卡分摊 optimizer state / gradient / weight
4. **LoRA / QLoRA**：只训 low-rank adapter，省 ~99% 显存
5. **Flash Attention**：attention 算法重写，省 attention 中间矩阵显存
6. **8-bit Adam (bitsandbytes)**：optimizer state 用 int8 存 → optimizer 显存减半

**实际工程组合**：
- 学生级：LoRA + bf16 + FlashAttention → 单 RTX 4090 训 7B
- 学术：bf16 + ZeRO-2 + gradient checkpointing → 8× A100 训 7B 全参 + RL
- 工业：bf16 + ZeRO-3 + 大 batch → 千卡训百 B

#### 2.9.8 把数字串起来：一个具体 PPO-RLHF 训练成本

假设训 **Qwen-7B PPO-RLHF**（10万 prompts，10 epochs）：

- 硬件：8× H100 80GB
- 时长：~2 周
- GPU-小时：$8 \times 24 \times 14 = 2688$ H100-小时
- 云成本：$2688 \times \$2.5 = \$6720$ ≈ 5 万 RMB

GRPO 同样规模：
- 硬件需求降到 2-4× H100
- 时长：~1 周
- 云成本：~$1500-2500 ≈ 1-2 万 RMB

**这就是为什么个人/小团队**用 GRPO + LoRA 在云上租 H100 跑一轮实验是**可行的**（万元级，几天）；而 PPO-RLHF 大规模复现门槛在 **5 万 RMB 起**。

---
