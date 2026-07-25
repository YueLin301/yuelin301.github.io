---
title: "From Noise to Data: An Introduction to Diffusion Models"
date: 2026-07-13 18:03:58 +0800
categories: [Uncategorized, Unfinished]
tags: [Tech, AI, Generative_Models, Diffusion]
math: True
---

> ⚠️ 本文尚在撰写中，内容未完成，后续会继续补充与修订。
{:.prompt-warning}

<!-- description: -->

> 本文由 **Claude Code** 生成。系统梳理扩散模型的核心脉络：前向加噪与反向去噪、DDPM 的变分推导与简化目标、分数匹配与 SDE 统一框架、DDIM 与引导采样、潜空间扩散等，关键名词附原文。仅为学习笔记，非权威综述。
{:.prompt-info}

扩散模型（diffusion models）是当前生成式建模的主导范式。从 Stable Diffusion、Midjourney、DALL·E 这些文生图系统，到视频生成、语音合成、蛋白质结构设计，背后都是同一个思想：与其让网络一步登天地从随机数变出数据，不如把生成拆成一长串小步骤，每一步只做一件简单的事，从带噪声的输入中把噪声去掉一点点。训练时，我们把真实数据逐步加噪直至面目全非；生成时，再让网络沿着相反方向逐步去噪，从纯噪声中一点点“显影”出数据。

这个想法有三条相对独立的源流。Sohl-Dickstein 等人 2015 年从非平衡热力学中借来正反两条马尔可夫链的框架；Song 与 Ermon 2019 年从分数匹配与朗之万动力学出发提出了噪声条件分数网络；Ho 等人 2020 年的 DDPM 则把变分推导简化为一个极其朴素的回归目标，扩散模型的图像质量自此起飞。2021 年，Song 等人用随机微分方程把这几条线索统一在同一张图景之下。下文沿这条主线展开：先交代生成式建模的背景，再依次推导前向过程、反向过程与 DDPM 的训练目标，然后引入分数匹配与 SDE 视角，最后讲采样加速、条件引导与潜空间扩散。

## 1. 背景：生成式建模的图景

### 1. 生成模型要做什么

判别式模型学习的是条件分布：给定输入，预测标签。生成式模型的野心更大：给定一堆样本（图像、文本、分子结构），学习产生这些样本的分布 $p(x)$ 本身，从而能够采样出以假乱真的新样本、评估任一样本的似然、或在给定部分观测时补全其余。困难在于，真实数据分布高维、多峰、支撑集弯曲（图像流形的维数远低于像素空间维数），任何显式的参数化都要在“表达力”“可采样”“可算似然”这三者之间做权衡。

### 2. 几大家族及其取舍

深度生成模型的几大家族，各自选择了不同的权衡方式。**自回归模型**（autoregressive models）把联合分布按链式法则拆成一串条件分布，似然精确、训练稳定，但采样必须逐维串行，对高维图像极慢。**变分自编码器**（VAE, Kingma & Welling, 2013）引入潜变量并优化证据下界，采样一步到位，但历史上生成的图像偏模糊。**生成对抗网络**（GAN, Goodfellow et al., 2014）用判别器与生成器的博弈绕开似然，样本锐利，但训练不稳定、容易模式坍缩，且没有似然可言。**标准化流**（normalizing flows）用可逆变换精确变换密度，但可逆性约束限制了架构自由度。

扩散模型的取舍是：用一个固定的、不需要学习的前向过程把推断问题变简单，把全部学习负担放在反向去噪上。它的训练目标是一个稳定的回归损失，没有对抗博弈；它有变分下界意义上的似然；代价是采样需要迭代多步，原始 DDPM 要跑上千次网络前向。后文的大量工作，正是围绕“如何少走几步”展开的。

| 家族 | 似然 | 采样 | 训练稳定性 | 典型弱点 |
| --- | --- | --- | --- | --- |
| 自回归 | 精确 | 串行、慢 | 稳定 | 高维图像采样代价大 |
| VAE | 下界 | 一步 | 稳定 | 样本偏模糊 |
| GAN | 无 | 一步 | 不稳定 | 模式坍缩、难评估 |
| 标准化流 | 精确 | 一步 | 稳定 | 架构受可逆性约束 |
| 扩散模型 | 下界（ODE 视角下可精确） | 迭代、慢 | 稳定 | 采样步数多 |

### 3. 三条源流

把时间线排开更清楚。2015 年，Sohl-Dickstein 等人的《利用非平衡热力学做深度无监督学习》（Deep Unsupervised Learning using Nonequilibrium Thermodynamics, ICML 2015）首次提出完整框架：一条固定的前向马尔可夫链逐步破坏数据结构，一条学习的反向链逐步恢复之，用变分下界训练。这篇论文超前于时代，沉寂了数年。2019 年，Song 与 Ermon 的**噪声条件分数网络**（NCSN, NeurIPS 2019）从另一个方向切入：学习数据分布的**分数函数**（score function），即对数密度的梯度，再用朗之万动力学采样；为了解决低密度区域分数估计不准的问题，他们用多个尺度的高斯噪声扰动数据，逐级退火采样。2020 年，Ho、Jain 与 Abbeel 的**去噪扩散概率模型**（DDPM, NeurIPS 2020）回到 Sohl-Dickstein 的框架，但把目标函数简化为“预测所加的噪声”这一朴素回归，并配上 U-Net 架构，首次在无条件图像生成上逼近 GAN 的质量。2021 年，Song 等人的《基于分数的 SDE 生成建模》（Score-Based Generative Modeling through Stochastic Differential Equations, ICLR 2021）证明 DDPM 与 NCSN 分别是同一连续时间框架的两种离散化，正反向过程统一为一对随机微分方程。

## 2. 前向过程：把数据逐步变成噪声

### 1. 马尔可夫加噪链

设数据 $x\_0 \sim q(x\_0)$。前向过程（forward process，又称扩散过程）是一条固定的马尔可夫链，共 $T$ 步（DDPM 取 $T=1000$），每一步向样本注入少量高斯噪声：

$$
q(x_t \mid x_{t-1}) = \mathcal{N}\left(x_t;\ \sqrt{1-\beta_t}\, x_{t-1},\ \beta_t \mathbf{I}\right),
$$

其中 $\beta\_t \in (0,1)$ 是预先给定的**噪声调度**（noise schedule）。DDPM 用从 $10^{-4}$ 线性增长到 $0.02$ 的调度；Nichol 与 Dhariwal（ICML 2021）后来提出的余弦调度在低分辨率图像上表现更好。系数 $\sqrt{1-\beta\_t}$ 不是装饰：它在注入噪声的同时把信号略微缩小，保证链的方差不爆炸，并使整条链的终点趋于标准高斯。

### 2. 任意时刻的封闭形式

这条链最重要的性质是：从 $x\_0$ 直接跳到任意时刻 $t$ 有封闭形式。记 $\alpha\_t = 1-\beta\_t$，$\bar{\alpha}\_t = \prod\_{s=1}^{t} \alpha\_s$，则高斯的复合仍是高斯：

$$
q(x_t \mid x_0) = \mathcal{N}\left(x_t;\ \sqrt{\bar{\alpha}_t}\, x_0,\ (1-\bar{\alpha}_t) \mathbf{I}\right).
$$

用重参数化写出来，就是训练时反复用到的一行式子：

$$
x_t = \sqrt{\bar{\alpha}_t}\, x_0 + \sqrt{1-\bar{\alpha}_t}\, \epsilon, \qquad \epsilon \sim \mathcal{N}(0, \mathbf{I}).
$$

随着 $t$ 增大，$\bar{\alpha}\_t$ 单调下降趋于 $0$：信号系数 $\sqrt{\bar{\alpha}\_t}$ 衰减，噪声系数 $\sqrt{1-\bar{\alpha}\_t}$ 增长，信噪比一路走低；当 $t=T$ 时 $x\_T$ 几乎就是纯噪声 $\mathcal{N}(0,\mathbf{I})$。生成的起点因此可以直接从标准高斯采样，与数据无关。

### 3. 这个设计的两处好处

其一是训练效率：因为 $q(x\_t \mid x\_0)$ 有封闭形式，训练时可以对每个样本随机抽一个 $t$，一步到位地造出 $x\_t$，不必逐步模拟整条链。其二是数学上的驯服：前向核是高斯，而且步长很小，这使得下一节的反向条件分布既有高斯近似的理论依据，又在给定 $x\_0$ 时有精确的高斯后验可作回归靶子。

## 3. 反向过程与变分下界

### 1. 反向马尔可夫链

生成即是逆转扩散：从 $x\_T \sim \mathcal{N}(0,\mathbf{I})$ 出发，逐步采样 $x\_{T-1}, \dots, x\_0$。真实的反向条件分布 $q(x\_{t-1} \mid x\_t)$ 依赖于整个数据分布，无法直接写出；但可以证明，当每步的 $\beta\_t$ 足够小时，它近似为高斯。于是我们用一条参数化的反向链去逼近：

$$
p_\theta(x_{t-1} \mid x_t) = \mathcal{N}\left(x_{t-1};\ \mu_\theta(x_t, t),\ \Sigma_\theta(x_t, t)\right),
$$

DDPM 把协方差固定为 $\sigma\_t^2 \mathbf{I}$（取 $\beta\_t$ 或下文的 $\tilde{\beta}\_t$，两者效果相近），只学习均值；Nichol 与 Dhariwal 后来证明把协方差也学出来对似然有帮助。

### 2. 变分下界

训练目标从最大似然出发。与 VAE 同理，对数似然有证据下界（ELBO），按时间步展开为：

$$
\mathbb{E}_q\left[ -\log p_\theta(x_0) \right] \le \mathbb{E}_q\Big[ \underbrace{D_{\mathrm{KL}}\big(q(x_T \mid x_0)\,\|\, p(x_T)\big)}_{L_T} + \sum_{t=2}^{T} \underbrace{D_{\mathrm{KL}}\big(q(x_{t-1} \mid x_t, x_0)\,\|\, p_\theta(x_{t-1} \mid x_t)\big)}_{L_{t-1}} \underbrace{-\log p_\theta(x_0 \mid x_1)}_{L_0} \Big].
$$

$L\_T$ 不含参数（前向链是固定的），$L\_0$ 是最后一步的重构项，真正的主体是中间那串 KL 散度：在每个时间步，让学到的反向核对齐一个**有闭式解的靶子**。

### 3. 高斯后验：靶子的闭式解

靶子之所以有闭式解，是因为给定 $x\_0$ 之后，前向链的条件后验是高斯（贝叶斯公式加高斯共轭）：

$$
q(x_{t-1} \mid x_t, x_0) = \mathcal{N}\left(x_{t-1};\ \tilde{\mu}_t(x_t, x_0),\ \tilde{\beta}_t \mathbf{I}\right),
$$

其中

$$
\tilde{\mu}_t(x_t, x_0) = \frac{\sqrt{\bar{\alpha}_{t-1}}\,\beta_t}{1-\bar{\alpha}_t}\, x_0 + \frac{\sqrt{\alpha_t}\,(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t}\, x_t, \qquad \tilde{\beta}_t = \frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t}\,\beta_t.
$$

两个高斯之间的 KL 散度有解析式，于是每一项 $L\_{t-1}$ 都化为均值之间的加权平方误差。整个变分推导到此结束，剩下的问题只有一个：网络到底该输出什么。

## 4. DDPM：预测噪声的简化目标

### 1. 参数化的选择

网络可以直接输出均值 $\mu\_\theta$，也可以输出对 $x\_0$ 的估计，还有第三种选择：输出前向过程所加的噪声 $\epsilon$。三者数学上等价，因为由重参数化式可以解出

$$
x_0 = \frac{1}{\sqrt{\bar{\alpha}_t}}\left(x_t - \sqrt{1-\bar{\alpha}_t}\,\epsilon\right),
$$

代回 $\tilde{\mu}\_t$ 便得到均值关于噪声的表达式。若让网络 $\epsilon\_\theta(x\_t, t)$ 预测噪声，则反向均值取

$$
\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\,\epsilon_\theta(x_t, t)\right).
$$

DDPM 的实验发现，预测 $\epsilon$ 的参数化在样本质量上明显占优。

### 2. 简化目标

把噪声参数化代入变分下界，每项 KL 化为对噪声预测误差的加权平方损失。DDPM 的关键一步是干脆把权重丢掉，得到**简化目标**（simplified objective）：

$$
L_{\text{simple}}(\theta) = \mathbb{E}_{t,\, x_0,\, \epsilon}\left[ \left\| \epsilon - \epsilon_\theta\big(\sqrt{\bar{\alpha}_t}\, x_0 + \sqrt{1-\bar{\alpha}_t}\,\epsilon,\ t\big) \right\|^2 \right],
$$

其中 $t$ 在 $1$ 到 $T$ 上均匀抽取。丢掉权重不是偷懒：相对于严格的变分权重，均匀加权实际上加重了大噪声时间步（更难的去噪任务）的比重，实验上样本质量反而更好。整个训练循环因此朴素得出奇：抽一张图，抽一个时刻，抽一份噪声，按封闭形式合成 $x\_t$，让网络猜噪声，回传均方误差。没有对抗，没有退火，没有内层循环。

### 3. 采样

训练完成后，生成按反向链逐步进行：从 $x\_T \sim \mathcal{N}(0,\mathbf{I})$ 出发，重复

$$
x_{t-1} = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\,\epsilon_\theta(x_t, t)\right) + \sigma_t z, \qquad z \sim \mathcal{N}(0, \mathbf{I}),
$$

直到 $t=1$（最后一步不加噪声）。每一步都要跑一次网络前向，$T=1000$ 步的代价正是扩散模型早期最受诟病的短板。

### 4. 架构一瞥

DDPM 用的骨干是 **U-Net**：多尺度的卷积编码-解码结构，跳连保留细节，中低分辨率层插入自注意力；时间步 $t$ 经正弦位置编码嵌入后注入每个残差块，使同一网络能在所有噪声水平上工作。这一选择统治了扩散模型三年，直到 Peebles 与 Xie 的 **DiT**（Diffusion Transformer, ICCV 2023）证明纯 Transformer 骨干在算力放大时扩展性更好；后来的 Sora 类视频生成模型即建立在 DiT 路线之上。

## 5. 分数匹配视角与 SDE 统一框架

### 1. 分数函数与朗之万动力学

换一条完全不同的思路。定义分布 $p(x)$ 的**分数函数**（score function）为对数密度的梯度 $\nabla\_x \log p(x)$。它指向密度上升最快的方向，而且不依赖归一化常数。若已知分数，**朗之万动力学**（Langevin dynamics）就能采样：从任意初始点出发，反复迭代

$$
x_{k+1} = x_k + \frac{\delta}{2}\, \nabla_x \log p(x_k) + \sqrt{\delta}\, z_k, \qquad z_k \sim \mathcal{N}(0, \mathbf{I}),
$$

步长 $\delta$ 足够小、步数足够多时，迭代所得样本的分布便收敛到 $p(x)$。于是生成问题化为：如何从样本中学出分数？这就是**分数匹配**（score matching, Hyvärinen, 2005）。但直接在原始数据上做分数匹配有两个障碍：数据集中在低维流形附近，流形之外密度为零，分数无定义；而低密度区域样本稀少，估计极不准，朗之万动力学恰恰要从那里出发、路过那里。Song 与 Ermon 的对策是：用一系列由大到小的高斯噪声扰动数据，让每个尺度下的加噪分布铺满全空间，训练一个以噪声水平为条件的分数网络，采样时从大噪声尺度逐级退火到小尺度。

### 2. 去噪与分数的等价

分数匹配与去噪之间有一条精确的桥，即 Vincent（2011）的**去噪分数匹配**（denoising score matching）：学习加噪分布的分数，等价于学习从加噪样本回归干净样本的去噪器。对扩散模型的高斯扰动核 $q(x\_t \mid x\_0)$ 求梯度立得

$$
\nabla_{x_t} \log q(x_t \mid x_0) = -\frac{x_t - \sqrt{\bar{\alpha}_t}\, x_0}{1-\bar{\alpha}_t} = -\frac{\epsilon}{\sqrt{1-\bar{\alpha}_t}},
$$

也就是说，DDPM 训练的噪声预测网络与分数只差一个缩放：$\epsilon\_\theta(x\_t, t)$ 学到的正是 $-\sqrt{1-\bar{\alpha}\_t}$ 乘以加噪分布的分数。DDPM 与 NCSN 由此殊途同归：一个从变分推断出发，一个从分数匹配出发，训练的是同一个对象。

### 3. SDE 框架：连续时间的统一

Song 等人（ICLR 2021）把离散的加噪链推广为连续时间的**随机微分方程**（SDE）。前向过程写成

$$
\mathrm{d}x = f(x, t)\,\mathrm{d}t + g(t)\,\mathrm{d}w,
$$

其中 $w$ 是标准布朗运动。DDPM 对应**方差保持**（variance preserving, VP）SDE，NCSN 对应**方差爆炸**（variance exploding, VE）SDE，两者只是漂移与扩散系数的不同取法。生成依赖一个经典结果（Anderson, 1982）：扩散过程的时间反演仍是扩散过程，且反向 SDE 为

$$
\mathrm{d}x = \left[ f(x, t) - g(t)^2\, \nabla_x \log p_t(x) \right] \mathrm{d}t + g(t)\,\mathrm{d}\bar{w},
$$

唯一未知量恰是各时刻边缘分布的分数 $\nabla\_x \log p\_t(x)$，由网络估计。更妙的是存在一条**概率流 ODE**（probability flow ODE）：

$$
\mathrm{d}x = \left[ f(x, t) - \frac{1}{2} g(t)^2\, \nabla_x \log p_t(x) \right] \mathrm{d}t,
$$

它与反向 SDE 有完全相同的边缘分布，却是确定性的：给定终点噪声，轨迹唯一。这带来三件礼物：可以用现成的 ODE 求解器少步采样；可以在数据与噪声之间来回编码解码（潜变量插值由此而来）；可以经由瞬时变量变换公式精确计算对数似然。

| 视角 | 训练目标 | 采样方式 | 代表工作 |
| --- | --- | --- | --- |
| 变分（离散链） | 变分下界化为噪声回归 | 反向链逐步采样 | Sohl-Dickstein 2015, DDPM 2020 |
| 分数匹配 | 多尺度去噪分数匹配 | 退火朗之万动力学 | NCSN 2019 |
| SDE（连续时间） | 连续时间分数匹配 | 反向 SDE 或概率流 ODE | Song et al. 2021 |

## 6. 采样加速与条件引导

### 1. DDIM：确定性的少步采样

Song、Meng 与 Ermon 的 **DDIM**（Denoising Diffusion Implicit Models, ICLR 2021）观察到：DDPM 的训练目标只依赖边缘分布 $q(x\_t \mid x\_0)$，而不依赖链的马尔可夫性。于是可以构造一族非马尔可夫的前向过程，边缘分布不变（训练好的网络照用），反向采样却可以取

$$
x_{t-1} = \sqrt{\bar{\alpha}_{t-1}}\, \hat{x}_0 + \sqrt{1-\bar{\alpha}_{t-1}-\sigma_t^2}\;\epsilon_\theta(x_t, t) + \sigma_t z,
$$

其中 $\hat{x}\_0$ 是由当前噪声预测反解出的对 $x\_0$ 的估计。把 $\sigma\_t$ 取为零，采样就完全确定，并且可以只在时间步的一个子序列上跳跃前进：实践中 20 至 50 步就能得到接近千步 DDPM 的质量。确定性的 DDIM 采样恰是概率流 ODE 的一种离散化，两种视角在此又汇合了。

### 2. 分类器引导

生成往往需要服从条件：类别标签、文本描述。按贝叶斯公式，条件分数拆成两项：

$$
\nabla_x \log p(x \mid y) = \nabla_x \log p(x) + \nabla_x \log p(y \mid x).
$$

Dhariwal 与 Nichol（NeurIPS 2021）据此提出**分类器引导**（classifier guidance）：额外训练一个在加噪图像上工作的分类器，采样时把其梯度乘上系数 $w$ 加到分数上，$w$ 越大越贴合条件、多样性越低。这篇论文（“扩散模型在图像合成上击败 GAN”）正是扩散模型全面超越 GAN 的标志。代价是要多训练一个分类器，而且它必须能应付所有噪声水平。

### 3. 无分类器引导

Ho 与 Salimans 的**无分类器引导**（classifier-free guidance, CFG）把分类器省掉：训练时以一定概率（如 10%）把条件置空，让同一网络兼学条件与无条件两个模型；采样时外推两者之差，

$$
\tilde{\epsilon}_\theta(x_t, c) = \epsilon_\theta(x_t, \varnothing) + s \cdot \big( \epsilon_\theta(x_t, c) - \epsilon_\theta(x_t, \varnothing) \big),
$$

引导系数 $s$ 大于 1 时把样本推向条件更“典型”的区域。CFG 几乎是所有现代文生图系统的标配，其代价同样是多样性换保真度，以及每步两次网络前向。

### 4. 蒸馏与一步生成

减少采样步数的另一条路是蒸馏。**渐进式蒸馏**（progressive distillation, Salimans & Ho, ICLR 2022）反复让学生网络用一步模仿教师两步，步数指数下降。**一致性模型**（consistency models, Song et al., ICML 2023）则训练网络把概率流 ODE 同一条轨迹上的任意点都映射到同一起点，从而支持一步或数步生成。这一方向发展极快，此处仅立路标。

## 7. 潜空间扩散与生态

### 1. 潜空间扩散：Stable Diffusion 的关键一步

在像素空间跑扩散，分辨率一高代价就不可承受。Rombach 等人的**潜空间扩散模型**（Latent Diffusion Models, CVPR 2022）先用一个感知压缩的自编码器把图像压到低维潜空间（如下采样 8 倍），在潜空间里做扩散，最后解码回像素。感知上不重要的高频细节交给自编码器，扩散模型专注于语义层面的组合，训练与推理代价骤降。配合在文本等条件上的**交叉注意力**（cross-attention）注入机制，这项工作以 Stable Diffusion 之名开源，成为扩散模型平民化的转折点。

### 2. 文生图系统的谱系

2021 至 2022 年间，文生图系统密集爆发：OpenAI 的 GLIDE 与 DALL·E 2（后者以 CLIP 图像嵌入为中介，又称 unCLIP），Google 的 Imagen（冻结的大型语言模型 T5 做文本编码器，级联超分辨率），以及开源的 Stable Diffusion 及其后继 SDXL。它们的共同配方是：强文本编码器加 CFG 加高分辨率策略（级联或潜空间）。评估上常用 FID 衡量图像质量、CLIP 分数衡量图文对齐，但两者都与人类偏好有明显偏差，人工评估仍不可替代。

### 3. 图像之外

扩散框架对数据形态的假设很弱，凡能加噪的地方都能扩散。视频生成把时间维并入 DiT 骨干（Sora 类系统）；语音与音乐合成早有 WaveGrad、DiffWave 一系；分子与蛋白质设计中，RFdiffusion 在骨架生成上已进入实际管线（Watson et al., Nature 2023）；机器人学里，扩散策略（diffusion policy）把动作序列的多峰分布建模得比高斯策略好得多；在离散的文本域，离散扩散与连续潜空间扩散都在活跃探索中，近来的扩散语言模型已开始挑战自回归范式的地位。

### 4. 新趋势：从扩散到流

一条正在改写默认选项的路线是**流匹配**（flow matching, Lipman et al., ICLR 2023）与**整流流**（rectified flow, Liu et al., ICLR 2023）：不再绕道 SDE，而是直接学习把噪声输运到数据的向量场，路径取直线，训练同样是简单回归。它保留了扩散的稳定性，路径更短更直，少步采样更自然，Stable Diffusion 3 与 Flux 等新一代模型已采用这一参数化。从 2015 年的热力学隐喻到今天的最优输运语言，这个领域的理论外衣换了几层，内核却始终未变：把困难的生成拆成一串简单的去噪。

至此，本文铺开的是扩散模型的主干：前向加噪的封闭形式、变分下界与噪声回归的等价、分数与 SDE 的统一视角、DDIM 与 CFG 这两件日用工具，以及潜空间扩散的工程转折。尚未展开的部分留待后续补充。

> 待补充章节：ELBO 的逐项完整推导；SDE 视角的似然计算与瞬时变量变换公式；余弦调度与 v-parameterization 等训练细节；flow matching 与 rectified flow 专节；一致性模型与对抗蒸馏；离散扩散与扩散语言模型。
{:.prompt-warning}
