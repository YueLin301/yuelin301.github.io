---
title: RL Toolbox
date: 2023-04-11 02:40:00 +0800
categories: [Code]
tags: [toolbox, reinforcement learning]
math: True
---

> This note will be consistently updated.
{: .prompt-info }

---

## Standardization

## Mask


## Gumbel-Softmax
- Reparameterization.
- Maintain gradients from the sampled variables.
- Commonly used in communication methods.

### What is gumbel-softmax for?
If $a_t\sim \pi_\theta(\cdot \mid s_t)$, then how to calculate $\nabla_\theta a_t$?

### What is reparameterization?
This trick decouples the deterministic part and the random part of a variable.

This concept can be best illustrated with the example of the Gaussian distribution.

If $z\sim \mathcal{N}(\mu,\sigma^2)$, then $z = \mu + \sigma \cdot \epsilon$, where $\epsilon\sim \mathcal{N}(0,1)$. In this way, $\frac{\partial z}{\partial \mu} = 1$ and $\frac{\partial z}{\partial \sigma} = \epsilon$. Usually $\mu$ and $\sigma$ are estimated by a neural network, and the following gradient can be automatically calculated by deep frameworks.

### What does Gumbel-Softmax do?

We often use neural networks to generate a probability simplex, i.e., a profile of probability where $0\le p_i$ and $\sum\limits_{i} p_i = 1$. Then we will sample an $x$ based on this distribution. 

An example scenario is in RL, where an agent needs to choose an action $a_t$. We output a distribution $\pi(\cdot \mid s_t)$ and then sample an action $a_t\sim \pi(\cdot \mid s_t)$ based on this distribution to execute.

Gumbel-Softmax is used to reparameterization this kind of categorical distribution. This technique allows samples to be drawn according to the original distribution and enables gradient computation.

$$
z\sim \arg\max\limits_i (\log(p_i) + g_i),
$$

where $g_i = -\log(-\log (u_i)), u_i\sim U(0,1)$. 

The argmax is non-differentiable, it can be replaced with softmax. $i = \arg\max\limits_{j} (x_j)$.

$$
\text{softmax}_T (x) = \frac{e^{x_j/T}}{\sum_k e^{x_k/T}}.
$$

If temperature $T$ is small enough, then the output of the softmax can be seen as a one-hot vector which indicates $i$.

### Example code
Check [my note](https://yuelin301.github.io/posts/Computation-Graph-Visualization/#example-5-nabla_theta-a-with-gumbel-softmax-reparameterization).

## Social Influence
- A MARL method.
- An intrinsic reward.
- Agent $i$ chooses the action that has the most impact on others.

$$
\begin{aligned}
    r_t^i 
    =& \sum\limits_{j\ne i} D_{KL}\left[\pi^j(a_t^j \mid s_t, a_t^i) \Big\Vert \sum\limits_{a_t^{i\prime}} \pi^j(a_t^j \mid s_t, a_t^{i\prime})\cdot \pi^i(a_t^{i\prime}\mid s_t) \right] \\
    =& \sum\limits_{j\ne i} D_{KL}\left[\pi^j(a_t^j \mid s_t, a_t^i) \Big\Vert P(a_t^j\mid s_t) \right]
\end{aligned}
$$

In the principal-agent communication:

$$
r^i = D_{KL}\left[ \pi^j(a^j\mid\sigma^i) \Big\Vert \sum\limits_{\sigma'}\varphi^i(\sigma^{i\prime}\mid s)\cdot \pi^j(a^j\mid\sigma^{i\prime})\right]
$$

## PPO 37 Tricks
> [The 37 Implementation Details of Proximal Policy Optimization](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/).
{: .prompt-info }

