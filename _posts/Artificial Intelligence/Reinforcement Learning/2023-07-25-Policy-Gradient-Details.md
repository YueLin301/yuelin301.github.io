---
title: Details on the Analysis of Policy Gradient Methods
date: 2023-07-25 02:40:00 +0800
categories: [Artificial Intelligence, Reinforcement Learning]
tags: [tech, convergence, policy gradient, reinforcement learning, tech]
math: True
pin: True
---

> The only way to make sense out of change is to plunge into it, move with it, and join the dance. *— Alan Watts.*

---



## Policy Gradient Theorem

> The proofs of the stochastic and deterministic policy gradient theorem are mainly summarized from [this blog](https://lilianweng.github.io/posts/2018-04-08-policy-gradient/#off-policy-policy-gradient) and the supplementary of the paper "[Deterministic Policy Gradient Algorithms](https://scholar.archive.org/work/v7bb4lgn2zhnta3bassfawrane/access/wayback/https://hal.inria.fr/hal-00938992/file/dpg-icml2014.pdf)," respectively.
{: .prompt-info }



### Stochastic Policy Gradient

#### Objectitve

$$
\max\limits_{\theta} J(\theta) = \mathbb{E}_{s_0\sim d_0}\left[ \textcolor{red}{V^{\pi_\theta} (s_0)} \right].
$$



#### Gradient

$$
\begin{aligned}
\nabla_\theta V^{\pi_\theta}(s_0) 
=& \frac{1}{(1-\gamma)} \sum\limits_{s} d^{\pi_\theta}(s\mid s_0) \sum\limits_{a} Q^{\pi_\theta}(s,a) \cdot \nabla_\theta \pi_\theta(a\mid s) \\
=& \frac{1}{(1-\gamma)} \sum\limits_{s} d^{\pi_\theta}(s\mid s_0) \sum\limits_{a} \pi(a\mid s) \cdot Q^{\pi_\theta}(s,a) \cdot \nabla_\theta \ln \pi_\theta(a\mid s) \\
=& \frac{1}{(1-\gamma)} \mathbb{E}_{s \sim d^{\pi_\theta}(\cdot \mid s_0)} \mathbb{E}_{a\sim \pi(\cdot \mid s)} \left[Q^{\pi_\theta}(s,a) \cdot \nabla_\theta \ln \pi_\theta(a\mid s)\right] 
\end{aligned}
$$



#### Proof

$$
\begin{aligned}
\nabla_\theta V^\pi(s) =& \nabla_\theta \sum\limits_{a} \pi(a\mid s) \cdot Q^\pi(s,a) \\
=& \sum\limits_{a} \left[Q^\pi(s,a) \cdot \nabla_\theta\pi(a\mid s) + \pi(a\mid s) \cdot \nabla_\theta Q^\pi(s,a)\right] \\
=& \sum\limits_{a} \left[Q^\pi(s,a) \cdot \nabla_\theta\pi(a\mid s) + \pi(a\mid s) \cdot \nabla_\theta \sum\limits_{s',r} P(s',r\mid s,a)\cdot \left(r+ \gamma \cdot V^\pi(s')\right)\right] \\
=& \sum\limits_{a} \left[Q^\pi(s,a) \cdot \nabla_\theta\pi(a\mid s) + \gamma \cdot  \pi(a\mid s) \cdot \sum\limits_{s'} P(s'\mid s,a)\cdot \nabla_\theta V^\pi(s')\right] \\
=& \sum\limits_{a} Q^\pi(s,a) \cdot \nabla_\theta\pi(a\mid s) + \gamma \sum\limits_{s',a} \pi(a\mid s) \cdot  P(s'\mid s,a)\cdot \nabla_\theta V^\pi(s') \\
\end{aligned}
$$

Note that the action $a$ is sampled from the parameterized policy $\pi_\theta.$ Thus $\nabla_\theta a$ is $0,$ without using the gumbel-softmax technique. In the deterministic policy, $\nabla_\theta a$ is not $0,$ and thus the derivation is different.

$$
\begin{aligned}
\nabla_\theta V^\pi(s) 
=& \sum\limits_{a} Q^\pi(s,a) \cdot \nabla_\theta\pi(a\mid s) + \sum\limits_{s',\textcolor{red}{a}} \textcolor{red}{\gamma \cdot \pi(a\mid s) \cdot  P(s'\mid s,a)} \cdot \nabla_\theta V^\pi(s') \\
=& \sum\limits_{a} Q^\pi(s,a) \cdot \nabla_\theta\pi(a\mid s) + \sum\limits_{s'} \textcolor{red}{\gamma \cdot \mathrm{Pr}(s\to s', k=1, \pi_\theta)} \cdot \nabla_\theta V^\pi(s') \\
=& \sum\limits_{a} Q^\pi(s,a) \cdot \nabla_\theta\pi(a\mid s) + \sum\limits_{s'} \gamma \cdot \mathrm{Pr}(s\to s', k=1, \pi_\theta) \left[\sum\limits_{a} Q^\pi(s',a) \cdot \nabla_\theta\pi(a\mid s') + \sum\limits_{s''} \gamma \cdot \mathrm{Pr}(s'\to s'', k=1, \pi_\theta)\cdot \nabla_\theta V^\pi(s'')\right] \\
=& \sum\limits_{a} Q^\pi(s,a) \cdot \nabla_\theta\pi(a\mid s) + \gamma \sum\limits_{s'} \mathrm{Pr}(s\to s', k=1, \pi_\theta) \sum\limits_{a} Q^\pi(s',a) \cdot \nabla_\theta\pi(a\mid s') \\
	&+ \textcolor{red}{\gamma^2 \sum\limits_{s'} \mathrm{Pr}(s\to s', k=1, \pi_\theta) \sum\limits_{s''} \mathrm{Pr}(s'\to s'', k=1, \pi_\theta)}\cdot \nabla_\theta V^\pi(s'') \\
=& \sum\limits_{a} Q^\pi(s,a) \cdot \nabla_\theta\pi(a\mid s) + \gamma \sum\limits_{s'} \mathrm{Pr}(s\to s', k=1) \sum\limits_{a} Q^\pi(s',a) \cdot \nabla_\theta\pi(a\mid s') \\
	&+ \textcolor{red}{\gamma^2 \sum\limits_{s'} \mathrm{Pr}(s\to s'', k=2, \pi_\theta) }\cdot \nabla_\theta V^\pi(s'') \\
=&\ldots \\
=& \sum\limits_{x\in S} \textcolor{blue}{\sum\limits_{k=0}^\infty \gamma^k \cdot \mathrm{Pr}(s\to x, k, \pi_\theta)} \cdot \sum\limits_{a} Q^\pi(x,a) \cdot \nabla_\theta\pi(a\mid x)
\end{aligned}
$$

The blue part is defined as the **discounted state visitation distribution** $$d^{\pi_\theta}(s\mid s_0) = (1-\gamma )\cdot \sum\limits_{k=0}^\infty \gamma^k \cdot \mathrm{Pr}(s_0\to s, k, \pi_\theta).$$

$$
\begin{aligned}
\sum\limits_{k=0}^\infty \gamma^k \cdot \mathrm{Pr}(s_0\to s, k, \pi_\theta) \le \sum\limits_{k=0}^\infty \gamma^k = \frac{1}{1-\gamma}
\end{aligned}
$$

The distribution should beshould lie within the range of $[0,1]$ and thus the coefficient $(1-\gamma)$ is is for normalization.

$$
\begin{aligned}
\nabla_\theta V^{\pi_\theta}(s_0) 
=& \textcolor{blue}{\frac{1}{(1-\gamma)}} \sum\limits_{s} \textcolor{blue}{d^{\pi_\theta}(s\mid s_0)} \sum\limits_{a} Q^{\pi_\theta}(s,a) \cdot \nabla_\theta \pi_\theta(a\mid s) \\
=& \frac{1}{(1-\gamma)} \sum\limits_{s} d^{\pi_\theta}(s\mid s_0) \sum\limits_{a} \pi(a\mid s) \cdot Q^{\pi_\theta}(s,a) \cdot \nabla_\theta \ln \pi_\theta(a\mid s) \\
=& \frac{1}{(1-\gamma)} \mathbb{E}_{s \sim d^{\pi_\theta}(\cdot \mid s_0)} \mathbb{E}_{a\sim \pi(\cdot \mid s)} \left[Q^{\pi_\theta}(s,a) \cdot \nabla_\theta \ln \pi_\theta(a\mid s)\right] 
& \blacksquare
\end{aligned}
$$



### Deterministic Policy Gradient

#### Basics

- $a = \mu_\theta (s)$
- $V^{\mu_\theta}(s) = Q^{\mu_\theta}(s,a) = Q^{\mu_\theta}(s,\mu_\theta(s))$
- $\nabla_\theta a \ne 0$
    - $\nabla_\theta r(s,a)\ne 0$
    - $\nabla_\theta P(s'\mid s,a)\ne 0$
- $P(s'\mid s,\mu_\theta(s)) = \mathrm{Pr}(s\to s', k=1, \mu_\theta)$




#### Gradient

$$
\begin{aligned}
\nabla_\theta V^{\mu_\theta}(s) 
=& \frac{1}{(1-\gamma)}\int_{x\in S} d^{\mu_\theta}(s) \cdot \nabla_\theta \mu_\theta(x)\cdot \nabla_a Q^{\mu_\theta} (x, \mu_\theta(x))\Big\vert_{a=\mu_\theta(x)} \,\mathrm{d} x \\
=&\frac{1}{(1-\gamma)} \mathbb{E}_{s\sim d^{\mu_\theta}} \bigg[ \nabla_\theta \mu_\theta(x)\cdot \nabla_a Q^{\mu_\theta} (x, \mu_\theta(x))\Big\vert_{a=\mu_\theta(x)}\bigg]
\end{aligned}
$$


#### Proof


$$
\begin{aligned}
\nabla_\theta V^{\mu_\theta}(s) 
=& \textcolor{blue}{\nabla_\theta Q^{\mu_\theta} (s, \mu_\theta(s))} \\
=& \nabla_\theta \left( r(s,\mu_\theta(s)) + \gamma \int_S P\left(s'\mid s,\mu_\theta(s)\right)\cdot V^{\mu_\theta}(s') \,\mathrm{d} s' \right) \\
=& \textcolor{red}{\nabla_\theta r(s,\mu_\theta(s))} + \gamma \int_S V^{\mu_\theta}(s') \cdot \textcolor{red}{\nabla_\theta P\left(s'\mid s,\mu_\theta(s)\right)} \,\mathrm{d} s' + \gamma \int_S P\left(s'\mid s,\mu_\theta(s)\right)\cdot \nabla_\theta V^{\mu_\theta}(s') \,\mathrm{d} s' \\
=& \nabla_\theta \mu_\theta(s)\cdot \textcolor{blue}{\nabla_a} \left( r(s,\mu_\theta(s)) + \gamma \int_S \textcolor{blue}{V^{\mu_\theta}(s')} \cdot P\left(s'\mid s,\mu_\theta(s)\right) \,\mathrm{d} s' \right)\Bigg\vert_{a=\mu_\theta(s)} + \gamma \int_S P\left(s'\mid s,\mu_\theta(s)\right)\cdot \nabla_\theta V^{\mu_\theta}(s') \,\mathrm{d} s' \\
=& \textcolor{blue}{\nabla_\theta \mu_\theta(s)\cdot \nabla_a Q^{\mu_\theta} (s, \mu_\theta(s))\Big\vert_{a=\mu_\theta(s)}} + \gamma \int_S \textcolor{green}{P\left(s'\mid s,\mu_\theta(s)\right)}\cdot \nabla_\theta V^{\mu_\theta}(s') \,\mathrm{d} s' \\
=& \nabla_\theta \mu_\theta(s)\cdot \nabla_a Q^{\mu_\theta} (s, \mu_\theta(s))\Big\vert_{a=\mu_\theta(s)} + \gamma \int_S \textcolor{green}{\mathrm{Pr}(s\to s', k=1, \mu_\theta)} \cdot \nabla_\theta V^{\mu_\theta}(s') \,\mathrm{d} s' \\
=& \nabla_\theta \mu_\theta(s)\cdot \nabla_a Q^{\mu_\theta} (s, \mu_\theta(s))\Big\vert_{a=\mu_\theta(s)} + \gamma \int_S \mathrm{Pr}(s\to s', k=1, \mu_\theta) \cdot \left( \nabla_\theta \mu_\theta(s')\cdot \nabla_a Q^{\mu_\theta} (s', \mu_\theta(s'))\Big\vert_{a=\mu_\theta(s')} + \gamma \int_S \mathrm{Pr}(s'\to s'', k=1, \mu_\theta) \cdot \nabla_\theta V^{\mu_\theta}(s'') \,\mathrm{d} s'' \right) \,\mathrm{d} s' \\
=& \nabla_\theta \mu_\theta(s)\cdot \nabla_a Q^{\mu_\theta} (s, \mu_\theta(s))\Big\vert_{a=\mu_\theta(s)} + \gamma \int_S \mathrm{Pr}(s\to s', k=1, \mu_\theta) \cdot \nabla_\theta \mu_\theta(s')\cdot \nabla_a Q^{\mu_\theta} (s', \mu_\theta(s'))\Big\vert_{a=\mu_\theta(s')} \,\mathrm{d} s' \\
	&+ \gamma^2 \int_S \mathrm{Pr}(s\to s', k=1, \mu_\theta) \int_S \mathrm{Pr}(s'\to s'', k=1, \mu_\theta) \cdot \nabla_\theta V^{\mu_\theta}(s'') \,\mathrm{d} s'' \,\mathrm{d} s' \\
=&\ldots \\
=& \int_{x\in S} \sum\limits_{k=0}^\infty \gamma^k \cdot \mathrm{Pr}(s\to x, k, \mu_\theta) \cdot \nabla_\theta \mu_\theta(x)\cdot \nabla_a Q^{\mu_\theta} (x, \mu_\theta(x))\Big\vert_{a=\mu_\theta(x)} \,\mathrm{d} x \\
=& \frac{1}{(1-\gamma)}\int_{x\in S} d^{\mu_\theta}(s) \cdot \nabla_\theta \mu_\theta(x)\cdot \nabla_a Q^{\mu_\theta} (x, \mu_\theta(x))\Big\vert_{a=\mu_\theta(x)} \,\mathrm{d} x \\
=&\frac{1}{(1-\gamma)} \mathbb{E}_{s\sim d^{\mu_\theta}} \bigg[ \nabla_\theta \mu_\theta(x)\cdot \nabla_a Q^{\mu_\theta} (x, \mu_\theta(x))\Big\vert_{a=\mu_\theta(x)}\bigg]
\end{aligned}
$$

Note that the Bellman equation here is different from the one in the stochastic case: the reward is not dependent on the next state. $\blacksquare$

> Calculating $\nabla_a Q(s,a)$ is the result of accounting for both $\nabla_\theta r(s,a)$ and $\nabla_\theta p(s'\mid s,a)$.
{: .prompt-tip }


## Performance Difference Lemma

For all policies $\pi, \pi^\prime$ and states
$s_0$,

$$\begin{aligned} V^\pi(s_0) - V^{\pi^\prime}(s_0) =& \mathbb{E}_{\tau \sim {\Pr}^\pi(\tau|s_0=s) } \left[\sum_{t=0}^\infty \gamma^t A^{\pi'}(s_t,a_t)\right] \\ =& \frac{1}{1-\gamma}\mathbb{E}_{s\sim d_{s_0}^\pi }\mathbb{E}_{a\sim \pi(\cdot|s) } \left[  A^{\pi^\prime}(s,a)\right]. \end{aligned} $$

> Kakade, Sham, and John Langford. "Approximately optimal approximate reinforcement learning." Proceedings of the Nineteenth International Conference on Machine Learning. 2002.
{: .prompt-info }

### Proof

The proof is provided in the appendix of "On the theory of policy gradient methods: Optimality, approximation, and distribution shift" and I just transcribed it here with additional details.

Let $\Pr^\pi(\tau \mid s_0 = s)$ denote the probability of observing a trajectory $\tau$ when starting in state $s$ and following the policy $\pi$. Using a telescoping argument, we have:

$$
\begin{aligned}
&V^\pi(s) - V^{\pi'}(s) \\
=&  \mathbb{E}_{\tau \sim {\Pr}^\pi(\tau|s_0=s) }
\left[\sum_{t=0}^\infty \gamma^t r(s_t,a_t)\right] - V^{\pi'}(s) \\
=& \mathbb{E}_{\tau \sim {\Pr}^\pi(\tau|s_0=s) }
\left[\sum_{t=0}^\infty \gamma^t \left(r(s_t,a_t)+V^{\pi'}(s_t)-V^{\pi'}(s_t) \right)\right]-V^{\pi'}(s)\\
\stackrel{(a)}{=}& \mathbb{E}_{\tau \sim {\Pr}^\pi(\tau|s_0=s) }
    \left[\sum_{t=0}^\infty \gamma^t \left(r(s_t,a_t)+\gamma V^{\pi'}(s_{t+1})-V^{\pi'}(s_t)\right)\right]\\
\stackrel{(b)}{=}&\mathbb{E}_{\tau \sim {\Pr}^\pi(\tau|s_0=s) }
    \left[\sum_{t=0}^\infty \gamma^t \left(r(s_t,a_t)+\gamma \mathbb{E}[V^{\pi'}(s_{t+1})|s_t,a_t]-V^{\pi'}(s_t)\right)\right]\\
\stackrel{(c)}{=}& \mathbb{E}_{\tau \sim {\Pr}^\pi(\tau|s_0=s) }
    \left[\sum_{t=0}^\infty \gamma^t A^{\pi'}(s_t,a_t)\right] \\
=& \frac{1}{1-\gamma}\mathbb{E}_{s'\sim d^\pi_s }\,\mathbb{E}_{a\sim \pi(\cdot | s')}
    \left[ A^{\pi'}(s',a) \right],
\end{aligned}
$$

where $(a)$ rearranges terms in the summation and cancels the $V^{\pi'}(s_0)$ term with the $-V^{\pi'}(s)$ outside the summation, and $(b)$ uses the tower property of conditional expectations and the final equality follows from the definition of $d^\pi_s$. $\blacksquare$

### Details

$(a)$:

$$
- a_0 +\sum\limits_{k=0}^{\infty} \left(a_k - b_k \right) = \sum\limits_{k=0}^{\infty} \left(a_{k+1} - b_k\right).
$$

$(b)$: The tower property of conditional expectations (or law of total probability):
If $\mathcal{H} \subseteq \mathcal{G}$, then

$$
\mathbb{E}\left[\mathbb{E}\left[X\mid \mathcal{G} \right] \mid \mathcal{H} \right] = \mathbb{E}\left[X\mid \mathcal{H} \right].
$$

Correspondingly, 
- $\mathcal{G} = \tau \sim {\Pr}^\pi(\tau \mid s_0=s)$, 
- $\mathcal{H} = (s_t,a_t)$.

$(c)$: Step $(b)$ is necessary. Note that

$$
Q^{\pi}(s, a) \ne r(s, a) + \gamma \cdot V^{\pi}(s').
$$

But

$$
Q^{\pi}(s, a) = r(s, a) + \gamma \cdot \sum\limits_{s'} P(s' \mid s,a) \cdot V^{\pi}(s').
$$

### Other proofs

> Check other proofs [here](https://people.cs.umass.edu/~akshay/courses/coms6998-11/files/lec7.pdf) and [here](https://wensun.github.io/CS4789_data/PDL.pdf).
{: .prompt-info }

###


## Convergence

### About

This section is based on the awesome paper:

> Agarwal, Alekh, et al. "On the theory of policy gradient methods: Optimality, approximation, and distribution shift." The Journal of Machine Learning Research 22.1 (2021): 4431-4506.
> {: .prompt-info }

And I will provide some omitted details here. 
The writing of the entire note may be somewhat **verbose**, and this is to familiarize myself with the content.



### Details of Setting

#### $V(s) \le \frac{1}{1-\gamma}$
$V(s)$ reaches its upper bound when $r(s,a)=1,\forall s,a$, which equals $\sum\limits_{t=0}^\infty \gamma^t$.

And it is a geometric progression:
- $a_n = a_0 \cdot \gamma^{n-1}$,
- $S_n = a_0 \cdot \frac{1-\gamma^n}{1-\gamma}$,
   - $S_n = a_0 + a_1 + \ldots + a_n$,
   - $\gamma\cdot S_n = a_1 + \ldots + a_n + a_{n+1}$,
   - $(1-\gamma)\cdot S_n = a_0\cdot (1 - \gamma^n)$.
- $\lim\limits_{n\to\infty}S_n = \frac{a_0}{1-\gamma} = \frac{1}{1-\gamma}$.

---

#### The famous theorem of Bellman and Dreyfus (1959)

> The famous theorem of Bellman and Dreyfus (1959) shows there exists a policy $\pi^\star$ which simultaneously maximizes $V^\pi(s_0)$, for all states $s_0\in S$.

I have read this referenced paper, and I do not find any theorem. This paper is mainly about trading additional computing time for additional memory capacity.

However this statement is intuitive and is not hard to understand. Assume there is a fixed $s_{-1}$ and can be transited to $s_0$ according to $\rho$, then this problem is equivalent to the one that has a fixed $s_0$.

---

#### Direct parameterization

$\theta\in\Delta(A)^{\vert S\vert}$ means for every state $s$ the parameters are  a point in a simplex. 

For eample, for state $s_0$, there are actions $a_1, a_2$, the parameters of the current policy $\pi_\theta(\cdot \mid s_0)$ are 
- $\theta_{s_0,a_1} = 0.2 = \pi_\theta(a_1 \mid s_0)$, and
- $\theta_{s_0,a_2} = 0.8 = \pi_\theta(a_2 \mid s_0)$.

---

#### Softmax parameterization

Sometimes it can be $\pi_\theta(a\mid s) = \frac{\exp(\tau\cdot \theta_{s,a})}{\sum\limits_{a'}\exp(\tau\cdot \theta_{s,a'})}$, which is called energy-based policy, where $\tau$ is the temperature parameter (inverse temperature) and $\theta_{s,a}$ is the energy function.

> Haarnoja, Tuomas, et al. "Reinforcement learning with deep energy-based policies." International conference on machine learning. PMLR, 2017.
{: .prompt-info }

---

#### $V^{\pi_\theta}(s)$ is non-concave (Lemma 1)
We want to **maximize** $V^{\pi_\theta}(s)$, so if $V^{\pi_\theta}(s)$ is **concave** then we can apply standard tools of convex optimization.  Unfortunately it is not.

As shown in the appendix, there is a MDP where exists policy points $\pi_1, \pi_2$ that $V^{\pi_1}(s)+V^{\pi_2}(s)> 2\cdot V^{\frac{1}{2}(\pi_1+\pi_2)}(s)$. This shows a property of convex, so $V^{\pi_\theta}(s)$ is non-concave.

---

#### Why is there a coefficient $(1-\gamma)$ in $(4)$?

$$
d_{s_0}^\pi(s) := (1-\gamma) \sum_{t=0}^\infty \gamma^t {\Pr}^\pi(s_t=s|s_0).
$$

Recall that [the derivation of the policy gradient theorem](https://lilianweng.github.io/posts/2018-04-08-policy-gradient/):

$$
\nabla_{\theta} V^{\pi_\theta}(s_0) = \sum\limits_{s} \sum\limits_{k=0}^{\infty} \gamma^k \cdot \text{Pr}(s_0\to s, k) \sum\limits_{a} \pi(a\mid s) \cdot Q^\pi(s,a)\cdot \nabla_{\theta}\ln \pi(a\mid s).
$$

> Policy Gradient
> - Williams, Ronald J. "Simple statistical gradient-following algorithms for connectionist reinforcement learning." Machine learning 8 (1992): 229-256.
> - Sutton, Richard S., et al. "Policy gradient methods for reinforcement learning with function approximation." Advances in neural information processing systems 12 (1999).
{: .prompt-info }

Note that $\lim\limits_{k\to\infty} \sum\limits_{k=0}^{\infty} \gamma^k = \frac{1}{1-\gamma} > 1$. The value of discounted state visitation distribution should not larger than $1$. So the coefficient $(1-\gamma)$ is for normalization.

---

#### Why is there a coefficient $\frac{1}{1-\gamma}$ in $(5)$?

$$
\begin{aligned}
\nabla_\theta V^{\pi_\theta}(s_0) 
=& \frac{1}{1-\gamma} \, \mathbb{E}_{s \sim d_{s_0}^{\pi_\theta} }\mathbb{E}_{a\sim \pi_\theta(\cdot | s) }
\big[\nabla_\theta \log
\pi_{\theta}(a| s) Q^{\pi_\theta}(s,a)\big] \\
=& \sum\limits_{s} d^\pi_{s_0}(s) \sum\limits_{a} \pi(a\mid s) \cdot Q^\pi(s,a)\cdot \nabla_{\theta}\log \pi(a\mid s).
\end{aligned}
$$

It is used to cancel that normalization.

---

#### Advantage

$$
\begin{aligned}
A^{\pi}(s,a)
:=& Q^\pi(s,a)-V^\pi(s) \\
=& Q^\pi(s,a) - \sum\limits_{a}\pi(a\mid s) \cdot Q^\pi(s,a).
\end{aligned}
$$

Given $s$ and $\pi$, $A^{\pi}(s,a)$ measures how much better the expected future return after selecting action $a$ is compared to the expected future return of sampling action based on the current policy $\pi$ in this state $s$.

---

#### Baseline
> This part partially use material from Prof. Wang's [Lecture note 18: Variance reduction](https://drive.google.com/drive/folders/1u1oyOMsvo4bJ765NE_2HSR5x40uXWwxD) and *Reinforcement learning: An introduction*.
{: .prompt-info }

Policy gradient is unbiased but with high variance. Recall that the form is

$$
\nabla_{\theta} V^{\pi_\theta}(s_0) =\frac{1}{1-\gamma} \mathbb{E}_{s\sim d_{s_0}^\pi}\mathbb{E}_{a\sim\pi(\cdot\mid s)}\left[ Q^\pi(s,a)\cdot \nabla_{\theta}\log \pi(a\mid s)\right].
$$

To reduce it, a natural solution is to subtract a baseline $b(s)$ from $Q^\pi$ which can be any function, even a random variable, as long as it does not depend on the action $a$, i.e., 

$$
\nabla_{\theta} V^{\pi_\theta}(s_0) =\frac{1}{1-\gamma} \mathbb{E}_{s\sim d_{s_0}^\pi}\mathbb{E}_{a\sim\pi(\cdot\mid s)}\left[ \left(Q^\pi(s,a) - b(s)\right)\cdot \nabla_{\theta}\log \pi(a\mid s)\right],
$$

$$
\nabla_{\theta} V^{\pi_\theta}(s_0) = \sum\limits_{s} d^\pi_{s_0}(s) \sum\limits_{a} \pi(a\mid s) \cdot \left(Q^\pi(s,a) - b(s)\right)\cdot \nabla_{\theta}\log \pi(a\mid s),
$$

or

$$
\nabla_{\theta} V^{\pi_\theta}(s_0) = \sum\limits_{s} d^\pi_{s_0}(s) \sum\limits_{a} \left(Q^\pi(s,a) - b(s)\right)\cdot \nabla_{\theta} \pi(a\mid s).
$$

This is still unbiased:

$$
\begin{aligned}
&\sum\limits_a b(s)\cdot \nabla_{\theta} \pi(a\mid s) \\
=& b(s) \cdot\nabla_\theta\sum\limits_a \pi(a\mid s) \\
=& b(s) \cdot\nabla_\theta 1 \\
=& 0. 
\end{aligned}
$$

But it has lower variance:
> Assume that:
> - $X = Q^\pi(s,a)\cdot \nabla_{\theta} \pi(a\mid s)$,
> - $Y = \nabla_{\theta} \pi(a\mid s)$,
> - $\mathbb{E} \left[ X \right] = \mu$,
> - $\mathbb{E} \left[ Y \right] = \eta = 0$,
> - $X' = X + c(Y-\eta)$.
> 
> Then:
> - $\mathbb{E} \left[ X' \right] = \mu$,
> - $$\begin{aligned} \mathbb{V} \left[ X' \right] =& \mathbb{V} \left[ X + c(Y-\eta) \right] \\ =& \mathbb{V} \left[ X \right] + c^2\cdot \mathbb{V} \left[ Y-\eta \right] +2c\cdot \text{Cov}(X,Y-\eta) \\ =& \mathbb{V} \left[ Y-\eta \right] \cdot c^2 + 2\cdot \text{Cov}(X,Y-\eta)\cdot c + \mathbb{V} \left[ X \right],\end{aligned}$$
> - $\min \mathbb{V} \left[ X' \right] = \left(1 - \text{Corr(X,Y)}\right)\cdot \mathbb{V} \left[ X \right]$.
<!-- $$
\mathbb{E}_x \left[ \mathbb{E}_y \left[ f(x,y) - b(x) \cdot g(x,y) \right] \right]
$$ -->

> Common usage: GAE (Generalized Advantage Estimation).
>
> Schulman, John, et al. "High-dimensional continuous control using generalized advantage estimation." arXiv preprint arXiv:1506.02438 (2015).
{: .prompt-info }

---

#### Equation (6) does not hold for the direct parameterization

$$
\begin{aligned}
   \sum\limits_a \nabla_\theta \pi(a) =& \left(\sum\limits_a\frac{\partial \pi(a)}{\partial \theta_1},\ldots, \sum\limits_a\frac{\partial \pi(a)}{\partial \theta_m}\right)\\ 
\end{aligned}
$$

If every $\frac{\partial \pi(a)}{\partial \theta_1}$ has the same variables, then $\sum\limits_a\frac{\partial \pi(a)}{\partial \theta_1} = \frac{\partial \sum\limits_a\pi(a)}{\partial \theta_1} = 0$. But in the case of the direct parameterization, this assumption does not hold, i.e., $\sum\limits_a\frac{\partial \pi(a)}{\partial \theta_1} = 1$.

---

#### Distribution mismatch coefficient (pass)
I think this concept is introduced too soon. Let's discuss it later.


---


### Details on Constrained Tabular Parameterization

> This algorithm is **projected gradient ascent** on the **direct policy parametrization** of the MDP.

#### Equation $(7)$
$\mu$ is a distribution of $s_0$.

$$
\begin{aligned}
& \nabla_\theta V^{\pi_\theta}(\mu) \\
=&  \frac{\partial V^{\pi_\theta}(\mu)}{\partial } \\
=& \frac{1}{1-\gamma} \, \mathbb{E}_{s \sim d_{s_0}^{\pi_\theta} }\mathbb{E}_{a\sim \pi_\theta(\cdot | s) }
\big[\nabla_\theta \log
\pi_{\theta}(a| s) Q^{\pi_\theta}(s,a)\big] \\
=& \sum\limits_{s} d^\pi_{s_0}(s) \sum\limits_{a} \pi(a\mid s) \cdot Q^\pi(s,a)\cdot \nabla_{\theta}\log \pi(a\mid s).
\end{aligned}
$$

> The following part has not been finished yet. One may check my [writing schedule](https://yuelin301.github.io/posts/Schedule/).
{: .prompt-warning }