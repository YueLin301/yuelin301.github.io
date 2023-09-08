---
title: Fictitious Self-Play and Zero-Shot Coordination
date: 2023-08-01 02:40:00 +0800
categories: [Economics & Game Theory]
tags: [game theory, fictitious play, self-play, multi agents, reinforcement learning]
math: True
pin: True
---

## What is Fictitious Play?
<!-- > 洪七公叫他取过树枝，将打狗棒法中一招“棒打双犬”细细说给了他听。杨过一学即会，当即照式演出。欧阳锋见棒招神奇，果然厉害，一时难以化解，想了良久，将一式杖法说给杨过听了。杨过依言演出。洪七公微微一笑，赞了声：“好！” —— 《神雕侠侣》第十一回 -->

1. Fictitious play is a learning rule. 
2. In it, each player presumes that the opponents are playing stationary (possibly mixed) strategies. 
3. At each round, each player thus best responds to the empirical frequency of strategy of their opponent. 
   - Belief of the opponent's strategy. Average of the history actions. Monte-Carlo style.
   - Best response.

([Wikipedia](https://en.wikipedia.org/wiki/Fictitious_play))

---

### Process

Given a game with $n$ players, where each player $i$ has a strategy $\pi^i$. 

1. Players initialize their beliefs about the strategies of the other players. And $\hat{\pi}^{i,j}$ means the belief of $i$ regarding the $j$'s strategy.
2. At each round $t$:
   - Player $i$ observes the action $a_{t-1}^j$ (or pure strategy) of every other player $j \neq i$.
   - Player $i$ updates their belief of player $j$'s strategy based on the empirical frequency: $\hat{\pi}\_t^{i,j} = \frac{1}{t-1} \sum\_{k=1}^{t-1} a_k^j$. The actions are one-hot encoded.
   - Player $i$ then plays a **best response** to $\hat{\pi}_t^{i,j}$.

---

### Convergence?
Fictitious play doesn't always converge to a Nash equilibrium in all games. It's been proven that fictitious play converges to the set of Nash equilibria in certain classes of games, like zero-sum games and potential games. However, there are games where fictitious play does not necessarily converge to a Nash equilibrium.

E.g., in the [Matching Pennies game](https://yuelin301.github.io/posts/Classic-Games/#matching-pennies), the trajectory of Fictitious Play tends to circle around the mixed-strategy Nash equilibrium rather than directly converging to it.

## Self-Play

> 小龙女奇道：“自己跟自己打架？怎生打法？” —— 《神雕侠侣》第二十五回

Self-play involves an agent (or a model) playing against itself or versions of itself. This can be thought of as a kind of **bootstrapping** method where an agent learns and refines its strategies through continuous iterations of gameplay against its own evolving strategies.

### Process

1. **Initialization**: Start with a randomly initialized agent or a naive version.
2. **Play**: Let the agent play games against itself or against past versions of itself.
3. **Update**: After each game, or a batch of games, update the agent's model based on the results, rewards, and feedback from the games.
4. **Iterate**: Repeat the Play and Update steps for a desired number of iterations or until the agent's strategy converges to an optimal or satisfactory level.


### Significance

1. **Unlimited Opponents**: Self-play provides an infinite supply of opponents since the agent is essentially playing against itself. This eliminates the need for external opponent data or human opponents, which can be limited or introduce variability.
2. **Evolving Difficulty**: As the agent improves, its past versions also present incrementally challenging opponents, ensuring that the agent is always pushed to improve and adapt.
3. **Consistency**: By playing against itself, the agent is exposed to a consistent level of gameplay, which can lead to more stable learning.

### Applications

The most famous application of self-play is perhaps in the training of **AlphaGo** and its successors by DeepMind. AlphaGo utilized self-play to achieve superhuman performance in the game of Go, a feat that was previously thought to be decades away. Following this, **AlphaZero** utilized a generalized version of this self-play approach to achieve superhuman performance not only in Go but also in Chess and Shogi.

### A resampling technique: bootstrapping
Resampling techniques are a class of statistical methods that involve creating new samples by repeatedly drawing observations from the original data sample.

Bootstrapping is a method where new "bootstrap samples" are created by drawing observations **with replacement** from the original sample.

In RL, a common example is the Temporal Difference (TD) learning. This method bootstraps from the current estimate of the value function.

#### Incremental mean
I have a dataset with $n$ samples $\{x_1, x_2, \ldots, x_n\}$. The expectation of $X$ is calculated as 

$$
\mu_n = \frac{1}{n}\sum\limits_{i=1}^n x_i
$$

Then I get a new sample $x_{n+1}$, then the expectation of $X$ should be updated. And it can be represented by the current expectation: 

$$
\mu_{n+1} = \mu_n + \frac{1}{n+1}\left(x_{n+1} - \mu_n \right)
$$

Derivation:

$$
\begin{aligned}
  \mu_{n+1} =& \frac{1}{n+1}\sum\limits_{i=1}^{n+1} x_i 
  =  \frac{1}{n+1}\left(x_{n+1} + \sum\limits_{i=1}^n x_i \right) \\
  =&  \frac{1}{n+1}x_{n+1} + \frac{n}{n+1} \sum\limits_{i=1}^n x_i \\
  =&  \frac{1}{n+1}x_{n+1} + \left(1 - \frac{1}{n+1}\right) \mu_n \\
  =& \mu_n + \frac{1}{n+1}\left(x_{n+1} - \mu_n \right)
\end{aligned}
$$

To reduce the impact of previous samples, the coefficient is fixed as a constant $\alpha$:

$$
\begin{aligned}
  \mu_{n+1} 
  =& \mu_n + \alpha\left(x_{n+1} - \mu_n \right) \\
  =& \alpha\cdot x_{n+1} - \left(1-\alpha\right)\cdot\mu_n 
\end{aligned}
$$


#### Temporal difference

According to the Bellman equation, the value function is

$$
V(s) = \mathbb{E}[R_{t+1} + \gamma V(S_{t+1}) | S_t = s]
$$

Now I get a new sample of $R_{t+1}$, I can use it to update $V(s_t)$.

$$
V(s_t) \gets V(s_t) + \alpha\left(x_{n+1} - V(s_t) \right)
$$


---

## Zero-Shot Coordination

> 宝玉看罢，因笑道：“这个妹妹我曾见过的。”贾母笑道：“可又是胡说，你又何曾见过她？”
> —— 《红楼梦》第三回

### Definition
Zero-shot coordination is about developing agents capable of coordinating with some other agents (even humans) they have never seen before.

1. **No Prior Coordination**: Agents do not have the opportunity to coordinate or train to learn how to work together before starting the task. Essentially, agents start “from scratch” without any prior coordination strategies.
2. **No Communication**: Agents cannot communicate with each other during task execution.

### The self-play is not good enough


---
## Other-Play
[Hu, Hengyuan, et al. "“other-play” for zero-shot coordination." International Conference on Machine Learning. PMLR, 2020.](http://proceedings.mlr.press/v119/hu20a/hu20a.pdf)



---

> The following part has not been finished yet. One may check my [writing schedule](https://yuelin301.github.io/posts/Schedule/).
{: .prompt-warning }

---

## To-Read

### Jakob Foerster

- [ ] Other-play  
  [Hu, Hengyuan, et al. "“other-play” for zero-shot coordination." International Conference on Machine Learning. PMLR, 2020.](http://proceedings.mlr.press/v119/hu20a/hu20a.pdf)
- [ ] A New Formalism, Method and OpenIssues  
  [Treutlein, Johannes, et al. "A new formalism, method and open issues for zero-shot coordination." International Conference on Machine Learning. PMLR, 2021.](http://proceedings.mlr.press/v139/treutlein21a/treutlein21a.pdf)
- [ ] Trajectory diversity  
  [Andrei Lupu, Brandon Cui, Hengyuan Hu, Jakob Foerster. "Trajectory diversity for zero-shot coordination." International conference on machine learning. PMLR, 2021.](http://proceedings.mlr.press/v139/lupu21a/lupu21a.pdf)
- [ ] Off-belief learning  
  [Hengyuan Hu, Adam Lerer, Brandon Cui, Luis Pineda, Noam Brown, Jakob Foerster. "Off-belief learning." International Conference on Machine Learning. PMLR, 2021.](http://proceedings.mlr.press/v139/hu21c/hu21c.pdf)

### Tencent

- [ ] Maximum Entropy Population-Based  
  [Zhao, Rui, et al. "Maximum entropy population-based training for zero-shot human-ai coordination." Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 37. No. 5. 2023.](https://ojs.aaai.org/index.php/AAAI/article/view/25758)

### Env
- [ ] Corridor [[paper](http://proceedings.mlr.press/v139/lupu21a/lupu21a.pdf)]
- [ ] Overcooked [[code](https://github.com/HumanCompatibleAI/overcooked_ai)]
- [ ] Hanabi [[code](https://github.com/deepmind/hanabi-learning-environment)] [[paper](https://www.sciencedirect.com/science/article/pii/S0004370219300116)]
