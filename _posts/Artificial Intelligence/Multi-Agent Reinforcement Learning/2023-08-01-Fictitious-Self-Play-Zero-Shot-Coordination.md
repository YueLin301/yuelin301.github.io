---
title: Fictitious Self-Play and Zero-Shot Coordination
date: 2023-08-01 02:40:00 +0800
categories: [Artificial Intelligence, Multi-Agent Reinforcement Learning]
tags: [Tech, AI, Multi Agents, RL, Game Theory, Framework]
math: True
---

## Fictitious Play
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

E.g., in the [Matching Pennies game]({{site.baseurl}}/posts/Classic-Games/#matching-pennies), the trajectory of Fictitious Play tends to circle around the mixed-strategy Nash equilibrium rather than directly converging to it.

## Self-Play

<!-- > 小龙女奇道：“自己跟自己打架？怎生打法？” —— 《神雕侠侣》第二十五回 -->

Self-play involves an agent (or a model) playing against itself or versions of itself. This can be thought of as a kind of [**bootstrapping**]({{site.baseurl}}/posts/RL-Toolbox/#td0) method where an agent learns and refines its strategies through continuous iterations of gameplay against its own evolving strategies.

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

---

## Zero-Shot Coordination

<!-- > 宝玉看罢，因笑道：“这个妹妹我曾见过的。”贾母笑道：“可又是胡说，你又何曾见过她？”
> —— 《红楼梦》第三回 -->

### Definition
Zero-shot coordination is about developing agents capable of coordinating in the testing phase with some other agents (even humans) that they have never seen in the training phase.

1. **No Prior Coordination**: Agents do not have the opportunity to coordinate or train to learn how to work together before starting the task. Essentially, agents start “from scratch” without any prior coordination strategies.
2. **No Communication**: Agents cannot communicate with each other during task execution.

### The self-play is not good enough

Self-play is a bootstrapping method. And
- **It assumes sample is representative**: Bootstrapping works under an assumption based on the original sample data, i.e., the sample data is randomly drawn and is representative of the population. If the original sample is not a good representative of the population, bootstrapping can yield misleading results.
- **It may cuases potential overfitting**: Given that bootstrapping is based on multiple resamples of the sample data, it may overemphasize certain features or outliers in the sample, leading to overfitting.

In the current setting, the self-play agent's policy may converge to overfit its own policy, thus forming a specialized **convention**. 

After training with self-play, agents might develop different conventions if there are multiple maxima in the sense of the joint policy, which can potentially be symmetric to each other. 
Thus, without a good way to **break the symmetries**, agents may fail to coordinate.

### Perspectives
In my understanding, the zero-shot coordination problem exists beacuse:
- There are symmetric descriptions in the tasks. Some quantities are not (and should not be) labeled.
- There are multiple maxima in the optimization problem of the task. Different pairs of agent cannot know which maxima their partner is at.
- In different runs of the same experimental code, the agents may have different random seeds.
- Self-play is a biased learning rule.

### Example: [Lever](https://proceedings.mlr.press/v119/hu20a/hu20a.pdf)

It is a matrix game. There are two agents with the same action space, each of size $m$. If the two agents choose the same actions, then they each receive a reward of 1; otherwise, they get 0.

|       | $a_1$ | $a_2$ | $a_3$ |
|-------|-------|-------|-------|
| $a_1$ | (1,1) | (0,0) | (0,0) |
| $a_2$ | (0,0) | (1,1) | (0,0) |
| $a_3$ | (0,0) | (0,0) | (1,1) |

If the two agents are trained through self-play successfully, then they will both choose the same action, a convention influenced by the initial random seeds used in the training process. Different pairs may converge to different outcomes, thus agents from different pairs may fail to coordinate.

By the way, the mixed strategy Nash equilibrium is $\left(\frac{1}{3}, \frac{1}{3}, \frac{1}{3}\right).$ The calculation is like the [Matching Pennies]({{site.baseurl}}/posts/Classic-Games/#matching-pennies) case.

In the paper "Other-Play," which proposed this task, there is another version of it:

|       | $a_1$ | $a_2$ | $a_3$     |
|-------|-------|-------|-------    |
| $a_1$ | (1,1) | (0,0) | (0,0)     |
| $a_2$ | (0,0) | (1,1) | (0,0)     |
| $a_3$ | (0,0) | (0,0) | (0.9,0.9) |

And the authors claimed that the most robust strategy is for everyone to choose the action $a_3$, which which would result in a payoff expectation $0.9.$ Otherwise, some pairs may choose $a_1$ and the others may choose $a_2$, it would lead to a payoff expectation $0.5.$

> [@fortytwo6364](https://www.youtube.com/watch?v=Sy2Z7alDgAE): But 0.9 is actually not the only unique answer to the lever problem, there is a unique 1.0 lever directly opposite to the .9 lever. This is stable to the symmetries implied by the problem statement where both players are shown the same set of levers, as well as being robust to different reflections and rotations being presented to different players (though not arbitrary permutations). So assuming whoever I am paired with is also optimizing we would both earn 1. This strategy doesn't work if there are an odd number of buttons to begin with.

The task introduced in the paper includes an illustration where the actions are circled up. However, the authors claim that these actions are not labeled. A more accurate description would be that the actions are uniformly sampled within a closed space, meaning that they cannot be identified by their positions.


> In my understanding, despite agents in zero-shot coordination never having interacted with others before, the designer assumes they are aware of the presence of others participating in the same task and are rational (and with some level of [theory of mind]({{site.baseurl}}/posts/ToM-MM/)). There is no free lunch here.
{: .prompt-tip }

### Equivalence mapping

This definition used to describe the symmetry is from the paper "other-play" and it is based on the **Dec-POMDPs**. It resembles the one used for [finding Nash equilibria in large games like poker](https://dl.acm.org/doi/abs/10.1145/1284320.1284324).

An equivalence mapping of can find the states and actions that share the same reward, transition probability, and observation. Formally,

$$
\begin{aligned}
  \phi \in \Phi \iff & P(\phi(s')\mid\phi(s),\phi(a)) 
  = P(s'\mid s,a) \\ 
  &\land R(\phi(s'), \phi(a), \phi(s)) = R(s', a, s) \\
  &\land O(\phi(o)\mid\phi(s), \phi(a), i) = O(o\mid s, a, i) \\
  &\forall s', s, a, o, i.
\end{aligned}
$$

And it's a shorthand for $\phi = \{\phi_S, \phi_A, \phi_O\}.$ It thus can be extended to trajectories and the policies.

- 
  $$
  \phi(\tau_{t}^{i}) = \{\phi(o_{0}^{i}), \phi(a_{0}^{i}), \phi(r_{0}), \ldots , \phi(o_{t}^{i})\},
  $$
- 
  $$
  \pi' = \phi(\pi) \iff \pi'(\phi(a)\mid\phi(\tau)) = \pi(a\mid\tau), \forall \tau, a.
  $$

In this way, a Dec-POMDP with two players has the following properties.

- 
  $$
  J(\pi_A, \pi_B) = J(\phi(\pi_{A}^1), \phi(\pi_{B}^2)), \forall \phi \in \Phi, \pi_A, \pi_B.
  $$
- 
  $$
  \{ \phi \cdot \phi' : \phi' \in \Phi \} = \Phi, \quad \forall \phi \in \Phi.
  $$

In my understanding, the `\cdot` here means function composition. That is, $(\phi \cdot \phi')(x)$ means $\phi(\phi'(x))$.


### Evaluation: Cross-Play

- The algorithm aimed to be tested is written in code, used to train the agents. 
- The code runs the traning experiment multiple times, each time with a different random seed.
- The agents in different runs (with different random seeds) are required to play the coordination task in the test time.

Success in cross-play is a **necessary condition** for the algorithm to achieve a level of zero-shot coordination.

![pic](/assets/img/23-08-01-zsc/cross-play.png){: width="400" height="400" }
_Illustration of Cross-Play from the paper "[A New Formalism, Method and Open Issues for Zero-Shot Coordination.](https://proceedings.mlr.press/v139/treutlein21a/treutlein21a.pdf)"_

---

## Other-Play
[Hu, Hengyuan, et al. "“other-play” for zero-shot coordination." International Conference on Machine Learning. PMLR, 2020.](https://proceedings.mlr.press/v119/hu20a/hu20a.pdf)

- Other-play is a learning rule.
- The optimization problem explicitly involves considerations related to policy symmetry.

---

Consider a Dec-POMDP with two players, the optimization problem in the sense of the other-play is

$$
\arg\max\limits_{\boldsymbol\pi} \mathbb{E}_{\phi \sim \Phi}\left[J(\pi^1, \phi(\pi^2)) \right],
$$

where the distribution is unifrom, and $\mathbb{E}\_{\phi \sim \Phi}\left[J(\pi^1, \phi(\pi^2)) \right]$ is denoted as $J\_{OP}(\boldsymbol\pi).$

> The expected OP return of $\boldsymbol{\pi}$ is equal to the expected return of each player independently playing a policy $\pi_\Phi^i$ which is the uniform mixture of $\phi(\pi^i)$ for all $\phi\in\Phi.$

$$
\begin{aligned}
  \mathbb{E}_{\phi \sim \Phi}\left[J(\pi^1, \phi(\pi^2)) \right]
  =& \mathbb{E}_{\phi_1 \sim \Phi, \phi_2 \sim \Phi}\left[J(\phi_1(\pi^1), \phi_1(\phi_2(\pi^2))) \right] \\
  =& \mathbb{E}_{\phi_1 \sim \Phi, \phi_2 \sim \Phi}\left[J(\phi_1(\pi^1), (\phi_2(\pi^2))) \right] \\
\end{aligned}
$$

> The distribution $\boldsymbol{\pi}^*\_{OP}$ produced by OP will be the uniform mixture $\boldsymbol{\pi}\_{\Phi}$ with the highest return $J(\boldsymbol{\pi}\_{\Phi}).$

This means the optimal solution $\boldsymbol{\pi}^\*\_{OP} = \arg\max\limits_{\boldsymbol\pi} \mathbb{E}\_{\phi \sim \Phi}\left[J(\pi^1, \phi(\pi^2)) \right]$ is the uniform mixture of the maxima of $\boldsymbol{\pi}^\* = \arg\max\limits_{\boldsymbol\pi} J(\pi^1, \pi^2).$

OP's best response is also OP. Since OP is a learning rule, the equilibrium reached can be seen as a kind of meta-equilibrium.

---

## Simplified Action Decoder

[Hu, Hengyuan, and Jakob N. Foerster. "Simplified Action Decoder for Deep Multi-Agent Reinforcement Learning." International Conference on Learning Representations. 2019.](https://arxiv.org/abs/1912.02288)

### Motivation

> Fundamentally, RL requires agents to explore in order to discover good policies. However, when done naively, this **randomness** will inherently make their actions **less informative** to others during training.

The more random the signal is, the less informative it will be. 

In my experience, this can be easily seen by the Recommendation Letter example from information design, or by the differential privacy.

A communication protocol is a mapping $f:X\to Y,$ where $X$ is the observation set, $Y$ is the signal set, and they are two random variables. 
"Informative" here refers to the degree to which $Y$ is related to $X.$ 
- If they are the same then the agent reveals all of the information.
- If $Y$ is irrelevant to $X$ then the agent reveals no information.

Thus the necessary randomness for exploration (of the sender) is harmful to the communication, as it might cause the receiver to lose faith in the sender and eventually learn to ignore the signals.

In the paper, equations $(3)$ to $(7)$ demonstrate that $\epsilon$-greedy exploration is harmful.

### Technique list
- CTDE (TODO)
- Joint Q-functions (VDN, QMIX) (TODO)
- Recurrent DQN
- Auxiliary tasks
- Theory of mind and Bayesian reasoning


### Bayesian reasoning

The basic setting is the Dec-POMDPs. 
- The complete trajectory $\tau = (s_0, \boldsymbol{a}_0, r_1, \ldots, r_T, S_T).$
- The partially observable trajectory of agent $i$ is $\tau^i = (o_0^i, \boldsymbol{a}_0, r_1, \ldots, r_T, S_T).$
  - The observation $o_t^i = O(s_t,i)$ is deterministic.
  - The agents are fully coorperative and they share the same reward at each timestep.

Each agent will calculate its prior guess about the complete trajectory, based on its own trajectory. That is $P(\tau_t\mid \tau_t^i),$ and it is denoted as $B(\tau_t).$

If the agent can observe another agent's action $a_t^j$, then it can update its belief.

$$
\begin{aligned}
  P(\tau_t \mid \tau_t^i, a_t^j) 
  =& \frac{P(a_t^j\mid \tau_t)\cdot P(\tau_t\mid \tau_t^i)}
  {P(a_t^j \mid \tau_t^i)}
  = \frac{P(a_t^j\mid \tau_t)\cdot P(\tau_t\mid \tau_t^i)}
  {\sum\limits_{\tau_t'} P(a_t^j \mid \tau_t')\cdot P(\tau_t'\mid \tau_t^i)} \\
  =& \frac{\pi^j\left(a_t^j\mid O(\tau_t,j) \right)\cdot B(\tau_t)}{\sum\limits_{\tau_t'} \pi^j\left(a_t^j\mid O(\tau_t',j) \right)\cdot B(\tau_t')}
\end{aligned}
$$

$\pi^j\left(a_t^j\mid O(\tau_t,j) \right)$ represents how agent $i$ perceives the strategy of agent $j$. And this is the theory of mind.
- Agent $i$ needs to have access to the policy of agent $j$ during training. Can be justified by CTDE.
- If this explicit belief is inputed into the network then it will cause higher order beliefs.
- The authors use RNN to learn it implicitly. (How? TODO.)

### Simplified belief

If agent $j$'s policy is with the $\epsilon$-greedy exploration, then agent $i$'s belief of it is like

$$
\pi^j\left(a_t^i\mid O(\tau_t, j)\right) 
= (1-\epsilon)\cdot \mathbf{I}\left(a^*(\tau_t) = a_t^j\right) 
+ \epsilon / \vert A \vert,
$$

where $\mathbf{I}\left(a^\*(\tau_t) = a_t^j\right)$ is the indicator function (one-hot encoded), and $a^\*(\tau_t) = \arg\max\limits_{a} Q^j\left(O(\tau_t, a), a\right)$.

Then, equations $(3)$ to $(7)$ demonstrate that $\epsilon$-greedy exploration is harmful, by substitute it into the equation

$$
\begin{aligned}
  P(\tau_t \mid \tau_t^i, a_t^j) 
  =& \frac{\pi^j\left(a_t^j\mid O(\tau_t,j) \right)\cdot B(\tau_t)}{\sum\limits_{\tau_t'} \pi^j\left(a_t^j\mid O(\tau_t',j) \right)\cdot B(\tau_t')}.
\end{aligned}
$$

Now, if agent $i$ can get agent $j$'s greedy action (in the centralized training phase), then $i$'s posterior can be

$$
\begin{aligned}
  P(\tau_t \mid \tau_t^i, a^{j*}) 
  =& \frac{\mathbf{I}\left(a^{j*}(\tau_t) = a^{j*}\right)\cdot B(\tau_t)}
  {\sum\limits_{\tau'} \mathbf{I}\left(a^{j*}(\tau') = a^{j*}\right)\cdot B(\tau')}.
\end{aligned}
$$

Note that the agent $i$ only takes $j$'s greedy action into account. And this stabilizes the training process.




---

## Trajectory Diversity

[Andrei Lupu, Brandon Cui, Hengyuan Hu, Jakob Foerster. "Trajectory diversity for zero-shot coordination." International conference on machine learning. PMLR, 2021.](https://proceedings.mlr.press/v139/lupu21a/lupu21a.pdf)

---

## Off-Belief Learning

[Hengyuan Hu, Adam Lerer, Brandon Cui, Luis Pineda, Noam Brown, Jakob Foerster. "Off-belief learning." International Conference on Machine Learning. PMLR, 2021.](https://proceedings.mlr.press/v139/hu21c/hu21c.pdf)

---

## Population-Based Training

[Zhao, Rui, et al. "Maximum entropy population-based training for zero-shot human-ai coordination." Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 37. No. 5. 2023.](https://ojs.aaai.org/index.php/AAAI/article/view/25758)

---

> The following part has not been finished yet. One may check my [writing schedule]({{site.baseurl}}/posts/Schedule/).
{: .prompt-warning }

---

## To-Read

### Jakob Foerster

<!-- - [ ] BAD  
  [Foerster, Jakob, et al. "Bayesian action decoder for deep multi-agent reinforcement learning." International Conference on Machine Learning. PMLR, 2019.](https://proceedings.mlr.press/v97/foerster19a.html) -->
- [x] SAD [[code](https://github.com/facebookresearch/hanabi_SAD)]  
  [Hu, Hengyuan, and Jakob N. Foerster. "Simplified Action Decoder for Deep Multi-Agent Reinforcement Learning." International Conference on Learning Representations. 2019.](https://arxiv.org/abs/1912.02288)
- [x] Other-Play [[code](https://github.com/facebookresearch/hanabi_SAD)]  
  [Hu, Hengyuan, et al. "“other-play” for zero-shot coordination." International Conference on Machine Learning. PMLR, 2020.](https://proceedings.mlr.press/v119/hu20a/hu20a.pdf)
- [ ] Trajectory Diversity  
  [Andrei Lupu, Brandon Cui, Hengyuan Hu, Jakob Foerster. "Trajectory diversity for zero-shot coordination." International conference on machine learning. PMLR, 2021.](https://proceedings.mlr.press/v139/lupu21a/lupu21a.pdf)
- [ ] Off-Belief Learning  
  [Hengyuan Hu, Adam Lerer, Brandon Cui, Luis Pineda, Noam Brown, Jakob Foerster. "Off-belief learning." International Conference on Machine Learning. PMLR, 2021.](https://proceedings.mlr.press/v139/hu21c/hu21c.pdf)

### Tencent

- [ ] Maximum Entropy Population-Based Training  
  [Zhao, Rui, et al. "Maximum entropy population-based training for zero-shot human-ai coordination." Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 37. No. 5. 2023.](https://ojs.aaai.org/index.php/AAAI/article/view/25758)

### Env
- [x] A matrix-game [[code](https://bit.ly/2P3YOyd)][[paper](https://proceedings.mlr.press/v97/foerster19a.html)]
  - communication through action challenge
  - BAD
- [x] Lever [[code](https://bit.ly/2vYkfI7)][[paper](https://proceedings.mlr.press/v119/hu20a/hu20a.pdf)]
  - Other-Play
- [x] Corridor [[paper](https://proceedings.mlr.press/v139/lupu21a/lupu21a.pdf)]
  - Trajectory Diversity
- [ ] Overcooked [[code](https://github.com/HumanCompatibleAI/overcooked_ai)]
- [x] Hanabi [[code](https://github.com/deepmind/hanabi-learning-environment)] [[paper](https://www.sciencedirect.com/science/article/pii/S0004370219300116)]
