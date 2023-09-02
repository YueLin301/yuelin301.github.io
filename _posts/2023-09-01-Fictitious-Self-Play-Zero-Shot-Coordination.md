---
title: Fictitious Self-Play and Zero-Shot Coordination
date: 2023-09-01 02:40:00 +0800
categories: [Economics & Game Theory]
tags: [game theory, fictitious play, self-play, multi agents, reinforcement learning]
math: True
---

## What is Fictitious Play?

*([Wikipedia](https://en.wikipedia.org/wiki/Fictitious_play))*

1. Fictitious play is a learning rule. 
2. In it, each player presumes that the opponents are playing stationary (possibly mixed) strategies. 
3. At each round, each player thus best responds to the empirical frequency of play of their opponent. 

---

Given a game with $n$ players, where each player $i$ has a strategy $\pi^i$. 

1. Players initialize their beliefs about the strategies of the other players. And $\hat{\pi}^{i,j}$ means the belief of $i$ regarding the $j$'s strategy.
2. At each round $t$:
   - Player $i$ observes the action $a_{t-1}^j$ (or pure strategy) of every other player $j \neq i$.
   - Player $i$ updates their belief of player $j$'s strategy based on the empirical frequency: $\hat{\pi}_t^{i,j} = \frac{1}{t-1} \sum_{k=1}^{t-1} a_k^j$. The actions are one-hot encoded.
   - Player $i$ then plays a **best response** to $\hat{\pi}_t^{i,j}$.


---

Fictitious play doesn't always converge to a Nash equilibrium in all games. It's been proven that fictitious play converges to the set of Nash equilibria in certain classes of games, like zero-sum games and potential games. However, there are games where fictitious play does not necessarily converge to a Nash equilibrium.

E.g., in the [Matching Pennies game](https://yuelin301.github.io/posts/Classic-Games/#matching-pennies), the trajectory of Fictitious Play tends to circle around the mixed-strategy Nash equilibrium rather than directly converging to it.

## Self-play

Self-play involves an agent (or a model) playing against itself or versions of itself. This can be thought of as a kind of bootstrapping method where an agent learns and refines its strategies through continuous iterations of gameplay against its own evolving strategies.

### Significance

1. **Unlimited Opponents**: Self-play provides an infinite supply of opponents since the agent is essentially playing against itself. This eliminates the need for external opponent data or human opponents, which can be limited or introduce variability.
2. **Evolving Difficulty**: As the agent improves, its past versions also present incrementally challenging opponents, ensuring that the agent is always pushed to improve and adapt.
3. **Consistency**: By playing against itself, the agent is exposed to a consistent level of gameplay, which can lead to more stable learning.

### Applications

The most famous application of self-play is perhaps in the training of **AlphaGo** and its successors by DeepMind. AlphaGo utilized self-play to achieve superhuman performance in the game of Go, a feat that was previously thought to be decades away. Following this, **AlphaZero** utilized a generalized version of this self-play approach to achieve superhuman performance not only in Go but also in Chess and Shogi.

### Process

1. **Initialization**: Start with a randomly initialized agent or a naive version.
2. **Play**: Let the agent play games against itself or against past versions of itself.
3. **Update**: After each game, or a batch of games, update the agent's model based on the results, rewards, and feedback from the games.
4. **Iterate**: Repeat the Play and Update steps for a desired number of iterations or until the agent's strategy converges to an optimal or satisfactory level.



- [ ] [Andrei Lupu, Brandon Cui, Hengyuan Hu, Jakob Foerster. "Trajectory diversity for zero-shot coordination." International conference on machine learning. PMLR, 2021.](http://proceedings.mlr.press/v139/lupu21a/lupu21a.pdf)
- [ ] [Hengyuan Hu, Adam Lerer, Brandon Cui, Luis Pineda, Noam Brown, Jakob Foerster. "Off-belief learning." International Conference on Machine Learning. PMLR, 2021.](http://proceedings.mlr.press/v139/hu21c/hu21c.pdf)
- [ ] Env
  - [ ] Hanabi [[code](https://github.com/deepmind/hanabi-learning-environment)] [[paper](https://www.sciencedirect.com/science/article/pii/S0004370219300116)]
  - [ ] Overcooked [[code](https://github.com/HumanCompatibleAI/overcooked_ai)]


> The following part has not been finished yet. One may check my [writing schedule](https://yuelin301.github.io/posts/Schedule/).
{: .prompt-warning }