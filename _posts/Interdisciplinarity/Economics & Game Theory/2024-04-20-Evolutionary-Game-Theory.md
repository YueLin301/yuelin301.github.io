---
title: Evolutionary Game Theory
date: 2024-04-20 14:40:00 +0800
categories: [Interdisciplinarity, Economics & Game Theory]
tags: [Tech, Interdisciplinarity, Economics, Game_Theory, Multi_Agents, Classic]
math: True
---

## Basic Symmetric Model with Stochastic Strategies
> This section is a summary of Chapter 29 of the book "Algorithmic Game Theory"[^AGT-Nisan].
{:.prompt-info}


1. Agents (organisms)
   1. The number of agents is infinite.
   2. Pure strategies (or action sets) are the same: $A.$
   3. Mixed strategies: $\Pi: \Delta(A).$
   4. In analysis, it is often assumed that there are only two mixed strategies. 
      1. The smaller group is referred to as `mutants`, with a population of $\epsilon.$
      2. The larger group is known as `incumbents`, with a population of $1-\epsilon.$
2. Env: Evolutionary game
   1. Uniformly sample 2 agents ($i$ and $j$) to play a symmetric `stage game`.
      1. Each agent just plays its current mixed strategies, $\pi^i$ and $\pi^j.$
      2. Fitness function (or reward function), the organisms ability to reproduce: $F: \Delta(A)\times\Delta(A) \to \mathbb{R}.$
      3. Agent $i$ gets a reward of $F(\pi^i\mid\pi^j)$ and agent $j$ gets $F(\pi^j\mid\pi^i).$
   2. Dynamics: Replication or Imitation.
      1. Replication: "Each organism reproduces asexually, and spawns a number of offspring proportional to its fitness."
      2. Imitation: "each agent initially plays some pure strategy. After interactions, if one agents sees the other agent earned a higher payoff, the agent with the lower payoff may adopt, or imitate, the strategy of the agent who earned the higher payoff."

### Two Main Characteristics
1. "The population is infinite."
2. "Players adopt a very simple, local dynamic, such as replication or imitation, for choosing and updating their strategies."

### The Goal
To find resilient strategies for a given environment.

### Evolutionarily Stable Strategy (ESS)

#### Description
"An evolutionarily stable strategy (ESS) is a strategy such that if all the members of a population adopt it, then no mutant strategy could overrun the population."

#### Expected Fitness
Incumbents and  mutants are denoted as $i$ and $j,$ respectively.

- An incumbent $i$
  - Meets an incumbemt: $(1-\epsilon)\cdot F(\pi^i\mid\pi^i)$
  - Meets an mutant: $\epsilon \cdot F(\pi^i\mid\pi^j)$
  - $\mathbb{E}_{\epsilon, \boldsymbol{\pi}}\left[f^i\right] = (1-\epsilon)\cdot F(\pi^i\mid\pi^i) + \epsilon \cdot F(\pi^i\mid\pi^j)$
- A mutant $j$
  - Meets an incumbemt: $(1-\epsilon)\cdot F(\pi^j\mid\pi^i)$
  - Meets an mutant: $\epsilon \cdot F(\pi^j\mid\pi^j)$
  - $\mathbb{E}_{\epsilon, \boldsymbol{\pi}}\left[f^j\right] = (1-\epsilon)\cdot F(\pi^j\mid\pi^i) + \epsilon \cdot F(\pi^j\mid\pi^j)$


#### Definition
> **Definition 29.1**[^AGT-Nisan] A strategy s is an evolutionarily stable strategy (ESS) for the 2-player, symmetric game given by fitness function F , if for every strategy $\pi^j\ne \pi^i,$ there exists an $\epsilon'$ such that for all $0<\epsilon<\epsilon',$ $\mathbb{E}\_{\epsilon, \boldsymbol{\pi}}\left[f^i\right] > \mathbb{E}\_{\epsilon, \boldsymbol{\pi}}\left[f^j\right],$ i.e., $(1-\epsilon)\cdot F(\pi^i\mid\pi^i) + \epsilon \cdot F(\pi^i\mid\pi^j) > (1-\epsilon)\cdot F(\pi^j\mid\pi^i) + \epsilon \cdot F(\pi^j\mid\pi^j).$
{:.prompt-info}

Definition 29.1 holds iff. **either** of 2 conditions on $\pi^i$ is satisfied $\forall\pi^j\ne\pi^i$
1. $F(\pi^i\mid\pi^i) > F(\pi^j\mid \pi^i),$ or
2. $F(\pi^i\mid\pi^i) = F(\pi^j\mid\pi^i)$ and $F(\pi^i\mid\pi^j) > F(\pi^j\mid\pi^j).$

Because $\epsilon$ can be approaching to $0.$

#### Corollaries 

> For $\pi^i$ to be an ESS, it must be the case that $F(\pi^i\mid\pi^i) \ge F(\pi^j\mid \pi^i), \forall \pi^j.$  
> It means: 
> - $\pi^i$ must be a best response to itself.
> - $\pi^i$ is an ESS $\to$ The strategy $(\pi^i, \pi^i)$ is a Nash equilibrium. (Equivalent to the previous one.)
{:.prompt-info}

> **Theorem 29.2**[^AGT-Nisan](An alternative definition of ESS) A strategy s is an ESS for the 2-player, symmetric game given by fitness function F , **iff.** $(\pi^i,\pi^i)$ is a Nash equilibrium of $F,$ and for every best response $\pi^j$ to $\pi^i$ where $\pi^j\ne\pi^i,$ $F(\pi^i\mid\pi^j) > F(\pi^j\mid \pi^j).$
{:.prompt-info}

Where
- "Profile $(\pi^i,\pi^i)$ is a Nash equilibrium of $F$" means "$F(\pi^i\mid\pi^i) \ge F(\pi^j\mid \pi^i), \forall \pi^j.$"
- "For every best response $\pi^j$ to $\pi^i$" means "$F(\pi^i\mid\pi^i) = F(\pi^j\mid\pi^i).$"

> In the long run, the fraction of mutants in the population will tend to $0$.
{:.prompt-info}

#### Memos
- "ESS are a refinement of Nash equilibria. ESS is more restrictive than Nash equilibrium."
- "Not all 2-player, symmetric games have an ESS."
- "A continuous time analysis of the replicator dynamics shows that every ESS is asymptotically stable."

### An Example: Hawks and Doves

The fitness matrix of the stage game is:

| Player1\Player2 	| Hawk              	| Dove      	|
|-----------------	|-------------------	|-----------	|
| **Hawk**        	| $(V-C)/2,(V-C)/2$ 	| $V,0$     	|
| **Dove**        	| $0,V$             	| $V/2,V/2$ 	|

Where $V$ means value, and $C$ means conflict.

The strategy profile $(D,D)$ is not a Nash equilibrium because one player can unilaterally deviate and play $H$  and increase its payoff from $V/2$ to $V.$

The strategy profile $(D,D)$ is not a Nash equilibrium $\to$ $D$ is not an ESS. (ESS are a refinement of Nash equilibria.)

Let $\pi$ denotes any policy with probability $0<p<1$ of playing $H.$

If $V > C$ then $H$ is an ESS: $F(\pi\mid H) = p\cdot (V-C) / 2 + (1-p)\cdot 0 < F(H\mid H) = (V-C)/2, \forall \pi\ne H.$

If $V\le C$ then $p=V/C$ is an ESS. The condition is, still, $F(p\mid p\*) < F(p^\*\mid p^\*), \forall p\ne p^\*.$


## Replicator Dynamics

**It requires the strategy set to be finite.**
<!-- Because of this restriction, I don't want to delve into this part yet. -->

**Replicator dynamics** is a mathematical model that describes how the proportion of individuals using certain strategies changes over time within a population. This dynamics is particularly important for studying Evolutionarily Stable Strategies (ESS) and the dynamic stability in game theory.

### Definition and Mathematical Formulation

Assume there is a set of strategies $\set{1, 2, \dots, n}$, where each strategy $i$ has a proportion $x_i$ in the population, with the constraint $\sum_{i=1}^n x_i = 1$. The fitness (or payoff) of each strategy $i$ is denoted as $f_i(x)$, which typically depends on the current distribution of all strategies $x = (x_1, x_2, \dots, x_n)$.

In replicator dynamics, the rate of change in the proportion of strategy $i$ over time is proportional to the deviation of its fitness from the average fitness. Specifically, the dynamics of strategy $i$ can be expressed as:

$$
\dot{x}_i = x_i (f_i(x) - \bar{f}(x))
$$

where $\bar{f}(x) = \sum_{j=1}^n x_j f_j(x)$ represents the expected fitness of the population.

### Intuitive Explanation

Intuitively, if a strategy has a fitness above the average, individuals adopting this strategy will reproduce at a faster rate, causing the proportion of this strategy in the population to increase. Conversely, strategies with a fitness below the average will decrease in proportion.

### Applications

Replicator dynamics is applied across various fields including ecology, economics, and social sciences, helping researchers understand how individual strategies evolve over time in competitive and cooperative environments. In evolutionary game theory, it serves as a key tool for analyzing the stability of strategies and the dynamics of group behavior.

## Other Dynamics

### Neutral Drift

Neutral drift is a genetic concept applied within evolutionary game theory to describe random changes in strategy frequencies when there are no significant adaptive advantages or disadvantages among the choices. 

If a strategy is adaptively equivalent to others (i.e., the differences in fitness between them are negligible), then their relative proportions may be determined solely by random factors. 

This phenomenon can theoretically be described using neutral drift models, which do not rely on fitness differences between strategies but emphasize the role of randomness and chance in evolutionary processes.

### Risk Dominance

In game theory, the concept of "risk dominance" is used to analyze how players choose strategies in games that have multiple equilibria. Risk dominance is based on the idea of minimizing the worst-case outcome—essentially minimizing potential losses—assuming opponents may choose different strategies.

Simply put, if in a game, strategy A is risk dominant over strategy B, it means that choosing strategy A ensures a higher minimum payoff compared to the minimum payoff of choosing strategy B, given the uncertainty of the opponent's strategy. This is a conservative decision-making approach under uncertainty.

A strategy $s_i$ is said to risk dominate another strategy $s_i'$ and can be mathematically expressed as follows:

Consider a two-player game where each player has two strategy options, $s_i$ and $s_i'$. For player $i$, the condition for strategy $s_i$ to risk dominate strategy $s_i'$ is:

$$
\min_{s_{-i} \in S_{-i}} u_i(s_i, s_{-i}) > \min_{s_{-i} \in S_{-i}} u_i(s_i', s_{-i}),
$$

where:
- $u_i(s_i, s_{-i})$ is the utility (or payoff) of player $i$ when they choose strategy $s_i$ and other players choose strategy $s_{-i}$.
- $S_{-i}$ represents the strategy set of all other players except player $i$.

This expression states that if player $i$ chooses strategy $s_i$, considering all possible strategies of the other players, the worst-case utility of $s_i$ is still higher than the worst-case utility of $s_i'$.

This criterion is a conservative strategy.

## References
[^AGT-Nisan]: Nisan, Noam and Roughgarden, Tim and Tardos, Eva and Vazirani, Vijay V. Algorithmic Game Theory. 2007.
[^GTE]: Gintis, Herbert. Game Theory Evolving: A Problem-Centered Introduction to Modeling Strategic Interaction - Second Edition. 2009.

