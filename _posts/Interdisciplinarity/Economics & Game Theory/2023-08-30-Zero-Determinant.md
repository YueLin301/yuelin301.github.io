---
title: Zero-Determinant Strategy
date: 2023-08-30 02:40:00 +0800
categories: [Interdisciplinarity, Economics & Game Theory]
tags: [Tech, Interdisciplinarity, Economics, Game_Theory, Social_Dilemma, Multi_Agents, Classic]
math: True
---

> This note aims to explain the parts omitted in this paper:  
> [Press, William H., and Freeman J. Dyson. "Iterated Prisoner’s Dilemma contains strategies that dominate any evolutionary opponent." Proceedings of the National Academy of Sciences 109.26 (2012): 10409-10413.](https://www.pnas.org/doi/pdf/10.1073/pnas.1206569109)
{: .prompt-info }


## Interesting Facts

1. As stated in the title: **Iterated Prisoner’s Dilemma contains strategies that dominate any evolutionary opponent.** And this kind of strategy is the Zero-Determinant strategy.
2. The "Iterated Prisoner’s Dilemma is an Ultimatum game".
3. "One player can enforce a unilateral claim to an unfair share of rewards", by the ZD strategy.
4. Any evolutionary agent will be exploited by the agent with the ZD strategy. "An evolutionary player’s best response is to accede to the extortion. Only a player with a theory of mind about his opponent can do better."
5. "For any strategy of the longer-memory player Y, shorter-memory X’s score is exactly the same as if Y had played a certain shorter-memory strategy."

---

## Iterated Prisoner's Dilemma

At each timestep, two agents are playing Prisoner's Dilemma:

| Player1\Player2      | Cooperate (Deny) | Defect (Confess) |
| -------------------- | ---------------- | ---------------- |
| **Cooperate (Deny)** | $R,R$            | $S,T$            |
| **Defect (Confess)** | $T,S$            | $P,P$            |

where $T > R > P > S$ and $2R > T+S$, and the meanings are as follows.
- $T$: Temptation
- $R$: Reward
- $P$: Punishment
- $S$: Sucker's payoff

The two agents repeatedly play this game $T$ times. It might be finite or infinite.

---

## Longer-Memory Strategies Offer No Advantage

- Player $i$:
  - Making decisions on short memories $\tau^i$.
  - $\pi^i(a^i\mid \tau^i)$
- Player $j$
  - Making decisions on long memories $(\tau^i,\Delta\tau)$.
  - $\pi^j(a^j\mid \tau^i,\Delta\tau)$

Derivation:

$$
\begin{aligned}
    \mathrm{Pr}(a^i, a^j)
    =&\sum\limits_{\tau^i,\Delta\tau} 
    \pi^i(a^i\mid \tau^i) \cdot
    \pi^j(a^j\mid \tau^i,\Delta\tau) \cdot
    \mathrm{Pr}(\tau^i,\Delta\tau) \\
    =& \sum\limits_{\tau^i} \pi^i(a^i\mid \tau^i) \cdot
    \left[
        \sum\limits_{\Delta\tau} \pi^j(a^j\mid \tau^i,\Delta\tau)
        \cdot \mathrm{Pr}(\Delta\tau\mid \tau^i) \cdot \mathrm{Pr}(\tau^i)
    \right] \\
    =& \sum\limits_{\tau^i} \pi^i(a^i\mid \tau^i)  \cdot
    \left[ \sum\limits_{\Delta\tau} \mathrm{Pr}(a^j, \Delta\tau\mid \tau^i) \right]
    \cdot \mathrm{Pr}(\tau^i)\\
    =& \sum\limits_{\tau^i} \pi^i(a^i\mid \tau^i) \cdot \mathrm{Pr}(a^j \mid \tau^i) \cdot \mathrm{Pr}(\tau^i)
\end{aligned}
$$

$\mathrm{Pr}(a^j \mid \tau^i)$ is the player $j$'s marginalized strategy. And

$$
\mathrm{Pr}(a^j \mid \tau^i) = \sum\limits_{\Delta\tau} \mathrm{Pr}(a^j, \Delta\tau\mid \tau^i).
$$

- After some plays, $j$ can estimate the expectations, and it can switch to an equivalent short-memory strategy.
- "$j$’s switching between a long- and short-memory strategy is completely undetectable (and irrelevant) to $i$."

> Is this conclusion only suitable for repeated games?
{: .prompt-tip }

## Zero-Determinant Strategy

Some tedious parts were automatically filled in with the help of ChatGPT-4.

### Notation of 4 outcomes

| Player1\Player2      | Cooperate (c)     | Defect (d)        |
| -------------------- | ----------------  | ----------------  |
| **Cooperate (c)**    | $\mathrm{cc}$ (1) | $\mathrm{cc}$ (2) |
| **Defect (d)**       | $\mathrm{dc}$ (3) | $\mathrm{cc}$ (4) |

### Notation of strategies
- There are two players, $i$ and $j$, with memory-one strategies.
- Strategies are based on the outcome of last play.

$$
\begin{aligned}
\mathbf{p} =& \pi^i(a_t^i=\mathrm{Cooperate}\mid a_{t-1}^i,a_{t-1}^j) \\
=&(p_{\mathrm{cc}}, p_{\mathrm{cd}}, p_{\mathrm{dc}}, p_{\mathrm{dd}}) \\
=&(p_1, p_2, p_3, p_4)
\end{aligned}
$$

$p_1, p_2, p_3, p_4$ are independent and range from $[0,1].$

$$
\begin{cases}
    p_{\mathrm{cc}} = \pi^i(a_t^i=\mathrm{Cooperate}\mid a_{t-1}^i =\mathrm{Cooperate},a_{t-1}^j=\mathrm{Cooperate}) \\
    p_{\mathrm{cd}} = \pi^i(a_t^i=\mathrm{Cooperate}\mid a_{t-1}^i=\mathrm{Cooperate},a_{t-1}^j=\mathrm{Defect}) \\
    p_{\mathrm{dc}} = \pi^i(a_t^i=\mathrm{Cooperate}\mid a_{t-1}^i=\mathrm{Defect},a_{t-1}^j=\mathrm{Cooperate}) \\
    p_{\mathrm{dd}} = \pi^i(a_t^i=\mathrm{Cooperate}\mid a_{t-1}^i=\mathrm{Defect},a_{t-1}^j=\mathrm{Defect})
\end{cases}
$$

$$
\mathbf{q} = \pi^j(a_t^j=\mathrm{Cooperate}\mid a_{t-1}^i,a_{t-1}^j)
$$

### Markov transition matrix: $\mathbf{M}(\mathbf{p}, \mathbf{q})$

- It is a transition kernel of an MDP.
- The rows indicates the current states.
- The columns indicates the next states.
- Each entry indicates the probability of the current row state transitioning to the next column state.


|               | $\mathrm{cc}$                              | $\mathrm{cd}$                           | $\mathrm{dc}$                           | $\mathrm{dd}$                           |
| ------------- | ------------------------------------------ | -------------------------------------- | -------------------------------------- | -------------------------------------- |
| $\mathrm{cc}$ | $\mathrm{Pr}(\mathrm{cc}\mid \mathrm{cc})$ | $\mathrm{Pr}(\mathrm{cd}\mid \mathrm{cc})$ | $\mathrm{Pr}(\mathrm{dc}\mid \mathrm{cc})$ | $\mathrm{Pr}(\mathrm{dd}\mid \mathrm{cc})$ |
| $\mathrm{cd}$ | $\mathrm{Pr}(\mathrm{cc}\mid \mathrm{cd})$ | $\mathrm{Pr}(\mathrm{cd}\mid \mathrm{cd})$ | $\mathrm{Pr}(\mathrm{dc}\mid \mathrm{cd})$ | $\mathrm{Pr}(\mathrm{dd}\mid \mathrm{cd})$ |
| $\mathrm{dc}$ | $\mathrm{Pr}(\mathrm{cc}\mid \mathrm{dc})$ | $\mathrm{Pr}(\mathrm{cd}\mid \mathrm{dc})$ | $\mathrm{Pr}(\mathrm{dc}\mid \mathrm{dc})$ | $\mathrm{Pr}(\mathrm{dd}\mid \mathrm{dc})$ |
| $\mathrm{dd}$ | $\mathrm{Pr}(\mathrm{cc}\mid \mathrm{dd})$ | $\mathrm{Pr}(\mathrm{cd}\mid \mathrm{dd})$ | $\mathrm{Pr}(\mathrm{dc}\mid \mathrm{dd})$ | $\mathrm{Pr}(\mathrm{dd}\mid \mathrm{dd})$ |


|               | $\mathrm{cc}$      | $\mathrm{cd}$      | $\mathrm{dc}$      | $\mathrm{dd}$         |
| ------------- | ------------------ | ------------------ | ------------------ | --------------------- |
| $\mathrm{cc}$ | $p_1\cdot q_1$     | $p_1\cdot (1-q_1)$ | $(1-p_1)\cdot q_1$ | $(1-p_1)\cdot(1-q_1)$ |
| $\mathrm{cd}$ | $p_2\cdot q_3$     | $p_2\cdot (1-q_3)$ | $(1-p_2)\cdot q_3$ | $(1-p_2)\cdot(1-q_3)$ |
| $\mathrm{dc}$ | $p_3\cdot q_2$     | $p_3\cdot (1-q_2)$ | $(1-p_3)\cdot q_2$ | $(1-p_3)\cdot(1-q_2)$ |
| $\mathrm{dd}$ | $p_4\cdot q_4$     | $p_4\cdot (1-q_4)$ | $(1-p_4)\cdot q_4$ | $(1-p_4)\cdot(1-q_4)$ |


<!-- - Each row sums up to $1.$ -->

### $\mathbf{M}$ has a unit eigenvalue

> This part requires some knowledge of stochastic processes. Check [my other note]({{site.baseurl}}/posts/Stochastic-Processes/).
{: .prompt-info }

- $\mathbf{M}$ is irreducible. It means that the entire state space is on communicating class, i.e., for every two states $i$ and $j$, it holds that $i\leftrightarrow j.$
- $\mathbf{M}$ is finite. It means that $\mathbf{M}$ has a finite number of states.

$\Rightarrow$ $\mathbf{M}$ is positive recurrent. It means that every state can return to itself in finite steps. Formally, $\mathbb{E}[T_i \mid X_0 = i] = m_i < \infty,$ where $T_i$ is the first return time and $T_i = \min \set{ n > 0 : X_n = i \mid X_0 = i }.$ (If the state space of an irreducible Markov chain is finite, then all its states are positive recurrent.) 

$\Rightarrow$ $\mathbf{M}$ has a stationary distribution $\mathbf{v}^\intercal = \mathbf{v}^\intercal \mathbf{M}.$ (If the irreducible Markov chain is positive recurrent, then a stationary distribution $\mathrm{v}$ exists, is unique, and is given by $\mathrm{v}_i = 1/m_i$, where $m_i$ is the expected return time to state $i$.)

$\Rightarrow$ $\mathbf{M}$ has a unit eigenvalue: $1\cdot \mathbf{v}^\intercal = \mathbf{v}^\intercal \mathbf{M}.$

### $\operatorname{det}(\mathbf{M'}) = 0$

$\mathbf{M}$ has a unit eigenvalue: $1\cdot \mathbf{v}^\intercal = \mathbf{v}^\intercal \mathbf{M}.$

$\Rightarrow$ $\mathbf{v}^\intercal \mathbf{M'} = \boldsymbol{0},$ where $\mathbf{M'} := \mathbf{M} - \mathbf{I}.$

$\Rightarrow$ $\mathbf{M'}$ is singular. Because $\mathbf{v}$ is not a zero vetor, meaning that there is a non-zero vector in the null space of $\mathbf{M'}.$

$\Rightarrow$ $\operatorname{det}(\mathbf{M'}) = 0,$ by definition.

### Every row of $\operatorname{Adj}(\mathbf{M}')$ is proportional to $\mathbf{v}$

$$
\mathbf{M} = \left[\begin{array}{llll}
p_1 q_1 & p_1\left(1-q_1\right) & \left(1-p_1\right) q_1 & \left(1-p_1\right)\left(1-q_1\right) \\
p_2 q_3 & p_2\left(1-q_3\right) & \left(1-p_2\right) q_3 & \left(1-p_2\right)\left(1-q_3\right) \\
p_3 q_2 & p_3\left(1-q_2\right) & \left(1-p_3\right) q_2 & \left(1-p_3\right)\left(1-q_2\right) \\
p_4 q_4 & p_4\left(1-q_4\right) & \left(1-p_4\right) q_4 & \left(1-p_4\right)\left(1-q_4\right)
\end{array}\right]
$$

$$
\mathbf{M'} = \left[\begin{array}{llll}
p_1 q_1 - 1 & p_1\left(1-q_1\right) & \left(1-p_1\right) q_1 & \left(1-p_1\right)\left(1-q_1\right) \\
p_2 q_3 & p_2\left(1-q_3\right) - 1 & \left(1-p_2\right) q_3 & \left(1-p_2\right)\left(1-q_3\right) \\
p_3 q_2 & p_3\left(1-q_2\right) & \left(1-p_3\right) q_2 - 1 & \left(1-p_3\right)\left(1-q_2\right) \\
p_4 q_4 & p_4\left(1-q_4\right) & \left(1-p_4\right) q_4 & \left(1-p_4\right)\left(1-q_4\right) - 1
\end{array}\right]
$$

The cofactor of $\mathbf{M'}:$

$$
C_{ij} = (-1)^{i+j} \cdot \operatorname{det}(\mathbf{M'}_{ij}),
$$

where $\mathbf{M'}\_{ij}$ is the $3 \times 3$ submatrix, with the row and column containing the element $\mathbf{M'}\_{ij}$ removed from the matrix $\mathbf{M'}.$

$$
\operatorname{Adj}(\mathbf{M'}) = \begin{bmatrix}
C_{11} & C_{21} & C_{31} & C_{41} \\
C_{12} & C_{22} & C_{32} & C_{42} \\
C_{13} & C_{23} & C_{33} & C_{43} \\
C_{14} & C_{24} & C_{34} & C_{44}
\end{bmatrix}
$$

A basic lemma in Linear Algebra:   

$$
\mathbf{A} \cdot \operatorname{Adj}(\mathbf{A}) = \operatorname{Adj}(\mathbf{A}) \cdot \mathbf{A} = \operatorname{det}(\mathbf{A}) \cdot \mathbf{I}.
$$

Here:

$$
\operatorname{Adj}(\mathbf{M'}) \cdot \mathbf{M'} = \operatorname{det}(\mathbf{M'}) \cdot \mathbf{I} = 0 \cdot \mathbf{I} = 0.
$$

Then we have $\operatorname{Adj}(\mathbf{M'}) \cdot \mathbf{M'} = \mathbf{v}^\intercal \mathbf{M'} = \boldsymbol{0}.$ It follows that every row of $\operatorname{Adj}(\mathbf{M'})$ and \mathbf{v} is in the null space of $\mathbf{M'}.$ And thus every row of $\operatorname{Adj}(\mathbf{M}')$ is proportional to $\mathbf{v}.$

### Adding the first column of $\mathbf{M′}$ into the second and third columns

$$
\mathbf{M}^{\prime\prime}=\left[\begin{array}{llll}
p_1 q_1 - 1 & p_1-1 & q_1-1   & \left(1-p_1\right)\left(1-q_1\right) \\
p_2 q_3 & p_2-1     & q_3     & \left(1-p_2\right)\left(1-q_3\right) \\
p_3 q_2 & p_3       & q_2 - 1 & \left(1-p_3\right)\left(1-q_2\right) \\
p_4 q_4 & p_4       & q_4     & \left(1-p_4\right)\left(1-q_4\right) - 1
\end{array}\right]
$$

The 4-th row of $\operatorname{Adj}(\mathbf{M'})$ is $(C_{14}, C_{24}, C_{34}, C_{44}).$ The signs and the determinants of these cofactors are not changed compared to the ones of $\mathbf{M}^{\prime\prime}.$

### $\mathbf{v'} \cdot \mathbf{f} \equiv \operatorname{D}(\mathbf{p}, \mathbf{q}, \mathbf{f})$

$$
\mathbf{v'} \cdot \mathbf{f} \equiv
\operatorname{D}(\mathbf{p}, \mathbf{q}, \mathbf{f}) = 
\operatorname{det}
\left[\begin{array}{llll}
p_1 q_1 - 1 & p_1-1 & q_1-1   & f_1 \\
p_2 q_3 & p_2-1     & q_3     & f_2 \\
p_3 q_2 & p_3       & q_2 - 1 & f_3 \\
p_4 q_4 & p_4       & q_4     & f_4 
\end{array}\right]
$$

Here $\mathbf{v'} = k\cdot\mathbf{v} = (C_{14}, C_{24}, C_{34}, C_{44}),$ where $k$ is an integer. The equation holds by the definition of determinant.




> The rest of the paper is not hard to parse.
{: .prompt-tip }