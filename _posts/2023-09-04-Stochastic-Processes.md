---
title: A Note on Stochastic Processes
date: 2023-09-04 02:40:00 +0800
categories: [Mathematics]
tags: [stochastic processes]
math: True
---

> This note partially uses the materials from [the notes of MATH2750](https://mpaldridge.github.io/math2750/S10-stationary-distributions.html).
{: .prompt-info }

## Transition Matrix

- The transition kernel $\mathbf{M}$ is a square matrix of size $\vert S\vert \times \vert S\vert$. 
- $\mathbf{M}_{ij}$ means the probability of the current state $i$ transitioning to the next state $j$.
- The exponent $k$ of the matrix power $\mathrm{M}^k$ represents the state distribution after $k$ consecutive state transitions from the current state distribution.
- $\mathbf{M}_{ij}^k$ means the probability of the current state $i$ transitioning to the state $j$ after $k$ timesteps.


## Class Structure
There may be some "irrelevant" states in a Markov chain. We can eliminate them to simplify the model.

### Accessible
These statements are equivalent:
- State $j$ is accessible from state $i;$
- $i$ can transition to $j$ in $n$ steps;
- Starting from $i$, there’s a positive chance that we’ll get to $j$ at some point in the future;
- $\exists n, \mathrm{s.t. }\mathbf{M}_{ij}^n > 0;$
- $i \to j.$

### Communicates with
#### Definition
These statements are equivalent:
- $i$ communicates with $j$;
- $i\to j$ and $j\to i$;
- $i \leftrightarrow j$.

#### Properties
- **Reflexive:** $i\leftrightarrow i, \forall i.$
- **Symmetric:** if $i \leftrightarrow j$ then $j \leftrightarrow i.$
- **Transitive:** if $i \leftrightarrow j$ and $j \leftrightarrow k$ then $i \leftrightarrow k.$

#### Proof of properties
##### Reflexive
$\mathrm{M}_{ii}^{n=0}=1>0$. In 0 step we stay where we are.

##### Symmetric
The definition of $i\leftrightarrow j$ is symmetric under swapping $i$ and $j$.

##### Transitive
- $i \leftrightarrow j$ means $\exists n_1\ge 1$ s.t. $\mathrm{M}_{ij}^{n_1}>0.$
- $j \leftrightarrow k$ means $\exists n_2\ge 1$ s.t. $\mathrm{M}_{jk}^{n_2}>0.$

$$
\begin{aligned}
& P(X_n = k\mid X_0 = i) \\
=& \sum\limits_{l,m\in S} P(X_n = k, X_{n-n_2} = l, X_{n_1} = m\mid X_0 = i) \\ 
=& \sum\limits_{l,m\in S} P(X_n = k \mid X_{n-n_2} = l) \cdot P(X_{n-n_2} = l \mid X_{n_1} = m) \cdot P(X_{n_1} = m\mid X_0 = i) \\ 
\ge& P(X_n = k \mid X_{n-n_2} = j) \cdot P(X_{n-n_2} = j \mid X_{n_1} = j) \cdot P(X_{n_1} = j\mid X_0 = i) \\
=& \mathrm{M}_{ij}^{n_1} \cdot \mathrm{M}_{jj}^{n-n_1-n_2} \cdot \mathrm{M}_{jk}^{n_2} \\
\ge& 0
\end{aligned}
$$


### Communicating class
All states $j$ that communicate with state $i$ are in the same equivalence class (or communicating class) as state $i.$

#### Closed communicating class
These statements are equivalent:
- Class $C\subset S$ is closed if whenever there exist $i\in C$ and $j\in S$ with $i\to j$, then $j\in C$ also.
- No state outside the class is accessible from any state within the class.
- Once you enter a closed class, you can’t leave that class.

> If a state $j$ is out of a communicating class where state $i$ in (i.e. $i \nleftrightarrow j$), it still may enter the communicating class (i.e. $j\to i$ but $i\nrightarrow j$).
{: .prompt-tip }

#### Open communicating class
If a communicating class is not closed, then it is open.

### Irreducible Markov chain

#### Definition
A Markov chain is irreducible if its entire state space $S$ is on communicating class.

#### How can I tell if a Markov chain is irreducible?
The mere presence of zeros in the transition matrix does **not** mean that the Markov chain is reducible. E.g.:

$$
\begin{bmatrix}
0.5 & 0.5 & 0.0 \\
0.0 & 0.5 & 0.5 \\
0.5 & 0.0 & 0.5 \\
\end{bmatrix}
$$

In my understanding, if a whole row (or a whole column) of a transition matrix is all zeros except for the pivot, then the Markov chain with this transition matrix is reducible, and the corresponding row (or column) state can be eliminated. 

E.g., this is such a transition matrix with which the Markov chain is reducible:

$$
\begin{bmatrix}
0.5 & 0.5 & 0.0 \\
0.4 & 0.5 & 0.1 \\
0.0 & 0.0 & 1.0 \\
\end{bmatrix}
$$

### Absorbing state
These statements are equivalent:
- State $i$ is an absorbing state.
- State $i$ is in a communicating class $\{i\}$ by itself and that class is closed.
- Once you reach state $i$, you can’t leave it.

### Periodicity

#### Definition

Periodicity describes a pattern of recurrence for a state. For a state $i$ in a Markov chain, its period $d(i)$ is defined as the greatest common divisor of the number of steps it can take to return to state $i$, excluding the possibility of 1-step self-transitions.

More formally, consider all $n$ such that $\mathbf{M}^n_{ii} > 0$ (i.e., the probability of transitioning from state $i$ to state $i$ in $n$ steps is positive). The period $d(i)$ is the greatest common divisor of all such $n$.

1. If $d(i) = 1$, then state $i$ is said to be aperiodic.
2. If $d(i) > 1$, then state $i$ is said to be periodic, with a period of $d(i)$.

The transition kernel is a square matrix. The exponent $k$ of the matrix power $\mathrm{M}^k$ represents the state distribution after $k$ consecutive state transitions from the current state distribution.

#### Examples

All states in the Markov chain with the following transition matrix are periodic with a period of 3.

$$
\mathbf{M} = \begin{bmatrix}
0 & 1 & 0 \\
0 & 0 & 1 \\
1 & 0 & 0 \\
\end{bmatrix}
$$


All states in the Markov chain with the following transition matrix are aperiodic.

$$ 
\mathbf{M} = \begin{bmatrix}
0.2 & 0.4 & 0.4 \\
0.6 & 0.2 & 0.2 \\
0.1 & 0.8 & 0.1 \\
\end{bmatrix}
$$

> In my understanding, this constraint is very lenient; as long as the probability of transitioning to oneself is greater than 0, it counts.
{: .prompt-tip }


---

## Hitting Times

### Hitting time


Let $(X_n)$ be a Markov chain on state space $S$. Let $H_{S'}$ be a random variable representing the hitting time to hit the set $S'\subset S$, given by

$$
H_{S'} = \min \{ n\in \{0, 1,2,\ldots\} : X_n \in S' \}
$$

$H_{S'}$ means the first timestep that any state in $S'$ appears.

If $S'$ contains only one state, i.e. $S' = \{i\}$, then the $H_{i}$ means the first timesstep that state $i$ appears.

$H_{S'} = \infty$ if $X_n\in S'$ for all $n,$ meaning that all the states in $S'$ will never appear.

### Hitting probability

The hitting probability $h_{iS'}$ of the set $S'$ from the state $i$ is

$$
\begin{aligned}
    h_{iS'} 
    =& P(X_n\in S'\text{ for some }n\ge n\mid X_0 = i) \\
    =& P(H_{S'} < \infty \mid X_0 = i)
\end{aligned}
$$

It means that the probability of $S'$ will appear in finite timesteps, given the starting state $i.$


If $S'$ contains only one state, i.e. $S' = \{i\}$, then the $h_{ij}$ means the probability of $i$ transitioning to $j$ in finite timesteps.

$$
    h_{ij} 
    = P(H_j < \infty \mid X_0 = i)
$$

### Expected hitting time

The expected hitting time $\eta_{iS'}$ of the set $S'$ starting from state $i$ is

$$
\eta_{iS'} = \mathbb{E}(H_{S'}\mid X_0 = i)
$$

$\eta_{iS'}$ can only be finite if $h_{iS'} = 1.$


### Return time

#### First return time

Given a state $i \in S$ of a Markov chain, the first return time $T_i$ to state $i$ is defined as:

$$ T_i = \min \{ n > 0 : X_n = i | X_0 = i \} $$

This is the number of time steps required to return to state $i$ for the first time, given that we start at $i$. Essentially, $T_i$ represents the first occurrence of the state $i$ after the initial time, assuming the chain started at state $i$.

#### Expected return time

The expected return time, $m_i$, to state $i$ is the average number of time steps it takes to return to state $i$ after initially starting at $i$. Mathematically, it's given by the expected value of the first return time $T_i$:

$$ 
\begin{aligned}
m_i =& \mathbb{E}[T_i | X_0 = i] \\
=& \sum\limits_{n=1}^\infty n\cdot \mathbf{M}^n_{ii}
\end{aligned}
$$

If the Markov chain is guaranteed to return to state $i$ and does so in an average of a finite number of steps, then $m_s$ is finite. If the Markov chain can return to state $i$ but takes, on average, an infinite number of steps, then $m_i$ is infinite. If the chain is not guaranteed to return to state $i$, then the expected return time is undefined (though it is often treated as infinite for certain analyses).


---

## Recurrence and Transience

### Recurrent and transient states

#### Recurrent state
A state $i \in S$ is recurrent if and only if the probability of eventually returning to $i$ starting from $i$ is 1, that is:

$$ P(T_i < \infty) = 1 $$

The following statements are equivalent:
1. State $i$ is a recurrent state;
2. $P(T_i < \infty) = 1;$
3. $P(R_i = \infty \mid X_0 = i) = 1$
  - or $P(R_i<\infty \mid X_0 = i) = 0$
  - The number of return of $i$ is infinite.
  - $R_i$ is the number of return of $i.$
4. $\mathbb{E}(R_i \mid X_0 = i) = \infty$

State $i$ is a recurrent state if and only if $\sum\limits_{n\ge 1}\mathbf{M}_{ii}^n = \infty$


#### Positive ecurrent state

If there exists a constant $m_i$ (expected return time) such that $\mathbb{E}[T_i | X_0 = i] = m_i < \infty$. This means that the expected time to return to state $i$ is finite.

- If the state space of a Markov chain is finite (has a finite number of states), then all its recurrent states are positive recurrent.
- **If the state space of an irreducible Markov chain is finite (has a finite number of states), then all its states are positive recurrent.**

#### Null recurrent state
If $\mathbb{E}[T_i | X_0 = i] = \infty$. This means that the expected time to return to state $i$ is infinite.

#### Transient State

A state $i \in S$ is transient if and only if the probability of eventually returning to $i$ starting from $i$ is less than 1, that is:
$$ P(T_i < \infty) < 1 $$

- [Comparison](https://mpaldridge.github.io/math2750/S09-recurrence-transience.html) of the two kinds of state:

|                                   | Recurrent states                                             | Transient states                                             |
| --------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **Visiting behavior**             | If we ever visit $s$, then we keep returning to $s$ again and again | We might visit $s$ a few times, but eventually we leave $s$ and never come back |
| **Expected number of visits**     | Starting from $s$, the expected number of visits to $s$ is infinite | Starting from $s$, the expected number of visits to $s$ is finite |
| **Certainty of number of visits** | Starting from $s$, the number of visits to $s$ is certain to be infinite | Starting from $s$, the number of visits to $s$ is certain to be finite |
| **Return probability**            | The return probability $m_s$ equals $1$                      | The return probability $m_s$ is strictly less than $1$       |

> Consider a Markov chain with transition matrix $\mathbf{M}$.
> - If the state $s$  is recurrent, then $\sum_{n=1}^{\infty} \mathbf{M}_{ss}(n) = \infty,$ and we return to state $s$ infinitely many times with probability 1.
> - If the state $s$ is transient, then $\sum_{n=1}^{\infty} \mathbf{M}_{ss}(n) < \infty,$ and we return to state $s$ infinitely many times with probability 0.
{: .prompt-info }

### Recurrent and transient classes
These statements are equivalent:
- Within a communicating class, either every state is transient or every state is recurrent.
- Let $i,j\in S$ be such that $i \leftrightarrow j$. If $i$ is recurrent, then $j$ is recurrent also; while if $i$ is transient, then $j$ is transient also.

If a Markov chain is irreducible (the entire state space is a single communicating class), we can refer to it as a “recurrent Markov chain” or a “transient Markov chain”.

- Every non-closed communicating class is transient. 
  - Non-closed communicating class $S'$ means that there exists $i\in S'$ such that $i\to j$ and $j\notin S'$, meaning that you can escape from $S'.$
  - Once you are escaped from $S'$, you will never come back. Beacause $i\to j$ and $j\notin S' \Rightarrow j\nrightarrow i.$
- Every finite closed communicating class is positive recurrent.
- Infinite closed classes can be positive recurrent, null recurrent, or transient.


---

## Stationary Distributions

### Definition
- The distribution is defined on the state set, representing the probability of each state occurring at a timestep.
- E.g., a state set is $\{s_1, s_2\}$, then a distribution can be $(0.7, 0.3)$, meaning that $s_1, s_2$ will appear with the probability of $0.7, 0.3$ respectively.
- If the current distribution is $\mathbf{v}$, then the distribution at the next timestep is $\mathbf{v} \mathbf{M}$.

If a distribution $\mathbf{v} = \mathbf{v} \mathbf{M}$, then it is a stationary distribution.

If a distribution $\mathbf{v}_i = \sum_j \mathbf{v}_i \cdot \mathbf{M}_{ij}$, then it is a stationary distribution.


A stationary distribution is a fixed point.


### Existence & Uniqueness

Consider an **irreducible** Markov chain.

- If the **irreducible** Markov chain is **positive recurrent** (i.e. the irreducible Markov chain is finite), then a stationary distribution $\mathrm{v}$ **exists**, is **unique**, and is given by $\mathrm{v}_i = 1/m_i$, where $m_i$ is the expected return time to state $i$.
- If the Markov chain is null recurrent or transient, then no stationary distribution exists.

Consider a Markov chain with multiple communicating classes.

- If none of the classes are positive recurrent, then no stationary distribution exists.
- If exactly one of the classes is positive recurrent (and therefore closed), then there exists a unique
stationary distribution, supported only on that closed class.
- If more the one of the classes are positive recurrent, then many stationary distributions will exist.
  - It can be divided; calculate their respective stationary distributions, and then combine them together.

### Proof of Existence

Let $X_n$ be the Markov chain, and let $\pi_j$ be the probability that the chain is in state $j$ at time $n$ given it starts in state $i$ at time 0, i.e., $\pi_j = P(X_n = j | X_0 = i)$.


<!-- From the definition of positive recurrence, it is also known that:
$$ \lim_{n \to \infty} \frac{1}{n} \sum_{k=1}^{n} P(X_k = i | X_0 = i) = \frac{1}{m_i} $$ -->

Recall the definitions and basic concepts associated with positive recurrence:

1. For state $i$, the first return time $T_i$ is defined as $T_i = \min\{ n > 0 : X_n = i \}$. In other words, it's the time required for the chain to return to state $i$ for the first time after starting from $i$.

2. State $i$ is said to be positively recurrent if $E(T_i | X_0 = i) = m_i < \infty$. That is, the expected time to return to $i$ starting from $i$ is finite.

Now, think about the long-time trajectory starting in state $i$. Given the positive recurrence, you'd expect the chain to return to state $i$ multiple times over a long period. Specifically, if you think of $m_i$ steps, you'd expect, on average, that one of these $m_i$ steps is in state $i$.

So, when you think over even longer durations, say $n$ steps, you'd expect the chain to be in state $i$ about $n/m_i$ times. In other words, the "proportion" of time the chain spends in state $i$ should converge to $\frac{1}{m_i}$.

From the discussion above, we have:

$$ 
\lim_{n \to \infty} \frac{1}{n} \sum_{k=1}^{n} P(X_k = i | X_0 = i) 
$$

This represents the average proportion of the first $n$ steps where the chain is in state $i$ given it started in state $i$. And indeed, this should equate to $\frac{1}{m_i}$.


Now, let's consider the average probability of being in state $j$ at time $n$ given we start in state $i$ at time 0:

$$ 
\lim_{n \to \infty} \frac{1}{n} \sum_{k=1}^{n} P(X_k = j | X_0 = i) = \pi_j 
$$

This $\pi_j$ is our desired stationary distribution for state $j$.

### Proof of Uniqueness

Suppose there are two stationary distributions $\pi$ and $\pi'$ for an irreducible Markov chain. Then, for each state $j$ and transition probability matrix $P$:

$$ 
\pi_j = \sum_i \pi_i P_{ij} 
$$

$$ 
\pi'_j = \sum_i \pi'_i P_{ij} 
$$

Consider a convex combination of $\pi$ and $\pi'$:

$$
\rho = \alpha \pi + (1 - \alpha) \pi' 
$$

For some $0 < \alpha < 1$.

We can then show:
$$
\rho_j = \sum_i \rho_i P_{ij} 
$$

This means $\rho$ is also a stationary distribution, but this contradicts the fact that the proportion of time the Markov chain spends in state $j$ (due to the ergodic theorem) should converge to the stationary probability of state $j$. Hence, $\pi$ and $\pi'$ cannot be distinct, so the stationary distribution is unique.

> TODO: what is the ergodic theorem?

---