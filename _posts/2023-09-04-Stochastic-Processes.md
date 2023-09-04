---
title: A Note on Stochastic Processes
date: 2023-09-04 02:40:00 +0800
categories: [Mathematics]
tags: [stochastic processes]
math: True
---

> This note partially uses the materials from [the notes of MATH2750](https://mpaldridge.github.io/math2750/S10-stationary-distributions.html).
{: .prompt-info }

## Class Structure
There may be some "irrelevant" states in a Markov chain. We can eliminate them to simplify the model.

### Accessible
These statements are equivalent:
- State $j$ is accessible from state $i;$
- $i$ can transition to $j$ in $n$ steps;
- Starting from $i$, there’s a positive chance that we’ll get to $j$ at some point in the future;
- $\exist n, \mathrm{s.t. }\mathbf{M}_{i,j}(n) > 0;$
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
A Markov chain is irreducible if its entire state space $S$ is on communicating class.

### Absorbing state
These statements are equivalent:
- State $i$ is in a communicating class $\{i\}$ by itself and that class is closed.
- Once you reach an absorbing state, you can’t leave that state.

### Periodicity

#### Definition

Periodicity describes a pattern of recurrence for a state. For a state $i$ in a Markov chain, its period $d(i)$ is defined as the greatest common divisor of the number of steps it can take to return to state $i$, excluding the possibility of 1-step self-transitions.

More formally, consider all $n$ such that $\mathbf{M}^n_{ii} > 0$ (i.e., the probability of transitioning from state $i$ to state $i$ in $n$ steps is positive). The period $d(i)$ is the greatest common divisor of all such $n$.

1. If $d(i) = 1$, then state $i$ is said to be aperiodic.
2. If $d(i) > 1$, then state $i$ is said to be periodic, with a period of $d(i)$.

#### Examples

All states in the Markov chain with the following transition matrix has are periodic with a period of 3.

$$
\mathbf{M} = \begin{bmatrix}
0 & 1 & 0 \\
0 & 0 & 1 \\
1 & 0 & 0 \\
\end{bmatrix}
$$


All states in the Markov chain with the following transition matrix has are aperiodic.

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

---

## Recurrence and Transience

---

## Stationary Distributions

---
## Draft
> **Theorem 10.1** Consider an **irreducible** Markov chain.
> 
> - If the Markov chain is **positive recurrent**, then a stationary distribution $\mathrm{v}$ exists, is unique, and is given by $v_s = 1/m_s$, where $m_s$ is the **expected return time** to state  
$s$. 
> - If the Markov chain is null recurrent or **transient**, then no stationar

#### My stack-style questions
- What is the definition of stationary distribution?
- Concepts
  - Recurrent? Transient?
    - Return time?
    - Positive recurrent?
    - Null recurrent?
    - Communicating class?
  - Irreducible?
- Why is $\mathrm{M}$ positive recurrent?
- How to prove the Existence and Uniqueness of the stationary distribution?

#### Stationary distribution
- The distribution is defined on the state set, representing the probability of each state occurring at a timestep.
- E.g., a state set is $\{s_1, s_2\}$, then a distribution can be $(0.7, 0.3)$, meaning that $s_1, s_2$ will appear with the probability of $(0.7, 0.3)$ respectively.
- If the current distribution is $\mathbf{v}$, then the distribution at the next timestep is $\mathbf{v} \mathbf{M}$.
- If a distribution $\mathbf{v} = \mathbf{v} \mathbf{M}$, then it is a stationary distribution.
- A stationary distribution is a fixed point.

<!-- #### Return probability
> Found in the [Definition 9.1](https://mpaldridge.github.io/math2750/S09-recurrence-transience.html)

Let $\left(X_n\right)$ be a Markov chain on a state space $S$. For $s\in S$, let $m_s$ be the **return probability**

$$ 
m_s = \mathrm{Pr}\left(X_n = s \text{ for some } n\ge 1 \mid X_0 = s\right). 
$$

Personally, I think this definition is vague. -->


#### Return time
> Generated by ChatGPT 4.

- **First Return Time**

   Given a state $s \in S$ of a Markov chain, the first return time $T_s$ to state $s$ is defined as:

   $$ T_s = \min \{ n > 0 : X_n = s | X_0 = s \} $$

   This is the number of time steps required to return to state $s$ for the first time, given that we start at $s$. Essentially, $T_s$ represents the first occurrence of the state $s$ after the initial time, assuming the chain started at state $s$.

- **Expected Return Time**

   The expected return time, $m_s$, to state $s$ is the average number of time steps it takes to return to state $s$ after initially starting at $s$. Mathematically, it's given by the expected value of the first return time $T_s$:
   $$ m_s = \mathbb{E}[T_s | X_0 = s] $$
   If the Markov chain is guaranteed to return to state $s$ and does so in an average of a finite number of steps, then $m_s$ is finite. If the Markov chain can return to state $s$ but takes, on average, an infinite number of steps, then $m_s$ is infinite. If the chain is not guaranteed to return to state $s$, then the expected return time is undefined (though it is often treated as infinite for certain analyses).


#### Recurrence & transience

- **Recurrent State**
    
    A state $s \in S$ is recurrent if and only if the probability of eventually returning to $s$ starting from $s$ is 1, that is:

    $$ P(T_s < \infty) = 1 $$

    Furthermore, recurrent states can be broken down into:

    - **Positive Recurrent:** If there exists a constant $m_s$ such that $\mathbb{E}[T_s | X_0 = s] = m_s < \infty$. This means that the expected time to return to state $s$ is finite.

    - **Null Recurrent:** If $\mathbb{E}[T_s | X_0 = s] = \infty$. This means that the expected time to return to state $s$ is infinite.

- **Transient State**

    A state $s \in S$ is transient if and only if the probability of eventually returning to $s$ starting from $s$ is less than 1, that is:
    $$ P(T_s < \infty) < 1 $$

- **Every state is either recurrent or transient.**
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

The transition kernel is a square matrix. The exponent $k$ of the matrix power $\mathrm{M}^k$ represents the state distribution after $k$ consecutive state transitions from the current state distribution.