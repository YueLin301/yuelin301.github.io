---
title: Contraction Mapping Theorem
date: 2023-10-20 02:40:00 +0800
categories: [Mathematics]
tags: [tech, math, contraction]
math: True
---

## Metric Space

### Definition of metric space

> [**Definition.**](https://en.wikipedia.org/wiki/Metric_space) 
> A metric space is an ordered pair $(M, d)$ where $M$ is a set and $d$ is a metric on $M$, i.e., a function $d: M\times M \to \mathbb{R}$ satisfying the following axioms for all points $x, y, z \in M:$
> 1. The distance from a point to itself is zero: $d(x,x) = 0.$
> 2. (Positivity) The distance between two distinct points is always positive: If $x\ne y,$ then $d(x,y)>0.$
> 3. (Symmetry) The distance from x to y is always the same as the distance from y to x: $d(x,y) = d(y,x)$
> 4. The triangle inequality holds: $d(x,z)\le d(x,y)+d(y,z).$
{: .prompt-info }


Note that $0 = d(x,x)\le d(x,y)+d(y,x),$ so the second axiom can be weakened to "If $x\ne y,$ then $d(x,y)\ne 0$" and combined with the first axiom to "$d(x,y) = 0 \Leftrightarrow x = y.$"

### Cauchy sequence

> [**Definition.**](https://en.wikipedia.org/wiki/Cauchy_sequence#In_a_metric_space)
> Let $\{x_t\}_{t}^\infty$ be a sequence in a metric space $(M,d),$ then it is a Cauchy sequence, if for every positive real number $\epsilon>0,$ there is a positive integer $N$ such that for all positive integers $m,n > N,$ the distance $d(x_m, x_n) < \epsilon.$
{: .prompt-info }

Symbolically, this is:

$$
\forall \epsilon > 0 (\exists N\in\mathbb{N} (\forall m,n\in \mathbb{N}(m,n\ge N \Rightarrow d(x_m,x_n)< \epsilon))).
$$

### Convergent sequence

> **Definition.**
> Let $\{x_t\}_{t}^\infty$ be a sequence in a metric space $(M,d),$ then it is a convergent sequence, if there is a $x\in M$ such that for every positive real number $\epsilon>0,$ there is a positive integer $N$ such that for all positive integer $n\ge N,$ the distance $d(x_n, x) < \epsilon.$
{: .prompt-info }

> [**Definition.**](https://en.wikipedia.org/wiki/Limit_of_a_sequence#Metric_spaces)
> A point $x$ of the metric space $(M, d)$ is the limit of the sequence $(x_n)$ if: For each $0<\epsilon\in\mathbb{R},$ there is $N\in\mathbb{N}$ such that, for every $N\le n\in \mathbb{N},$ we have $d(x_n, x)<\epsilon.$
{: .prompt-info }

Symbolically, this is:

$$
\forall \epsilon > 0 (\exists N\in\mathbb{N} (\forall n\in \mathbb{N}(n\ge N \Rightarrow d(x_n,x)< \epsilon))).
$$

### Completeness

> [**Definition.**](https://en.wikipedia.org/wiki/Cauchy_sequence#Completeness)
> A metric space $(M,d)$ is called complete, if every Cauchy sequence in it converges to an element of $M.$
{: .prompt-info }

Examples:
1. $\mathbb{R}$ with usual distance is complete;
2. $\mathbb{Q}$ (the set of rational numbers) with usual distance is not complete. Because a sequence of rational numbers can converge to an irrational number, e.g. $\pi.$

## Contraction Mapping

> [**Definition.**](https://en.wikipedia.org/wiki/Contraction_mapping)
> A contraction mapping on a metric space $(M,d)$ is a function $f:M\to M$ with the property that there is some real number $0\le k< 1$ such that for all $x$ and $y$ in $M,$ 
> 
> $$d(f(x),f(y))\le k\cdot d(x,y).$$
{: .prompt-info }

## Fixed Point

> **Definition.**
> Let $(M,d)$ be a metric space and $f:M\to M$ be a function. If $x\in M$ and $x=f(x),$ then $x$ is called a fixed point of $f.$
{: .prompt-info }

## Contraction Mapping → Cauchy Sequence

### Lemma

> **Lemma.**
> Let $(M,d)$ be a metric space, $f:M\to M$ be a function, $x_0\in M$, and $x_{t+1} = f(x_t), t=0,1,\ldots,$ then $\{x_t\}_{t}^\infty$ is a Cauchy sequence. 
{: .prompt-info }

### Proof

Since $f$ is a contraction mapping, there must exist a $\beta \in (0,1)$ such that 

$$
d(x_2, x_1) = d(f(x_1), f(x_0)) \leq \beta d(x_1, x_0)
$$ 

and

$$
d(x_3, x_2) \leq \beta d(x_2, x_1) \leq \beta^2 d(x_1, x_0), \cdots.
$$

Generally, we have

$$
d(x_{t+1}, x_t) \leq \beta d(x_t, x_{t-1}) \leq \cdots \leq \beta^t d(x_1, x_0).
$$

Let $N$ be a positive integer. Let positive integers $m, n$ satisfy $m > n \geq N$. Then by the triangle inequality, we have

$$
\begin{aligned}
d(x_m, x_n) & \leq d(x_m, x_{m-1}) + d(x_{m-1}, x_{m-2}) + \cdots + d(x_{n+2}, x_{n+1}) + d(x_{n+1}, x_n) \\
& \leq \left[\beta^{m-1} + \beta^{m-2} + \cdots + \beta^{n+1} + \beta^n\right] d(x_1, x_0) \\
& = \beta^n \left[\beta^{m-1-n} + \beta^{m-2-n} + \cdots + \beta^1 + 1\right] d(x_1, x_0) \\
& = \beta^n \frac{1-\beta^{m-n}}{1-\beta} d(x_1, x_0) \\
& = \frac{\beta^n - \beta^m}{1-\beta} d(x_1, x_0) \\
& \leq \frac{\beta^N}{1-\beta} d(x_1, x_0) .
\end{aligned}
$$

Let $\epsilon > 0$. If $d(x_1, x_0) > 0$ (the case of $d(x_1, x_0) = 0$ will be discussed later), then for a positive integer $N$ satisfying 

$$
N > \frac{\ln \frac{(1-\beta) \epsilon}{d(x_1, x_0)}}{\ln \beta},
$$

we have for any $m, n \geq N$, that $d(x_m, x_n) \leq \frac{\beta^N}{1-\beta} d(x_1, x_0) < \epsilon$, hence $\{x_t\}\_{t=0}^{\infty}$ is a Cauchy sequence. 

If $d(x_1, x_0) = 0$, then $\{x_t\}\_{t=0}^{\infty}$ is a constant sequence, and therefore also a Cauchy sequence. $\blacksquare$

## Contraction Mapping Theorem

### Theorem
> **Theorem.** Let $(M, d)$ be a complete metric space, and $f: M \to M$ be a contraction mapping. Then $f$ has a unique fixed point $x^\* \in M$. Furthermore, starting from any point $x_0 \in M$, the sequence $\left(x_0, f(x_0), f(f(x_0)), \cdots\right)$ converges to $x^\*$.
{: .prompt-info }

### Proof

**Uniqueness**: First, let's prove that if a point is a fixed point of $f$, then it is the only fixed point of $f$. Suppose $x_1, x_2 \in M$ and $x_1=f(x_1)$, $x_2=f(x_2)$. Since $f$ is a contraction mapping, there exists a $\beta \in (0, 1)$ such that 

$$
d(x_1, x_2) = d(f(x_1), f(x_2)) \leq \beta d(x_1, x_2).
$$

The above inequality holds only when $d(x_1, x_2) = 0$, i.e., $x_1 = x_2$. 

**Convergence**: Since $M$ is complete, there exists a point $x^\* \in M$ such that the sequence $\{x_t\}$ converges to $x^\*$.

**Fixed Point**: Let $\epsilon > 0$. Since $\{x_t\}$ is a Cauchy sequence and converges to $x^\*$, there exists a positive integer $T$ such that for any $t \geq T$, we have $d(x^\*, x_t) < \frac{\epsilon}{3}$ and $d(x_{t+1}, x_t) < \frac{\epsilon}{3}$. For $t \geq T$, by the triangle inequality, we have

$$
d(x^*, f(x^*)) \leq d(x^*, x_t) + d(x_t, f(x_t)) + d(f(x_t), f(x^*)).
$$

Note that the middle term on the right-hand side is equal to $d(x_t, x_{t+1})$. Hence, the first two terms on the right-hand side are both less than $\frac{\epsilon}{3}$ by our earlier discussion. Now, consider the third term. We have 

$$
d(f(x_t), f(x^*)) \leq \beta d(x_t, x^*) < \frac{\epsilon}{3}.
$$ 

Thus, $d(x^\*, f(x^\*)) < \epsilon$. Since $\epsilon$ was arbitrary, we have $d(x^\*, f(x^\*)) = 0$, i.e., $x^\* = f(x^\*)$. $\blacksquare$