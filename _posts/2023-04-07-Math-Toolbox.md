---
title: Math Toolbox
date: 2023-04-07 02:40:00 +0800
categories: [Mathematics]
tags: [English, toolbox]
math: True
---

> This note will be consistently updated.
{: .prompt-info }

---

## Logic

## Inequality

## Calculus

## Probability

### Expectation

$$
\mathbb{E}[X] = \sum\limits_{i}p_i\cdot x_i
$$

$$
\mathbb{E}[X] = \int_{-\infty}^{\infty} x\cdot f(x) \, dx
$$

Linearity:

$$
\mathbb{E}[X+Y] = \mathbb{E}[X]+\mathbb{E}[Y],
$$

$$
\mathbb{E}[aX] = a\mathbb{E}[X].
$$

If $X$ and $Y$ are independent:

$$
\mathbb{E}[XY] = \mathbb{E}[X]\cdot \mathbb{E}[Y].
$$

If $X$ and $Y$ are dependent:

$$
\mathbb{E}[XY] \ne \mathbb{E}[X]\cdot \mathbb{E}[Y].
$$

If $X = c$, where $c\in \mathbb{R}$, then $\mathbb{E}[X] = c$. Thus

$$
\mathbb{E}[\mathbb{E}[X]] = \mathbb{E}[X].
$$

### Variance

$$
\begin{aligned}
   \mathbb{V}\left[X \right] =& \mathbb{E}\left[ (X - \mathbb{E}[X])^2 \right] \\
   =& \mathbb{E}\left[ X^2 - 2\,X\cdot \mathbb{E}[X] +\mathbb{E}[X]^2 \right]\\
   =& \mathbb{E}\left[ X^2 - 2\,\mathbb{E}[X]\cdot \mathbb{E}[X] +\mathbb{E}[X]^2 \right]\\
   =& \mathbb{E}\left[X^2\right] - \mathbb{E}\left[X\right]^2
\end{aligned}
$$

$$
\mathbb{V}\left[X \right] = \sum\limits_{i} p_i\cdot (x_i - \mathbb{E}[X])^2
$$

$$
\mathbb{V}\left[X \right] = \int_{\mathbb{R}} f(x) \cdot (x - \mathbb{E}[X])^2 \, dx
$$

$$
\mathbb{V}\left[X \right] = \int_{\mathbb{R}} x^2\cdot f(x)\, dx - \mathbb{E}\left[X\right]^2
$$

$$
\mathbb{V}\left[X \right] = \text{Cov}(X,X)
$$

$$
\mathbb{V}[X] \ge 0
$$

$$
\mathbb{V}[X+a] = \mathbb{V}[X]
$$

$$
\mathbb{V}[aX] = a^2\mathbb{V}[X]
$$

$$
\mathbb{V}[aX+bY] = a^2\, \mathbb{V}[X] + b^2\, \mathbb{V}[Y] + 2ab\, \text{Cov}[X]
$$

If there is a set of random variables $\set{X_1,\ldots X_N}$,then

$$
\begin{aligned}
   \mathbb{V}\left[\sum\limits_i X_i\right] =& \sum\limits_{i,j} \text{Cov}(X_i, X_j) \\
   =& \sum\limits_{i}\mathbb{V} [X_i] +\sum\limits_{i\ne j}\text{Cov} (X_i, X_j).
\end{aligned}
$$

### Covariance

$$
\text{Cov}(X,Y) = \mathbb{E}\left[ X - \mathbb{E}[X] \right] \cdot \mathbb{E}\left[ Y - \mathbb{E}[Y] \right]
$$

### Moment

## Algebra

