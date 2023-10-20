---
title: Writing Buffer
date: 2023-04-01 02:40:00 +0800
categories: [Misc Notes]
tags: [misc note, writing buffer]
---

> This post is used for storing temporary content, which will be classified later. I don't want my thought process to be interrupted by format-related issues while composing.
{: .prompt-info }


## Eigenvalue





## Sets

The supremum of a nonempty set $X \subset \mathbb{R}$ is the smallest scalar $y$ such that
$$
y \geq x \text { for all } x \in X
$$
The infimum of a set $X \subset \mathbb{R}$ is the largest scalar $y$ such that
$$
y \leq x \text { for all } x \in X
$$
In the case $\sup X \in X(\inf X \in X)$, we write $\sup X=\max X(\inf X=\min X)$. We now briefly consider an example to illustrate the definition of sup and inf. The supremum and infimum of the set $\left\{\frac{1}{n}: n \geq 1\right\}$ are given by
$$
\sup \{1 / n: n \geq 1\}=\max \{1 / n: n \geq 1\}=1, \quad \inf \{1 / n: n \geq 1\}=0 .
$$
For $\epsilon>0$ and $x \in \mathbb{R}^n$ we define $B_\epsilon(x)=\left\{y \in \mathbb{R}^n:\|x-y\|<\epsilon\right\}$ to be open ball with radius $\epsilon$ and center $x$. Next, we collect further properties and terminologies for sets:
- A set $X \subset \mathbb{R}^n$ is called open if for every $x \in X$ there exists $\epsilon>0$ such that $B_\epsilon(x) \subset X$.
- A set $X \subset \mathbb{R}^n$ is closed if $\mathbb{R}^n \backslash X$ is open. Alternatively, we can define closedness of set as follows: For every sequence $\left(x^k\right)$ with $x^k \in X$ for all $k$ and $x^k \rightarrow x$, we have $x \in X$.
- A set $X \subset \mathbb{R}^n$ is bounded if there exists $B \in \mathbb{R}$ with $\|x\| \leq B$ for all $x \in X$.
- A bounded and closed set is called compact.


Consider subsets of $\mathbb{R}^2$:
- The set $\{(𝑥,𝑦)\mid 𝑥^2+𝑦^2<1\}$ is bounded but not closed. 
- The set $\{(𝑥,𝑦)\mid 𝑥\ge0\}$ is closed but not bounded. 