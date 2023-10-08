---
title: Theoretical Computer Science (TCS)
date: 2023-06-17 20:00:01 +0800
categories: [Mathematics]
tags: [theoretical computer science, computational complexity theory, reduction, complexity classes]
math: True
---

> This note will be consistently updated.
{: .prompt-info }

## What is TCS?
*([Wikipedia](https://en.wikipedia.org/wiki/Theoretical_computer_science))*   
Theoretical computer science (TCS) is a subset of general computer science and mathematics that focuses on **mathematical aspects of computer science** such as the theory of computation, lambda calculus, and type theory.

Topics that I might come across:
1. Algorithms
2. **Computational complexity theory**
3. Computational learning theory
4. Information-based complexity
5. Information theory
6. Machine learning

> currently I mainly focus on the computational complexity theory.
{: .prompt-info }


## What is Computational Complexity Theory?
*([Wikipedia](https://en.wikipedia.org/wiki/Computational_complexity_theory))*  
1. Computational complexity theory is a branch of the theory of computation that focuses on **classifying comiputational problems according to their inherent diffculty, and relating those classes to each other**. 
2. **A computational problem** is understood to be a task that is in principle amenable to **being solved by a computer**, which is equivalent to stating that the problem may be solved by mechanical application of mathematical steps, such as an **algorithm**.
3. **A problem is regarded as inherently difficult if its solution requires significant resources, whatever the algorithm used.** The theory formalizes this intuition, by introducing mathematical models of computation to study these problems and quantifying the amount of resources needed to solve them, such as **time** and **storage**. Other complexity measures are also used, such as the amount of communication (used in communication complexity), the number of gates in a circuit (used in circuit complexity) and the number of processors (used in parallel computing). 
4. One of the roles of computational complexity theory is to **determine the practical limits on what computers can and cannot do.**

## Computational Problems
### Problem instances

### Representing problem instances

### Complexity classes

## Misc Important Concepts
### Reduction
*([Wikipedia](https://en.wikipedia.org/wiki/Reduction_(complexity)))*  
In computability theory and computational complexity theory, **a reduction is an algorithm for transforming one problem into another problem.** A sufficiently efficient reduction from one problem to another may be used to show that the second problem is **at least as difficult as the first.**

### Intractability


### Big $O$ notation
> This part partially uses material from [this website](https://www.freecodecamp.org/news/big-o-notation-why-it-matters-and-why-it-doesnt-1674cfa8a23c/) and [Wikipedia](https://en.wikipedia.org/wiki/Big_O_notation).
{: .prompt-info }

> Big-oh is about finding an asymptotic upper bound.

$f(x) = O\left(g(x)\right)$, iff (if and only if) $\exists 0<k, 0<x_0$, s.t.$f(x)\le k\cdot g(x), \forall x_0\le x$.

After $x_0$, there is a $k\cdot g(x)$ which is the upper bound of $f(x)$. (So in my understanding, the complexity means the upper bound.)

It is read "$f(x)$ is big O of $g(x)$".

Comparison:

1. $f(x) = O\left(g(x)\right)$, iff $\exists 0<k, 0<x_0$, s.t.$f(x)\le k\cdot g(x), \forall x_0\le x$.
   1. The upper bound of $f(x)$ after $x_0$.
   2. $\lim\limits_{x\to\infty} \frac{f(x)}{g(x)} < \infty$.
2. $f(x) = \Omega\left(g(x)\right)$, iff $\exists 0<k, 0<x_0$, s.t.$f(x)\ge k\cdot g(x), \forall x_0\le x$.
   1. The lower bound of $f(x)$ after $x_0$.
   2. $\lim\limits_{x\to\infty} \frac{f(x)}{g(x)} > 0$.
3. $f(x) = \Theta\left(g(x)\right)$, iff $f(x) = O\left(g(x)\right)$ and $f(x) = \Omega\left(g(x)\right)$. 
   1. The exact bound of $f(x)$.
   2. $\lim\limits_{x\to\infty} \frac{f(x)}{g(x)} \in \mathbb{R}_{>0}$.
4. $f(x) = o\left(g(x)\right)$, iff $f(x) = O\left(g(x)\right)$ and $f(x)$ is not $\Theta\left(g(x)\right)$. 
   1. The upper bound of $f(x)$ excluding the exact bound.
   2. $\lim\limits_{x\to\infty} \frac{f(x)}{g(x)} = 0$.

Big O can also be used to describe the error term in an  approximation to a mathematical function.

$$
\begin{aligned}
e^x =& 1+x+\frac{x^2}{2!}+\frac{x^3}{3!}+\frac{x^4}{4!}+\dots & \forall x \\
=& 1+x+\frac{x^2}{2!}+O(x^3) & x\to 0\\
=& 1+x+O(x^2) & x\to 0  \\
\end{aligned}
$$

Compared with $x^2$, $x^3$ is closer to $0$, when $x\to 0$.


> Disclaimer: The above content is summarized from Wikipedia and other sources. Corresponding links or references have been provided.
{: .prompt-danger }