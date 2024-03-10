---
title: Convergence Analysis of Gradient Descent
date: 2023-10-22 02:40:00 +0800
categories: [Mathematics]
tags: [Tech, Math, Convergence, Theory]
math: True
---

> The following part has not been finished yet.
{: .prompt-warning }

## Gradient Descent

### The goal

We want to solve this unconstrained minimization problem

$$
\min _x f(x) \quad \text { s.t. } \quad x \in \mathbb{R}^n .
$$

<!-- ### An equivalent solution

Assume that the objective function is continuously differentiable on $\mathbb{R}^n$. The first order necessary optimality condition:
$$
x^* \text { is a local minimizer or maximizer } \Longrightarrow \nabla f\left(x^*\right)=0. 
$$

Theorem 3.1: Necessary First-Order Optimality Conditions
Let $f: X \rightarrow \mathbb{R}$ be differentiable $\left(X=\mathbb{R}^n\right.$ or $X \subset \mathbb{R}^n$ open $)$ and let $x^* \in X$ be a local minimum of $f$. Then, it follows $\nabla f\left(x^*\right)=0$.

Proof. Let $h \in \mathbb{R}^n$ be arbitrary. Since $x^*$ is a local minimizer, we have $f\left(x^*+t h\right) \geq f\left(x^*\right)$ for all $t>0$ sufficiently small. Rearranging the terms, dividing by $t$ and taking the limit, we thus obtain
$$
0 \leq \lim _{t \rightarrow 0} \frac{f\left(x^*+t h\right)-f\left(x^*\right)}{t}=\nabla f\left(x^*\right)^{\top} h .
$$

Choosing $h=-\nabla f\left(x^*\right)$ finishes the proof.

We want to 

In principle, the optimal solution of the problem can be obtained by calculating all stationary points of $f$ and by comparing the objective function values. Unfortunately, such a procedure has several drawbacks:
- It might not be possible or too difficult to solve the equation $\nabla f(x)=0$ analytically.
- There might be infinitely many stationary points and finding the lowest function value is another (maybe challenging) optimization problem.

We will consider iterative algorithms for finding stationary points. In this section, we will consider algorithms of the form:
$$
x^{k+1}=x^k+\alpha_k d^k, \quad k=0,1,2, \ldots
$$
where $d^k$ is the so-called direction and $\alpha_k$ is the step size. We will mainly work with descent directions:

Definition 5.1: Descent Directions
Let $f: \mathbb{R}^n \rightarrow \mathbb{R}$ be continuously differentiable. A vector $d \in \mathbb{R}^n \backslash\{0\}$ is called descent direction of $f$ at $x$ if
$$
\nabla f(x)^{\top} d<0 .
$$

The most important property of descent directions is that taking a small enough step along these directions leads to a decrease of the objective function.

Lemma 5.2: Descent Property
Let $f: \mathbb{R}^n \rightarrow \mathbb{R}$ be a continuously differentiable function and suppose that $d$ is a descent direction of $f$ at $x$. Then, there is $\varepsilon>0$ such that
$$
f(x+\alpha d)<f(x) \quad \forall \alpha \in(0, \varepsilon] .
$$

Proof. Using the directional derivative condition, we have
$$
\lim _{\alpha \rightarrow 0} \frac{f(x+\alpha d)-f(x)}{\alpha}=\nabla f(x)^{\top} d<0
$$

The policy descent is updating $x^{(k)}$ by 

$$
x^{(k+1)} \gets x^{(k)} + \alpha^{(k)} \cdot d^{(k)},
$$

where \alpha^{(k)}is

## Basic Questions

### What is convergence?

> **Definition.**
> Let $\set{x_t}_{t}^\infty$ be a sequence in a metric space $(M,d),$ then it is a convergent sequence, if there is a $x\in M$ such that for every positive real number $\epsilon>0,$ there is a positive integer $N$ such that for all positive integer $n\ge N,$ the distance $d(x_n, x) < \epsilon.$
{: .prompt-info }

> [**Definition.**](https://en.wikipedia.org/wiki/Limit_of_a_sequence#Metric_spaces)
> A point $x$ of the metric space $(M, d)$ is the limit of the sequence $(x_n)$ if: For each $0<\epsilon\in\mathbb{R},$ there is $N\in\mathbb{N}$ such that, for every $N\le n\in \mathbb{N},$ we have $d(x_n, x)<\epsilon.$
{: .prompt-info }

Symbolically, this is:

$$
\forall \epsilon > 0 (\exists N\in\mathbb{N} (\forall n\in \mathbb{N}(n\ge N \Rightarrow d(x_n,x)< \epsilon))).
$$

### Whose convergence do we want to analyze?

We want to prove that $f(x^{(k)})$ will converges to $f(x^{*})$

### Under what conditions can we conclude convergence?

## Convergence of Gradient Descent with Fixed Step Size

Theorem 6.1 Suppose the function $f: \mathbb{R}^{n} \rightarrow \mathbb{R}$ is convex and differentiable, and that its gradient is Lipschitz continuous with constant $L>0$, i.e. we have that $\|\nabla f(x)-\nabla f(y)\|_{2} \leq L\|x-y\|_{2}$ for any $x, y$. Then if we run gradient descent for $k$ iterations with a fixed step size $t \leq 1 / L$, it will yield a solution $f^{(k)}$ which satisfies

$$
f\left(x^{(k)}\right)-f\left(x^{*}\right) \leq \frac{\left\|x^{(0)}-x^{*}\right\|_{2}^{2}}{2 t k}
$$

where $f\left(x^{*}\right)$ is the optimal value. Intuitively, this means that gradient descent is guaranteed to converge and that it converges with rate $O(1 / k)$.

Proof: Our assumption that $\nabla f$ is Lipschitz continuous with constant $L$ implies that $\nabla^{2} f(x) \preceq L I$, or equivalently that $\nabla^{2} f(x)-L I$ is a negative semidefinite matrix. Using this fact, we can perform a quadratic expansion of $f$ around $f(x)$ and obtain the following inequality:

$$
\begin{aligned}
f(y) & \leq f(x)+\nabla f(x)^{T}(y-x)+\frac{1}{2} \nabla^{2} f(x)\|y-x\|_{2}^{2} \\
& \leq f(x)+\nabla f(x)^{T}(y-x)+\frac{1}{2} L\|y-x\|_{2}^{2}
\end{aligned}
$$

Now let's plug in the gradient descent update by letting $y=x^{+}=x-t \nabla f(x)$. We then get:

$$
\begin{aligned}
f\left(x^{+}\right) & \leq f(x)+\nabla f(x)^{T}\left(x^{+}-x\right)+\frac{1}{2} L\left\|x^{+}-x\right\|_{2}^{2} \\
& =f(x)+\nabla f(x)^{T}(x-t \nabla f(x)-x)+\frac{1}{2} L\|x-t \nabla f(x)-x\|_{2}^{2} \\
& =f(x)-\nabla f(x)^{T} t \nabla f(x)+\frac{1}{2} L\|t \nabla f(x)\|_{2}^{2} \\
& =f(x)-t\|\nabla f(x)\|_{2}^{2}+\frac{1}{2} L t^{2}\|\nabla f(x)\|_{2}^{2} \\
& =f(x)-\left(1-\frac{1}{2} L t\right) t\|\nabla f(x)\|_{2}^{2}
\end{aligned}
$$

Using $t \leq 1 / L$, we know that $-\left(1-\frac{1}{2} L t\right)=\frac{1}{2} L t-1 \leq \frac{1}{2} L(1 / L)-1=\frac{1}{2}-1=-\frac{1}{2}$. Plugging this in to (??), we can conclude the following:

$$
f\left(x^{+}\right) \leq f(x)-\frac{1}{2} t\|\nabla f(x)\|_{2}^{2}
$$

Since $\frac{1}{2} t\|\nabla f(x)\|_{2}^{2}$ will always be positive unless $\nabla f(x)=0$, this inequality implies that the objective function value strictly decreases with each iteration of gradient descent until it reaches the optimal value $f(x)=f\left(x^{*}\right)$. Note that this convergence result only holds when we choose $t$ to be small enough, i.e. $t \leq 1 / L$. This explains why we observe in practice that gradient descent diverges when the step size is too large.

Next, we can bound $f\left(x^{+}\right)$, the objective value at the next iteration, in terms of $f\left(x^{*}\right)$, the optimal objective value. Since $f$ is convex, we can write

$$
\begin{aligned}
& f\left(x^{*}\right) \geq f(x)+\nabla f(x)^{T}\left(x^{*}-x\right) \\
& f(x) \leq f\left(x^{*}\right)+\nabla f(x)^{T}\left(x-x^{*}\right)
\end{aligned}
$$

where the first inequality yields the second through simple rearrangement of terms. Plugging this in to (??), we obtain:

$$
\begin{aligned}
& f\left(x^{+}\right) \leq f\left(x^{*}\right)+\nabla f(x)^{T}\left(x-x^{*}\right)-\frac{t}{2}\|\nabla f(x)\|_{2}^{2} \\
& f\left(x^{+}\right)-f\left(x^{*}\right) \leq \frac{1}{2 t}\left(2 t \nabla f(x)^{T}\left(x-x^{*}\right)-t^{2}\|\nabla f(x)\|_{2}^{2}\right) \\
& f\left(x^{+}\right)-f\left(x^{*}\right) \leq \frac{1}{2 t}\left(2 t \nabla f(x)^{T}\left(x-x^{*}\right)-t^{2}\|\nabla f(x)\|_{2}^{2}-\left\|x-x^{*}\right\|_{2}^{2}+\left\|x-x^{*}\right\|_{2}^{2}\right) \\
& f\left(x^{+}\right)-f\left(x^{*}\right) \leq \frac{1}{2 t}\left(\left\|x-x^{*}\right\|_{2}^{2}-\left\|x-t \nabla f(x)-x^{*}\right\|_{2}^{2}\right)
\end{aligned}
$$

where the final inequality is obtained by observing that expanding the square of $\left\|x-t \nabla f(x)-x^{*}\right\|_{2}^{2}$ yields $\left\|x-x^{*}\right\|_{2}^{2}-2 t \nabla f(x)^{T}\left(x-x^{*}\right)+t^{2}\|\nabla f(x)\|_{2}^{2}$. Notice that by definition we have $x^{+}=x-t \nabla f(x)$. Plugging this in to (??) yields:

$$
f\left(x^{+}\right)-f\left(x^{*}\right) \leq \frac{1}{2 t}\left(\left\|x-x^{*}\right\|_{2}^{2}-\left\|x^{+}-x^{*}\right\|_{2}^{2}\right)
$$

This inequality holds for $x^{+}$on every iteration of gradient descent. Summing over iterations, we get:

$$
\begin{aligned}
\sum_{i=1}^{k} f\left(x^{(i)}-f\left(x^{*}\right)\right. & \leq \sum_{i=1}^{k} \frac{1}{2 t}\left(\left\|x^{(i-1)}-x^{*}\right\|_{2}^{2}-\left\|x^{(i)}-x^{*}\right\|_{2}^{2}\right) \\
& =\frac{1}{2 t}\left(\left\|x^{(0)}-x^{*}\right\|_{2}^{2}-\left\|x^{(k)}-x^{*}\right\|_{2}^{2}\right) \\
& \leq \frac{1}{2 t}\left(\left\|x^{(0)}-x^{*}\right\|_{2}^{2}\right)
\end{aligned}
$$

where the summation on the right-hand side disappears because it is a telescoping sum. Finally, using the fact that $f$ decreasing on every iteration, we can conclude that

$$
\begin{aligned}
f\left(x^{(k)}\right)-f\left(x^{*}\right) & \leq \frac{1}{k} \sum_{i=1}^{k} f\left(x^{(i)}\right)-f\left(x^{*}\right) \\
& \leq \frac{\left\|x^{(0)}-x^{*}\right\|_{2}^{2}}{2 t k}
\end{aligned}
$$

where in the final step, we plug in (??) to get the inequality from (??) that we were trying to prove.

#### 0.1.2. Convergence of gradient descent with adaptive step size

We will not prove the analogous result for gradient descent with backtracking to adaptively select the step size. Instead, we just present the result with a few comments.

Theorem 6.2 Suppose the function $f: \mathbb{R}^{n} \rightarrow \mathbb{R}$ is convex and differentiable, and that its gradient is Lipschitz continuous with constant $L>0$, i.e. we have that $\|\nabla f(x)-\nabla f(y)\|_{2} \leq L\|x-y\|_{2}$ for any $x, y$. Then if we run gradient descent for $k$ iterations with step size $t_{i}$ chosen using backtracking line search on each iteration $i$, it will yield a solution $f^{(k)}$ which satisfies

$$
f\left(x^{(k)}\right)-f\left(x^{*}\right) \leq \frac{\left\|x^{(0)}-x^{*}\right\|_{2}^{2}}{2 t_{\min } k}
$$

where $t_{\min }=\min \{1, \beta / L\}$

Notice that the only difference between Theorems ?? and ?? is that the fixed step size $t$ is replaced by $t_{\min }$. Notice that if we choose $\beta$ to be large enough, the rate of convergence is very similar to what we got for gradient descent with fixed step size.

#### 0.1.3. Convergence rates for gradient descent

Convex $f$. From Theorem ??, we know that the convergence rate of gradient descent with convex $f$ is $O(1 / k)$, where $k$ is the number of iterations. This implies that in order to achieve a bound of $f\left(x^{(k)}\right)-f\left(x^{*}\right) \leq$ $\epsilon$, we must run $O(1 / \epsilon)$ iterations of gradient descent. This rate is referred to as "sub-linear convergence."

Strongly convex $f$. In contrast, if we assume that $f$ is strongly convex, we can show that gradient descent converges with rate $O\left(c^{k}\right)$ for $0<c<1$. This means that a bound of $f\left(x^{(k)}\right)-f\left(x^{*}\right) \leq \epsilon$ can be achieved using only $O(\log (1 / \epsilon))$ iterations. This rate is typically called "linear convergence."

#### 0.1.4. Pros and cons of gradient descent

The principal advantages and disadvantages of gradient descent are:

- Simple algorithm that is easy to implement and each iteration is cheap; just need to compute a gradient
- Can be very fast for smooth objective functions, i.e. well-conditioned and strongly convex
- However, it's often slow because many interesting problems are not strongly convex
- Cannot handle non-differentiable functions (biggest downside)


### 0.2. Subgradients

Subgradients are the analog to gradients for non-differentiable functions. They are one of the fundamental mathematical concepts underlying convexity.

Definition 6.3 A subgradient of a convex function $f: \mathbb{R}^{n} \rightarrow \mathbb{R}$ at some point $x$ is any vector $g \in \mathbb{R}^{n}$ that achieves the same lower bound as the tangent line to $f$ at $x$, i.e. we have

$$
f(y) \geq f(x)+g^{T}(y-x) \quad \forall x, y
$$

The subgradient $g$ always exists for convex functions on the relative interior of their domain. Furthermore, if $f$ is differentiable at $x$, then there is a unique subgradient $g=\nabla f(x)$. Note that subgradients need not exist for nonconvex functions (for example, cubic functions do not have subgradients at their inflection points).

#### 0.2.1. Examples of subgradients

absolute value. $f(x)=|x|$. Where $f$ is differentiable, the subgradient is identical to the gradient, $\operatorname{sign}(x)$. At the point $x=0$, the subgradient is any point in the range $[-1,1]$ because any line passing through $x=0$ with a slope in this range will lower bound the function.

$\ell_{2}$ norm. $f(x)=\|x\|_{2}$. For $x \neq 0, f$ is differentiable and the unique subgradient is given by $g=x /\|x\|_{2}$. For $x=0$, the subgradient is any vector whose $\ell_{2}$ norm is at most 1 . This holds because, by definition, in order for $g$ to be a subgradient of $f$ we must have that

$$
f(y)=\|y\|_{2} \geq f(x)+g^{T}(y-x)=g^{T} y \quad \forall y .
$$

In order for $\|y\|_{2} \geq g^{T} y$ to hold, $g$ must have $\|g\|_{2} \leq 1$.

$\ell_{1}$ norm. $f(x)=\|x\|_{1}$. Since $\|x\|_{1}=\sum_{i=1}^{n}\left|x_{i}\right|$, we can consider each element $g_{i}$ of the subgradient separately. The result is very analogous to the subgradient of the absolute value function. For $x_{i} \neq 0$, $g_{i}=\operatorname{sign}\left(g_{i}\right)$. For $x_{i}=0, g_{i}$ isanypointin $[-1,1]$.

maximum of two functions. $f(x)=\max \left\{f_{1}(x), f_{2}(x)\right\}$, where $f_{1}$ and $f_{2}$ are convex and differentiable. Here we must consider three cases. First, if $f_{1}(x)>f_{2}(x)$, then $f(x)=f_{1}(x)$ and therefore there is a unique subgradient $g=\nabla f_{1}(x)$. Likewise, if $f_{2}(x)>f_{1}(x)$, then $f(x)=f_{2}(x)$ and $g=\nabla f_{2}(x)$. Finally, if $f_{1}(x)=f_{2}(x)$, then $f$ may not be differentiable at $x$ and the subgradient will be any point on the line segment that joints $\nabla f_{1}(x)$ and $\nabla f_{2}(x)$.

#### 0.2.2. Subdifferential

Definition 6.4 The subdifferential of a convex function $f: \mathbb{R}^{n} \rightarrow \mathbb{R}$ at some point $x$ is the set of all subgradients of $f$ at $x$, i.e. we say

$$
\partial f(x)=\left\{g \in \mathbb{R}^{n}: g \text { is a subgradient of } f \text { at } x\right\}
$$

An important property of the subdifferential $\partial f(x)$ is that it is a closed and convex set, which holds even for nonconvex $f$. To verify this, suppose we have two subgradients $g_{1}, g_{2} \in \partial f(x)$. We need to show that $g_{0}=\alpha g_{1}+(1-\alpha) g_{2}$ is also in $\partial f(x)$ for arbitrary $\alpha$. If we write the following inequalities,

$$
\begin{aligned}
\alpha(f(y) & \left.\geq f(x)+g_{1}^{T}(y-x)\right) \alpha \\
(1-\alpha)(f(y) & \left.\geq f(x)+g_{2}^{T}(y-x)\right)(1-\alpha)
\end{aligned}
$$

which follow from the definition of subgradient applied to $g_{1}$ and $g_{2}$, we can add them together to yield $f(y) \geq f(x)+\alpha g_{1}^{T}(y-x)+(1-\alpha) g_{2}^{T}(y-x)=g_{0}^{T}(y-x)$.

#### 0.2.3. Connection between sugradients and convex geometry

Suppose we have a convex set $C \subseteq \mathbb{R}^{n}$ and consider the indicator function $\mathbb{I}_{C}: \mathbb{R}^{n} \rightarrow \mathbb{R}$, defined by

$$
\mathbb{I}_{C}(x)=\mathbb{I}\{x \in C\}= \begin{cases}0 & \text { if } x \in C \\ \infty & \text { if } x \notin C\end{cases}
$$

We would like to determine the subgradients of $\mathbb{I}_{C}$. If $x \in C, \partial \mathbb{I}_{C}=N_{C}(x)$, where $N_{C}(x)$ is the normal cone of $C$ at $x$ and is defined as

$$
N_{C}(x)=\left\{g \in \mathbb{R}^{n}: g^{T} x \geq g^{T} y \text { for any } y \in C\right\}
$$

This result comes directly out of the definition of subgradient $\mathbb{I}_{C}(y) \geq \mathbb{I}_{C}(x)+g^{T}(y-x)$.

The subgradients of indicator functions are interesting for the following reason. Eventually, we will prove that a point $x^{*}$ minimizes a particular function $f$ if and only if $0 \in \partial f\left(x^{*}\right)$. Now suppose we want to know whether $x^{*}$ minimized $f(x)$ subject to the constraint that $x$ be a member of the set $C$. We can rewrite this optimization problem as

$$
\min _{x} f(x)+\mathbb{I}_{C}(x)
$$

Thus we can determine whether $x^{*}$ is a solution to this problem by checking whether $0 \in \partial\left(f\left(x^{*}\right)+\mathbb{I}_{C}\left(x^{*}\right)\right)$.

#### 0.2.4. Subgradient calculus

Subgradients can be computed by knowing the subgradients for a basic set of functions and then applying the rules of subgradient calculus. Here are the set of rules.

Scaling. $\partial(a f)=a \cdot \partial f$ provided that $a>0$

Addition. $\partial\left(f_{1}+f_{2}\right)=\partial f_{1}+\partial f_{2}$

Affine composition. If $g(x)=f(A x)+b$ then

$$
\partial g(x)=A^{T} \partial f(A x+b)
$$

Finite pointwise maximum. If $f(x)=\max _{i=1, \ldots, m} f_{i}(x)$ then

$$
\partial f(x)=\operatorname{conv}\left(\bigcup_{i: f_{i}(x)=f(x)} \partial f_{i}(x)\right)
$$

General pointwise maximum. If $f(x)=\max _{s \in \mathcal{S}} f_{s}(x)$ then

$$
\partial f(x) \supseteq \operatorname{cl}\left\{\operatorname{conv}\left(\bigcup_{s: f_{s}(x)=f(x)} \partial f_{s}(x)\right)\right\}
$$
 -->
