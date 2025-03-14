---
title: Math Toolbox
date: 2023-04-08 02:40:00 +0800
categories:
  - Mathematics
  - Math Miscs
tags:
  - Tech
  - Math
  - Toolbox
math: true
---

> This note will be consistently updated.
{: .prompt-info }

---

## Optimization

### Basics
The standard form for an optimization problem (the primal problem) is the following:

$$
\begin{aligned}
&\min\limits_{x} \quad f_0(x)  \\
&\begin{array}{cc}
\mathrm{s.t.} 	&f_i(x) \le 0,	& i=1,2,\ldots,m\\
				&h_i(x) = 0,		& i=1,2,\ldots,p\\
\end{array}
\end{aligned}
$$

1. optimization variable: $x\in \mathbb{R}^n$

2. $\mathrm{dom}(x)=D=\bigcap\limits_{i=1}^m \mathrm{dom}(f_i) \cap \bigcap\limits_{i=1}^p\mathrm{dom}(h_i)$

3. objective function (cost function): $f_0(x):\mathbb{R}^n\to \mathbb{R}$

4. $x^*=\arg\min\limits_{x\in D} f_0(x),$ subjects to the constraints.

5. $p^\*=f_0(x^\*)$

### Convex

Convex set:

Epigraphs: $\{(x,y):y\ge f(x)\}$

Convex function: a function is convex if and only if its epigraph is convex, and the epigraph of a pointwise supremum is the intersection of the epigraphs. Hence, the pointwise supremum of convex func tions is convex.

### Duality

The Lagrangian function $L:\mathbb{R}^n \times \mathbb{R}^m \times \mathbb{R}^p \to \mathbb{R}$
$$
L(x,\lambda,v)=f_0(x) + \sum\limits_{i=0}^m \lambda_i f_i(x) + \sum\limits_{i=0}^p v_i h_i(x)
$$

The Lagrange dual function $g:\times \mathbb{R}^m \times \mathbb{R}^p \to \mathbb{R}$
$$
g(\lambda,v)=\inf\limits_{x\in D} L(x,\lambda,v)
$$

**The dual function is concave even when the optimization problem is not convex**, since the dual function is the pointwise infimum of a family of affine functions of $(\lambda,v)$.

**Infimum = Greatest Lower Bound**: the infimum of a subset $S$ partially ordered set $P$ is a greatest element in $P$ that is less than or equal to all elements of $S$, if such an element exists.

The Lagrange dual problem, then, is to maximize this dual function:

$$
\begin{align*}
\max\limits_{\lambda,v} \quad & g(\lambda,v) \\
\mathrm{s.t.} \quad & \lambda_i \ge 0, \, i = 1, \ldots, m \\
\end{align*}
$$

Formulating a dual problem gives a lower bound for the primal problem.

$$
f_0(x) \ge L(x, \lambda, \nu) \ge g(\lambda, \nu)
$$

Solving the dual problem can provide useful insights into the primal problem, and in some circumstances, it is easier to find the solution to the primal problem through solving its dual.

In certain cases (under certain conditions such as the Slater condition), the solutions to the original and dual problems will match, showcasing “strong duality”.

### Slater's Condition

Slater's condition is a criterion in convex optimization used to ensure strong duality. It imposes requirements on the existence of interior points concerning the constraints of the primal problem. Specifically, it requires there to be a point $x \in \text{relint}(D)$ (where $\text{relint}(D)$ is the relative interior of the domain $D$) such that:

$$
\begin{align*}
f_i(x) & < 0, & i = 1, \ldots, m \\
h_i(x) & = 0, & i = 1, \ldots, p \\
\end{align*}
$$

Meeting Slater's condition ensures that there is zero duality gap between the primal and dual problems, **affirming strong duality**.

Slater's condition ensures that there is a point that lies strictly within all inequality constraints and satisfies all equality constraints. This condition is employed to assure strong duality, i.e., the optimal value of the primal problem equals the optimal value of the dual problem.

1. **Existence**: By assuring there is a feasible point that meets all constraint conditions, Slater's condition guarantees that both the primal and dual problems are solvable.

2. **Gap-Free**: Slater's condition ensures a zero duality gap, i.e., there is no "gap" between the optimal solution of the primal problem and that of the dual problem, thus assuring strong duality.

### KKT Conditions

The KKT (Karush-Kuhn-Tucker) conditions are a set of equations and inequalities necessary for finding the optimal solutions to a nonlinear constrained optimization problem. The KKT conditions comprise the following equations and inequalities:

1. **Stationarity condition**:
   $$
   \nabla f_0(x) + \sum_{i=1}^m \lambda_i \nabla f_i(x) + \sum_{i=1}^p v_i \nabla h_i(x) = 0
   $$

2. **Primal feasibility**:
   $$
   \begin{align*}
   f_i(x) & \leq 0, & i = 1, \ldots, m \\
   h_i(x) & = 0, & i = 1, \ldots, p \\
   \end{align*}
   $$

3. **Dual feasibility**:
   $$
   \lambda_i \geq 0, \, i = 1, \ldots, m
   $$

4. **Complementary slackness**:
   $$
   \lambda_i f_i(x) = 0, \, i = 1, \ldots, m
   $$

If a set of solutions $x^*$, $\lambda^*$, and $v^*$ satisfy these conditions, then $x^*$ is a local optimal solution to the primal problem.

1. **Primal Feasibility and Dual Feasibility**: These two conditions ensure that the solution we find satisfies all constraint conditions of the primal and dual problems, respectively.

2. **Complementary Slackness**: This condition implies that for each inequality constraint, either it is tight (i.e., the equality holds) or its corresponding Lagrange multiplier is zero. This ensures that at the optimal solution, the solutions to the primal and dual problems are "aligned," thereby affirming strong duality.

3. **Stationarity Condition**: This condition, by setting the gradient of the Lagrangian to zero, provides us with a system to solve for the potential optimal solutions.

## Notation & Operators

### Misc
- $[[n]] = \set{1, \ldots, n}$.
- $A := B$ means $A$ is defined as $B$.
- $(f\circ g)(x) = f(g(x))$. Function composition. `\circ`.
- $A^\intercal$. vector/matrix transpose. `\intercal`.
- Norm $\Vert x \Vert$. `\Vert x \Vert`.
- Operator: $\operatorname{det}(\mathbf{M}).$ `\operatorname{det}(\mathbf{M})`.

### Greek alphabet
> Check [here](https://www.overleaf.com/learn/latex/List_of_Greek_letters_and_math_symbols).
{: .prompt-info }

- $\alpha$ (`\alpha`). /ˈælfə/.
- $\beta$ (`\beta`). UK: /ˈbiːtə/, US: /ˈbeɪtə/.
- $\gamma$ (`\gamma`), $\Gamma$ (`\Gamma`), $\varGamma$ (`\varGamma`). /ˈɡæmə/.
- $\delta$ (`\delta`), $\Delta$ (`\Delta`), $\varDelta$ (`\varDelta`). /ˈdɛltə/.
- $\epsilon$ (`\epsilon`), $\varepsilon$ (`\varepsilon`). /ˈɛpsɪlɒn, ɛpˈsaɪlən/.
- $\zeta$ (`\zeta`). UK: /ˈziːtə/,[1] US: /ˈzeɪtə/.
- $\eta$ (`\eta`). /ˈiːtə, ˈeɪtə/.
- $\theta$ (`\theta`), $\Theta$ (`\Theta`), $\vartheta$ (`\vartheta`), $\varTheta$ (`\varTheta`). UK: /ˈθiːtə/, US: /ˈθeɪtə/.
- $\iota$ (`\iota`). /aɪˈoʊtə/.
- $\kappa$ (`\kappa`), $\varkappa$ (`\varkappa`). /ˈkæpə/.
- $\lambda$ (`\lambda`), $\Lambda$ (`\Lambda`), $\varLambda$ (`\varLambda`). /ˈlæmdə/.
- $\mu$ (`\mu`). /ˈm(j)uː/.
- $\nu$ (`\nu`). /ˈnjuː/.
- $\xi$ (`\xi`), $\Xi$ (`\Xi`), $\varXi$ (`\varXi`). /zaɪ, ksaɪ/.
- $o$ (`o`), $O$ (`O`). Omicron, /ˈoʊmɪkrɒn, ˈɒmɪkrɒn, oʊˈmaɪkrɒn/.
- $\pi$ (`\pi`), $\Pi$ (`\Pi`), $\varpi$ (`\varpi`), $\varPi$ (`\varPi`). /ˈpaɪ/.
- $\rho$ (`\rho`), $\varrho$ (`\varrho`). /ˈroʊ/.
- $\sigma$ (`\sigma`), $\Sigma$ (`\Sigma`), $\varsigma$ (`\varsigma`), $\varSigma$ (`\varSigma`). /ˈsɪɡmə/.
- $\tau$ (`\tau`). /ˈtɔː, ˈtaʊ/.
- $\upsilon$ (`\upsilon`), $\Upsilon$ (`\Upsilon`), $\varUpsilon$ (`\varUpsilon`). /ˈʌpsɪˌlɒn, ˈ(j)uːp-, -lən/.
- $\phi$ (`\phi`), $\Phi$ (`\Phi`),  $\varphi$ (`\varphi`), $\varPhi$ (`\varPhi`). /faɪ/.
- $\chi$ (`\chi`). /ˈkaɪ, ˈxiː/.
- $\psi$ (`\psi`), $\Psi$ (`\Psi`), $\varPsi$ (`\varPsi`). /ˈ(p)saɪ, ˈ(p)siː/.
- $\omega$ (`\omega`), $\Omega$ (`\Omega`), $\varOmega$ (`\varOmega`). /oʊˈmiːɡə, oʊˈmɛɡə, oʊˈmeɪɡə, əˈmiːɡə/.

`var` indicates that this quantity is a variable.

### Fonts

> Adapted from [this site](https://tex.stackexchange.com/questions/58098/what-are-all-the-font-styles-i-can-use-in-math-mode).

- `\mathnormal{}` is the normal math italic font. It is the default font.
- `\mathbb{}`. 
  - $\mathbb{E}[X]$, the expectation of $X$.
  - $\mathbb{R}$, the set of real numbers.
- `\mathcal{}` is the special calligraphic font for uppercase letters only.
  - The standard normal distribution $\mathcal{N}(0,1)$.
- `\mathbf{}` gives upright Roman boldface letters.
  - $\mathbf{v}$, a vector.
  - $\mathbf{A}$, a matrix.
- `\mathrm{}` is the normal upright Roman font. 
  - $\mathrm{e}$, the constant "e".
  - $\mathrm{sin}$, the sine function.
  - $\mathrm{softmax}$, an operators.
  - $\mathrm{m}$, the unit "meter".
  - $\mathrm{cov}(X, Y)$, the covariance of $X$ and $Y$.
- `\mathtt{}` gives upright letters from the typewriter type font.  It is typically used to represent computer code, variable names, function names, and other elements that need to be displayed in a fixed-width font.
  - `\mathtt{print("Hello, World!")}`
  - $\mathtt{print("Hello, World!")}$
- `\mathit{}` gives text italic letters.
- `\mathsf{}` gives upright sans serif letters.

### Multiple variables

- Addition and subtraction of **matrices** are element-wise.

$$
\begin{bmatrix}
1 & 2 \\
3 & 4 
\end{bmatrix}
+
\begin{bmatrix}
5 & 6 \\
7 & 8 
\end{bmatrix}
=
\begin{bmatrix}
6 & 8 \\
10 & 12 
\end{bmatrix}
$$

- Multiplication of a **matrix** with a **scalar**:

$$
2 \cdot
\begin{bmatrix}
1 & 2 \\
3 & 4 
\end{bmatrix}
=
\begin{bmatrix}
2 & 4 \\
6 & 8 
\end{bmatrix}
$$

- Multiplication of **matrices**. 
  - $AB$ or $A\cdot B$. 
  - $A$ is of $m\times n$ shape, and $B$ is of $n\times k$.

$$
\begin{bmatrix}
1 & 2 \\
3 & 4 
\end{bmatrix}
\cdot
\begin{bmatrix}
5 & 6 \\
7 & 8 
\end{bmatrix}
=
\begin{bmatrix}
1\times 5+2\times 7 & 1\times 6+2\times 8 \\
3\times 5+4\times 7 & 3\times 9+7\times 8
\end{bmatrix}
$$

- Hadamard (element-wise) product: $\odot$ (`\odot`) or $\circ$ (`\circ`).

$$
\begin{bmatrix}
1 & 2 \\
3 & 4 
\end{bmatrix}
\odot
\begin{bmatrix}
5 & 6 \\
7 & 8 
\end{bmatrix}
=
\begin{bmatrix}
5 & 12 \\
21 & 28 
\end{bmatrix}
$$

- Dot product of **vectors**. Or inner product, scalar product.
  - $\mathbf{u} \cdot \mathbf{v} = \sum\limits_{i}^n u_i\cdot v_i$
- Outer product of **vectors**. 
  - $\mathbf{u} = [u_1,\ldots, u_m]^\intercal$
  - $\mathbf{v} = [v_1,\ldots, v_n]^\intercal$
  - $\mathbf{A} = \mathbf{u} \otimes \mathbf{v} = \mathbf{u} \mathbf{v}^\intercal$.
  - $\mathbf{A}$ is $m\times n$.

$$
\mathbf{u} \otimes \mathbf{v}=
\begin{bmatrix}
   u_1\cdot v_1 & u_1\cdot v_2 & \ldots & u_1\cdot v_n \\
   u_2\cdot v_1 & u_2\cdot v_2 & \ldots & u_2\cdot v_n \\
   \vdots & \vdots & \ddots & \vdots \\
   u_m\cdot v_1 & u_m\cdot v_2 & \ldots & u_m\cdot v_n 
\end{bmatrix}
$$

- [Outer product of **tensors**](https://en.wikipedia.org/wiki/Outer_product#The_outer_product_of_tensors).

Given two tensors $\mathbf{u}$ and $\mathbf{v}$ with dimensions $(k_1, k_2, \ldots, k_m)$ and $l_1,l_2, \ldots, l_n$, their outer product is a tensor with dimensions $(k_1, k_2, \ldots, k_m, l_1,l_2, \ldots, l_n)$. and entries 

$$
(\mathbf{u} \otimes \mathbf{v})_{i_1,\ldots,i_m, j_1,\ldots, j_m} = u_{i_1,\ldots,i_m}\cdot v_{j_1, \ldots, j_n}
$$


## Inequality

### Log-sum inequality

$$
\sum\limits_{i=1}^n a_i \cdot \log \frac{a_i}{b_i} \ge 
\left( \sum\limits_{i=1}^n a_i \right) \cdot
\log \frac{\sum\limits_{i=1}^n a_i}{\sum\limits_{i=1}^n b_i}$$

### Jensen's inequality

$$
\varphi\left(\mathbb{E}[X]\right) \le \mathbb{E}[\varphi(X)],
$$

where $X$ is a random variable and $\varphi$ is a convex function.

Jensen gap: $\mathbb{E}[\varphi(X)] - \varphi\left(\mathbb{E}[X]\right)$.

$$
\varphi\left(\frac{\sum a_i\cdot x_i}{\sum a_i}\right) \le \frac{\sum a_i \cdot \varphi(x_i)}{\sum a_i}
$$

Equality holds iif. $x_1 = \ldots =x_n$ or $\varphi$ is linear.


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
\mathbb{V}\left[X \right] = \mathrm{Cov}(X,X)
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
\mathbb{V}[aX+bY] = a^2\, \mathbb{V}[X] + b^2\, \mathbb{V}[Y] + 2ab\, \mathrm{Cov}[X]
$$

If there is a set of random variables $\set{X_1,\ldots X_N}$,then

$$
\begin{aligned}
   \mathbb{V}\left[\sum\limits_i X_i\right] =& \sum\limits_{i,j} \mathrm{Cov}(X_i, X_j) \\
   =& \sum\limits_{i}\mathbb{V} [X_i] +\sum\limits_{i\ne j}\mathrm{Cov} (X_i, X_j).
\end{aligned}
$$

### Covariance

$$
\mathrm{Cov}(X,Y) = \mathbb{E}\left[ X - \mathbb{E}[X] \right] \cdot \mathbb{E}\left[ Y - \mathbb{E}[Y] \right]
$$

### Moment

### Probability simplex and its illustration

![](../../../assets/img/2023-04-08-Math-Toolbox/img_2024-04-27-22-07-52.png)
_https://inst.eecs.berkeley.edu/~ee127/sp21/livebook/exa_prob_simplex.html_

## Information Theory

### Self-information
$$
I(x) = -\log p(x)
$$

信息源发消息，信息是消息中的语义，是抽象的，信号是消息的物理表示；

信息源发什么消息是不确定的（确定的话就没有通信的必要了），所以消息有一个样本空间，和一个对应的概率分布；

信息描述事件的不确定性。一个事件发生，会带来信息；事件发生的概率越小，则其出现后所带来的信息越大。事件的信息=事件的发生概率的某个函数，$I(a_i) = f[P(a_i)]$，这个叫自信息，自信息度量了随机事件信息量的大小，the amount of information；

根据信息的特性找出这个函数：
- 事件$x_i$发生的概率越小，其发生后带来的信息越大
- 事件$x_i$发生概率为1，则其发生不带来信息
- 事件$x_i$发生概率为0，则其发生带来无穷大的信息
- 信息是关于事件发生概率的**递减**函数
- 两个事件都发生的概率为两个事件的发生概率的乘积，两个事件都发生的信息为这两个事件发生的信息的和

然后一些证明，自信息的函数为$-\log$，即 $I(x_i) = -\log P(x_i)$

我看到[有说这个也叫log-perplexity](https://www.youtube.com/watch?v=EzYXbU-ZE3s&list=PLmd_zeMNzSvRRNpoEWkVo6QY_6rR3SHjp&index=2)，困惑程度。
- 概率越大，越有可能发生，越符合习惯，困惑程度低，比如"Mary had a little lamb."
- 概率越小，越不可能发生，越奇怪，困惑程度高，比如"Correct horse battery stapler."

### Entropy
$$
\begin{aligned}
   H(X) =& \mathbb{E}\_{X}\left[I(x)\right]\\
   =& -\sum\limits_{x} p(x) \cdot \log p(x)
\end{aligned}
$$

- 信源的样本空间是$X$，也就是能发送的信号（事件的集合）
- 信源的概率空间是$\set{X, P(X)}$
- 事件$x_i$以$P(x_i)$的概率发生，发生后带来$I(x_i)$的信息
- **熵 = 信源能带来的平均自信息**
- 熵一定是正的（概率大于0小于1看看就知道）
- 熵是凸函数，用Jensen不等式证明
- 均匀分布的时候，熵最大

一个事件的自信息表示了该事件发生的不确定性，该事件发生的概率越小，则其发生的不确定性越大，则其发生带来的信息越大，则其自信息越大。信息熵是自信息的期望，表示一个消息的概率分布确定时这个信源能带来的信息的多少，也是平均每个信源符号（发送一次信息，出现一次发信息事件）所携带的信息量。

#### The uniform distribution has the max entropy

要证明在给定条件下均匀分布具有最大熵，我们可以使用拉格朗日乘数法来找到概率分布的最优解。假设我们有一个离散的概率分布$p(x)$，其中$x$可以取$n$个不同的值，优化问题是

$$
\begin{aligned}
\max\limits_{p}\quad &H(p) = - \sum_{i=1}^{n} p(x_i) \log p(x_i) \\
\textrm{s.t.}\quad &\sum_{i=1}^{n} p(x_i) = 1, \\
\quad &p(x_i) > 0, \forall i.
\end{aligned}
$$

首先构造拉格朗日函数如下：

$$
L(p, \lambda) = -\sum_{i=1}^{n} p(x_i) \log p(x_i) - \lambda \left( \sum_{i=1}^{n} p(x_i) - 1 \right),
$$

其中 $\lambda$ 是拉格朗日乘数。

接下来，我们将对$L$分别对$p(x_i)$和$\lambda$求偏导，并将其设为0以找到驻点。得到：

$$
\frac{\partial L}{\partial p(x_i)} = -\log p(x_i) -1 - \lambda = 0, \quad \forall i,
$$

$$
\frac{\partial L}{\partial \lambda} = - \sum_{i=1}^{n} p(x_i) + 1 = 0.
$$

从第一个偏导数方程中我们可以解出：

$$
-\log p(x_i) -1 - \lambda = 0 \implies \log p(x_i) = -\lambda - 1 \implies p(x_i) = e^{-1-\lambda}.
$$

接着，我们可以将这个解代回约束条件

$$
\sum_{i=1}^{n} p(x_i) = 1 \implies \sum_{i=1}^{n} e^{-1-\lambda} = 1 \implies n e^{-1-\lambda} = 1 \implies e^{-1-\lambda} = \frac{1}{n}.
$$

现在我们找到了$\lambda$的值：

$$
-\lambda -1 = \log \frac{1}{n} \implies \lambda = -\log \frac{1}{n} + 1.
$$

然后我们可以找到$p(x_i)$的解：

$$
p(x_i) = e^{-1-\lambda} \implies p(x_i) = e^{-1 - (-\log \frac{1}{n} + 1)} = \frac{1}{n}.
$$

我们验证这确实是一个最大点，通过证明Hessian矩阵是负定的。最终我们得到最优解是均匀分布：

$$
p(x_i) = \frac{1}{n}, \quad \forall i.
$$


#### The more random the signal is, the less informative it will be

熵和KL散度是信息论中的两个核心概念。熵是用来衡量一个随机变量的不确定性的量，而KL散度用来衡量两个概率分布之间的差异。我们可以使用KL散度来证明一个信号的分布越随机，则其熵越高。以下是证明步骤：

1. 为了证明一个分布越随机其熵越高，我们可以假设有一个分布 $P$ 与一个完全均匀分布 $U$，其中 $U$ 的每个状态的概率都是 $\frac{1}{n}$。

2. 然后我们计算 $P$ 与 $U$ 之间的 KL 散度：
   $$
   D_{\text{KL}}(P||U) = \sum_{i=1}^{n} p(x_i) \log \frac{p(x_i)}{\frac{1}{n}} = \sum_{i=1}^{n} p(x_i) \log (n p(x_i)) - \log n \sum_{i=1}^{n} p(x_i)
   $$
   
3. 我们可以发现：
   $$
   D_{\text{KL}}(P||U) = H(U) - H(P) + \log n
   $$

4. 由于KL散度总是非负的，我们有：
   $$
   H(U) - H(P) + \log n \geq 0 \quad \Rightarrow \quad H(P) \leq H(U) + \log n
   $$

5. 由于均匀分布的熵是最大的，所以我们可以得出结论：一个分布越随机，其熵就越高。



### Joint entropy
$$
H(X, Y) = \sum\limits_{i,j} P(x_i,y_j)\cdot [- \log P(x_i,y_j)]
$$

把空间改成二维的，$x_i \in X$ 变成 $(x_i,y_j)\in X\times Y$

### Conditional entropy (channel equivocation)

假设我们原本对$X$的认识是先验概率$P$，算的这个熵也叫**先验熵**

$$
H(X) = \sum\limits_{i} P(x_i)\cdot [- \log P(x_i)]
$$

接收到一个信号$y_j$，我们就有了后验概率，对$X$的认识就改变了，可以算一个**后验熵**

$$
H(X\mid y_j) =  \sum\limits_{i} P(x_i\mid y_j)\cdot [- \log P(x_i\mid y_j)]
$$

可以对随机变量$Y$求期望，把后验熵变成**条件熵**，也叫**信道疑义度**或者**损失熵**

$$
\begin{aligned}
   H(X\mid Y)
   =& \sum\limits_{j} P(y_j) \sum\limits_{i} P(x_i\mid y_j)\cdot [- \log P(x_i\mid y_j)] \\
   =& \sum\limits_{i,j} P(x_i,y_j)\cdot [- \log P(x_i\mid y_j)] 
\end{aligned}
$$

- 如果发的信号$y_j$能唯一确定一个$x_i$，那么$P(x_i\mid y_j)=1$，$H(X\mid Y)=0$
- 如果发的信号$y_j$和$x_i$无关/独立，则￼$P(x_i\mid y_j) = P(x_i)$，则：

$$H(X\mid Y)= \sum\limits_{i,j} P(x_i)\cdot P(y_j)\cdot [- \log P(x_i)] = H(X)$$

结果和先验熵一样，我们对$X$的认识没有在收到$Y$后而改变

### Conditioning never increases entropy

条件增益性，等号只有$X$和$Y$独立时取到

$$
H(X\mid Y) \le H(X)
$$

### Chain rule for entropy

$$
H(X,Y) = H(X) + H(Y\mid X)
$$

$$
\begin{aligned}
& H(X) + H(Y\mid X) \\
=& \sum\limits_{i} P(x_i)\cdot [- \log P(x_i)]
+ \sum\limits_{i,j} P(x_i,y_j)\cdot [- \log P(y_i\mid x_j)]  \\
=& \sum\limits_{i,j} P(x_i,y_j) \cdot [- \log P(x_i)]
+ \sum\limits_{i,j} P(x_i,y_j)\cdot [- \log P(y_i\mid x_j)]  \\
=& \sum\limits_{i,j} P(x_i,y_j)\cdot [- \log P(x_i,y_j)] \\
=& H(X,Y)
\end{aligned}
$$

### Mutual information

平均互信息：接收到信号$Y$消除掉的不确定性，也是获得的信息，也是$X$和$Y$的相关性

$$
\begin{aligned}
I(X;Y) &= H(X) - H(X\mid Y)\\
&=\sum\limits_{i} P(x_i)\cdot [- \log P(x_i)] - \sum\limits_{i,j} P(x_i,y_j)\cdot [- \log P(x_i\mid y_j)] \\
&= \sum\limits_{i,j} P(x_i,y_j)\cdot [\log P(x_i\mid y_j) - \log P(x_i)] \\
&= \sum\limits_{i,j} P(x_i,y_j)\cdot \log \frac{P(x_i,y_j)}{P(x_i)\cdot P(y_j)} \\
&= D_{KL} (P_{(X,Y)} \mid \mid  P_X \otimes P_Y)
\end{aligned}
$$

- 可以看到是对称的
- 可以写成KL散度
- 互信息就是先验概率的熵减去后验概率的熵，所以先验和后验差距越大，越相关


### KL divergence
$$
\begin{aligned}
   D_{KL} \left[p(X) \Vert q(X)\right]
   =& \sum\limits_{x\in X} - p(x)\cdot \log q(x)
   -p(x)\cdot \log p(x) \\
   =& \sum\limits_{x\in X} p(x)\cdot \log \frac{p(x)}{q(x)}
\end{aligned}
$$


| 发送的信号是哪个/事件：$X$                    | $a_1$ | $a_2$ |
| --------------------------------------------- | ----- | ----- |
| 信源1发送信号的概率/事件发生的概率1：$P_1(X)$ | 0.99  | 0.01  |
| 信源2发送信号的概率/事件发生的概率2：$P_2(X)$ | 0.2   | 0.8   |

不同的信源发不同的消息带来的自信息不同，因为他们发信息的概率不一样。（相同事件在不同概率分布下发生，带来的自信息不同，因为事件在不同概率分布下的概率不一样）

KL散度就是描述这种差异，衡量两个信源发信号带来信息差的期望，也是两个概率分布的差异程度

假如知道这个表，那么信源1发$a_1$信号/$a_1$事件发生，带来的信息/消除的不确定性/自信息是$-\log P_1(a_1)$，而信源2发$a_1$信息，自信息是$-\log P_2(a_1)$

KL散度：$D_{KL}(P||Q) = \mathbb{E}_{x\sim P}\left[-\log Q(x) - \left(-\log P(x)\right) \right] = \mathbb{E}_{x\sim P} \left[\log\frac{P(x)}{Q(x)}\right]$

上述例子中：$D_{KL}(P_1||P_2) = P_1(a_1)\cdot \log \frac{P(a_1)}{Q(a_1)} + P_1(a_2)\cdot \log \frac{P(a_2)}{Q(a_2)}$

在ML中，一般$D_{KL}(P||Q)$是以$P$分布为真实分布/目标分布，衡量$Q$分布离$P$分布差多少。事件发生的概率是按$P$发生的，所以以$P$分布的概率作为求期望的分布。在真实分布/目标分布$P$下，事件按这样的分布$P$发生，发生后会带来xxx的自信息，而$Q$则带来了xxx的自信息，所以是相减然后对$P$求分布

KL散度虽然是代表差异，但不是真正意义上的距离，因为它不是对称的：$D_{KL}(P||Q)\ne D_{KL}(Q||P)$

在ML中优化时，是以最小化KL散度作为优化目标，让模型输出的$Q$分布接近目标分布$P$。那么其实可以发现，式子中的一部分是不用优化的，所以引出了交叉熵：$H(P,Q)=-\mathbb{E}_{x\sim P}\log Q(x)$。“最小化KL散度”等同于“最小化交叉熵”

## Set

### Mapping
1. **单射 (Injective)**：
    一个函数$f: A \to B$是单射的，当且仅当对于$A$中的任意两个不同的元素$a_1$和$a_2$（即$a_1 \ne a_2$），我们都有$f(a_1) \neq f(a_2)$。简单来说，不同的输入被映射到不同的输出。

2. **满射 (Surjective)**：
    一个函数$f: A \to B$是满射的，当且仅当对于集合$B$中的每一个元素$b$，都存在集合$A$中的某个元素$a$使得$f(a) = b$。简单来说，$B$中的每一个元素都是$f$的某个输出。

3. **双射 (Bijective)**：
    一个函数$f: A \to B$是双射的，当且仅当它既是单射的又是满射的。这意味着$f$为$A$中的每一个元素与$B$中的某一个元素建立了一一对应的关系，而且每一个元素在这种对应中都是独一无二的。如果存在这样的双射函数，则我们可以说集合$A$和$B$具有相同的势（cardinality）。

### Isomorphic 同构映射

首先定义“同构映射”是一个双射，它还保持某些基本的代数或其他结构特性。我们可以根据具体的结构来定义它。这样的映射通常被表示为 $f: A \rightarrow B$，其中 A 和 B 是我们所考虑的集合。

#### 在不同的数学结构中的同构

- **群同构**：在群论中，如果存在一个映射 $f: G \rightarrow H$ 使得对所有的 $g_1, g_2 \in G$，都有 $f(g_1 \cdot g_2) = f(g_1) \cdot f(g_2)$，其中“$\cdot$”表示群的操作，则称$G$和$H$是同构的。
- **环同构**：在环论中，一个环同构是一个保持加法和乘法操作的双射。即如果存在一个映射 $f: R \rightarrow S$ 使得对所有的 $r_1, r_2 \in R$，都有 $f(r_1 + r_2) = f(r_1) + f(r_2)$ 和 $f(r_1 \cdot r_2) = f(r_1) \cdot f(r_2)$，则称 R 和 S 是同构的。
- **向量空间同构**：在线性代数中，一个线性变换可以是一个向量空间到另一个向量空间的同构，如果它是双射且保持向量加法和标量乘法。

#### 性质和结果

- **唯一性**：如果两个结构是同构的，那么它们在结构上是“相同”的，这意味着它们的性质是相同的。因此，一个结构中的定理也将适用于另一个结构。
- **反对称性和传递性**：同构具有反对称性和传递性。这意味着如果 $A$ 是与 $B$ 同构的，那么 $B$ 也是与 $A$ 同构的；如果 $A$ 是与 $B$ 同构的，并且 $B$ 是与 $C$ 同构的，那么 $A$ 是与 $C$ 同构的。

#### 例子

- **整数与偶数的集合**：我们可以构建一个映射 $f: \mathbb{Z} \rightarrow 2\mathbb{Z}$，通过 $f(n) = 2n$ 来定义。这里 $2\mathbb{Z}$ 是偶数的集合。这个映射是一个同构，因为它是双射且保持加法运算。


## Topology

## Logic

## Calculus

## Combinatorics

1. Permutation:
    Permutation refers to the **arrangement** of a certain number of elements from a set **in a specific order**. The number of permutations of $k$ elements taken from a set of $n$ elements is denoted as $P(n, k)$, and it is calculated using the formula:
    $$ P(n, k) = \frac{n!}{(n-k)!} $$ 
2. Combination:
    Combination refers to the selection of a certain number of elements from a set **without considering the order**. The number of combinations of $k$ elements taken from a set of $n$ elements is denoted as $C(n, k)$, and it is calculated using the formula:
    $$ C(n, k) = \binom{n}{k} = \frac{n!}{k!(n-k)!} $$