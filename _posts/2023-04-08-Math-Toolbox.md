---
title: Math Toolbox
date: 2023-04-08 02:40:00 +0800
categories: [Mathematics]
tags: [math, toolbox]
math: True
---

> This note will be consistently updated.
{: .prompt-info }

---

## Optimization

### Basics
The standard form for an optimization problem is the following:

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

4. $x^*=\arg\min\limits_{x\in D} f_0(x)$

5. $p^\*=f_0(x^\*)$

### Convex

Convex set:

Epigraphs: $\{(x,y):y\ge f(x)\}$

Convex function: a function is convex if and only if its epigraph is convex, and the epigraph of a pointwise supremum is the intersection of the epigraphs. Hence, the pointwise supremum of convex func tions is convex.

### Duality

The Lagrangian function $L:\mathbb{R}^n \times \mathbb{R}^m \times \mathbb{R}^p \to \mathbb{R}$
$$
L(x,\lambda,v)=f_0(x) + \sum\limits_{i=0}^m \lambda_i f_i(x) + \sum\limits_{i=0}^m v_i h_i(x)
$$

The Lagrange dual function $g:\times \mathbb{R}^m \times \mathbb{R}^p \to \mathbb{R}$
$$
g(\lambda,v)=\inf\limits_{x\in D} L(x,\lambda,v)
$$

**The dual function is concave even when the optimization problem is not convex**, since the dual function is the pointwise infimum of a family of affine functions of $(\lambda,v)$.

**Infimum = Greatest Lower Bound**: the infimum of a subset $S$ partially ordered set $P$ is a greatest element in $P$ that is less than or equal to all elements of $S$, if such an element exists.

### KKT condition

## Notation & Operators

### Misc
- $[[n]] = \set{1, \ldots, n}$.
- $A := B$ means $A$ is defined as $B$.
- $(f\circ g)(x) = f(g(x))$. Function composition. `\circ`.
- $A^\intercal$. vector/matrix transpose. `\intercal`.
- Norm $\Vert x \Vert$. `\Vert x \Vert`.

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

一个事件的自信息表示了该事件发生的不确定性，该事件发生的概率越小，则其发生的不确定性越大，则其发生带来的信息越大，则其自信息越大。信息熵是自信息的期望，表示一个消息的概率分布确定时这个信源能带来的信息的多少，也是平均每个信源符号（发送一次信息，出现一次发信息事件）所携带的信息量。

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

# 

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

## Algebra

### Learning resources
- [Linear Algebra Done Right](https://linear.axler.net) (Book).
- [Videos by Gilbert Strang (reposted on bilibili)](https://www.bilibili.com/video/BV18K4y1R7MP/?spm_id_from=333.337.search-card.all.click&vd_source=da10e270ab81b4c1a3343cc28f4d6a37).
- [The Matrix Cookbook](https://www.math.uwaterloo.ca/~hwolkowi/matrixcookbook.pdf) (Book).

### Inverse


1. A square matrix $A$ is invertible 
2. = $A$ has n pivots
3. = $A$ is not singular
4. = the columns/rows of $A$ are independent
5. = the columns/rows are linearly independent
6. = elimination can be completed: $PA=LDU$, with all n pivots
7. = the nullspace of these vectors are $\{\mathbf 0\}$ 
8. = the determinant of $A$ is not 0
9. = the rank of $A$ is n
10. = 0 is not an eigenvalue of $A$
11. = $A^TA$ is positive definite

### Rank

rank of a matrix = the number of independent columns/rows of this matrix = the number of pivots of this matrix

方程组求解 -> 写成矩阵形式 -> 知道什么是线性组合 -> 知道什么是生成子空间 -> 矩阵的秩是生成子空间的维度

#### 方程组求解 -> 写成矩阵形式：

$$
a_{11}x_1+a_{12}x_2+a_{13}x_3=b_1\\ a_{21}x_1+a_{22}x_2+a_{23}x_3=b_2\\ a_{31}x_1+a_{32}x_2+a_{33}x_3=b_3
$$
可以写成$\mathbf{A}\mathbf{x}=\mathbf{b}$，矩阵和向量的乘法就是这么定义的，方便表达

#### 知道什么是线性组合 -> 知道什么是生成子空间：

$$
\begin{bmatrix} a_{11} \\ a_{21} \\ a_{31}  \end{bmatrix} x_1 + \begin{bmatrix} a_{12} \\ a_{22} \\ a_{32}  \end{bmatrix} x_2 + \begin{bmatrix} a_{13} \\ a_{23} \\ a_{33}  \end{bmatrix} x_3  = \begin{bmatrix} b_{1} \\ b_{2} \\ b_{3}  \end{bmatrix}
$$

矩阵拆成列向量，一个列向量代表一个方向，每个列向量的乘积代表沿着这个列向量的方向走多远。

解方程组的问题变成：在空间中给出几个方向，也给出了终点和起点（相对位置），要分别沿着这些方向走多远，可以从起点到达终点？

看几个情况，欠定方程组的系数矩阵为矮矩阵，即系数矩阵$\mathbf{A}\in \mathbb{R}^{m\times n}$的行数$m$小于列数$n$，方程组有无穷多解，比如：

$$
\begin{bmatrix} a_{11} \\ a_{21} \end{bmatrix} x_1 + \begin{bmatrix} a_{12} \\ a_{22} \end{bmatrix} x_2 + \begin{bmatrix} a_{13} \\ a_{23}  \end{bmatrix} x_3  = \begin{bmatrix} b_{1} \\ b_{2}  \end{bmatrix}
$$

如果这三个列向量互相不成比例，那么他们是线性无关的，由于每个列向量是两个元素，可以在一个平面中画出来。我们知道在平面中任意两个不共线的向量的线性组合，就可以表示这个平面上的任意一个向量了，那么现在有3个，多了，想怎么走就怎么走了

另一种情况，稀疏矩阵的列数小于行数，比如：

$$
\begin{bmatrix} a_{11} \\ a_{21} \\ a_{31}  \end{bmatrix} x_1 + \begin{bmatrix} a_{12} \\ a_{22} \\ a_{32}  \end{bmatrix} x_2  = \begin{bmatrix} b_{1} \\ b_{2} \\ b_{3}  \end{bmatrix}
$$

给的方向还有目标点都是在三维空间中的，但是只给了两个方向，怎么组合那也只能是在一个平面里折腾，如果恰好$b$这个目标点在这个平面上，那么可以完成任务，否则是不能完成的，方程组也就无解

每一种$(x_1,x_2)$的取值都对应了一种$\mathbf{A}$的列向量的**线性组合 (linear combination)**，线性组合是对向量的操作，矩阵的列向量的线性组合 = 这个矩阵的**列空间（column space）** 或者是这个矩阵的值域（range）

$\mathbf{A}\mathbf{x}=\mathbf{b}$，如果$\mathbf{b}$不在$\mathbf{A}$的列空间里，那么这个问题就无解了。所以如果要有解，那么首先维度得对得上，即$\mathbf{A}$的列空间的维度要大于等于$\mathbf{b}$的维度


#### 

一组向量的**生成子空间（span）**就是关于它的线性组合能到达的点，一个矩阵的列空间（列向量的生成子空间）的维度就是**秩（rank）**


### Linear Independency

If $\exists k\in \mathbb{R}$, s.t. $\mathbf{x_1} = k\cdot\mathbf{x_2}$, then $x_1$ and $x_2$ are  linear dependent.

如果一个矩阵中有两个列向量是线性相关的，那么其中一个被删掉了也不会改变这个矩阵的列空间

<!-- $\mathbf{A}\mathbf{x}=\mathbf{b}$，如果要对任意一个$\mathbf{b} \in \mathbb{R}^m$都有解，那么$\mathbf{A}$的列空间要覆盖$\mathbb{R}^m$，那么 -->

### Eigenvalue

### Determinant

### Singular

### Cramer’s rule

### Adjugate matrix

## Set

> Generated by ChatGPT 4

1. **单射 (Injective)**：
    一个函数$f: A \to B$是单射的，当且仅当对于$A$中的任意两个不同的元素$a_1$和$a_2$（即$a_1 \ne a_2$），我们都有$f(a_1) \neq f(a_2)$。简单来说，不同的输入被映射到不同的输出。

2. **满射 (Surjective)**：
    一个函数$f: A \to B$是满射的，当且仅当对于集合$B$中的每一个元素$b $，都存在集合$A$中的某个元素$a$使得$f(a) = b$。简单来说，$ B$中的每一个元素都是$f$的某个输出。

3. **双射 (Bijective)**：
    一个函数$f: A \to B$是双射的，当且仅当它既是单射的又是满射的。这意味着$f$为$A$中的每一个元素与$B$中的某一个元素建立了一一对应的关系，而且每一个元素在这种对应中都是独一无二的。如果存在这样的双射函数，则我们可以说集合$A$和$B$具有相同的势（cardinality）。

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