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

## Notation & Operators

### Misc
- $[[n]] = \set{1, \ldots, n}$.
- $A := B$ means $A$ is defined as $B$.
- $(f\circ g)(x) = f(g(x))$. Function composition. `\circ`.
- $A^\intercal$. vector/matrix transpose. `\intercal`.

### Greek alphabet
> Check [here](https://www.overleaf.com/learn/latex/List_of_Greek_letters_and_math_symbols).
{: .prompt-info }

- $\alpha$ (`\alpha`), $\Alpha$ (`\Alpha`). /ˈælfə/.
- $\beta$ (`\beta`), $\Beta$ (`\Beta`). UK: /ˈbiːtə/, US: /ˈbeɪtə/.
- $\gamma$ (`\gamma`), $\Gamma$ (`\Gamma`), $\varGamma$ (`\varGamma`). /ˈɡæmə/.
- $\delta$ (`\delta`), $\Delta$ (`\Delta`), $\varDelta$ (`\varDelta`). /ˈdɛltə/.
- $\epsilon$ (`\epsilon`), $\Epsilon$ (`\Epsilon`),  $\varepsilon$ (`\varepsilon`). /ˈɛpsɪlɒn, ɛpˈsaɪlən/.
- $\zeta$ (`\zeta`), $\Zeta$ (`\Zeta`). UK: /ˈziːtə/,[1] US: /ˈzeɪtə/.
- $\eta$ (`\eta`), $\Eta$ (`\Eta`). /ˈiːtə, ˈeɪtə/.
- $\theta$ (`\theta`), $\Theta$ (`\Theta`), $\vartheta$ (`\vartheta`), $\varTheta$ (`\varTheta`). UK: /ˈθiːtə/, US: /ˈθeɪtə/.
- $\iota$ (`\iota`), $\Iota$ (`\Iota`). /aɪˈoʊtə/.
- $\kappa$ (`\kappa`), $\Kappa$ (`\Kappa`), $\varkappa$ (`\varkappa`). /ˈkæpə/.
- $\lambda$ (`\lambda`), $\Lambda$ (`\Lambda`), $\varLambda$ (`\varLambda`). /ˈlæmdə/.
- $\mu$ (`\mu`), $\Mu$ (`\Mu`). /ˈm(j)uː/.
- $\nu$ (`\nu`), $\Nu$ (`\Nu`). /ˈnjuː/.
- $\xi$ (`\xi`), $\Xi$ (`\Xi`), $\varXi$ (`\varXi`). /zaɪ, ksaɪ/.
- $o$ (`o`), $O$ (`O`). Omicron, /ˈoʊmɪkrɒn, ˈɒmɪkrɒn, oʊˈmaɪkrɒn/.
- $\pi$ (`\pi`), $\Pi$ (`\Pi`), $\varpi$ (`\varpi`), $\varPi$ (`\varPi`). /ˈpaɪ/.
- $\rho$ (`\rho`), $\Rho$ (`\Rho`), $\varrho$ (`\varrho`). /ˈroʊ/.
- $\sigma$ (`\sigma`), $\Sigma$ (`\Sigma`), $\varsigma$ (`\varsigma`), $\varSigma$ (`\varSigma`). /ˈsɪɡmə/.
- $\tau$ (`\tau`), $\Tau$ (`\Tau`). /ˈtɔː, ˈtaʊ/.
- $\upsilon$ (`\upsilon`), $\Upsilon$ (`\Upsilon`), $\varUpsilon$ (`\varUpsilon`). /ˈʌpsɪˌlɒn, ˈ(j)uːp-, -lən/.
- $\phi$ (`\phi`), $\Phi$ (`\Phi`),  $\varphi$ (`\varphi`), $\varPhi$ (`\varPhi`). /faɪ/.
- $\chi$ (`\chi`), $\Chi$ (`\Chi`). /ˈkaɪ, ˈxiː/.
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

$I(x_i)$是事件$x_i$的自信息，事件$x_i$发生的概率越小，其发生后带来的信息越大
- 事件$x_i$发生概率为1，则其发生不带来信息
- 事件$x_i$发生概率为0，则其发生带来无穷大的信息
- 信息是关于事件发生概率的**递减**函数
- 两个事件都发生的概率为两个事件的发生概率的乘积，两个事件都发生的信息为这两个事件发生的信息的和

$$
I(x_i) = -\log p(x_i)
$$
这个函数可以满足以上条件

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

## Algebra

## Topology

## Logic

## Calculus
