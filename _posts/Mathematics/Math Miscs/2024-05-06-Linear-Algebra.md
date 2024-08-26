---
title: Linear Algebra
date: 2024-05-06 10:40:00 +0800
categories:
  - Mathematics
  - Math Miscs
tags:
  - Tech
  - Math
math: true
---

## Learning resources

- [Linear Algebra Done Right](https://linear.axler.net) (Book).
- [Videos by Gilbert Strang (reposted on bilibili)](https://www.bilibili.com/video/BV18K4y1R7MP/?spm_id_from=333.337.search-card.all.click&vd_source=da10e270ab81b4c1a3343cc28f4d6a37).
- [The Matrix Cookbook](https://www.math.uwaterloo.ca/~hwolkowi/matrixcookbook.pdf) (Book).

## Inverse

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

## Rank

rank of a matrix = the number of independent columns/rows of this matrix = the number of pivots of this matrix

方程组求解 -> 写成矩阵形式 -> 知道什么是线性组合 -> 知道什么是生成子空间 -> 矩阵的秩是生成子空间的维度

### 方程组求解 -> 写成矩阵形式：

$$
a_{11}x_1+a_{12}x_2+a_{13}x_3=b_1\\ a_{21}x_1+a_{22}x_2+a_{23}x_3=b_2\\ a_{31}x_1+a_{32}x_2+a_{33}x_3=b_3
$$
可以写成$\mathbf{A}\mathbf{x}=\mathbf{b}$，矩阵和向量的乘法就是这么定义的，方便表达

### 知道什么是线性组合 -> 知道什么是生成子空间：

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


### 知道什么是生成子空间 -> 知道什么是秩

一组向量的**生成子空间（span）**就是关于它的线性组合能到达的点，一个矩阵的列空间（列向量的生成子空间）的维度就是**秩（rank）**


## Linear Independency

If $\exists k\in \mathbb{R}$, s.t. $\mathbf{x_1} = k\cdot\mathbf{x_2}$, then $x_1$ and $x_2$ are  linear dependent.

如果一个矩阵中有两个列向量是线性相关的，那么其中一个被删掉了也不会改变这个矩阵的列空间

<!-- $\mathbf{A}\mathbf{x}=\mathbf{b}$，如果要对任意一个$\mathbf{b} \in \mathbb{R}^m$都有解，那么$\mathbf{A}$的列空间要覆盖$\mathbb{R}^m$，那么 -->

## Eigenvalue

## Determinant

## Singular

## Cramer’s rule

## Adjugate matrix


#### Algebraic Cofactor
The algebraic cofactor of an element $a_{ij}$ (located at the $i$th row and $j$th column) is defined as follows:

1. **Remove the $i$th row and $j$th column**: First, remove the row and column containing the element $a_{ij}$ from the matrix $A$, resulting in a $(n-1) \times (n-1)$ submatrix.
2. **Calculate the determinant of the submatrix**: Compute the determinant of this reduced matrix, denoted as $ \text{det}(A_{ij}) $.
3. **Multiply by $ (-1)^{i+j} $**: The final cofactor $C_{ij}$ is the product of $ (-1)^{i+j} $ and the determinant of the submatrix:

$$
C_{ij} = (-1)^{i+j} \cdot \text{det}(A_{ij})
$$

This value represents the determinant of the remaining matrix after removing the $i$th row and $j$th column from $A$, adjusted for sign.

#### Adjugate Matrix

The adjugate matrix, also known as the adjoint matrix or classical adjoint, is closely related to the original matrix and is primarily used for calculating the inverse of a matrix. The elements of the adjugate matrix are the algebraic cofactors of the corresponding elements of the original matrix, assembled and then transposed. For a matrix $A$, the adjugate matrix $ \operatorname{Adj}(A) $ is defined as follows:

1. **Compute the algebraic cofactors for each element**: For each element $a_{ij}$ of $A$, compute its algebraic cofactor $C_{ij}$.
2. **Construct the adjugate matrix**: Arrange all $C_{ij}$ in a new matrix according to their positions in $A$, then **transpose** this matrix:

$$
\operatorname{Adj}(A) = \begin{bmatrix}
C_{11} & C_{21} & \cdots & C_{n1} \\
C_{12} & C_{22} & \cdots & C_{n2} \\
\vdots & \vdots & \ddots & \vdots \\
C_{1n} & C_{2n} & \cdots & C_{nn}
\end{bmatrix}
$$

Thus, the $(i, j)$ element of the adjugate matrix is actually the algebraic cofactor of the $(j, i)$ position in the original matrix $A$. The adjugate matrix is crucial for the computation of the inverse of a matrix because for any non-singular matrix $A$:

$$
A^{-1} = \frac{1}{\text{det}(A)} \cdot \operatorname{Adj}(A)
$$

If $A$ is singular (i.e., its determinant is zero), this formula does not apply as division by zero is undefined.


#### $A \cdot \operatorname{Adj}(A) = \operatorname{Adj}(A) \cdot A = \text{det}(A) \cdot I$

This is a consequence of the Laplace expansion of the determinant.

$$
\operatorname{det}(A)=\sum_{j=1}^n C_{ij} a_{ij}
$$

