---
title: Mathematica Memos
date: 2024-04-27 22:40:00 +0800
categories: [Interdisciplinarity, Mathematics]
tags: [Tech, Interdisciplinarity, Math]
math: True
---

> 听说python里用sympy也能做一些推导和化简，之后去看看；mathematica占硬盘太多地了
{:.prompt-info}

## 基础
1. $\epsilon$这种输入是`Epsilon`，首字母大写
2. `Enter`是换行，`Shift + Enter`是执行
3. 区分大小写，大小写不同的量是两个量
4. 函数调用，参数用中括号框起来
5. 表达式结尾加分号`;`能让这个表达式的结果不输出
6. `*`是逐元素乘法，句号`.`是线性代数的乘法，矩阵相乘
7. `(* Comments *)`
8. 把单个文件关掉再打开，不会重启内核，之前的变量都还在
9. 退出内核：`Evaluation` -> `Quit Kernel` -> `Local`
10. 索引总是从 1 开始计数，而不是从 0 开始

## 求解

### 等式
```mathematica
Solve[ { x+y==2, x-y==0 }, { x,y }] 
```

带约束
```mathematica
Solve[ef[[1]] - eff == 0 && 0 < l < h < 1/2 , c]
```

### 不等式

```mathematica
Reduce[{ x^2 + y^2 - 2 x <= 1}, x]
```

## 向量和矩阵

### 定义
{% raw %}
```mathematica
matrix = {{a, b}, {c, d }};  (* Replace a, b, c, d with actual numbers or symbols *)
vector = {x, y};            (* Replace x, y with actual numbers or symbols *)
result = matrix . vector;
```
{% endraw %}

### 索引

使用双重中括号 `[[...]]` 来索引矩阵或向量的元素

对于向量：

```mathematica
vector = {v1, v2, v3};
element = vector[[2]];  (* 这会返回向量的第二个元素，v2 *)
```

对于矩阵：

{% raw %}
```mathematica
matrix = {{a1, b1}, {a2, b2} };
element = matrix[[1, 2]];  (* 这会返回矩阵第一行第二列的元素，b1 *)
```
{% endraw %}

索引矩阵的某一行或某一列：

- **索引某一行**：只给出行号

```mathematica
row = matrix[[1]];  (* 返回第一行，{a1, b1} *)
```

- **索引某一列**：使用 `All` 关键字来表示所有行，并给出列号

```mathematica
column = matrix[[All, 2]];  (* 返回第二列，{b1, b2} *)
```

要记住的重要事项是，这种索引方法返回的是原始数据结构中的实际元素，而不是它们的副本，这意味着如果你修改返回的元素，那么原始矩阵或向量中对应的元素也会被修改。如果你只想获得一个副本而不影响原始数据，你应该在索引操作后使用 `Copy` 函数。 -->

## 删除变量

删除单个
```mathematica
Clear[x]
```

删除多个
```mathematica
Clear[x, y, z]
```

删除全部
```mathematica
ClearAll["Global`*"]
```

反引号是用来指定上下文的特殊符号。上下文是 Mathematica 中用于管理符号名称的一种机制，它可以帮助避免不同程序包或笔记本中的符号名称冲突

## 集合
$c\in \mathbb{R}$
```mathematica
Element[c, Reals]
```


## 画图

```mathematica
Plot[x^2, {x, -2, 2}]
```

{% raw %}
```mathematica
plot = Plot[{1 - x, 2/3 - x}, {x, 0, 1}, 
   PlotRange -> {{0, 1}, {0, 1}}, AspectRatio -> Automatic];
points = Graphics[{PointSize[Large], Point[{1/3, 1/3}], 
    Point[{2/3, 0}]}];
Show[plot, points]
```
{% endraw %}

```mathematica
Plot3D[x^2 + y^2, {x, -2, 2}, {y, -2, 2}]
```

```mathematica
Plot3D[x*y, {x, -5, 5}, {y, -5, 5}]
```

```mathematica
Plot3D[x*y, {x, 0, 10}, {y, 0, 10}, 
 RegionFunction -> Function[{x, y, z}, y < 5/x], PlotRange -> All, 
 Axes -> True, AxesOrigin -> {0, 0, 0}, 
 AxesStyle -> Directive[Red, Thick], AxesLabel -> {"x", "y", "z"}, 
 AxesStyle -> Arrowheads[0.03], Boxed -> False]
```

```mathematica
Plot3D[x*y, {x, 0, 10}, {y, 0, 10}, 
 RegionFunction -> Function[{x, y, z}, Abs[y - 1] < 5/Abs[x - 1]], 
 PlotRange -> All, Axes -> True, AxesOrigin -> {0, 0, 0}, 
 AxesStyle -> Directive[Red, Thick], AxesLabel -> {"x", "y", "z"}, 
 AxesStyle -> Arrowheads[0.03], Boxed -> False]
```

```mathematica
ContourPlot[y^2 == 1/x^2, {x, -2, 2}, {y, -2, 2}]
```

Multiple lines:
```mathematica
Plot[{1 - x, 0.25/x, x, 0.3/x}, {x, 0, 1}, PlotRange -> {All, {0, 1}},
  PlotLegends -> "Expressions", AspectRatio -> Automatic]
```

Fill with color:
{% raw %}
```mathematica
Plot[1 - x, {x, 0, 1.2}, Filling -> Axis, 
 PlotRange -> {{0, 1.2}, {0, 1.2}}, AspectRatio -> Automatic]
```
{% endraw %}