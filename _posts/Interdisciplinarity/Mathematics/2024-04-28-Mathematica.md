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

这块做不了笔记，出现两个右大括号jekyll就会抽风

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