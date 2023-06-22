---
title: TCS Basics
date: 2023-06-17 20:00:01 +0800
categories: [Mathematics]
tags: [theoretical computer science, computational complexity theory, reduction, complexity classes]
math: True
---

> This note has not been finished yet. One may check my [writing schedule](https://yuelin301.github.io/posts/Schedule/).
{: .prompt-warning }

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

> This note is basically a compilation of information from Wikipedia. 😂  
> And currently I mainly focus on the computational complexity theory.
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
