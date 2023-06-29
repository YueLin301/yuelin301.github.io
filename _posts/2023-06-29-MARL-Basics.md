---
title: MARL Basics
date: 2023-06-29 17:38:00 +0800
categories: [Multi-Agent Reinforcement Learning]
tags: [multi agents, reinforcement learning, Nash equilibrium]
math: True
---

> This note has not been finished yet. One may check my [writing schedule](https://yuelin301.github.io/posts/Schedule/).
{: .prompt-warning }

## Reading List
1. Multi-Agent Reinforcement Learning: A survey.  
   L. Busoniu, R. Babuska, B. De Schutter.
   *International Conference on Con- trol, Automation, Robotics and Vision (ICARCV) 2006*.
2. Convergence and No-Regret in Multiagent Learning.  
   Michael Bowling.  
   *Advances in Neural Information Processing Systems (NIPS) 2004*.
3. Multi-Agent Reinforcement Learning: A Critical Survey.  
   Yoav Shoham, Rob Powers, Trond Grenager.
   *Technical report, Stanford University 2003*.
4. An Overview of Multi-Agent Reinforcement Learning from Game Theoretical Perspective.  
   Yaodong Yang, Jun Wang.  
   *arXiv 2020*.
5. A Survey and Critique of Multiagent Deep Reinforcement Learning.  
   Pablo Hernandez-Leal, Bilal Kartal, Matthew E. Taylor.  
   *Autonomous Agents and Multi-Agent Systems (AAMAS) 2019*.
6. Multi-Agent Reinforcement Learning: A Selective Overview of Theories and Algorithms.  
   Kaiqing Zhang, Zhuoran Yang, Tamer Basar.  
   *Handbook of reinforcement learning and control 2021*.
   

## Learning Goals
*(Busoniu 2006)*

### Stability  
(of the learning process)  
Stability essentially means the convergence to stationary policies.
   1. Convergence to equilibria (a requirement)
### Adaptation  
(to the dynamic behavior of the other agents)  
Adaptation ensures that performance is maintained or improved.
   1. Rationality: It requires the agent to converge to a best response when other agents remain stationary. (an adaptation criterion)
   2. No-regret: It prevents the learner from 'being exploited' by the other agents. (an alternative to rationality)