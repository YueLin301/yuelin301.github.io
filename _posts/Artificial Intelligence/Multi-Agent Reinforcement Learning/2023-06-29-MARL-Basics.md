---
title: MARL Basics
date: 2023-06-29 17:38:00 +0800
categories: [Artificial Intelligence, Multi-Agent Reinforcement Learning]
tags: [Tech, AI, Multi_Agents, RL, Markov_Models, Classic]
math: True
---

> This note has not been finished yet.
{: .prompt-warning }


## Markov Models

1. MDP
   - Markov decision process
   - $(S, A, \mathcal{P}, R, \gamma)$
   - Single-agent, fully observable, and dynamic.
2. POMDP
   - Partially observable Markov decision process
   - $(S, A, \mathcal{P}, R, O, \mathcal{Q}, \gamma)$
   - Single-agent, partially observable, and dynamic.
3. Dec-POMDP
   - Decentralized partially observable Markov decision process
   - $(I, S,\set{A^i}\_{i\in I}, \mathcal{P}, R, \set{O^i}\_{i\in I}, \mathcal{Q}, \gamma)$
   - Multi-agent, fully observable, cooperative, and dynamic.
4. Stochastic games (or Markov games)
   - $(I, S,\set{A^i}\_{i\in I}, \mathcal{P}, \set{R^i}\_{i\in I}, \gamma)$
   - Multi-agent, fully observable, C/A/M (meaning that it can be cooperative, adversarial, or mixed-motive), and dynamic.
   ```
   % Markov Game
   @incollection{MarkovGame1994,
   title={Markov games as a framework for multi-agent reinforcement learning},
   author={Littman, Michael L},
   booktitle={Machine learning proceedings 1994},
   pages={157--163},
   year={1994},
   publisher={Elsevier}
   }
   % Stochastic Game (= Markov Game)
   @book{StochasticGame2013,
   title={Game theory},
   author={Owen, Guillermo},
   year={2013},
   publisher={Emerald Group Publishing}
   }
   ```

where
- $I$ is a set of agents (and $\vert I\vert = n$, where $n\ge 2$),
- $S$ is the state space, 
- $A$ is the action space, 
- $P: S\times A \to \Delta(S)$ is the state transition function, 
- $R: S\times A$
- (TODO)

1. Normal-form game (or repeated games)
   - $(I,\set{A^i}\_{i\in I}, \set{R^i}\_{i\in I})$
   - Multi-agent, C/A/M, and static.
2. Extensive-form game
   - $(I, X, Z, f,\set{A^i}\_{i\in I}, \set{R^i}\_{i\in I})$
   - Multi-agent, C/A/M, and dynamic.
3. Bayesian game

Check this note, [A Memo on Game Theory]({{site.baseurl}}/posts/Game-Theory-Memo/).


(TODO)
Some potentially confusing concepts:
- Repeated game
- Matrix game
- Static game
- Dynamic game
- Normal-form game
- Extensive-form game
- Bayesian game




## Learning Goals
The learning goals of MARL are mainly stability and adaptation.

### Stability  
Stability is an inherent characteristic of the learning process within the algorithm we are examining. It signifies the achievement of convergence towards stationary policies or other measurable criteria. 

Here are some common criteria (in repeated games, where $t$ means the number of times the agents have played) listed, with their strength gradually increasing *(Bowling 2004)*:
1. Average reward $\sum\limits_{t\le T} (r_t\cdot \pi_t)/T$;
2. Empirical distribution of actions $\sum\limits_{t\le T} (\pi_t)/T$;
3. Expected reward $r_t\cdot \pi_t$;
4. Policies $\pi_t$.

The analysis of equilibrium should take dynamic environment into account.

> I have a question: Do the resulting policies form an equilibrium when convergence occurs? What if the it converges to a suboptimal case or failed due to insufficient exploration?
{: .prompt-tip }

### Adaptation  
This concept includes several aspects:
1. The performance of agents should be maintained or improved.
2. Agents can be adaptive to the **update** of the others' policies since the others are learning too.
3. Agents can avoid exploitation by the others.
4. When agents have several optimal actions in coordination cases (which are common), they can achieve coordination to some extent.

To check whether the designed algorithm for agents are adaptive or not, there are two criteria:
1. Rationality: Whether each agent converges to a best response when the other agents remain stationary.
   - But even in this kind of equilibrium, agents can be exploited by the others. Rationality cannot prevent the learner from 'being exploited' by the other agents.
2. No-regret: An algorithm has no-regret $\Leftrightarrow$ the average regret is less than or equal to $0$ against all others' policies.
   - No-regret can prevent the learner from 'being exploited' by the other agents.
   - Regret $\mathscr{R}$




## Scenarios (or Tasks)
There are three types of scenarios based on their properties: fully cooperative, fully competitive, and mixed scenarios. These scenarios are classified according to the **reward functions** assigned to the agents: a scenario is
1. fully cooperative if all the reward functions are the same function (i.e. $r^i=r^j\ \forall i, j\in I$),
   - (Without a centralized controller, a coordination problem will arise. Discussed below.)
2. fully competitive if $r^i = -r^j$ (two-agent scenarios are mostly discussed in this case),
   - (Games in this case are called zero-sum.)
3. and mixed otherwise.

Some additional points:
1. If there is no constraint on the reward functions, the game is called general-sum game, and pure cooperative games and zero-sum games are special cases. 
2. The second and the third case are called noncooperative games. 
3. In coordination games, rewards are always positively related.


## Algorithms
MARL algorithms are designed for variant tasks. Some of them are built relying on the assumptions of the tasks, making them applicable only to those particular tasks. 

### For fully cooperative tasks
In this case, if a centralized controller is available, the task reduces to a MDP. Otherwise the agents take actions independently, and a coordination problem arises. When there are multiple equilibria, the agents should coordinate to break ties in a same way.

The algorithms designed for the fully cooperative scenarios without centralized controller (i.e. the coordination problems) can be classified into three categories based on the dimension of coordination:
1. Coordination-free methods: teammate-independent
2. Direct coordination methods: 
   1. teammate-independent if it rely on common-knowledge, 
   2. and teammate-aware if they use negotiation.
3. Indirect coordination methods: team-aware

(TODO)
1. Team-Q: It is the Q-learning algorithm, assuming that the optimal joint action are unique (which will rarely be the case).
2. Distributed-Q: It is deterministic. Every one can see the 
3. OAL
4. JAL
5. FMQ

### For fully cooperative tasks
1. minimax-Q

### For mixed-motive tasks
#### Static
1. Fictitious play
2. MetaStrategy
3. IGA
4. WoLF-IGA
5. GIGA
6. GIGA-WoLF
7. AWESOME
8. Hyper-Q

#### Dynamic
1. Single-agent RL
2. Nash-Q
3. CE-Q
4. Asymmetric-Q
5. NSCP
6. EIRA
7. WoLF-PHC
8. PD-WoLF
9. EXORL

## Reading List
- [x] Convergence and No-Regret in Multiagent Learning.  
   Michael Bowling.  
   *Advances in Neural Information Processing Systems (NIPS) 2004*.
   - GIGA-WoLF (Generalized Infinitesimal Gradient Ascent, Win or Learn Fast)
- [x] Multi-Agent Reinforcement Learning: A survey.  
   L. Busoniu, R. Babuska, B. De Schutter.  
   *International Conference on Control, Automation, Robotics and Vision (ICARCV) 2006*.
   - This note is basically built on the framework of this paper.
- [ ] Playing is Believing: The Role of Beliefs in Multi-Agent Learning.  
   Yu-Han Chang, Leslie Pack Kaelbling.  
   Advances in Neural Information Processing Systems (NIPS) 2001.
- [ ] Nash Q-Learning for General-Sum Stochastic Games.
   Junling Hu, Michael P. Wellman.  
   Journal of machine learning research (JMLR) 2003.
- [ ] [Convergence of Q-learning: A Simple Proof](https://users.isr.ist.utl.pt/~mtjspaan/readingGroup/ProofQlearning.pdf).  
   Francisco S. Melo.
- [ ] Online Convex Programming and Generalized Infinitesimal Gradient Ascent.  
   Martin Zinkevich.  
   International Conference on Machine Learning (ICML) 2003.
- [ ] Value-Function Reinforcement Learning in Markov Games.  
   Michael L. Littman.  
   Cognitive Systems Research 2001.
- [ ] Multi-Agent Reinforcement Learning: A Critical Survey.  
   Yoav Shoham, Rob Powers, Trond Grenager.  
   *Technical report, Stanford University 2003*.
- [ ] An Overview of Multi-Agent Reinforcement Learning from Game Theoretical Perspective.  
   Yaodong Yang, Jun Wang.  
   *arXiv 2020*.
- [ ] A Survey and Critique of Multiagent Deep Reinforcement Learning.  
   Pablo Hernandez-Leal, Bilal Kartal, Matthew E. Taylor.  
   *Autonomous Agents and Multi-Agent Systems (AAMAS) 2019*.
- [ ] Multi-Agent Reinforcement Learning: A Selective Overview of Theories and Algorithms.  
   Kaiqing Zhang, Zhuoran Yang, Tamer Basar.  
   *Handbook of reinforcement learning and control 2021*.

> Disclaimer: The above content partially uses materials from the cited papers. Corresponding links or references have been provided.
{: .prompt-danger }