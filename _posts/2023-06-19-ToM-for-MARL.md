---
title: Theory of Mind for MARL
date: 2023-06-19 20:00:01 +0800
categories: [Multi-Agent Reinforcement Learning]
tags: [theory of mind]
math: True
---

> This note has not been finished yet. One may check my [writing schedule](https://yuelin301.github.io/posts/Schedule/).
{: .prompt-warning }

## What is Theory of Mind?
In psychology, theory of mind refers to **the capacity to understand other people by ascribing mental states to them** (that is, surmising what is happening in their mind). This includes the knowledge that others' beliefs, desires, intentions, emotions, and thoughts may be different from one's own.[^wiki-tom]

Possessing a functional theory of mind is considered crucial for success in everyday human social interactions. People use such a theory when analyzing, judging, and inferring others' behaviors. The discovery and development of theory of mind primarily came from studies done with animals and infants.[^wiki-tom]

**Empathy**—the recognition and understanding of the states of mind of others, including their beliefs, desires, and particularly emotions—is a related concept. Empathy is often characterized as the ability to **"put oneself into another's shoes"**. Recent neuro-ethological studies of animal behaviour suggest that even rodents may exhibit empathetic abilities。 **While empathy is known as emotional perspective-taking, theory of mind is defined as cognitive perspective-taking.**[^wiki-tom]

In my understanding, theory of mind refers to the ability of an individual modeling others' decision making processes based on others' partial observations.


---


## Settings

### Dec-POMDP

A **decentralized partially observable Markov decision process** (Dec-POMDP) is a tuple $(I, S,\{A^i\}\_{i\in I}, \mathcal{P}, R, \{O^i\}\_{i\in I}, \mathcal{Q}, \gamma)$, where
- $I$ is a set of agents ($\vert I\vert=n$ and they are **cooperative**),
- $S$ is a set of global states of the environment (and agents cannot see the sampled state at any time, but they know the state set), 
- $A^i$ is a set of actions for agent $i$, with $A=\times_{i\in I} A^i$ is the set of joint actions,
- $\mathcal{P}:S\times A\to\Delta(S)$ is the state transition probability where $\Delta(S)$ is a set of distributions over $S$,
- $R:S\times A\to \mathbb{R}$ is a reward function (not $\mathbb{R}^n$ since the agents are **cooperative**),
- $O^i$ is a set of observations for agent $i$, with $O = \times_{i\in I} O^i$ is the set of joint observations,
- $\mathcal{Q}:S\times A\to\Delta(O)$ is an observation emission function (sometimes $\mathcal{Q}:S\to\Delta(O)$), and
- $\gamma\in[0,1]$ is the discount factor.

One step of the process is: $s_{t}\to a_{t}\to (s_{t+1}, o_{t}, r_{t})$. At each timestep $t$,  
- each agent takes an action $a_t^i\in A^i$,
- $s_{t+1}\sim \mathcal{P}(\cdot \mid s_t, a_t)$, 
- $o_t \sim \mathcal{Q}(\cdot \mid s_{t+1}, a_{t})$ and a reward is generated for **the whole team** based on the reward function $R(s,a)$.

These timesteps repeat until some given horizon (called finite horizon) or forever (called infinite horizon). The discount factor $\gamma$ maintains a finite sum in the infinite-horizon case ($\gamma \in [0,1)$). 
The goal is to maximize expected cumulative reward over a finite or infinite number of steps.[^wiki-tom]

What is different from what I previously thought is that the observations are sampled after agents make decisions at each timestep.

> This definition is adapted from that of Wikipedia[^wiki-Dec-POMDP]. When discussing Dec-POMDP, these papers[^Dec-POMDP-Bernstein][^Dec-POMDP-Oliehoek] are often referenced.
{: .prompt-info }

### MDP & POMDP

Markov decision processes (MDPs) and partially observable Markov decision processes (POMDPs) are degenerated cases of Dec-POMDPs:
- POMDP: $(S, A, \mathcal{P}, R, O, \mathcal{Q}, \gamma)$, a single-agent version of Dec-POMDP.
- MDP: $(S, A, \mathcal{P}, R, \gamma)$, a fully observable version of POMDP.

> One may check this slides[^mdps-slides] for understanding the comparison between MDP, POMDP, and Dec-POMDP. 
{: .prompt-info }


### Belief

> So how does the agent choose its action wisely in **POMDPs**?
{: .prompt-tip }

Since it is the state that affects payoffs and the state transitions (thus the future payoffs) rather than the observation, the agent needs to estimate the current state $s_t$ by its previous observations before choosing $a_t$.

The state is Markovian by assumption, meaning that maintaining a belief over the current states $s_t$ solely requires knowledge of 
- the previous belief state $s_{t-1}$, 
- the taken action $a_{t-1}$, 
- and the current observation (generated at the end of the last step) $o_{t-1}$.

The belief $b$ is a distribution over states. The probability of a current state $s_t$ is $b(s_t)$ can be recursively defined as follows: Given $b(s_{t-1})$, 

$$
b(s_{t}) = \frac
    {
        \mathcal{Q}(o_{t-1}\mid s_t, a_{t-1}) 
        \sum\limits_{s_{t-1}} 
        \mathcal{P}(s_{t}\mid s_{t-1}, a_{t-1}) 
        \cdot b(s_{t-1})
    }
    {
        \sum\limits_{s_{t}}
        \mathcal{Q}(o_{t-1}\mid s_t, a_{t-1}) 
        \sum\limits_{s} 
        \mathcal{P}(s_{t}\mid s_{t-1}, a_{t-1})
        \cdot b(s_t-1)
    }.
$$

<!-- P(s_t\mid s_{t-1}, o_{t-1}, a_{t-1}) =  -->


> This definition is adapted from that of Wikipedia[^wiki-POMDP].
{: .prompt-info }

---

## References

[^wiki-tom]: Wikipedia: [theory of mind](https://en.wikipedia.org/wiki/Theory_of_mind).
[^Fuchs2019]: Andrew Fuchs, Michael Walton, Theresa Chadwick, Doug Lange. "Theory of mind for deep reinforcement learning in hanabi." *NeurIPS Workshop (2019)*.
[^wiki-Dec-POMDP]: Wikipedia: [Dec-POMDP](https://en.wikipedia.org/wiki/Decentralized_partially_observable_Markov_decision_process).
[^mdps-slides]: Alina Vereshchaka's [slides about MDPs](https://cse.buffalo.edu/~avereshc/rl_fall19/lecture_23_MDP_POMDP_DecPOMDP.pdf).
[^Dec-POMDP-Bernstein]: Daniel S Bernstein, Robert Givan, Neil Immerman, Shlomo Zilberstein. "The complexity of decentralized control of markov decision processes." *Mathematics of operations research (2002)*.
[^Dec-POMDP-Oliehoek]:Frans A Oliehoek, Christopher Amato. "A concise introduction to decentralized POMDPs." *Springer (2016)*.
[^wiki-POMDP]: Wikipedia: [POMDP](https://en.wikipedia.org/wiki/Partially_observable_Markov_decision_process).