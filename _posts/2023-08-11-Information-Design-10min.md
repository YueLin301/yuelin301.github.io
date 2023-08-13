---
title: Information Design in 10 Minutes
date: 2023-08-11 05:30:00 +0800
categories: [Economics & Game Theory]
tags: [information design, bayesian persuasion, obedience, multi agents, incentive compatibility]
math: True
pin: True
---

## What problems does information design study?

Information design focuses on scenarios of **mixed-motive** unidirectional communication, where one self-interested sender with informational advantage attempts to **persuade** a self-interested rational receiver to take actions that the sender prefers.

- The "**informational advatage**" means that the sender has something that the receiver wants to know (i.e. which affects the receiver's payoff) but cannot know, 
- "**self-interested**" refers to the agent being concerned only about its own expected payoff, and 
- "**rational**" means that when it believes one action's payoff is greater than another's, the agent will choose the action with higher expected payoff.

Additionally, information design with a sender and a receiver is known as Bayesian persuasion. And the flow of an one-step Bayesian persuasion process is as follows:
1. The sender commits a signaling scheme to the receiver. The receiver will use this to calculate its posterior expected payoff. (This is referred to as the **commitment assumption**.);
2. The nature generates a state $s$. The sender observes the state $s$ and then samples a message according to the distribution of the committed signaling scheme; and
3. Receiving the message, the receiver calculates a posterior and chooses an optimal action for itself. Given the current state and the receiver's chosen action, the sender and the receiver get rewards from the nature.

The key to the sender successfully persuading a receiver with whom it has an interest conflict lies in **obedience constraints**. To introduce it, let's simplify the problems first.

Assuming that the sender's signal set is equal to the receiver's action set, the sender's signals can be interpreted as recommending the receiver to take a specific action. This common assumption is without loss of generality according to the **revelation principle**, i.e., there is an optimal signaling scheme that does not require more signals than the number of actions available to the receiver. 

Under this premise, **obedience constraints** can be formalized as:

$$
\sum\limits_{s} \mu_0(s) 
  \cdot \varphi( a\mid s )
  \cdot \Big( r^j(s, a) - r^j(s, a') \Big) \ge 0,
$$

where $s\in S$ is the state which is only observable by the sender, $\mu_0$ is a prior distribution which is a common knowledge (both know, both know both know, etc.), $a\in A$ is the receiver's action space, $\varphi$ is the sender's signaling scheme, and $r^j$ is the receiver's reward function that depends on the state and the receiver's chosen action.

**The obedience constraints ensure that the receiver will definitely follow the sender's recommendations.** A simple derivation is as follows: 

$$
\begin{aligned}
  & \sum\limits_{s} \mu_0(s) 
  \cdot \varphi( a\mid s )
  \cdot \Big( r^j(s, a) - r^j(s, a') \Big) \ge 0 \\
  \Leftrightarrow &
  \sum\limits_{s} \frac{\mu_0(s) \cdot \varphi( a\mid s )}
  { \sum\limits_{s'}\mu_0(s') \cdot \varphi( a\mid s')}
  \cdot \Big( r^j(s, a) - r^j(s, a') \Big) \ge 0 , \forall a'\in A.\\
  \Leftrightarrow &
  \sum\limits_{s} \mu(s\mid a)
  \cdot \Big( r^j(s, a) - r^j(s, a') \Big) \ge 0 , \forall a'\in A.\\
  \Leftrightarrow &
  \sum\limits_{s} \mu(s\mid a)
  \cdot r^j(s, a)  \ge 
  \sum\limits_{s} \mu(s\mid a)
  \cdot r^j(s, a'), \forall a'\in A.
\end{aligned}
$$

where $\mu$ represents the posterior probability. Therefore, a self-interested and rational receiver will definitely follow the sender's recommendations, because the posterior expected payoff of the action recommended by the sender is greater than or equal to the posterior expected payoffs of all other actions. 

This greatly simplifies the problem, allowing **the sender to choose the receiver's action that maximizes its expected payoff, while ensuring that the receiver obeys, and then recommend the receiver to take that action**. Thus, the specific representation of the sender's optimization goal is:

$$
\begin{aligned}
\max\limits_{\varphi} \mathbb{E}_{\varphi}[\ r^i(s, a) \ ],\;\;\textrm{s.t. Obedience Constraints.}
\end{aligned}
$$

## What are its applications in real-world scenarios?

Information design was first introduced in this paper (and it has been cited 2460 times to date):
> Kamenica, Emir, and Matthew Gentzkow. "Bayesian persuasion." *American Economic Review* (2011).
{: .prompt-info }

In practical economic scenarios, persuasion is ubiquitous and plays a crucial role. As stated in the title and conclusion of McCloskey and Klamer’s paper, **“one quarter of the GDP is persuasion”**. This kind of persuasion demonstrates that communication is conceivable in mixed-motive scenarios.

> Research in this second strand includes applications to (in no particular order): financial sector stress tests, grading in schools, employee feedback, law enforcement deployment, censorship, entertainment, financial over-the-counter markets, voter coalition formation, research procurement, contests, medical testing, medical research, matching platforms, price discrimination, financing, insurance, transparency in organizations, and routing software.  

>  Kamenica, Emir. "Bayesian persuasion and information design." *Annual Review of Economics* (2019).
{: .prompt-info }

Among them, **routing software** is an relatively easily imaginable example. 
1. The central routing software and user vehicles are in a **mixed-motive** scenario, where the routing software aims to optimize the overall traffic speed, while each user wants to increase their individual speed. 
2. Moreover, the routing software has more **informational advantage** than user vehicles; it can know the congestion status of all relevant road segments, while user vehicles are not allowed to do so. Each user is self-interested, which may lead to a decrease in collective benefits, known as a social dilemma. 

A common example of this is **Braess's paradox**. Using information design can resolve this issue. For specific details, please refer to the article mentioned in this paper: 

> Das, Sanmay, Emir Kamenica, and Renee Mirka. "Reducing congestion through information design." *2017 55th annual allerton conference on communication, control, and computing (allerton)*. IEEE, 2017.
{: .prompt-info }