---
title: Information Design in 10 Minutes
date: 2023-08-11 05:30:00 +0800
categories: [Interdisciplinarity, Economics & Game Theory]
tags: [Tech, Interdisciplinarity, Economics, Game Theory, Multi Agents, Information Design]
math: True
pin: True
---

> This note provides a brief introduction to the basic concepts of information design. More details can be found in [my other note on this topic]({{site.baseurl}}/posts/Information-Design/).
{: .prompt-info }

<!-- ## A Brief Introduction -->

> "Sometimes, the truth is not good enough." *— Batman, The Dark Knight (2008).*

Information design focuses on scenarios of **mixed-motive** unidirectional communication, where one self-interested sender with informational advantage attempts to **persuade** a self-interested rational receiver to take actions that the sender prefers.

- The **"informational advatage"** means that the sender has something that the receiver wants to know (i.e. which affects the receiver's payoff) but cannot know, 
- **"self-interested"** refers to the agent being concerned only about its own expected payoff, and 
- **"rational"** means that when it believes one action's payoff is greater than another's, the agent will choose the action with higher expected payoff.

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

This greatly simplifies the problem, allowing **the sender to choose the receiver's action that maximizes its expected payoff, while ensuring that the receiver obeys, and then recommend the receiver to take that action.** Thus, the specific representation of the sender's optimization goal is:

$$
\begin{aligned}
\max\limits_{\varphi} \mathbb{E}_{\varphi}[\ r^i(s, a) \ ],\;\;\textrm{s.t. Obedience Constraints.}
\end{aligned}
$$

<!-- ## A View of D&D Alignment

![](../../assets/img/2023-08-11-Information-Design-10min/img_2023-12-09-05-39-14.png){: width="500" }
_Illustration from this [web page](https://www.reddit.com/r/DnD/comments/xd16rh/a_different_interpretation_of_the_alignment_chart/)_

![](../../assets/img/2023-08-11-Information-Design-10min/img_2023-12-09-05-39-51.png){: width="300" }

> This section is entirely based on my understanding, so please consider it accordingly.
{: .prompt-danger}

The circle denotes the boundary of interests. The left circle represents self-interest and the right circle represents altruistic interest. The intersection of the two circles signifies the mixed-motive part. 
<!-- The vertical line defines what is good or evil. The area to the left of the line is considered evil and can cause harm to others, while the area to the right is seen as good and will benefit others. -->
<!-- With these in mind, we can say that... -->

<!-- The sender's original motivation is $\text{Chaotic Evil}$, as the basic assumption is that it is self-interested and rational. 
Acting upon this, the sender won't do anything that benefits others out of the consideration for altruism.

However, through information design, the sender is constrained to send messages that cause no harm to the receiver thereby showing care for others and thus eliminating the left part of $\text{Chaotic Evil}$. 
In this way, we can say that the sender's actions can be described as ranging from $(\text{Neutral Good} - \text{Lawful Good})$.  -->

<!-- ![](../../assets/img/2023-08-11-Information-Design-10min/img_2023-12-09-06-58-37.png){: width="100" }
_$(\text{Neutral Good} - \text{Lawful Good})$_ 
-->
