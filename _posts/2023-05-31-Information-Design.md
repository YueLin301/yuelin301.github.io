---
title: Information Design (A Note)
date: 2022-05-31 20:00:01 +0800
categories: [Economics & Game Theory]
tags: [information design, bayesian persuasion, bayes plausible, concavification, obedience, bayes correlated equilibrium, bayes nash equilibrium]
---

 > This note is not finished yet. 
{: .prompt-warning }

 
## What's information design?

> "Sometimes, the truth is not good enough." - Batman, The Dark Knight (2008)
{: .prompt-info }

Communication does not just happen in fully cooperative scenarios. 
In some cases, the sender can persuade the receiver by "strategically deceiving", to increase its own expected payoff. 
Actually, "one quarter of gdp is persuasion." (McCloskey & Klamer 1995). (e.g. )

Nontrivially, "successful partially deceiving" is a better equilibrium than "saying nothing" and "revealing all information".
Information design is the study of this phenomenon of persuasion. And Bayesian persuasion is a special case of information design, which consists of a sender and a receiver.

 > An ethical justification: I do not think information design is immoral. Information is a kind of property of the sender, and it is legal for it to profit from its information.  Furthermore, in those cases where the sender can improve its own expected payoff through information design, the receiver's payoff is not worse than that of the sender not reveal information at all. Nevertheless, practice use of information design should take the sender's objective function into some serious consideration.
{: .prompt-tip }




## Papers
The following part of this note is to summarize the essence of these papers:
1. Bayesian Persuasion (Kamenica & Gentzkow 2011)
2. Bayes Correlated Equilibrium and the Comparison of Information Structures in Games (Bergemann & Morris 2016)
3. Surveys:
   1. Bayesian Persuasion and Information Design (Kamenica 2019)
   2.  Algorithmic Information Structure Design: A survey (Dughmi 2019)

<!-- ## Notations
The notations in this note are adapted from the common RL papers.

| Meaning          | Symbol in this note | Symbol(s) in other papers |
| ---------------- | ------------------- | ------------------------- |
| state            | $s\in S$            | $\omega$,                 |
| signal (message) |                     |                           |
| action           |                     |                           | -->



## Timing
Consider a persuasion between a sender and a receiver (named as Bayesian persuasion). The timing of it is:
1. The sender chooses a signaling scheme $\varphi$. (Given a state $s$, $\varphi(\cdot|s)$ is a distribution of $\sigma$.)
2. The receiver observes this $\varphi$.
3. The environment generates a state $s$.
4. A signal $\sigma$ is sampled accroding to the commited $\varphi(\cdot|s)$.
5. The receiver calculates its posterior belief $\mu(\cdot | \sigma)$, and then chooses an action $a$ that maximizes $\mathbb{E}_{\mu}(r^j)$. (Thus the term Bayesian persuasion.)
6. The sender and the receiver get rewards $r^i(s,a)$ and $r^j(s,a)$, respectively.

## Examples
1. Recommendation Letter (Example 2.1 in Dughmi 2019)
2. Courtroom (The first example in Section 2.2 of Kamenica 2019)


## Assumptions
1. The sender and the receiver are rational.
2. The receiver's strategy is based on Bayes' rule. (Timing 5)
3. Commitment: The sender will *honestly* commit a signaling scheme to the receiver *before* the interaction with the receiver. (Timing 1-2)
   1. It is this that makes Bayesian persuasion (or information design) different from other communication models, e.g. cheap talk, verifiable message, signaling games. (Kamenica 2019)
   2. Justifications:
      1. Reputation
      2. (Justifications vary across applications)
4. An analysis analogues to the revelation principle: The optimal scheme needs no more signals than the number of states of nature.


## Properties and geometric interpretations
1. Bayes plausible: If an arbitrary $\tau$ satisfies $\mathbb{E}_{\mu\sim\tau}(\mu) = \mu_0$, then this $\tau$ is Bayes plausible. (Kamenica 2019)
   1. Every $\tau_{\varphi}$ (a $\tau$ induced by the $\varphi$) is Bayes plausible. (It can be proved by the law of iterated expectations. Kamenica 2019)
   2. If a $\tau$ is Bayes plausible, then it can be induced by a $\varphi$. (Kamenica 2019. Proved in Kamenica & Gentzkow 2011)
   3. Probability simplex: 
2. Concavification
   1. Special cases:
      1. If reward functions are identical (i.e. $r^i = r^j$), then the sender's objective function is convex. The optimal signaling scheme is to reveal all the information.
      2. If $(r^i+r^j)(s,a) = k, \forall s,a$, where $k \in \mathbb{R}$, then the sender's objective function is concave. The optimal signaling scheme is to reveal nothing (In this case, $\mu = \mu_0$). 


## Extensions
1. Multiple senders
2. Multiple receivers
3. Dynamic environment
4. Others
   1. The receiver has private information
   2. 