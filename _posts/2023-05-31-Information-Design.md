---
title: Information Design (A Note)
date: 2022-05-31 20:00:01 +0800
categories: [Economics & Game Theory]
tags: [information design, bayesian persuasion, bayes plausible, concavification, obedience, bayes correlated equilibrium, bayes nash equilibrium]
math: True
---

 > This note is not finished yet. 
{: .prompt-warning }

 
## What is information design?

> "Sometimes, the truth is not good enough." - Batman, The Dark Knight (2008)
{: .prompt-info }

Communication does not just happen in fully cooperative scenarios. 
In some cases, the sender can persuade the receiver by "strategically deceiving", to increase its own expected payoff. 
Actually, "one quarter of gdp is persuasion." (McCloskey & Klamer 1995).

Nontrivially, "successful partially deceiving" is a better equilibrium than "saying nothing" and "revealing all information".
Information design is the study of this persuasion. And Bayesian persuasion is a special case of information design, which consists of a sender and a receiver.


## Papers
The following part of this note is to summarize the essence of these papers:
1. [Bayesian Persuasion](https://www.aeaweb.org/articles?id=10.1257/aer.101.6.2590) (Kamenica & Gentzkow 2011)
2. [Bayes Correlated Equilibrium and the Comparison of Information Structures in Games](https://onlinelibrary.wiley.com/doi/abs/10.3982/TE1808) (Bergemann & Morris 2016)
3. Surveys:
   1. [Bayesian Persuasion and Information Design](https://www.annualreviews.org/doi/abs/10.1146/annurev-economics-080218-025739) (Kamenica 2019)
   2.  [Algorithmic Information Structure Design: A survey](https://dl.acm.org/doi/abs/10.1145/3055589.3055591) (Dughmi 2019)

<!-- ## Notations
The notations in this note are adapted from the common RL papers.

| Meaning          | Symbol in this note | Symbol(s) in other papers |
| ---------------- | ------------------- | ------------------------- |
| state            | $s\in S$            | $\omega$,                 |
| signal (message) |                     |                           |
| action           |                     |                           | -->



## Timing
Consider a persuasion between a sender and a receiver (named as Bayesian persuasion). The timing of it is:
1. The sender chooses a signaling scheme $\varphi$. (Given a state $s$, $\varphi(\cdot \mid s)$ is a distribution of $\sigma$.)
2. The receiver observes this $\varphi$.
3. The environment generates a state $s$.
4. A signal $\sigma$ is sampled accroding to the commited $\varphi(\cdot\mid s)$.
5. The receiver calculates its posterior belief $\mu(\cdot \mid \sigma)$, and then chooses an action $a$ that maximizes $\mathbb{E}_{\mu}(r^j)$. (Thus the term Bayesian persuasion.)
6. The sender and the receiver get rewards $r^i(s,a)$ and $r^j(s,a)$, respectively.

 > The core of information design is to find the optimal signaling scheme $\varphi$ for the sender and to study the nature of this problem. But the sender's payoff is determined by the receiver's action, so it has to use its information advantage to influence the receiver's behavior (by designing $\varphi$), thereby indirectly optimizing its own expected payoff.
{: .prompt-tip }



## Important examples
1. Recommendation Letter (Example 2.1 in Dughmi 2019)
2. Courtroom (The first example in Section 2.2 of Kamenica 2019)


## Assumptions
1. The receiver is self-interested.
   - The receiver has its objective function to optimize (In my understanding it does not need to be consistent with its epected environmental reward), and the sender wants to influence the receiver's behavior.
   - The sender's objective function does not need to be consistent with its environmental reward, e.g., it can be social welfare $r^i+r^j$ (in this way, the sender is altruistic).
   - Anyway, there is no limit to the goals of the sender and the receiver. They may be fully cooperative, mixed motived or adversarial.
2. The receiver's strategy is based on Bayes' rule. (Timing 5)
3. **Commitment**: The sender will *honestly* commit a signaling scheme to the receiver *before* the interaction with the receiver. (Timing 1-2)
   - It is this that makes Bayesian persuasion (or information design) different from other communication models, e.g. cheap talk, verifiable message, signaling games. (Kamenica 2019)
   - Justifications:
      - Reputation
      - (Justifications vary across applications)
4. An analysis analogues to the **revelation principle**: The optimal scheme needs no more signals than the number of states of nature.


## Properties and geometric interpretations
1. **Bayes plausible**: If an arbitrary $\tau$ satisfies $\mathbb{E}_{\mu\sim\tau}(\mu) = \mu_0$, then this $\tau$ is Bayes plausible. (Kamenica 2019)
   - A necessary and sufficient condition:
     - Every $\tau_{\varphi}$ (a $\tau$ induced by the $\varphi$) is Bayes plausible. (It can be proved by the law of iterated expectations. Kamenica 2019)
     - If a $\tau$ is Bayes plausible, then it can be induced by a $\varphi$. (Kamenica 2019. Proved in Kamenica & Gentzkow 2011)
   - Probability simplex: 
     - aaa
2. **Concavification**
   - Special cases:
      - If reward functions are identical (i.e. $r^i = r^j$), then the sender's objective function is convex. The optimal signaling scheme is to reveal all the information.
      - If $(r^i+r^j)(s,a) = k, \forall s,a$, where $k \in \mathbb{R}$, then the sender's objective function is concave. The optimal signaling scheme is to reveal nothing (In this case, $\mu = \mu_0$). 


## Extensions
1. Multiple senders
2. Multiple receivers
3. Dynamic environment
4. Others
   1. The receiver has private information

## Closing Remarks

 > An ethical justification: I do not think information design is immoral. Information is a kind of property of the sender, and it is legal for it to profit from its information.  Furthermore, in those cases where the sender can improve its own expected payoff through information design, the receiver's payoff is not worse than that of the sender not reveal information at all. Nevertheless, practice use of information design should take the sender's objective function into some serious consideration.
{: .prompt-tip }