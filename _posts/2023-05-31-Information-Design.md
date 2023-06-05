---
title: Information Design (A Note)
date: 2022-05-31 20:00:01 +0800
categories: [Economics & Game Theory]
tags: [information design, bayesian persuasion, bayes plausible, concavification, obedience, bayes correlated equilibrium, bayes nash equilibrium]
math: True
pin: True
---

 > This note is not finished yet. 
{: .prompt-warning }

 
## What is Information Design?

> "Sometimes, the truth is not good enough." *- Batman, The Dark Knight (2008)*
{: .prompt-info }

Communication does not just happen in fully cooperative scenarios. 
In some cases, the sender can persuade the receiver by "strategically deceiving", to increase its own expected payoff. 
Actually, "one quarter of gdp is persuasion." *(McCloskey & Klamer 1995)*.

Nontrivially, "successful partially deceiving" is a better equilibrium than "saying nothing" and "revealing all information".
Information design is the study of this persuasion. And Bayesian persuasion is a special case of information design, which consists of a sender and a receiver.

The problem of information design can be equivalently approached from various perspectives:
1. $\max_{\varphi} \mathbb{E}_{s \sim \mu_0}\left[\mathbb{E}_{\sigma \sim \varphi(\cdot\mid s)}\left[r^i(s,a^*(\sigma))\right]\right]$
2. $\max_{\varphi} \mathbb{E}_{\mu(\cdot\mid\sigma)\sim\tau}\Big[\mathbb{E}_{s\sim\mu(\cdot\mid\sigma)}\big[r^i(s,a^*(\sigma))\big]\Big]$, s.t. $\mathbb{E}_{\mu(\cdot\mid\sigma)\sim\tau}(\mu(\cdot\mid\sigma)) = \mu_0$.
3. In a two-signal case: The sender manipulates the receiver's posterior beliefs (each corresponding to a sent signal) to find the **highest intersection point** of the line segment $(\mu_1 - \mu_2, \hat{v}_1 - \hat{v}_2)$ and $x = \mu_0$
4. The sender select an optimal Bayes correlated equilibrium given an objective function.

## Papers
The following part of this note is to summarize the essence of these papers:
1. [Bayesian Persuasion](https://www.aeaweb.org/articles?id=10.1257/aer.101.6.2590) *(Kamenica & Gentzkow 2011)*
2. [Bayes Correlated Equilibrium and the Comparison of Information Structures in Games](https://onlinelibrary.wiley.com/doi/abs/10.3982/TE1808) *(Bergemann & Morris 2016)*
3. Surveys:
   1. [Bayesian Persuasion and Information Design](https://www.annualreviews.org/doi/abs/10.1146/annurev-economics-080218-025739) *(Kamenica 2019)*
   2.  [Algorithmic Information Structure Design: A Survey](https://dl.acm.org/doi/abs/10.1145/3055589.3055591) *(Dughmi 2019)*

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



## Important Examples
1. Recommendation Letter *(Example 2.1 in Dughmi 2019)*.
2. Courtroom *(The first example in Section 2.2 of Kamenica 2019)*.
3. Routing Game *(Example 3.2 in Dughmi 2019)* or [Braess's paradox](https://en.wikipedia.org/wiki/Braess%27s_paradox).


## Assumptions
1. The prior $\mu_0$ of states is common knowledge.
2. The receiver is self-interested.
   - The receiver has its objective function to optimize (In my understanding it does not need to be consistent with its epected environmental reward), and the sender wants to influence the receiver's behavior.
   - The sender's objective function does not need to be consistent with its environmental reward, e.g., it can be social welfare $r^i+r^j$. (In this way, the sender is altruistic.)
   - Anyway, there is no limit to the goals of the sender and the receiver. They may be fully cooperative, mixed motived or adversarial.
3. The receiver's strategy is based on Bayes' rule. (Timing 5)
4. **Commitment**: The sender will *honestly* commit a signaling scheme to the receiver *before* the interaction with the receiver. (Timing 1-2)
   - It is this that makes Bayesian persuasion (or information design) different from other communication models, e.g. cheap talk, verifiable message, signaling games. *(Kamenica 2019)*
   - Justifications:
      - Reputation
      - The signaling schemes are common knowledge, e.g., the sender (prosecutor) can choose investigative means (forensic tests, calling witness, etc.) to partially reveal the truth, which are public to everyone. *(Courtroom)*
      - (Justifications vary across applications. See *Section 2.2 in Kamenica 2019*)
5. An analysis analogues to the **revelation principle**: The optimal scheme needs no more signals than the number of states of nature.


## Simplification and Geometric Interpretations
### Reformulation
- The distribution of posterior beliefs $\tau$
   - **Every sent signal induces a specific posterior belief**: Given a committed signaling scheme $\varphi$ and a sent signal $\sigma$, the receiver's posterior belief is $\mu(\cdot\mid\sigma)$. Calculated as $\mu(s_i \mid\sigma) = \frac{\mu_0(s_i)\cdot \varphi(\sigma\mid s_i)}{\sum\limits_{s_j}\mu_0(s_j)\cdot \varphi(\sigma\mid s_j)}$. 
   - The mapping from signals to posterior beliefs is **many-to-one**.
   - **A distribution of signals corresponds to a distribution of posterior beliefs**: Before the signal $\sigma$ realized, the receiver can only estimate its distribution by the committed signaling scheme: $\sigma\sim p_{\mu_0, \varphi}(\cdot)  = \sum\limits_{s}\mu_0(s)\cdot \varphi(\cdot\mid s)$.
   - A $\tau$ induced by a $\varphi$ is denoted as $\tau_{\mu_0,\varphi}$. And $\tau_{\mu_0,\varphi}(\mu(\cdot\mid\sigma)) = p_{\mu_0, \varphi}(\cdot)  = \sum\limits_{s}\mu_0(s)\cdot \varphi(\cdot\mid s)$.
- The sender's optimization problem
   - The receiver's strategy: 
     - Given a sent $\sigma$, its posterior belief is $\mu(\cdot\mid \sigma)$.
     - Then it chooses $a^*(\sigma) = \argmax_{a}\mathbb{E}_{s\sim\mu(\cdot\mid\sigma)}\left[r^j(s,a)\right]$.
   - Original optimization: $\max_{\varphi} \mathbb{E}_{s\sim\mu_0}\Big[\mathbb{E}_{\sigma\sim\varphi(\cdot\mid s)}\big[r^i(s,a^*(\sigma))\big]\Big]$.
   - Reformulation:
     - An equivalent objective function: $\mathbb{E}_{\mu(\cdot\mid\sigma)\sim\tau}\Big[\mathbb{E}_{s\sim\mu(\cdot\mid\sigma)}\big[r^i(s,a^*(\sigma))\big]\Big]$.
     - The only contraint: $\mathbb{E}_{\mu(\cdot\mid\sigma)\sim\tau}(\mu(\cdot\mid\sigma)) = \mu_0$. (i.e. $\tau = \tau_{\mu_0,\varphi}$. This is named as Bayes plausible. See the next subsection.)

### Bayes plausible
 > The sender's optimization problem can be approached from the perspective of **any arbitrary** distribution of induced posterior beliefs **that satisfies the Bayes plausible**. Understanding this is important for grasping the concepts of geometric interpretation (concavification) in the next subsection.
{: .prompt-tip }

If an arbitrary $\tau$ satisfies $\mathbb{E}_{\mu\sim\tau}(\mu) = \mu_0$, then this $\tau$ is Bayes plausible. (Kamenica 2019)
- What is necessary and sufficient:
  - Every $\tau_{\mu_0,\varphi}$ (a $\tau$ induced by the $\varphi$) is Bayes plausible. (It can be proved by the law of iterated expectations. *Kamenica 2019*)
  - If a $\tau$ is Bayes plausible, then it can be induced by a $\varphi$. *(Kamenica 2019. Proved in Kamenica & Gentzkow 2011.)*
- A geometric interpretation: probability simplex *(Figure 2 of Dughmi 2019)*. 
  - A point in the probability simplex represents a distribution of singals (a convex combination of signals). 
  - The blue point is the prior belief. The red points are posterior beliefs. 
  - Bayes plausible = The blue point is inside the simplex of the red points.


### Concavification
 > Does the sender benefit from persuasion? What is an optimal mechanism?
{: .prompt-tip }

This technique is best described with a **two-signal** example *(Figure 1 of Kamenica 2011, 2019)*:
- Assume that there are $n$ states. Then a posterior belief $\mu$ is a point in $\mathbb{R}^{n-1}$. *(2011)*
- The values of $x$-axis are in $\mathbb{R}^{n-1}$. Each represents a posterior belief $\mu$. *(2011)*
- A $\mu$ induces an expected payoff of the sender $\hat{v}(\mu(\cdot\mid\sigma)) = \mathbb{E}_{s\sim\mu(\cdot\mid\sigma)}\big[r^i(s,a^*(\sigma))\big]$. The black line in the figure denotes this function. *(2011,2019)*
- $\sigma_1$ and $\sigma_2$ induce posterior beliefs $\mu_1(\cdot\mid\sigma_1)$ and $\mu_2(\cdot\mid\sigma_2)$ respectively. These two $\mu$ correspond to two values on the $x$-axis, and indicate two expected payoff of the sender $\hat{v}_1$ and $\hat{v}_2$. *(2019)*
- $\mu_1$ and $\mu_2$ are **almost arbitrary**: The distribution of $\mu_1$ and $\mu_2$ should be Bayes plausible, i.e., their expectation should be $\mu_0$. 
  - Assume that the distribution of $\mu$ is $\tau = (k, 1-k)$. Then $k\cdot\mu_1+(1-k)\cdot\mu_2 = \mu_0$. Thus if $\mu_{i}$ is on the left side of $\mu_0$ on the $x$-axis, then $\mu_{1-i}$ is on the right side. Unless they both coincide with $\mu_0$.
  - Also, $\mathbb{E}_{\mu\sim\tau}(\hat{v}(\mu)) = k\cdot \hat{v}_1+(1-k)\cdot\hat{v}_2$, which is the sender's expected payoff (obejective). It means that: When the receiver's posterior beliefs are $\mu_1,\mu_2$ (after receiving $\sigma_1, \sigma_2$ respectively), the sender will gets $\mathbb{E}_{\mu\sim\tau}(\hat{v}(\mu))$.
  - Since the coordinates are proportional, connecting $(\mu_1, \hat{v}_1)$ and $(\mu_2, \hat{v}_2)$ to get a line segment $l$, $(\mu_0, \mathbb{E}_{\mu\sim\tau}(\hat{v}(\mu)))$ is on $l$. In this way, the sender needs to manipulate the receiver's posterior beliefs to find the **highest intersection point**.

 > Please re-read the previous part of this section to make sure you understand this important example.
{: .prompt-tip }

Special situations *(Dughmi 2019)*:
  - If reward functions are identical (i.e. $r^i = r^j$), then the sender's objective function is convex. The optimal signaling scheme is to reveal all the information.
  - If $(r^i+r^j)(s,a) = k, \forall s,a$, where $k \in \mathbb{R}$, then the sender's objective function is concave. The optimal signaling scheme is to reveal nothing (In this case, $\mu = \mu_0$). 

## From an Equilibrium Perspective
1. Obedience
2. Bayesian correlated equilibrium
3. Bayesian Nash equilibrium


## Extensions
1. Multiple senders
2. Multiple receivers
3. Dynamic environment
4. Others:
   1. The receiver has private information

## Closing Remarks

 > An ethical justification: I do not think information design is immoral. Information is a kind of property of the sender, and it is legal for it to profit from its information.  Furthermore, in those cases where the sender can improve its own expected payoff through information design, the receiver's payoff is not worse than that of the sender not reveal information at all. Nevertheless, practice use of information design should take the sender's objective function into some serious consideration.
{: .prompt-tip }