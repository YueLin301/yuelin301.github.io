---
title: Zero-Determinant Strategy
date: 2023-08-30 02:40:00 +0800
categories: [Multi-Agent Reinforcement Learning]
tags: [game theory, sequential social dilemma, multi agents]
math: True
---

> This note aims to summarize the essence of this paper:  
> [Press, William H., and Freeman J. Dyson. "Iterated Prisoner’s Dilemma contains strategies that dominate any evolutionary opponent." Proceedings of the National Academy of Sciences 109.26 (2012): 10409-10413.](https://www.pnas.org/doi/pdf/10.1073/pnas.1206569109)
{: .prompt-info }


## Interesting Facts

1. As stated in the title: **Iterated Prisoner’s Dilemma contains strategies that dominate any evolutionary opponent.** And this kind of strategy is the Zero-Determinant strategy.
2. The "Iterated Prisoner’s Dilemma is an Ultimatum game".
3. "One player can enforce a unilateral claim to an unfair share of rewards", by the ZD strategy.
4. Any evolutionary agent will be exploited by the agent with the ZD strategy. "An evolutionary player’s best response is to accede to the extortion. Only a player with a theory of mind about his opponent can do better."
5. "For any strategy of the longer-memory player Y, shorter-memory X’s score is exactly the same as if Y had played a certain shorter-memory strategy."

---

## Iterated Prisoner's Dilemma

At each timestep, two agents are playing Prisoner's Dilemma:

| Player1\Player2      | Cooperate (Deny) | Defect (Confess) |
| -------------------- | ---------------- | ---------------- |
| **Cooperate (Deny)** | $R,R$            | $S,T$            |
| **Defect (Confess)** | $T,S$            | $P,P$            |

where $T > R > P > S$, and the meanings are as follows.
- $T$: Temptation
- $R$: Reward
- $P$: Punishment
- $S$: Sucker's payoff

The two agents repeatedly play this game $T$ times. It might be finite or infinite.

---

## Longer-memory Strategies Offer No Advantage



> The following part has not been finished yet. One may check my [writing schedule](https://yuelin301.github.io/posts/Schedule/).
{: .prompt-warning }

