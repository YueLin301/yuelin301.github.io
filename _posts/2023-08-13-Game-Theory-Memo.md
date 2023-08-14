---
title: A Memo on Game Theory
date: 2023-08-13 014:30:00 +0800
categories: [Economics & Game Theory]
tags: [game theory]
math: True
---


> This note will be consistently updated.
{: .prompt-info }

## Rationality
A rational player is one who chooses his action, to maximize his payoff consistent with his beliefs about what is going on in the game.[^intro-book]

## Normal-form game

**Definition 3.3**[^intro-book] A **normal-form game** includes three components as follows:
1. A finite **set of players**, $N = \set{1,2,\ldots,n}$.
2. A **collection of sets of pure strategies**, $\set{S_1, S_2, \ldots, s_n}$.
3. A **set of payoff functions**, $\set{v_1,v_2,\ldots, v_n}$, each assigning a payoff value to each combination of chosen strategies, that is, a set of functions $v_i: S_1\times S_2, \times \dots \times S_n \to \mathbb{R}$ for each $i\in N$.

Each of the players $i\in N$ must simultaneously choose a possible strategy $s_i\in S_i$.[^intro-book]

> In my understanding, it's like $(I,\set{A^i}\_{i\in I}, \set{R^i}\_{i\in I})$ in the notation of MARL.
{: .prompt-tip }

## Information Set


## Extensive-Form Games

> Often credit this paper:  
> H. W. Kuhn. Extensive games and the problem of information. Contributions to the Theory of Games, 2:193–216, 1953.
{: .prompt-info }

## Information: Complete v.s. Perfect

> I've looked up these two concepts so many times already, but every time I read about them, I forget shortly afterward. 😅
{: .prompt-tip }

### Complete Information
> Tadelis, Steven. Game theory: an introduction. Princeton university press, 2013.
{: .prompt-info }

> A game of complete information requires that the following four components be common knowledge among all the players of the game:
> 1. all the possible actions of all the players,
> 2. all the possible outcomes,
> 3. how each combination of actions of all players affects which outcome will materialize, and
> 4. the preferences of each and every player over outcomes.

> **Preferences** describe how the player ranks the set of possible outcomes, from most desired to least desired.

> In my understanding, a game is with complete information **if the utility functions (with their domain) are common knowledge.** Generally, rationality is a fundamental assumption, so the fourth point in the definition is usually satisfied.
{: .prompt-tip }

---

Each player has full information about others. Common knowledge includes[^wiki-complete]
- utility functions (including risk aversion), 
- payoffs,
- strategies, and
- "types" of players ("private" information).

> I'm not sure what strategies being common knowledge looks like. And it reads that Chess is with incomplete information because of this. I think there is a conflict with the definition in the book.
{: .prompt-tip }

But
- players may not know the others' actions (e.g. the initial placement of ships in  **Battleship**), and
- the game may has chance element (card games).

### Perfect Information

> Tadelis, Steven. Game theory: an introduction. Princeton university press, 2013.
{: .prompt-info }

> **Definition 7.3** A game of complete information in which every information set is a **singleton** and there are no moves of Nature is called a **game of perfect information.**

Each player is perfectly informed of all the events that have previously occurred.[^wiki-perfect]


### Examples

Perfect and complete:
- Chess, Tic-Tac-Toe, Go.

Perfect but incomplete:
- Bayesian game.

Complte but imperfect:
- Card games, where each player's cards are hidden from other players but objectives are known. The dealing event is not public (imperfect). 




## Public Goods v.s. Common Recourses


---

## References

[^intro-book]: Tadelis, Steven. Game theory: an introduction. Princeton university press, 2013.
[^wiki-complete]: [Wikipedia: Complete Information](https://en.wikipedia.org/wiki/Complete_information)
[^wiki-perfect]: [Wikipedia: Perfect Information](https://en.wikipedia.org/wiki/Perfect_information)


> Disclaimer: The above content is summarized from Wikipedia and other sources. Corresponding links or references have been provided.
{: .prompt-danger }