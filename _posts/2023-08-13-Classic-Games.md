---
title: Classic Games
date: 2023-08-13 16:30:00 +0800
categories: [Economics & Game Theory]
tags: [game theory, matrix game]
math: True
---


> This note will be consistently updated.
{: .prompt-info }

## [Prisoner's Dilemma](https://en.wikipedia.org/wiki/Prisoner%27s_dilemma)

> Two members of a criminal organization are arrested and imprisoned. Each prisoner is in solitary confinement with no means of communicating with the other. The prosecutors lack sufficient evidence to convict the pair on the principal charge, but they have enough to convict both on a lesser charge. Simultaneously, the prosecutors offer each prisoner a bargain. Each prisoner is given the opportunity either to betray the other by testifying that the other committed the crime, or to cooperate with the ot	her by remaining silent.

> Poundstone, William. Prisoner's dilemma: John von Neumann, game theory, and the puzzle of the bomb. Anchor, 1993.
{: .prompt-info }

### Normal form

| Player1\Player2      | Cooperate (Deny) | Defect (Confess) |
| -------------------- | ---------------- | ---------------- |
| **Cooperate (Deny)** | $b,b$            | $d,a$            |
| **Defect (Confess)** | $a,d$            | $c,c$            |

$a \gt b \gt c \gt d$.

Attractor of player1: Reversed N-like (or N-like)

| $b\downarrow$ | $d\downarrow$   |
| -----------   | -----------     |
| $a$           | $c(\nwarrow)$   |

Attractor of player2:

| $b\rightarrow$   | $a$             |
| -----------      | -----------     |
| $d\rightarrow$   | $c(\nwarrow)$   |

The "confess" action is the dominant one.

### Nash equilibrium

Can be found by iterated elimination of strictly dominated strategies (**IESDS**). For player1: check the 1st num up-and-down. For player2: check 2nd num left-and-right.

| Player1\Player2      | Cooperate (Deny)               | Defect (Confess)              |
| -------------------- | ------------------------------ | ----------------------------- |
| **Cooperate (Deny)** | $b(\downarrow),b(\rightarrow)$ | $d(\downarrow),a(\checkmark)$ |
| **Defect (Confess)** | $a(\checkmark),d(\rightarrow)$ | $c(\checkmark),c(\checkmark)$ |

### Pareto optimal 

(check 3 times for each state)

| Player1\Player2      | Cooperate (Deny)         | Defect (Confess)          |
| -------------------- | ------------------------ | ------------------------- |
| **Cooperate (Deny)** | $b,b\ldots (\checkmark)$ | $d,a\ldots (\checkmark)$  |
| **Defect (Confess)** | $a,d\ldots (\checkmark)$ | $c,c \ldots(\nwarrow)$    |


---



## Stag Hunt

> In the simple, matrix-form, two-player Stag Hunt each player makes a choice between a risky action (hunt the stag) and a safe action (forage for mushrooms). Foraging for mushrooms always yields a safe payoff while hunting yields a high payoff if the other player also hunts but a very low payoff if one shows up to hunt alone.

> - Peysakhovich, Alexander, and Adam Lerer. "Prosocial Learning Agents Solve Generalized Stag Hunts Better than Selfish Ones." Proceedings of the 17th International Conference on Autonomous Agents and MultiAgent Systems. 2018.
> - Harsanyi, John C., and Reinhard Selten. "A general theory of equilibrium selection in games." MIT Press Books 1 (1988).
{: .prompt-info }

### Normal form

| Player1\Player2      | Cooperate (Hunt) | Defect (Forage) |
| -------------------- | ---------------- | --------------- |
| **Cooperate (Hunt)** | $a,a$            | $d,b$           |
| **Defect (Forage)**  | $b,d$            | $c,c$           |

$a \gt b \ge c \gt d$.

Attractor of player1: Reversed u-like (or n-like)

| $a$           | $d\downarrow$   |
| ------------- | -----------     |
| $b\uparrow$   | $c(\leftarrow)$ |

Attractor of player2:

| $a$           | $b\leftarrow$   |
| ------------- | -----------     |
| $d\rightarrow$   | $c(\uparrow)$ |

### Nash equilibrium

| Player1\Player2      | Cooperate (Hunt)              | Defect (Forage)               |
| -------------------- | ----------------------------- | ----------------------------- |
| **Cooperate (Hunt)** | $a(\checkmark),a(\checkmark)$ | $d(\downarrow),b(\leftarrow)$ |
| **Defect (Forage)**  | $b(\uparrow),d(\rightarrow)$  | $c(\checkmark),c(\checkmark)$ |

### Pareto optimal

| Player1\Player2      | Cooperate (Hunt)        | Defect (Forage)         |
| -------------------- | ----------------------- | ----------------------- |
| **Cooperate (Hunt)** | $a,a\ldots(\checkmark)$ | $d,b\ldots(\leftarrow)$ |
| **Defect (Forage)**  | $b,d\ldots(\uparrow)$   | $c,c\ldots(\nwarrow)$   |



---



## Snowdrift

> The name Snowdrift game refers to the situation of two drivers caught with their cars in a snow drift. If they want to get home, they have to clear a path. The fairest solution would be for both of them to start shoveling (we assume that both have a shovel in their trunk). But suppose that one of them stubbornly refuses to dig. The other driver could do the same, but this would mean sitting through a cold night. It is better to shovel a path clear, even if the shirker can profit from it without lifting a finger.

> Sigmund, Karl. The calculus of selfishness. Princeton University Press, 2010.
{: .prompt-info }

This game has the same payoff pattern as the games **Chicken** and **Hawk–Dove**. Alternatively, I could say that these two games share the same **potential game**?

### Normal form

| Player1\Player2      | Cooperate        | Defect           |
| -------------------- | ---------------- | ---------------- |
| **Cooperate**        | $b,b$            | $c,a$            |
| **Defect**           | $a,c$            | $d,d$            |

$a \gt b \gt c \gt d$.

Attractor of player1: Reversed n-like (or u-like)

| $b\downarrow$ | $c(\leftarrow)$ |
| ------------  | -------------   |
| $a$           | $d\uparrow$     |

Attractor of player2:

| $b\rightarrow$ | $a$             |
| ------------   | -------------   |
| $c(\uparrow)$  | $d\leftarrow$   |

### Nash equilibrium

| Player1\Player2      | Cooperate                      | Defect                        |
| -------------------- | ------------------------------ | ----------------------------- |
| **Cooperate**        | $b(\downarrow),b(\rightarrow)$ | $c(\checkmark),a(\checkmark)$ |
| **Defect**           | $a(\checkmark),c(\checkmark)$  | $d(\uparrow),d(\leftarrow)$   |

### Pareto optimal

| Player1\Player2      | Cooperate               | Defect                                  |
| -------------------- | ----------------------- | --------------------------------------- |
| **Cooperate**        | $b,b\ldots(\checkmark)$ | $c,a\ldots(\checkmark)$                 |
| **Defect**           | $a,c\ldots(\checkmark)$ | $d,d\ldots(\leftarrow\nwarrow\uparrow)$ |


---

## Battle of the Sexes
Also known as **Bach or Stravinsky**.


---


## Matching Pennies

It is zero-sum (at each entry). Players are fully competitive.

### Normal form

| Player1\Player2      | Heads            | Tails            |
| -------------------- | ---------------- | ---------------- |
| **Heads**            | $1,-1$           | $-1,1$           |
| **Tails**            | $-1,1$           | $1,-1$           |

### Nash equilibrium

No pure strategy works.

| policy   |                     | q         | 1-q       |
| -------- | ---------           | --------  | --------- |
|          | **Player1\Player2** | **Heads** | **Tails** |
| **p**    | **Heads**           | $1,-1$    | $-1,1$    |
| **1-p**  | **Tails**           | $-1,1$    | $1,-1$    |

If player1 chooses "Heads", its expected payoff is

$$
\mathbb{E}\left(r^1\mid a^1 = \text{``Heads''}\right) = q + (-1)\cdot (1-q) = 2q-1.
$$

And if it chooses "Tails", its expected payoff is $(-2q+1)$. It should be indifferent about playing heads or tails, otherwise it can improve its expected payoff by increasing the probability of the action that causes higher expected payoff. In this way,

$$
\begin{aligned}
    \quad & \mathbb{E}\left(r^1\mid a^1 = \text{``Heads''}\right) = 
        \mathbb{E}\left(r^1\mid a^1 = \text{``Tails''}\right) \\
    \Rightarrow \quad & 2q-1 = -2q+1 \\
    \Rightarrow \quad & q = 0.5
\end{aligned}
$$



##  Rock paper scissors

It is zero-sum (at each entry). Players are fully competitive.

### Normal form

| Player1\Player2  | Rock     | Paper   | Scissors |
| ---------------- | -------- | ------- | -------- |
| **Rock**         | $0,0$    | $-1,1$  | $1,-1$   |
| **Paper**        | $1,-1$   | $0,0$   | $-1,1$   |
| **Scissors**     | $-1,1$   | $1,-1$  | $0,0$    |

### Nash equilibrium

No pure strategy works.

$$
\begin{aligned}
    \quad & \mathbb{E}\left(r^1\mid a^1 = \text{``Rock''}\right) = 
        \mathbb{E}\left(r^1\mid a^1 = \text{``Paper''}\right) = 
        \mathbb{E}\left(r^1\mid a^1 = \text{``Scissors''}\right) \\
    \Rightarrow \quad & q = \frac{1}{3}.
\end{aligned}
$$


## Muddy Children Puzzle

Check my other note: [Dynamic Epistemic Logic](https://yuelin301.github.io/posts/Dynamic-Epistemic-Logic/#the-muddy-children-puzzle).


## Trust
> 1. In the first stage, the Donor (or Investor) receives a certain endowment by the experimenter, and can decide whether or not to send a part of that sum to the Recipient (or Trustee), knowing that the amount will be tripled upon arrival: each euro spent by the Investor yields three euros on the Trustee’s account.
> 2. In the second stage, the Trustee can return some of it to the Investor’s account, on a one-to-one basis: it costs one euro to the Trustee to increase the Investor’s account by one euro.
> 3. This ends the game. Players know that they will not meet again.

## Ultimatum
> The experimenter assigns a certain sum, and the Proposer can offer a share of it to the Responder. If the Responder (who knows the sum) accepts, the sum is split accordingly between the two players, and the game is over. If the Responder declines, the experimenter withdraws the money. Again, the game is over: but this time, neither of the two players gets anything.

## Braess's Paradox
A computational example about Braess's Paradox is in `Figure 18.2` of [this chapter](https://timroughgarden.org/papers/rg.pdf):

> Roughgarden, Tim. "Routing games." Algorithmic game theory 18 (2007): 459-484.
{: .prompt-info }

## [Iterated Prisoner's Dilemma](https://en.wikipedia.org/wiki/Prisoner%27s_dilemma#The_iterated_prisoner's_dilemma)

> Axelrod, Robert, and William D. Hamilton. "The evolution of cooperation." science 211.4489 (1981): 1390-1396.
{: .prompt-info }

### Successful strategy conditions

> Stated by Axelrod. The following conditions are adapted from [Wikipedia]((https://en.wikipedia.org/wiki/Prisoner%27s_dilemma#Axelrod's_contest_and_successful_strategy_conditions)).

- Nice/optimistic: The strategy will not defect before its opponent does.
- Retaliating: The strategy must sometimes retaliate. Otherwise it will be exploited by the opponent.
- Forgiving: Though players will retaliate, they will cooperate again if the opponent does not continue to defect.
- Non-envious: The strategy must not strive to score more than the opponent.



### Strategies
- Tit-for-tat.
- Win-stay, lose-switch.
- [Zero-determinant strategy](https://www.pnas.org/doi/pdf/10.1073/pnas.1206569109).

### Zero-determinant strategy





> Disclaimer: The description of games is from Wikipedia and other sources (books or papers). Corresponding links or references have been provided. The content regarding attractors is my personal understanding.
{: .prompt-danger }