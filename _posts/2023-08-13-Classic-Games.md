---
title: Classic Games
date: 2023-08-13 016:30:00 +0800
categories: [Economics & Game Theory]
tags: [game theory, matrix game]
math: True
---


> This note will be consistently updated.
{: .prompt-info }

## Prisoner's Dilemma (Donation)

> Two members of a criminal organization are arrested and imprisoned. Each prisoner is in solitary confinement with no means of communicating with the other. The prosecutors lack sufficient evidence to convict the pair on the principal charge, but they have enough to convict both on a lesser charge. Simultaneously, the prosecutors offer each prisoner a bargain. Each prisoner is given the opportunity either to betray the other by testifying that the other committed the crime, or to cooperate with the ot	her by remaining silent.

Normal form:

| Player1\Player2      | Cooperate (Deny) | Defect (Confess) |
| -------------------- | ---------------- | ---------------- |
| **Cooperate (Deny)** | $b,b$            | $d,a$            |
| **Defect (Confess)** | $a,d$            | $c,c$            |

$a \gt b \gt c \gt d$

> if player1 defects and player2 cooperate
>
> then player1 and player2 will get payoffs $a$ and $d$, respectively

Attractor of player1: Reversed N-like (or N-like)

| $b\downarrow$ | $d\downarrow$   |
| -----------   | -----------     |
| $a$           | $c(\nwarrow)$   |

Attractor of player2:

| $b\rightarrow$   | $a$             |
| -----------      | -----------     |
| $d\rightarrow$   | $c(\nwarrow)$   |


Nash Equilibrium: (For player1: check the 1st num up-and-down. For player2: check 2nd num left-and-right)

| Player1\Player2      | Cooperate (Deny)               | Defect (Confess)              |
| -------------------- | ------------------------------ | ----------------------------- |
| **Cooperate (Deny)** | $b(\downarrow),b(\rightarrow)$ | $d(\downarrow),a(\checkmark)$ |
| **Defect (Confess)** | $a(\checkmark),d(\rightarrow)$ | $c(\checkmark),c(\checkmark)$ |

Pareto optimal: (check 3 times for each state)

| Player1\Player2      | Cooperate (Deny)         | Defect (Confess)          |
| -------------------- | ------------------------ | ------------------------- |
| **Cooperate (Deny)** | $b,b\ldots (\checkmark)$ | $d,a\ldots (\checkmark)$  |
| **Defect (Confess)** | $a,d\ldots (\checkmark)$ | $c,c \ldots(\nwarrow)$    |


---



## Stag Hunt

> In the simple, matrix-form, two-player Stag Hunt each player makes a choice between a risky action (hunt the stag) and a safe action (forage for mushrooms). Foraging for mushrooms always yields a safe payoff while hunting yields a high payoff if the other player also hunts but a very low payoff if one shows up to hunt alone.

Normal form:

| Player1\Player2      | Cooperate (Hunt) | Defect (Forage) |
| -------------------- | ---------------- | --------------- |
| **Cooperate (Hunt)** | $a,a$            | $d,b$           |
| **Defect (Forage)**  | $b,d$            | $c,c$           |

$a \gt b \ge c \gt d$

Attractor of player1: Reversed u-like (or n-like)

| $a$           | $d\downarrow$   |
| ------------- | -----------     |
| $b\uparrow$   | $c(\leftarrow)$ |

Attractor of player2:

| $a$           | $b\leftarrow$   |
| ------------- | -----------     |
| $d\rightarrow$   | $c(\uparrow)$ |

Nash Equilibrium:

| Player1\Player2      | Cooperate (Hunt)              | Defect (Forage)               |
| -------------------- | ----------------------------- | ----------------------------- |
| **Cooperate (Hunt)** | $a(\checkmark),a(\checkmark)$ | $d(\downarrow),b(\leftarrow)$ |
| **Defect (Forage)**  | $b(\uparrow),d(\rightarrow)$  | $c(\checkmark),c(\checkmark)$ |

Pareto optimal:

| Player1\Player2      | Cooperate (Hunt)        | Defect (Forage)         |
| -------------------- | ----------------------- | ----------------------- |
| **Cooperate (Hunt)** | $a,a\ldots(\checkmark)$ | $d,b\ldots(\leftarrow)$ |
| **Defect (Forage)**  | $b,d\ldots(\uparrow)$   | $c,c\ldots(\nwarrow)$   |



---



## Snowdrift

> The name Snowdrift game refers to the situation of two drivers caught with their cars in a snow drift. If they want to get home, they have to clear a path. The fairest solution would be for both of them to start shoveling (we assume that both have a shovel in their trunk). But suppose that one of them stubbornly refuses to dig. The other driver could do the same, but this would mean sitting through a cold night. It is better to shovel a path clear, even if the shirker can profit from it without lifting a finger.

This game has the same payoff pattern as the games **Chicken**. Alternatively, I could say that these two games share the same **potential game**?

Normal form:

| Player1\Player2      | Cooperate (Deny) | Defect (Confess) |
| -------------------- | ---------------- | ---------------- |
| **Cooperate (Deny)** | $b,b$            | $c,a$            |
| **Defect (Confess)** | $a,c$            | $d,d$            |

$a \gt b \gt c \gt d$

Attractor of player1: Reversed n-like (or u-like)

| $b\downarrow$ | $c(\leftarrow)$ |
| ------------  | -------------   |
| $a$           | $d\uparrow$     |

Attractor of player2:

| $b\rightarrow$ | $a$             |
| ------------   | -------------   |
| $c(\uparrow)$  | $d\leftarrow$   |

Nash Equilibrium:

| Player1\Player2      | Cooperate (Deny)               | Defect (Confess)              |
| -------------------- | ------------------------------ | ----------------------------- |
| **Cooperate (Deny)** | $b(\downarrow),b(\rightarrow)$ | $c(\checkmark),a(\checkmark)$ |
| **Defect (Confess)** | $a(\checkmark),c(\checkmark)$  | $d(\uparrow),d(\leftarrow)$   |

Pareto optimal:

| Player1\Player2      | Cooperate (Deny)        | Defect (Confess)                        |
| -------------------- | ----------------------- | --------------------------------------- |
| **Cooperate (Deny)** | $b,b\ldots(\checkmark)$ | $c,a\ldots(\checkmark)$                 |
| **Defect (Confess)** | $a,c\ldots(\checkmark)$ | $d,d\ldots(\leftarrow\nwarrow\uparrow)$ |


---

## Battle of the Sexes (or )
Also known as **Bach or Stravinsky**.


---

## Matching Pennies

##  Rock paper scissors


## Muddy Children Puzzle

Check this [note](https://yuelin301.github.io/posts/Dynamic-Epistemic-Logic/#the-muddy-children-puzzle).


## The Trust Game
> 1. In the first stage, the Donor (or Investor) receives a certain endowment by the experimenter, and can decide whether or not to send a part of that sum to the Recipient (or Trustee), knowing that the amount will be tripled upon arrival: each euro spent by the Investor yields three euros on the Trustee’s account.
> 2. In the second stage, the Trustee can return some of it to the Investor’s account, on a one-to-one basis: it costs one euro to the Trustee to increase the Investor’s account by one euro.
> 3. This ends the game. Players know that they will not meet again.

## Ultimatum
> The experimenter assigns a certain sum, and the Proposer can offer a share of it to the Responder. If the Responder (who knows the sum) accepts, the sum is split accordingly between the two players, and the game is over. If the Responder declines, the experimenter withdraws the money. Again, the game is over: but this time, neither of the two players gets anything.
