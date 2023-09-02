---
title: A Memo on Game Theory
date: 2023-08-13 14:30:00 +0800
categories: [Economics & Game Theory]
tags: [game theory]
math: True
---


> This note will be consistently updated.
{: .prompt-info }

## Rationality
A rational player is one who chooses his action, to maximize his payoff consistent with his beliefs about what is going on in the game.[^intro-book]

> - **"self-interested"** refers to the agent being concerned only about its own expected payoff, and 
> - **"rational"** means that when it believes one action's payoff is greater than another's, the agent will choose the action with higher expected payoff.
{: .prompt-tip }

**Bounded rationality** is the idea that rationality is limited when individuals make decisions, and under these limitations, rational individuals will select a decision that is **satisfactory** rather than optimal.[^wiki-boundedration]


## Normal-form game

**Definition 3.3**[^intro-book] A **normal-form game** includes three components as follows:
1. A finite **set of players**, $N = \set{1,2,\ldots,n}$.
2. A **collection of sets of pure strategies**, $\set{S_1, S_2, \ldots, s_n}$.
3. A **set of payoff functions**, $\set{v_1,v_2,\ldots, v_n}$, each assigning a payoff value to each combination of chosen strategies, that is, a set of functions $v_i: S_1\times S_2, \times \dots \times S_n \to \mathbb{R}$ for each $i\in N$.

Each of the players $i\in N$ must simultaneously choose a possible strategy $s_i\in S_i$.[^intro-book]

> In my understanding, it's like $(I,\set{A^i}\_{i\in I}, \set{R^i}\_{i\in I})$ in the notation of MARL.
{: .prompt-tip }


## Game Tree

**Definition 3.1**[^MIT-opencourse] A tree is a set of nodes and directed edges connecting these nodes such
that
1. there is an initial node, for which there is no incoming edge;
2. for every other node, there is exactly one incoming edge;
3. for any two nodes, there is a unique path that connect these two nodes.

And:
- The edges are actions. 
- Each non-terminal node has been defined with who is to perform the action at this moment.
- Payoffs for each player are defined at each terminal node.

## Information Set

**Definition 3.4**[^MIT-opencourse] An **information set** is a collection of nodes such that
1. the same player $i$ is to move at each of these nodes; 
2. the same moves are available at each of these nodes.

> In my understanding, the nodes in an information set share the same parent node, and they are at the same depth of the tree. The player taking actions at the current depth cannot figure out which node it is at. It cannot see the opponent's previous move.
{: .prompt-tip }

**Definition 3.5**[^MIT-opencourse] An **information partition** is an allocation of each non-terminal node of the tree to an information set; the starting node must be "alone".

## Extensive-Form Games

> Often credit this paper:  
> H. W. Kuhn. Extensive games and the problem of information. Contributions to the Theory of Games, 2:193–216, 1953.
{: .prompt-info }

The extensive-form representation of a game contains all the information about the game explicitly, by defining who moves when, what each player knows when he moves, what moves are available to him, and where each move leads to, etc. This is done by use of a **game tree** and **information sets**–as well as more basic information such as players and the payoffs.[^MIT-opencourse]


**Definition 3.3**[^MIT-opencourse] (Extensive form) A Game consists of
- a set of players, 
- a tree,
- an allocation of non-terminal nodes of the tree to the players,
- an informational partition of the non-terminal nodes, and
- payoffs for each player at each terminal node.


---

In my understanding, the nodes in extensive-form games are not equivalent to the states in MDP. Because the nodes are not Markovian. 

> I'm so confused. What should the corresponding quantity of "state" be? This could be a gap between the MARL and Game Theory frameworks.
{: .prompt-tip }

## Information: Complete v.s. Perfect

> I've looked up these two concepts so many times already, but every time I read about them, I forget shortly afterward. 😅
{: .prompt-tip }

### Complete Information

A game of complete information requires that the following four components be common knowledge among all the players of the game:[^intro-book]
1. all the possible actions of all the players,
2. all the possible outcomes,
3. how each combination of actions of all players affects which outcome will materialize, and
4. the preferences of each and every player over outcomes.

**Preferences** describe how the player ranks the set of possible outcomes, from most desired to least desired.[^intro-book]

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

But[^wiki-complete]
- players may not know the others' actions (e.g. the initial placement of ships in  **Battleship**), and
- the game may has chance element (card games).

### Perfect Information

**Definition 7.3**[^intro-book] A game of complete information in which every information set is a **singleton** and there are no moves of Nature is called a **game of perfect information.**

Each player is perfectly informed of all the events that have previously occurred.[^wiki-perfect] There is no hidden information.


### Examples[^wiki-complete] [^wiki-perfect]

- Perfect and complete: Chess, Tic-Tac-Toe, Go.
- Perfect but incomplete: Bayesian game.
- Complte but imperfect: Card games, where each player's cards are hidden from other players but objectives are known. The dealing event is not public (imperfect). 


## Social Choice Function
It aims to align the interests of all agents.


## Price of Anarchy v.s. Price of Stability

1. PoA = Price of Anarchy
    1. Anarchy = 无政府主义、混乱
    2. 如果不要中心化的authority来组织，那要付出多大的代价（相比性能下降多少）（我的理解）
2. notation
    1. game $G = (N,S,u)$
    2. 玩家有$N$个，单个某个玩家记为$i$
    3. 单个某个玩家的策略集合为$S_i$（这里只假定做的策略是确定性的，只选择某一个动作）
    4. 单个某个玩家的收益为$u_i:S\to \mathbb{R}$；其中$S=S_1\times \ldots\times S_N$
3. PoA衡量“玩家的自私行为”恶化“系统表现”的“效率”
    1. “系统”指的是所有玩家构成的那个抽象的整体（我的理解）
    2. “系统表现”根据我们的目的来定义，记为$\text{Welf}:S\to\mathbb{R}$，比如
        1. social welfare（utilitarian objective）：$\text{Welf}=\sum\limits_{i\in N}u_i(s)$；所有玩家收益的和
        2. fairness（egalitarian objective）：$\text{Welf}=\min\limits_{i\in N}u_i(s)$
        3. 系统表现是我们想控制得到的最大化的目标
        4. 如果是想最小化某个目标，那就应该是$\text{Cost}$
4. 社会困境：个体只注重自己收益期望的优化不一定能让系统表现优化
    1. 如果有中心化的authority：把所有人作为一个抽象的整体来分析，可以让某些人牺牲之类的，这样比较好分析得到一个$\text{Welf}$很高的$s_{\text{centralized}}$（或多个）
    2. 如果没有中心化的authority，每个人只优化自己，最终的结果会到达equilibrium，也就是某个$s_{\text{equilibrium}}$（比如常见的Nash equilibria）。equilibria集合记为$\text{Equil}$。
    3. $\text{PoA} = \frac{\max\limits_{s\in S}\text{Welf}(s)} {\min\limits_{s\in \text{Equil}}\text{Welf}(s)}$，分母是fully decentralized设置下的最坏情况的收益。（也有分子分母反过来定义的说法）
    4. $\text{PoA} = \frac{\max\limits_{s\in \text{Equil}}\text{Cost}(s)} {\min\limits_{s\in S}\text{Cost}(s)}$
    5. 另一个概念是PoS，the Price of Stability，
    6. $\text{PoS} = \frac{\max\limits_{s\in S}\text{Welf}(s)} {\max\limits_{s\in \text{Equil}}\text{Welf}(s)}$，分子和PoA一样，分母是fully decentralized设置下的最好情况的收益。
    7. $\text{PoS} = \frac{\min\limits_{s\in \text{Equil}}\text{Cost}(s)} {\min\limits_{s\in S}\text{Cost}(s)}$
    8. 根据定义，$\text{PoA}\ge\text{PoS}\ge1$，wiki里是这么说的，是负数我不知道咋办，比如：分母<0<分子<-分母，这样加绝对值也不对。
5. 例子：[Prisoner's Dilemma](https://yuelin301.github.io/posts/Classic-Games/#prisoners-dilemma)
    1. 希望最大化的东西是social walfare，即，我们想优化$\text{Welf}=u_1(s_1,s_2)+u_2(s_1,s_2)$
    2. 如果有中心化的authority，则最优的情况肯定是$(s_1=\text{Cooperate},s_2=\text{Cooperate})$，此时$\text{Welf}=2b$。
    3. 如果没有中心化的authority，每个人只优化自己，那么此时对于两个玩家来说，$\text{Defect}$都是占优策略（别人选择合作，那我背叛收益高；别人选择背叛，那我背叛收益高；不论别人怎么选择，我选背叛收益高）。会收敛到$(s_1=\text{Defect},s_2=\text{Defect})$的Nash equilibrium。$\text{Welf}$相比有中心化的情况，降低了。
    4. $\text{PoA} = \frac{2b}{2c}=\frac{b}{c}$，其跟表中的数值有关。

## Shapley Value

In my understanding, the Shapley value reflects how significant an individual's effective contribution is to the total value of a certain coupling in a group. A real-world application is to determine how to distribute money: those who work more and contribute more effectively to the group should receive more money.

Given a cooperative game with a set $N$ of players and a value function $v: 2^N \to \mathbb{R}$, which assigns a value to each coalition of players, the Shapley value of player $i$, denoted as $\phi_i(v)$, is defined as:

$$
\phi_i(v) = \sum_{S \subseteq N \setminus \{i\}} \frac{\vert S\vert! (\vert N\vert - \vert S\vert - 1)!}{\vert N\vert!} \left[v(S \cup \{i\}) - v(S)\right]
$$

Where:

- $N$ is the set of all players.
- $S$ is a subset of players not including player $i$.
- $\vert S\vert$ is the number of players in subset $S$.
- $\vert N\vert$ is the total number of players.
- $v(S)$ is the value function, i.e., the value of coalition $S$.
- The term $\left[v(S \cup \{i\}) - v(S)\right]$ represents the marginal contribution of player $i$ to coalition $S$.

In words, the Shapley value of player $i$ is the average of its marginal contributions over all possible coalitions.



## Regret

## Potential Game

## Backward Induction

## Forward Induction


## Solution concept

### Nash equilibrium

### Subgame perfect equilibrium

### Perfect Bayesian equilibrium

---

## References

[^intro-book]: Tadelis, Steven. Game theory: an introduction. Princeton university press, 2013.  
[^wiki-complete]: [Wikipedia: Complete Information](https://en.wikipedia.org/wiki/Complete_information).  
[^wiki-perfect]: [Wikipedia: Perfect Information](https://en.wikipedia.org/wiki/Perfect_information).  
[^wiki-boundedration]: [Wikipedia: Bounded Rationality](https://en.wikipedia.org/wiki/Bounded_rationality).  
[^MIT-opencourse]: [MIT OpenCourseWare: 14.12 Economic Applications of Game Theory](https://ocw.mit.edu/courses/14-12-economic-applications-of-game-theory-fall-2012/b90ef0b930888cc7a1828d5eaf91f5c9_MIT14_12F12_chapter3.pdf).


> Disclaimer: The above content is summarized from Wikipedia and other sources. Corresponding links or references have been provided.
{: .prompt-danger }