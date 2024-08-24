---
title: Dynamic Epistemic Logic
date: 2023-06-23 02:00:00 +0800
categories: [Interdisciplinarity, Mathematics]
tags: [Tech, Interdisciplinarity, Math, Logic, Multi_Agents]
math: True
---

> Three logicians walk into a bar.  
The bartender asks: "Do you all want a drink?"  
The first logician says: "I don't know."  
The second logician says: "I don't know."  
The third logician says: "Yes."



## What is Dynamic Epistemic Logic?[^wiki-Dynamic-Epistemic-Logic]

> Dynamic epistemic logic (DEL) is a logical framework dealing with knowledge and information change.  
Typically, DEL focuses on situations involving multiple agents and studies how their knowledge changes when events occur.

There are two kinds of events:
- Ontic events: These events can change factual properties of the actual world, e.g., a red card is painted in blue. 
- Epistemic events: These events can change agent's knowledge without changing factual properties of the world, e.g., a card is revealed publicly (or privately) to be red.  

> Originally, DEL focused on epistemic events.   
> In computer science, DEL is for example very much related to multi-agent systems, which are systems where multiple intelligent agents interact and exchange information.

I discovered that this topic shares some relevance with the theory of mind in certain aspects. Both of them concern the knowledge of agents and involve analyzing how the agents modeling others.

However this is perhaps not what I am looking for, for it being a **modal logic**. So this blog simply aims to provide a brief overview of the topic, focusing on the game-theoretic part of it.

> Epistemic logic is a modal logic dealing with the notions of knowledge and belief. As a logic, it is concerned with understanding the process of *reasoning* about knowledge and belief.

Some relevant keywords:
- Common knowledge (logic),
- Induction puzzles, 
- Dynamic epistemic logic.


## Induction Puzzles[^wiki-Induction-Puzzles]

A good way to understand this topic is through examples of the problems that it aims to investigate, namely induction puzzles.

> Induction puzzles are logic puzzles, which are examples of multi-agent reasoning, where the solution evolves along with the principle of induction.
A puzzle's scenario always involves multiple players with the same reasoning capability, who go through the same reasoning steps.

A logical feature:
> According to the principle of induction, a solution to the simplest case makes the solution of the next complicated case obvious. Once the simplest case of the induction puzzle is solved, the whole puzzle is solved subsequently.

Basic settings:
1. Players know informaiton about others but does not know their owns.
   > Typical tell-tale features of these puzzles include any puzzle in which each participant has a given piece of information (usually as common knowledge) about all other participants but not themselves.  
2. Players are smart and are capaple of theory of mind.
   > Also, usually, some kind of hint is given to suggest that the participants can trust each other's intelligence — they are capable of theory of mind (that "every participant knows modus ponens" is common knowledge).  
3. "Doing nothing" is also an action and indicates something.
   > Also, the inaction of a participant is a non-verbal communication of that participant's lack of knowledge, which then becomes common knowledge to all participants who observed the inaction.

> The introduction of "common knowledge" will be presented in [the next section](#common-knowledge).
{: .prompt-tip }

### The Muddy Children Puzzle
This is one of the most classic examples of inductin puzzles. This problem also has other descriptive forms and variations, such as the Blue-Eyed Islanders and cheating wives/husbands puzzles.

#### Description
> There is a set of attentive children. They think perfectly logically. The children consider it possible to have a muddy face. None of the children can determine the state of their own face themselves. But, every child knows the state of all other children's faces. A custodian tells the children that at least one of them has a muddy face. The children are each told that they should step forward if they know that their own face is muddy. Hereafter, the custodian starts to count and after every stroke, every muddy child has an opportunity to step forward.

Let us extract the core of it.

What is the goal?
Players (children) want to find out a piece of information (whether their face is clean or not) that they do not know but others do.

What are the assumptions?
1. Before the game starts, there will be an outsider who provides some indirect information.
2. Players know informaiton about others but does not know their owns.
3. There are occasions when making decisions openly (the chances for children to step forward), and the actions taken become common knowledge. 
4. Players are smart and are capaple of theory of mind.
5. If players know their own information, they will reveal it to their own best interests.
6. "Doing nothing" is also an action and indicates something. Specifically, it means that a player does not know its own information. And this is new information to others. 


#### Logical solution
Assume that there are $n$ children and $k$ of them are dirty where $k>1$.

> Let's assume that there are just 2 children: Alice and Bob. If only Alice is dirty, she will step forward at the first stroke, because she does not see any other dirty faces. The same is true for Bob. If Alice sees Bob not stepping forward at the first stroke, she must conclude that he certainly sees another muddy child and they will step forward simultaneously at the second stroke.  
Let's assume that there are just 3 children: Alice, Bob, and Charly. If there are less than 3 muddy children, the puzzle evolves like in the case with 2 children. If Charly sees that Alice and Bob are muddy and not stepping forward at the second stroke, they all together will step forward at the third stroke.  
It can be proven that $k$ muddy children will step forward after $k$ strokes.

Denote Alice as $i$ and Bob as $j$. There are three cases:
1. $i$ is dirty and $j$ is not.
   - $i$ knows $j$ is not dirty. So $i$ can infer that it is dirty. Thus $i$ will step forward at the first stroke.
   - $j$ knows $i$ is dirty but $j$ cannot infer its situation. But after the first stroke $j$ will know that "$i$ can infer itself is dirty", and thus $j$ can infer that $j$ is not dirty.
2. $j$ is dirty and $i$ is not. The analysis is symmetrical to that of the first case.
3. $i$ and $j$ are both dirty.
   - $i$ knows $j$ is dirty but $i$ cannot infer its situation. Thus $i$ will do nothing at the first stroke.
   - $j$ knows $i$ is dirty but $j$ cannot infer its situation. Thus $j$ will do nothing at the first stroke.
   - After the first stroke, $i$ will know that $j$ cannot infer anything, meaning that it is not the first case (not "You are dirty but I am not"). So $i$ can infer that it is dirty. Thus $i$ will step forward at the second stroke.
   - So do $j$. Thus $j$ will step forward at the second stroke.

> There is a straightforward proof by induction that the first $k − 1$ times he asks the question, they will all say “No,” but then the $k$th time the children with muddy foreheads will all answer “Yes.”[^Common-knowledge-revisited]

So what information does "At least one of you has mud on your forehead" bring to players? **The common knowledge.**
> Let us denote the fact “at least one child has a muddy forehead” by $p$. Notice that if $k > 1$, i.e., more than one child has a muddy forehead, then every child can see at least one muddy forehead, and the children initially all know $p$. Thus, it would seem that the father does not provide the children with any new information, and so he should not need to tell them that $p$ holds when $k > 1$.  
> But this is false! What the father provides is common knowledge. If exactly $k$ children have muddy foreheads, then it is straightforward to see that $E^{k−1}p$ holds before the father speaks, but $E^kp$ does not (here $E^k\varphi$ means $\varphi$, if $k = 0$, and everyone knows $E^{k-1}\varphi$, if $k \ge 1$). The father’s statement actually converts the children’s state of knowledge from $E^{k−1}p$ to $Cp$ (here $Cp$ means that there is common knowledge of $p$). With this extra knowledge, they can deduce whether their foreheads are muddy.  
> In the muddy children puzzle, the children do not actually need common knowledge; Ekp suffices for them to figure out whether they have mud on their foreheads.[^Common-knowledge-revisited]

I found an alternative and more general notation:
- $p$: A piece of information.
- $K_i p$: Agent $i$ knows $p$.
- $B_i p$: Agent $i$ believes $p$.
- $E p := \land_{i} K_i p$: All agents know $p$. 
- $C p := Ep \land EEp \land EEEp \land \ldots$: Common knowledge.

> Two interaction axioms: $K_i p \to B_i p$ (i.e. if $i$ knows $p$ then it believes $p$) and $B_i p \to KB_i p$ (i.e. if $i$ believes $p$ then it knows that it believes $p$).
{: .prompt-tip }

#### Game-theoretic solution

> Muddy children puzzle can also be solved using **backward induction** from game theory.

The reference is from [this paper](https://www.qucosa.de/api/qucosa%3A22752/attachment/ATT-0/), written in German.

> Muddy children puzzle can be represented as an extensive form game of imperfect information. Every player has two actions — stay back and step forwards. There is a move by nature at the start of the game, which determines the children with and without muddy faces. Children do not communicate as in non-cooperative games. Every stroke is a simultaneous move by children. It is a sequential game of unlimited length. The game-theoretic solution needs some additional assumptions:  
> 1. All children are rational and all children's rationality is common knowledge. This means that Alice is rational, Alice knows that Bob is rational and Alice knows that Bob knows that Charly is rational and so on and vice versa.
> 2. Stepping forward without having a muddy face results in a big penalty.
> 3. Stepping forward with a muddy face results in a reward.
> 4. Every stroke results in minor negative penalty aka discount factor for every child until any of them stepped forward. Any multiple of the minor penalty is always a lesser evil than the big penalty.
> 
> If only Alice is muddy, the last assumption makes it irrational for her to hesitate. If Alice and Bob are muddy, Alice knows that Bob's only reason of staying back after the first stroke is the apprehension to receive the big penalty of stepping forward without a muddy face. In the case with $k$ muddy children, receiving $k$ times the minor penalty is still better than the big penalty.

This is a non-cooperative extensive-form (sequential) game of imperfect information.
1. Agents: finite, rational (common knowledge), and not hesitant.
2. States: All agents' face situations.
3. Actions: Do nothing, or step forward. Everyone moves  simultaneously at every timestep.
4. Observations: The others' face situations.
5. Reward function:
   1. $R^i(s^i=\text{clean face}, a^i=\text{step forward}) = -\infty$,
   2. $R^i(s^i=\text{muddy face}, a^i=\text{step forward}) = r$, where $r$ is a predefined constant reward.
   3. $R^i(\cdot, a^i=\text{step forward}) = 0$, I guess?

So this is just formulation. What about the method "backward induction"?

> I found a paper about this, cited 181 times. I will read it later:  
> Rational Dynamics and Epistemic Logic in Games.  
Johan Van Benthem.  
*International Game Theory Review 2007*.
{: .prompt-tip }


### Hat Puzzles
> One type of induction puzzle concerns the wearing of colored hats, where each person in a group can only see the color of those worn by others, and must work out the color of their own.

#### The King's Wise Men Hat Puzzle
> The King called the three wisest men in the country to his court to decide who would become his new advisor. He placed a hat on each of their heads, such that each wise man could see all of the other hats, but none of them could see their own. Each hat was either white or blue. The king gave his word to the wise men that at least one of them was wearing a blue hat; in other words, there could be one, two, or three blue hats, but not zero. The king also announced that the contest would be fair to all three men. The wise men were also forbidden to speak to each other. The king declared that whichever man stood up first and correctly announced the colour of his own hat would become his new advisor. The wise men sat for a very long time before one stood up and correctly announced the answer. What did he say, and how did he work it out?






## Common Knowledge

> The notion of common knowledge, where **everyone knows, everyone knows that everyone knows, etc.**, has proven to be fundamental in various disciplines, including Philosophy, Artificial Intelligence, Game Theory, Psychology, and Distributed Systems. This key notion was first studied by the philosopher **David Lewis** in the context of **conventions**. Lewis pointed out that in order for something to be a convention, it must in fact be common knowledge among the members of a group. (For example, the convention that green means “go” and red means “stop” is presumably common knowledge among the drivers in our society.)[^Common-knowledge-revisited]

> Common knowledge also arises in discourse understanding. Suppose Ann asks Bob “What did you think of the movie?” referring to a showing of Monkey Business they have just seen. Not only must Ann and Bob both know that “the movie” refers to Monkey Business, **but Ann must know that Bob knows (so that she can be sure that Bob will give a reasonable answer to her question), Bob must know that Ann knows that Bob knows (so that Bob knows that Ann will respond appropriately to his answer)**, and so on. In fact, by a closer analysis of this situation, it can be shown that there must be common knowledge of what movie is meant in order for Bob to answer the question appropriately.[^Common-knowledge-revisited]


> This note is incomplete, and I have no intention of completing it because I realized it is not the topic I am looking for.
{: .prompt-warning }


## References
[^wiki-Dynamic-Epistemic-Logic]: Wikipedia: [Dynamic Epistemic Logic](https://en.wikipedia.org/wiki/Dynamic_epistemic_logic).
[^wiki-Induction-Puzzles]: Wikipedia: [Induction Puzzles](https://en.wikipedia.org/wiki/Induction_puzzles)
[^Common-knowledge-revisited]: Ronald Fagin, Joseph Y. Halpern, Yoram Moses, Moshe Y. Vardi. "Common knowledge revisited." *Annals of Pure and Applied Logic (1999)*.
<!-- [^Andrew-Backward-Induction]: Andrew M. Colman. "Rationality assumptions of game theory and the
backward induction paradox."  -->