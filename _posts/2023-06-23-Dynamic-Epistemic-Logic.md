---
title: Dynamic Epistemic Logic
date: 2023-06-23 02:00:00 +0800
categories: [Mathematics]
tags: [epistemic logic, common knowledge, logic, induction puzzles]
math: True
---

> This note has not been finished yet. One may check my [writing schedule](https://yuelin301.github.io/posts/Schedule/).
{: .prompt-warning }

> Three logicians walk into a bar.  
The bartender asks: "Do you all want a drink?"  
The first logician says: "I don't know."  
The second logician says: "I don't know."  
The third logician says: "Yes."
{: .prompt-info }



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
2. Players is smart and is capaple of theory of mind.
   > Also, usually, some kind of hint is given to suggest that the participants can trust each other's intelligence — they are capable of theory of mind (that "every participant knows modus ponens" is common knowledge).  
3. "Doing nothing" is also an action and indicates something.
   > Also, the inaction of a participant is a non-verbal communication of that participant's lack of knowledge, which then becomes common knowledge to all participants who observed the inaction.

### The Muddy Children Puzzle
This is one of the most classic examples of inductin puzzles. And there are 

> There is a set of attentive children. They think perfectly logically (and this is common knowledge). The children consider it possible to have a muddy face. None of the children can determine the state of their own face themselves. But, every child knows the state of all other children's faces. A custodian tells the children that at least one of them has a muddy face. The children are each told that they should step forward if they know that their own face is muddy. Hereafter, the custodian starts to count and after every stroke, every muddy child has an opportunity to step forward.[^wiki-Induction-Puzzles]

### Hat Puzzles
> One type of induction puzzle concerns the wearing of colored hats, where each person in a group can only see the color of those worn by others, and must work out the color of their own.[^wiki-Induction-Puzzles]








## Common Knowledge

> The notion of common knowledge, where **everyone knows, everyone knows that everyone knows, etc.**, has proven to be fundamental in various disciplines, including Philosophy, Artificial Intelligence, Game Theory, Psychology, and Distributed Systems. This key notion was first studied by the philosopher **David Lewis** in the context of **conventions**. Lewis pointed out that in order for something to be a convention, it must in fact be common knowledge among the members of a group. (For example, the convention that green means “go” and red means “stop” is presumably common knowledge among the drivers in our society.)[^Common-knowledge-revisited]

> Common knowledge also arises in discourse understanding. Suppose Ann asks Bob “What did you think of the movie?” referring to a showing of Monkey Business they have just seen. Not only must Ann and Bob both know that “the movie” refers to Monkey Business, **but Ann must know that Bob knows (so that she can be sure that Bob will give a reasonable answer to her question), Bob must know that Ann knows that Bob knows (so that Bob knows that Ann will respond appropriately to his answer)**, and so on. In fact, by a closer analysis of this situation, it can be shown that there must be common knowledge of what movie is meant in order for Bob to answer the question appropriately.[^Common-knowledge-revisited]





## References
[^wiki-Dynamic-Epistemic-Logic]: Wikipedia: [Dynamic Epistemic Logic](https://en.wikipedia.org/wiki/Dynamic_epistemic_logic).
[^wiki-Induction-Puzzles]: Wikipedia: [Induction Puzzles](https://en.wikipedia.org/wiki/Induction_puzzles)
[^Common-knowledge-revisited]: Ronald Fagin, Joseph Y. Halpern, Yoram Moses, Moshe Y. Vardi. "Common knowledge revisited" *Annals of Pure and Applied Logic (1999)*.