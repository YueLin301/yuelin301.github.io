---
title: Fictitious Play
date: 2023-06-14 20:00:00 +0800
categories: [Economics & Game Theory]
tags: [game theory, fictitious play, multi agents, reinforcement learning]
math: True
---

## What is Fictitious Play?
*([Wikipedia](https://en.wikipedia.org/wiki/Fictitious_play))*

1. Fictitious play is a learning rule. 
2. In it, each player presumes that the opponents are playing stationary (possibly mixed) strategies. 
3. At each round, each player thus best responds to the empirical frequency of play of their opponent. 

*(ChatGPT)*

**Fictitious play is a concept in game theory that describes a learning process in which players make decisions based on their beliefs about the actions of their opponents.** It was first introduced by George W. Brown in 1951.

In fictitious play, each player assumes that their opponents are playing fixed, deterministic strategies and updates their beliefs about these strategies based on observed outcomes of the game. Players do not directly reason about the psychological motivations or thought processes of their opponents; instead, they update their beliefs solely based on the observed history of play.

The basic idea behind fictitious play is that players continually revise their beliefs about the strategies being employed by their opponents. They do this by assuming that their opponents are playing the strategy that has been most frequently observed in the past. Each player then chooses their best response to this assumed strategy, and the process continues iteratively.

Over time, as players repeatedly update their beliefs and adjust their strategies accordingly, it is believed that fictitious play can converge to a stable strategy profile. This profile represents a set of strategies where no player has an incentive to deviate from their chosen strategy given their beliefs about the strategies of others.

Fictitious play has been used to analyze various types of games, including both cooperative and non-cooperative games. It provides insights into how players might learn and adapt their strategies in dynamic environments, even when they have limited information about the true intentions and strategies of their opponents.

> The following part has not been finished yet. One may check my [writing schedule](https://yuelin301.github.io/posts/Schedule/).
{: .prompt-warning }