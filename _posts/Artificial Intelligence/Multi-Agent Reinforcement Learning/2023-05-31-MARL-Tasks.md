---
title: MARL Tasks
date: 2023-05-31 20:00:01 +0800
categories: [Artificial Intelligence, Multi-Agent Reinforcement Learning]
tags: [Tech, AI, Multi_Agents, RL, Environment, Social_Dilemma]
math: True
---

> This note will be consistently updated.
{: .prompt-info }

## List
- StarCraft II
   1. [SMAC (StarCraft Multi-Agent Challenge)](https://github.com/oxwhirl/smac). SMAC is WhiRL's environment for research in the field of collaborative multi-agent reinforcement learning (MARL) based on Blizzard's StarCraft II RTS game.
   2. [PySC2 (StarCraft II Learning Environment)](https://github.com/deepmind/pysc2). PySC2 is DeepMind's Python component of the StarCraft II Learning Environment (SC2LE).
   3. [gym-starcraft](https://github.com/apsdehal/gym-starcraft). An OpenAI Gym interface to StarCraft.
- [MAagent](https://github.com/geek-ai/MAgent)
   1. persuit: predators pursuit preys
   2. gather: agents rush to gather food
   3. battle: battle between two armies
   4. arrange: arrange agents into some characters
   5. [A demo video](https://www.youtube.com/watch?v=HCSm0kVolqI)
- Overcooked
- MiniRTS
   1. [Paper](https://arxiv.org/pdf/1707.01067.pdf)
   2. [GitHub](https://github.com/facebookresearch/ELF)
- Cards & Chess
   1. Go
   2. Texas Hold'em
   3. Mahjong
   4. Hanabi [[code](https://github.com/deepmind/hanabi-learning-environment)] [[paper](https://www.sciencedirect.com/science/article/pii/S0004370219300116)]
- Communication
- Melting Pot (Deep Mind)
   1. [Sequential Social Dilemma (SSD)](https://github.com/eugenevinitsky/sequential_social_dilemma_games)
   2. Harvest
   3. Clean Up
-  Level-Based Foraging

## Overcooked

This environment is designed to test the human-ai coordination, or used as a zero-shot coordination task.

### Resources
1. [[GitHub Repo: overcooked_ai]](https://github.com/HumanCompatibleAI/overcooked_ai/tree/master)
2. [[Online Game]](https://humancompatibleai.github.io/overcooked-demo/)
3. [[An Environment Wrapper: PantheonRL]](https://github.com/Stanford-ILIAD/PantheonRL)

<!-- It seems that PantheonRL is out of date now (2023-12-26). To run the provided experiments, the environment needs some  -->
<!-- The provided example code cannot run directly, for it calls a deprecated function. When I was fixing that, I found the environment wrapper was rebuilt in the repo `overcooked_ai`. -->



## Hanabi

### **Objective**

Players work collaboratively to put on a firework show by placing a series of cards in the correct order. The objective is to complete a series of five cards of the same color in ascending numerical order (1 to 5) for each color. 

- **Cards Facing Outwards**: One distinctive rule of Hanabi is that players hold their cards facing outwards, meaning they can see everyone else's cards but their own. This facilitates a cooperative environment where players rely on each other's hints to figure out their cards.
- **Communication**: Communication is restricted to the formal hint-giving process to maintain the game's difficulty and collaborative spirit.

### **Components**

- **Deck**: Consists of 50 cards, distributed into 5 different colors (red, yellow, green, blue, white), with each color having numbers from 1 to 5 (1s x3, 2s x2, 3s x2, 4s x2, and 5s x1).
- **Hint tokens**: 8 in total, used to give hints to other players.
- **Fuse tokens**: 3 in total, representing the players' "lives".

### **Setup**

1. **Number of players**: 2-5 players.
2. **Hand size**: Depending on the number of players, each player starts with a hand of cards (5 cards for 2-3 players, 4 cards for 4-5 players).
3. **Initial layout**: Players hold their cards facing outwards so they can't see their cards but can see others'.
   
### **Gameplay**

Players take turns clockwise and can choose to perform one of the following actions on their turn:

1. **Give a hint**: A player may give a hint to another player about the contents of their hand. The hint must be about either the color or the number of the cards, not both. It consumes a hint token. It must relate to at least one card in the other player’s hand, and it must give information about all the cards of the chosen characteristic. 

2. **Discard a card**: A player may choose to discard a card from their hand to regain a hint token. The discarded card is placed in the discard pile and the player draws a new card from the draw pile.

3. **Play a card**: A player may choose to play a card from their hand onto the table. If the card successfully fits into a fireworks display (i.e., it is the next number in sequence of the same color), it stays on the table. Otherwise, it goes to the discard pile and a fuse token is removed.

### **Game Ending Conditions**

The game can end in the following ways:

1. **Successful Completion**: Players successfully play all five cards in each color.
2. **Fuse Tokens Exhausted**: Players make three mistakes and lose all fuse tokens.
3. **Deck Exhaustion**: The draw pile is exhausted. Players get one final round before the game ends.

### **Scoring**

At the end of the game, the score is calculated based on the number of cards successfully played in the fireworks displays. The highest possible score is 25 (if all fireworks are successfully completed). The score is determined by summing the highest value of cards played for each color.

### **Strategy and Tips**

- **Memory and Deduction**: Players must rely heavily on their memory and deduction skills to remember the hints given and to figure out which cards they hold.
- **Non-verbal cues**: Players are not allowed to give extra hints through non-verbal cues or suggestive comments. All hints must be given using hint tokens officially in one’s turn.
- **Hint Efficiency**: Given the limited number of hint tokens, players should strive to give hints that convey the maximum amount of information.
