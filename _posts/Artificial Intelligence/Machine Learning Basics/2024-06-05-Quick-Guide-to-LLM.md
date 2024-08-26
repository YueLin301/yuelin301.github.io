---
title: A Quick Guide to LLMs
date: 2024-06-05 12:40:00 +0800
categories:
  - Artificial Intelligence
  - Machine Learning Basics
tags:
  - Tech
  - AI
  - ML
  - NLP
  - Transformer
  - GPT
  - LLM
  - Classic
math: true
pin: false
description: LLM = Large Language Model
---

> This note has not been finished yet.
{: .prompt-warning }

## Main Storyline

### Transformer
> Google. 2017.6. "Attention is all you need."

For more detailed introduction, see [my blog]({{site.baseurl}}/posts/Transformer/).

<!-- [obsidian test](_posts/Artificial%20Intelligence/Machine%20Learning%20Basics/2024-06-06-Transformer) -->

1. Encoder-Decoder Framework. No RNNs.
2. Positional Encoding. 
   1. Introduce periodicity using trigonometric functions, allowing the model to handle longer input lengths that it may not have seen during training.
   2. The encoded positional vector can be added directly to the input because they can be seen as linear independent in the embedding space.
3. Self-Attention: $QKV$ 
   1. Input a query, return the value whose key is the most similar to the query.
   2. Dot product can be used to calculate the similarity of $Q$ and $K$ because they are embedded and the product can be seen as the inner product of two vectors.
   3. The results of $QK$ are weights which reflect the similarity. And $(QK)V$ is extracting the content of input.
4. The output of $QK$ in decoder is masked so each input vector cannot see the vectors after it.
5. The output layer is softmax. The output is the largest number of the softmax output. Or it is done by beam search.
6. The time complexity is $O(n^2d),$ and RNNs' is $O(nd^2),$ where $n$ is the sequence length and $d$ is the model dimension. And the attention can done in parallel.

### GPT-1
> OpenAI. 2018. "Improving language understanding by generative pre-training."

For more detailed introduction, see [my blog]({{site.baseurl}}/posts/GPT-1-2/).

1. GPT-1 = Decoder (in Transformer, with learnable positional encoding) + Pre-Training + Fine-Tuning.
2. Pre-Training: Unsupervised learning. The model is trained using unlabeled data to predict the next word. It is used to make the model to be familiar with human common knowledge.
3. Fine-Tuning: Supervised learning. The model is trained using labeled data for specific downstream NLP tasks.
   
### BERT
> Google. 2018.10. "Bert: Pre-training of deep bidirectional transformers for language understanding."

1.  BERT = Encoder (in Transformer) + Pre-Training + Fine-Tuning + Masked Input.
2.  The inputs are masked before they come through the self-attention part. E.g. `"I love singing because it is fun." -> "I [?] singing because it is fun."`
3.  BERT uses the encoder so the self-attention part does not have a mask layer.
4.  BERT is good at NLU (Understanding), while GPT is good at NLG (Generation). NLG is harder because it has open ending.

### GPT-2
> OpenAI. 2019. "Language models are unsupervised multitask learners."

For more detailed introduction, see [my blog]({{site.baseurl}}/posts/GPT-1-2/#gpt-2).

1. GPT-2 = Decoder (in Transformer) + Pre-Training + Turning Fine-tuning to Pre-Training + More Parameters.
2. Enhanced pre-training. Eliminated fine-tuning.
  1. The unsupervised objective of the earlier pre-training is demonstrated to be the same as the supervised objective of the later fine-tuning.
  2. The downstream tasks can be reconstructed to be descripted in the form used in pre-training.
  3. A competent generalist is not an agregation of narrow experts.
3. The scaling law is initially emerging: The more parameters, the better the performance, and the improvement is very stable.
4. The Number of Parameters: 1.5B

### Scaling Law
> OpenAI. 2020.1. "Scaling Laws for Neural Language Models."


### GPT-3
> OpenAI. 2020.5. "Language models are few-shot learners."

## Code

1. Ollama
   1. [Homepage](https://ollama.com/)
   2. [GitHub](https://github.com/ollama/ollama)
2. OpenSpiel
   1. [GitHub: Chat Games](https://github.com/google-deepmind/open_spiel/tree/master/open_spiel/python/games/chat_games)
   2. [GitHub: A PSRO Example](https://github.com/google-deepmind/open_spiel/blob/master/open_spiel/python/examples/chat_game_psro_example.py)


## Memo
1. The number of parameters.
   1. 1K = 1,000
   2. 1M = 1,000,000 = 一百万
      1. 20M = 两千万
      2. 200M = 两亿
   3. 1B = 1,000,000,000 = 十亿