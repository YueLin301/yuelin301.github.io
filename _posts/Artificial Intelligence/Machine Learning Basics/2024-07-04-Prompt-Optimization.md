---
title: Prompt Optimization
date: 2024-07-04 12:30:00 +0800
categories: [Artificial Intelligence, Machine Learning Basics]
tags: [Tech, AI, ML, NLP, LLM]
math: True
---

## Verbalized Machine Learning

> [Verbalized Machine Learning: Revisiting Machine Learning with Language Models](https://arxiv.org/pdf/2406.04344)

两个大模型
- 一个大模型用来执行任务
- 一个大模型用来输出前一个大模型的prompt

执行任务大模型的表现可以算loss，
然后把这个loss以文本形式告诉给prompt大模型，
有了这个信息那么prompt大模型就可以根据不同的loss来改变输出的prompt

有了这个反馈链就可以优化了，就是这个反馈不是数值形式算梯度的，而是文本化的

## TextGrad

[TextGrad: Automatic “Differentiation” via Text](https://arxiv.org/pdf/2406.07496)


三个大模型
- 一个大模型用来执行任务
- 一个大模型用来输出 执行大模型的prompt
- 一个大模型用来评估 执行大模型的loss

