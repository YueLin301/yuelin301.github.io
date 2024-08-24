---
title: "Some Insights of Misc Economics Papers"
date: 2024-06-06 14:40:00 +0800
categories: [Interdisciplinarity, Economics & Game Theory]
tags: [Tech, Interdisciplinarity, Economics, Game_Theory, Multi_Agents]
math: True
---

## 前言

都是查老师艾特我看的，看了不能白看，简单用中文记录一下；一些涉及idea的就不放了

## Honesty Is the Best Policy: Defining and Mitigating AI Deception

> Ward, Francis, et al. "Honesty is the best policy: defining and mitigating AI deception." Advances in Neural Information Processing Systems 36 (2024).

这篇文章的标题"Honesty Is the Best Policy"对我来说很mind blowing，因为这和info design是相悖的结论

起初我以为是他们的setting或者definition不太一样，但是他们没有关于info design的讨论；而且我过了一遍definition 3.7 deception和definition 3.1 belief，发现info design的setting是满足这俩定义的。这样的表述应该等于是在否定info design的分析，但是后面也没靠谱的理论分析，而是intuitively提了定义然后用实验empirically说明，那就显得很奇怪了；是我miss了哪里吗？

## Social Cost

> Chandra, Kartik, et al. "Cooperative Explanation as Rational Communication." (2024).

这个框架很直观

如果发生了我当前认知model下概率很小的事件，我就会提问，因为感觉这不太符合我的认知；然后让别人根据我的提问用贝叶斯算我当前的认知model，再纠正我的认知model

别人的优化目标是提高我改进model后的未来对这个事件发生概率的收益期望，减去一个cost；cost包含一个social cost，这个值的意思是如果解释者跟我说了一些我已经知道的东西，我就可能会感到尴尬，或者我会认为他误会了我的认知model

我觉得提出这个cost应该是他们的主要贡献吧？

提出的情景也很真实，整个看起来是个fully cooperative的任务，alice带bob走路，因为alice知道一些路况信息但bob不知道，所以bob可能以为alice在绕远路，因此就会问她，alice就会回答来更新bob的认知model。这看起来像个社科的文章，他们还雇人来玩这个游戏来统计结果

## AI Economists

> Zhang, Edwin, et al. "Position: Social Environment Design Should be Further Developed for AI-based Policy-Making." Forty-first International Conference on Machine Learning.

这篇是 AI Economists 的后续工作，还是同一批人做的

preferences aggregation是让agents选一种social welfare作为principal的优化目标：

- 比如agents可能会想让principal优化utilitarian目标（max sum u_i），让社会总体福利最大化；
- 或者agents也可能会想让principal优化egalitarian目标（max min u_i），让社会更公平

这个outer loop我觉得是让agents自己投票说明自己这个种群的倾向是什么，比如是更功利还是更注重公平，或者是其他social welfare函数

这里的principal自然就被设置为altruistic了，优化目标就直接被设置为这个投票出来的wocial welfare


