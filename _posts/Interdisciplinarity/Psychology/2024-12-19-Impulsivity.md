---
title: Impulsivity
date: 2024-12-18 12:30:00 +0800 
categories: [Interdisciplinarity, Psychology]
tags: [Life, Interdisciplinarity, Psychology, Personality]
math: True
# pin: True
---

这学期参加了8次seminar，这是比较感兴趣的一个

> Huang, Yuqi, et al. "Impulsivity is a stable, measurable, and predictive psychological trait." Proceedings of the National Academy of Sciences 121.24 (2024): e2321758121.
{:.prompt-info}


从这个入手可以看一下一个personality trait是怎么develop出来的，很适合入门
- Impulsivity的定义不清晰，现在规范定义是multi-dimensional的，包含好几个维度
- Measurements
    - self-report
    - behavior tasks
    - (internet-based platform, tasks)
    - (give incentives and feedback. tell them how impulsive they are, compared to other players)
- Correlation
    - in their outcomes, behavior tasks has nearly zero correlation with self-report outcomes
    - 方差解释度
- 信度和效度
    - 信度
        - temporal stability
        - test-retest reliability of scales
    - 效度
        - the measure with the highest correlation with a behavior
        - how can we predict behaviors using correlations? Supervised machine learning (random forest is the best among 7 algos, posterior); 比较直接，他们claim说是第一个应用的
- Develop a new scale to measure: Adjustable Impulsivity Scale
    - 所以我的理解是一个trait要对应设计一套调查问卷去measure


里面的behavior tasks里居然有bandit，还有几个打气球的。我觉得可能有点怪，因为这个和经验和对任务的熟悉程度有关，一个risk neutral的也可以有不同的表现程度，所以根据表现程度来评价impusivity或者说risk态度可能有点问题

后续的工作里还特别提到了在中国的监狱里测试，看看impusivity区别，他们认为大家会先验地认为囚犯的impusivity比较大，给的奖励是零食和活动时间

给囚犯分了几类，诈骗犯的impusivity和平常人没多大区别，其他几类的impusivity比较大