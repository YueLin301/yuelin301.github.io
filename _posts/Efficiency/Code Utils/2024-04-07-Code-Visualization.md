---
title: Code Visualization
date: 2024-04-07 14:40:00 +0800
categories: [Efficiency, Code Utils]
tags: [Tech, Efficiency, Code_Utils, Visualization]
math: True
---


## Function Call Graph

Not working:
- `pyan3`
- `pycallgraph`
- `pycallgraph2`

## Inheritance Visualization

- Example 1: See [my blog]({{site.baseurl}}/posts/Python-Toolbox/#inheritance-visualization).
- Example 2:
```bash
pyreverse -o png -p outputed_diagram main.py Agent.py Buffer.py Model.py Nets.py
```

## Computational Graph Visualization

See [my blog]({{site.baseurl}}/posts/Computation-Graph-Visualization/).
