---
title: Computation Graph Visualization
date: 2023-06-25 02:00:01 +0800
categories: [Code]
tags: [computation graph, visualization, torchviz, torchsummary, graphviz, reparameterization]
math: True
---

## PyTorchviz
### Basics
1. Install
   1. `brew install graphviz` (or [here](https://graphviz.org/download/#mac))
   2. `pip install torchviz`
2. Documentation: [Github](https://github.com/szagoruyko/pytorchviz)
3. Official examples: [Colab](https://colab.research.google.com/github/szagoruyko/pytorchviz/blob/master/examples.ipynb)

> If a node represents a backward function, it is gray. Otherwise, the node represents a tensor and is either blue, orange, or green:
> - Blue: reachable leaf tensors that requires grad (tensors whose `.grad` fields will be populated during `.backward()`)
> - Orange: saved tensors of custom autograd functions as well as those saved by built-in backward nodes
> - Green: tensor passed in as outputs
> - Dark green: if any output is a view, we represent its base tensor with a dark green node.

### Example 1: Basics

```python
import torch
from torchviz import make_dot

if __name__ == '__main__':
    batch_size, input_size, hidden_size, output_size = 100, 4, 2, 1

    critic = torch.nn.Sequential(
        torch.nn.Linear(in_features=input_size, out_features=hidden_size, bias=False, dtype=torch.float),
        torch.nn.Tanh(),
        torch.nn.Linear(in_features=hidden_size, out_features=output_size, bias=True, dtype=torch.float),
    )
    states = torch.rand(batch_size, input_size, dtype=torch.float)
    values = critic(states).squeeze()

    graph = make_dot(values, params=dict(critic.named_parameters()), show_attrs=True, show_saved=True)
    graph.view()

```

### Example 2: Without Net

```python
import torch
from torchviz import make_dot

if __name__ == '__main__':
    x = torch.rand(100, 5, requires_grad=True)
    f1 = 2 * x + 3

    f2 = torch.softmax(f1, dim=-1)
    g = 5 * f1

    h = f2+g
    graph = make_dot(h, show_attrs=True, show_saved=True)
    graph.view()
```

### Example 3: Detach

```python
import torch
from torchviz import make_dot

if __name__ == '__main__':
    x = torch.rand(100, 5, requires_grad=True)
    f1 = 2 * x + 3

    f2 = torch.softmax(f1, dim=-1)
    g = 5 * f1.detach()

    h = f2+g
    graph = make_dot(h, show_attrs=True, show_saved=True)
    graph.view()
```

### Example 4: $\nabla_\theta a$ ❌
```python
import torch
from torchviz import make_dot

if __name__ == '__main__':
    batch_size, input_size, output_size = 10, 4, 2

    actor = torch.nn.Sequential(
        torch.nn.Linear(in_features=input_size, out_features=output_size, bias=False, dtype=torch.float),
        torch.nn.Softmax(dim=-1)
    )

    states = torch.rand(batch_size, input_size, dtype=torch.float)
    pi_list = actor(states)
    distributions = torch.distributions.Categorical(pi_list)
    actions = distributions.sample().squeeze(dim=0)

    graph = make_dot(actions, params=dict(actor.named_parameters()), show_attrs=True, show_saved=True)
    graph.view()
```
No gradient.


### Example 5: $\nabla_\theta a$ with Gumbel-Softmax (Reparameterization)
```python
import torch
from torchviz import make_dot

if __name__ == '__main__':
    batch_size, input_size, output_size = 10, 4, 2
    temperature = 1

    actor = torch.nn.Sequential(
        torch.nn.Linear(in_features=input_size, out_features=output_size, bias=False, dtype=torch.float),
        torch.nn.Softmax(dim=-1)
    )

    states = torch.rand(batch_size, input_size, dtype=torch.float)

    pi_list = actor(states)
    logits = torch.log(pi_list)
    actions = torch.nn.functional.gumbel_softmax(logits, tau=temperature, hard=True)

    graph = make_dot(actions, params=dict(actor.named_parameters()), show_attrs=True, show_saved=True)
    graph.view()
```



## Model Structure

If the visualization object is an `nn.Module`, then its structure can be easily inspected using the approaches in this section.

### Print
```python
import torch

if __name__ == '__main__':
    batch_size, input_size, hidden_size, output_size = 100, 4, 2, 1

    critic = torch.nn.Sequential(
        torch.nn.Linear(in_features=input_size, out_features=hidden_size, bias=False, dtype=torch.float),
        torch.nn.Tanh(),
        torch.nn.Linear(in_features=hidden_size, out_features=output_size, bias=True, dtype=torch.float),
    )
    print(critic)

```

```
Sequential(
  (0): Linear(in_features=4, out_features=2, bias=False)
  (1): Tanh()
  (2): Linear(in_features=2, out_features=1, bias=True)
)

Process finished with exit code 0
```

### Torchsummary
`pip install torchsummary`

```python
import torch
from torchsummary import summary

if __name__ == '__main__':
    batch_size, input_size, hidden_size, output_size = 100, 4, 2, 1

    critic = torch.nn.Sequential(
        torch.nn.Linear(in_features=input_size, out_features=hidden_size, bias=False, dtype=torch.float),
        torch.nn.Tanh(),
        torch.nn.Linear(in_features=hidden_size, out_features=output_size, bias=True, dtype=torch.float),
    )
    
    summary(critic, input_size=(batch_size, 4))

```

```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Linear-1               [-1, 100, 2]               8
              Tanh-2               [-1, 100, 2]               0
            Linear-3               [-1, 100, 1]               3
================================================================
Total params: 11
Trainable params: 11
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.00
Params size (MB): 0.00
Estimated Total Size (MB): 0.01
----------------------------------------------------------------

Process finished with exit code 0
```