---
title: Code Toolbox
date: 2023-04-10 02:40:00 +0800
categories: [Code]
tags: [toolbox]
math: True
---

> This note will be consistently updated.
{: .prompt-info }

---

## Random

### Integer

```python
import torch
torch.randint(1,5,[2])	# 1到4的整数，生成2个数，可重复
torch.randint(5,[1,2])	# 0到4的整数，结果是类似这样的：tensor([[2, 4]])
```

```python
import random
random.randint(-1,2) # -1到2的整数，包含-1和2，生成1个
```

```python
import random
items = [i for i in range(1, 6)] # 1到5的整数列表
random.shuffle(items)	#打乱顺序
print(items[:k])		#保留前k个；k自己取
```

### Real

#### Uniform distribution
```python
torch.rand(2, 3) # size (2,3)，每一个数都是从[0,1)均匀采样的
```

#### Normal distribution
```python
torch.randn(2, 3) # size (2,3)，每一个数都是从正态分布N(0,1)采样的
```

> Check the [pytorch documentation](https://pytorch.org/docs/stable/distributions.html).
{: .prompt-info }

## Customized Module Template
```python
import torch
import torch.nn as nn
import os


class net_base(nn.Module):
    def __init__(self, n_channels, config, name, device=None):
        super().__init__()
        self.n_channels = n_channels
        # padding for keeping the width and height of input unchanged: kernel=3, padding=1; kernel=5, padding= 2; ...
        self.conv_layer = nn.Sequential(
            nn.Conv2d(n_channels, config.nn.n_filters, config.nn.kernel, config.nn.stride,
                      padding=int((config.nn.kernel - 1) / 2), dtype=torch.double), nn.ReLU(),
        )

        obs_vector = config.env.map_height * config.env.map_width * config.nn.n_filters
        self.mlp = nn.Sequential(
            nn.Linear(obs_vector, config.nn.hidden_width, dtype=torch.double), nn.ReLU(),
            nn.Linear(config.nn.hidden_width, config.nn.hidden_width, dtype=torch.double), nn.ReLU(),
        )

        self.name = name
        self.checkpoint_file = os.path.join(config.path.saved_models, config.main.exp_name, name)
        # print(os.getcwd())
        if not os.path.exists(os.path.join(config.path.saved_models, config.main.exp_name)):
            os.makedirs(os.path.join(config.path.saved_models, config.main.exp_name), exist_ok=True)

        self.device = device
        self.to(self.device)

    def forward(self, x):
        return self.mlp(self.conv_layer(x).view(x.shape[0], -1))

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)
        return

    def load_checkpoint(self, path=None):
        if not path:
            self.load_state_dict(torch.load(self.checkpoint_file, map_location=self.device))
        else:
            load_path = os.path.join(path, self.name)
            self.load_state_dict(torch.load(load_path, map_location=self.device))

```

