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

## Per-Sample Gradient

- $\mathrm{batch\_size} = n$,
- $\boldsymbol x \to \mathrm{net}(\boldsymbol w) \to \boldsymbol y \to \boldsymbol L \to L_{scalar}$
  - $\boldsymbol w \gets \boldsymbol w + \frac{\alpha}{n}\cdot \frac{\partial L_{scalar}}{L_i} \cdot \frac{L_i}{\partial \boldsymbol w}$
- Accomplishing it by `for` costs lots of time.

### [Hook]()
- PyTorch中，可以自己定一个hook函数，给nn.Module登记
  - 登记完后，nn.Module在forward的时候会触发这个hook函数
  - 也可以选择让其在backward的时候触发hook函数
- hook函数的参数是固定的：(module, grad_input, grad_output)
  - hook函数被触发后，自动搜集当前触发状态下的这3个参数，因此可以用hook实现搜集一些中间量
  - grad_input是反向传播的量对module的input的梯度
- $\frac{\partial L}{\partial w} = \sum\limits_i \frac{\partial L}{\partial L_i} \cdot\frac{\partial L_i}{\partial y_i}\cdot\frac{\partial y_i}{\partial w}$
  - $\frac{\partial L}{\partial L_i} \cdot\frac{\partial L_i}{\partial y_i}=\mathrm{grad\_output}$

### Opacus
- 让PyTorch训练模型时能做差分隐私的一个库
- DP-SGD (Differentially-Private Stochastic Gradient Descent)
  - 要让Loss对每个sample的grad都做一个clip，再加个噪声
  - 所以要求per-sample gradient
- 他们也是用的hook来做的，但是是封装好了，可以直接用

### vmap
- v = vectorization
- 新函数 = `vmap`(要做的批量操作的函数，输入的量按哪个维度作分割)
- 批量操作的结果 = 新函数(批量的原函数的输入)
- 现在要批量求梯度，那么要给vmap传入个求梯度的函数
- vmap不支持autograd，但有函数代替
- 具体写在了22.9.14的实验进展里


## Python Profile
- Used to find performance bottlenecks.
- Can be easily done by clicking the button in the upper right corner, if you are using `PyCharm (Professional Edition)`.

> Check this [website](https://realpython.com/python-profiling/).
{: .prompt-info }

## Tmux
- `tmux ls`
- `tmux attach-session -t 0`

## Terminal Python Environment Initialization
```bash
source ~/.bash_profile
conda activate rlbasic
```

## Check the status of GPU or CPU
(... in a terminal)
- GPU: `nvidia-smi`
- CPU: `top`

## Github

### Create a repo
1. Click the green button `New` on the GitHub repo website.
2. Do **not** check the `Add a README file`.
3. Copy the link with the `.git` extension.
4. Create a directory locally and enter it in a terminal.
5. `git init`
6. `git remote add origin xxx.git`

### Lazy commit

Create a `snippet` in the software `Terminus`:

```bash
git add .
git commit -m "quick commit"
git push origin main
```

Then enter your `github name` and your `git temporary token`.

## Random

### Integer

```python
import torch
torch.randint(1,5,[2])	# [1,4]的整数，生成2个数，可重复
torch.randint(5,[1,2])	# [0,4]的整数，结果是类似这样的：tensor([[2, 4]])
```

```python
import random
random.randint(-1,2) # [-1,2]的整数，生成1个
```


```python
import random
numbers = random.choices(range(101), k=10) # [0,100]，生成10个，可以重复
print(numbers)

```

```python
import random
numbers = random.sample(range(101), 10) # [0,100]的整数，生成10个，不会重复
print(numbers)
```

```python
import random
items = [i for i in range(1, 6)] # [1,5]的整数列表
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

> Check the [PyTorch documentation](https://pytorch.org/docs/stable/distributions.html).
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

## Web Crawler

### Resources

1. [A nice blog cuiqingcai (in Chinese)](http://cuiqingcai.com/categories/Python/爬虫/).
2. 