---
title: Misc Code Toolbox
date: 2023-04-10 02:40:00 +0800
categories:
  - Efficiency
  - Code Utils
tags:
  - Tech
  - Efficiency
  - Code_Utils
  - Toolbox
math: true
---

> This note will be consistently updated.
{: .prompt-info }

---

## Tmux
太久没连服务器连这个怎么用都快忘了...不要想太复杂的操作，我用这个的原因就只有两个，第一个原因是用这个在服务器上运行python文件后，我再断开服务器的连接，这个还能在后台跑；第二个原因是，可以只用ssh连服务器一次就可以用tmux来用多个shell，比如同时跑两个python文件，这个应噶就是终端复用的意思

Tmux有3个逻辑层次，分别是：session会话、window窗口、panel窗格。打开这个进程后，底下显示的是session会话的名字，进去session之后能输入终端命令的那个直接交互的地方就是panel窗格

进入一个session会话默认有一个window窗口并在这个窗口里有一个panel窗格，我根本不需要用到多窗口和多窗格；如果我要跑多个python文件我只要多个会话就行了

### 会话操作

#### 创建会话

下面这个命令：如果tmux里啥也没有就创建个名字是0的会话，如果tmux里面已经有创建的会话了的话会怎么样就不好说了，不用管（我试过有时候是重连到最新的那个会话，有时候是创建个新的名字是编号的会话；可能和“有没有再创建新会话”、“新创建的会话名字是不是编号”有关）

```bash
tmux
```

创建一个会话，名字是默认的数字，这个数字具体是多少不用管，反正是递增的（应该是`历史创建过多少名字是编号数字的会话数 -1 +1`，其中`-1`是因为编号从0开始，`+1`是因为是新多加了个会话）

```bash
tmux new
```

创建指定名字的会话，`-s`是指的`session_name`的意思

```bash
tmux new -s ${session_name}
```

#### 从会话中脱离

快捷键`Command + B` `D`，等价于下面这个命令

```bash
tmux detach
```

#### 显示所有会话

```bash
tmux ls
```

#### 重新连到指定会话

这里的`-t`表示的是`target`

```bash
tmux attach-session -t ${session_name}
```

简写是

```bash
tmux at -t ${session_name}
```

#### 重命名会话

把指定编号的会话名重命名，这里的`-t`表示的也是`target`

```bash
tmux rename-session -t ${session_name} ${session_new_name}
```

#### 关闭会话

关闭指定会话，这里的`-t`表示的也是`target`

```bash
tmux kill-session -t ${session_name}
```

关闭所有会话

```bash
tmux kill-server
```

### 窗口操作

我不用

### 窗格操作

少用但是记一下备用

#### 翻页

进入翻页模式的快捷键是`Control + B` `[`，然后`page up/down`和`up/down`和鼠标触控板就都可以用来翻页了；退出翻页模式是`Control + C`

#### 新窗格

1. 右边创建个新窗格，左右分屏：`Control + B` `%`
2. 下面创建个新窗格，上下分屏：`Control + B` `"`

#### 关闭当前窗格

`Control + B` `x`

#### 选择窗格

1. 选上次用过的窗格：`Control + B` `;`
2. 根据屏幕位置来选窗格：`Control + B` `（方向键）`
3. 根据编号连选窗格：`Control + B` `q`，此时会显示窗格编号，然后按下数字就行

#### 切换布局

就是可以把水平分屏换成垂直分屏这种

1. 换成下一个布局：`Control + B` `space`
2. 换成指定布局：`Control + B` `Option + (1-5)`
3. 最大化当前窗格：`Control + B` `Z`；再按一次就又恢复成之前的布局


## Terminal Python Environment Initialization
```bash
source ~/.bash_profile
conda activate rlbasic
```

## Check the status of GPU or CPU
(... in a terminal)
- GPU: `nvidia-smi`
- CPU: `top`


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

1. [A nice blog cuiqingcai (in Chinese)](https://cuiqingcai.com/categories/Python/爬虫/).



## argparse

> - [A tutorial](https://docs.python.org/3/howto/argparse.html).
> - [Documentation](https://docs.python.org/3/library/argparse.html).
{: .prompt-info }

`argparse_test.py`:
```python
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("positional_arg1", help="A message to the user.")
    parser.add_argument("positional_arg2")
    parser.add_argument("-o1", "--optional_arg1")
    parser.add_argument("-o2", "--optional_arg2", action="store_true", help="Input -o2 will set it True.")
    parser.add_argument("-o3", "--optional_arg3", default="wuwu")
    args = parser.parse_args()
    print(args, args.positional_arg1, args.positional_arg2, sep="\n")

```

Run (help):
- `python3 argparse_test.py -h`
- `python3 argparse_test.py --help`

Result:
```bash
usage: argparse_test.py [-h] [-o1 OPTIONAL_ARG1] [-o2] [-o3 OPTIONAL_ARG3] positional_arg1 positional_arg2

positional arguments:
  positional_arg1       A message to the user.
  positional_arg2

optional arguments:
  -h, --help            show this help message and exit
  -o1 OPTIONAL_ARG1, --optional_arg1 OPTIONAL_ARG1
  -o2, --optional_arg2  Input -o2 will set it True.
  -o3 OPTIONAL_ARG3, --optional_arg3 OPTIONAL_ARG3
```

Run:
```bash
python3 argparse_test.py xixi haha
```

Result:
```bash
Namespace(optional_arg1=None, optional_arg2=False, optional_arg3='wuwu', positional_arg1='xixi', positional_arg2='haha')
xixi
haha
```

Run:
```bash
python3 argparse_test.py xixi haha -o2 -o1 heihei
```

Result:
```bash
Namespace(optional_arg1='heihei', optional_arg2=True, optional_arg3='wuwu', positional_arg1='xixi', positional_arg2='haha')
xixi
haha
```

## Change Dir to the Project Dir

> Make sure that there is a `README.md` file under the project directory. This file is used as an identifier.
{: .prompt-warning }

```python
import os


def find_project_directory(identifier_file):
    current_path = "."

    while True:
        if identifier_file in os.listdir(current_path):
            return os.path.abspath(current_path)

        parent_path = os.path.join(current_path, "..")
        if os.path.abspath(parent_path) == os.path.abspath(current_path):
            return None

        current_path = parent_path


def cd_project_directory():
    project_directory = find_project_directory("README.md")

    if project_directory:
        os.chdir(project_directory)
    else:
        print("Project directory not found.")

    return project_directory


if __name__ == '__main__':
    project_directory = cd_project_directory()
    print('All done.')
```


## Code Visualization Tools

TODO

[Sourcetrail](https://github.com/CoatiSoftware/Sourcetrail)

## Python Type

For python 3.8:

```python
import torch
import numpy as np
from typing import Union, Tuple, List


def make_decision(
    state: Union[np.ndarray, torch.Tensor], info: str
) -> Tuple[int, List[int]]:
    action, pi = function(state)
    return action, pi

```

- `state` is a `np.ndarray` or `torch.Tensor`
- `info` is a `str`
- The output should be an `int` and a `List` of `int`


```python
state = ...  # 你的变量

if type(state) is int:
    print("处理整数")
elif type(state) is str:
    print("处理字符串")
elif type(state) is list:
    print("处理列表")
else:
    print("处理其他类型")
```

在Python中，当你直接使用`list`、`int`、`str`等作为类型时，你是在引用内置的类型。这些是Python语言的一部分，用于表示最基本的数据结构。例如，当你检查一个变量的类型是否是列表时，使用`type(state) is list`是检查这个变量是否正好是内置的`list`类型。

另一方面，当你在使用类型注解（Type Hints）或者在处理类型检查时，特别是在使用Python的标准类型提示模块`typing`时，`List`（大写L）是`typing`模块中定义的一个泛型类型。它用于在类型提示中表示一个列表，其中可以包含特定类型的元素。例如，`List[int]`表示一个整数列表。这种用法主要是为了提供更详细的类型信息，帮助静态类型检查器（如mypy）和IDE进行更准确的代码分析。

简单来说，`list`是Python的内置列表类型，而`List`（来自`typing`模块）是用于类型提示的泛型版本，允许指定列表中元素的类型。它们的用途不同：

- 使用`type(state) is list`是在运行时判断变量的类型。
- 使用`List`（例如，在函数定义中）是为了提供静态类型信息，以改善代码的可读性和安全性，但它不会影响代码的运行时行为。