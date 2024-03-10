---
title: Misc Code Toolbox
date: 2023-04-10 02:40:00 +0800
categories: [Code Utils]
tags: [Tech, Code Utils, Toolbox]
math: True
---

> This note will be consistently updated.
{: .prompt-info }

---

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
6. `git remote add origin [xxx.git]`

- The content inside `[]` is a variable.
- The `origin` is a default name that refers to the original location (i.e., the remote repository's URL) from which you cloned the repository. When you use the `git clone [URL]` command to clone a repository, Git automatically names the remote repository's URL as `origin`.

### Lazy commit

Create a `snippet` in the software `Termius`:

```bash
git add .
git commit -m "[commit_info]"
git push origin [branch_name]
```

Then enter your `github name` and your `git temporary token`.

### Download
1. Create a new terminal at the folder where you want to download the repo. The downloaded repo will be a subfolder, and its contents are what you see on the webpage.
2. `git clone [repo_URL(xxx.git)]` (Download.) 
3. Enter the subfolder.

The `git clone` will create a subfolder (named after the repo) in your current folder.

### Branch

- `git branch -a` (List all the braches.)
- `git checkout [branch_name]` (Switch to a branch.) 
- `git checkout -b [branch_name]` (Create a branch.)

### Get updated

#### Way 1
1. `git fetch origin` (Retrieve the changes from all branches.)
2. `git merge origin/[remote_branch_name] [local_branch_name]`


#### Way 2
`pull` = `fetch` + `merge`

`git pull origin [remote_branch_name]` (Update the code in your current local branch.)


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

## print

### Separator line
```python
import shutil

terminal_columns = shutil.get_terminal_size().columns
print('=' * terminal_columns)
```


## Code Visualization Tools

TODO

[Sourcetrail](https://github.com/CoatiSoftware/Sourcetrail)

## ConfigDict

### Source

`configdict.py`

```python
# https://github.com/google-research/exoplanet-ml/blob/master/exoplanet-ml/tf_util/configdict.py
# Copyright 2018 The Exoplanet ML Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# 1/8/2020 - edited for Python 3

"""Configuration container for TensorFlow models.
A ConfigDict is simply a dict whose values can be accessed via both dot syntax
(config.key) and dict syntax (config['key']).
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


def _convert_sub_configs(value):
    if isinstance(value, dict):
        return ConfigDict(value)

    if isinstance(value, list):
        return [_convert_sub_configs(subvalue) for subvalue in value]

    return value


class ConfigDict(dict):
    """Configuration container class."""

    def __init__(self, initial_dictionary=None):
        """Creates an instance of ConfigDict.
        Args:
          initial_dictionary: Optional dictionary or ConfigDict containing initial
          parameters.
        """
        if initial_dictionary:
            for field, value in initial_dictionary.items():
                initial_dictionary[field] = _convert_sub_configs(value)
            super().__init__(initial_dictionary)
        else:
            super().__init__()

    def __setattr__(self, attribute, value):
        self[attribute] = _convert_sub_configs(value)

    def __getattr__(self, attribute):
        try:
            return self[attribute]
        except KeyError as e:
            raise AttributeError(e)

    def __delattr__(self, attribute):
        try:
            del self[attribute]
        except KeyError as e:
            raise AttributeError(e)

    def __setitem__(self, key, value):
        super().__setitem__(key, _convert_sub_configs(value))
```

### Example

```python
from util_configdict import ConfigDict

config = ConfigDict()

config.main = ConfigDict()
config.main.exp_name = 'exp1a_aligned_honest_map3'

# ==================================================
config.env = ConfigDict()
config.env.map_height = 3
config.env.map_width = 3
config.env.max_step = 50

# ==================================================
config.train = ConfigDict()
config.train.n_episodes = 200000
config.train.period = 500

if __name__ == '__main__':
    print(f"config: {config}")
    print(f"type(config): {type(config)}")
    print(f"config.env: {config.env}")

    print(f"config.env.max_step: {config.env.max_step}")
    print(f"config.env['max_step']: {config.env['max_step']}")
    print(f"config['env']['max_step']: {config['env']['max_step']}")
    print(f"config['env'].max_step.: {config['env'].max_step}")
```

```
config: {'main': {'exp_name': 'exp1a_aligned_honest_map3'}, 'env': {'map_height': 3, 'map_width': 3, 'max_step': 50}, 'train': {'n_episodes': 200000, 'period': 500}}
type(config): <class 'util_configdict.ConfigDict'>
config.env: {'map_height': 3, 'map_width': 3, 'max_step': 50}
config.env.max_step: 50
config.env['max_step']: 50
config['env']['max_step']: 50
config['env'].max_step.: 50
```

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