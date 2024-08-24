---
title: LyPythonToolbox
date: 2024-04-05 14:40:00 +0800
categories: [Efficiency, Code Utils]
tags: [Tech, Efficiency, Code_Utils, Toolbox, Library]
math: True
---

## Resources
1. [Github Repo](https://github.com/YueLin301/LyPythonToolbox)
2. [My Full Code Toolbox]({{site.baseurl}}/categories/code-utils/)

## Install

Install:
```bash
pip install LyPythonToolbox
```

Update:
```bash
pip install --upgrade LyPythonToolbox
```

## Print Tricks

### `lyprint_separator`

```python
from LyPythonToolbox import lyprint_separator
lyprint_separator()  # print "=" * terminal_width
lyprint_separator("-")  # print "-" * terminal_width
```

### `lyprint_flash`

```python
from LyPythonToolbox import lyprint_flash
import time

for i in range(20):
    lyprint_flash(f"Fake Epoch: {i}; Info: {str(i) * 240}")
    time.sleep(0.5)
```

### `@lyprint_elapsed_time`

```python
from LyPythonToolbox import lyprint_elapsed_time

@lyprint_elapsed_time
def print_howmanyhaha(times):
    for i in range(times):
        print("haha")
        result = i
    return result

result = print_howmanyhaha(2)
print(f"result={result}")
```

```
haha
haha
Elapsed Time of print_howmanyhaha: 2.7894973754882812e-05s
result=1
```


## ConfigDict

### Copyright

> [exoplanet-ml/exoplanet-ml/tf_util/configdict.py](https://github.com/google-research/exoplanet-ml/blob/master/exoplanet-ml/tf_util/configdict.py)
{:.prompt-info}

```python
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
```

### Example

```python
from LyPythonToolbox import ConfigDict

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

### Global Access

In Python, mutable data types like lists and dictionaries are passed by reference. When you pass a list or dictionary to a function, what you are actually passing is a reference to the original data. Therefore, when you modify the data inside the function, it affects the same piece of data, resulting in changes to the data outside the function as well.

```python
def modify_data(my_list, my_dict):
    my_list.append(4)  # 向列表中添加一个元素
    my_dict['d'] = 4  # 向字典中添加一个键值对

# 定义列表和字典
original_list = [1, 2, 3]
original_dict = {'a': 1, 'b': 2, 'c': 3}

# 调用函数
modify_data(original_list, original_dict)

# 查看修改后的列表和字典
print(original_list)  # 输出: [1, 2, 3, 4]
print(original_dict)  # 输出: {'a': 1, 'b': 2, 'c': 3, 'd': 4}
```

```python
from LyPythonToolbox import ConfigDict

config = ConfigDict()
config.haha = "haha"


class a_class:
    def __init__(self, input_config) -> None:
        self.config = input_config

    def cat1(self):
        self.config.haha += "wuwu"
        return
    
a = a_class(config)
b = a_class(config)
a.cat1()
print(f"a.config.haha: {a.config.haha}")
print(f"b.config.haha: {b.config.haha}")
print(f"config.haha: {config.haha}")
print(f"id(a.config.haha) == id(b.config.haha) == id(config.haha): {id(a.config.haha) == id(b.config.haha) == id(config.haha)}")
print(f"a.config.haha is config.haha and b.config.haha is config.haha: {a.config.haha is config.haha and b.config.haha is config.haha}")
```

```
a.config.haha: hahawuwu
b.config.haha: hahawuwu
config.haha: hahawuwu
id(a.config.haha) == id(b.config.haha) == id(config.haha): True
a.config.haha is config.haha and b.config.haha is config.haha: True
```