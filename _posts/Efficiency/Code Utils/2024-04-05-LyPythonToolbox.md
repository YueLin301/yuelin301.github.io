---
title: LyPythonToolbox
date: 2024-04-05 14:40:00 +0800
categories: [Efficiency, Code Utils]
tags: [Tech, Efficiency, Code Utils, Toolbox]
math: True
---

## Resources
1. [Github Repo](https://github.com/YueLin301/LyPythonToolbox)
2. [My Full Code Toolbox]({{site.baseurl}}/categories/code-utils/)

## Install

```bash
pip install LyPythonToolbox
```

## Print Tricks

### `lyprint_separator`

```python
from LyPythonToolbox import lyprint_separator
lyprint_separator()  # print "=" * terminal_width
```

```python
from LyPythonToolbox import lyprint_separator
lyprint_separator("-")  # print "-" * terminal_width
```

### `lyprint_flash`

```python
from LyPythonToolbox import lyprint_flash
if __name__ == "__main__":
    import time
    for i in range(100):
        lyprint_flash(f"Fake Epoch: {i}; Info: {str(i) * 240}")
        time.sleep(0.5)
```

### `@lyprint_elapsed_time`

```python
from LyPythonToolbox import lyprint_elapsed_time
if __name__ == "__main__":

    @lyprint_elapsed_time
    def print_howmanyhaha(times=2):
        for i in range(times):
            print("haha")
            r = i
        return r

    r = print_howmanyhaha(3)
    print(r)
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