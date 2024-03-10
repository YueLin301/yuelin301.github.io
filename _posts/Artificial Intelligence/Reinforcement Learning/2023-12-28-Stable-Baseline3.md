---
title: Stable Baseline 3
date: 2023-12-28 10:40:00 +0800
categories: [Artificial Intelligence, Reinforcement Learning]
tags: [Tech, AI, RL, Framework]
math: True
---

## Getting Started

> Stable Baselines3 (SB3) is a set of reliable implementations of reinforcement learning algorithms in PyTorch. — Stable Baseline3 Docs.

### Resources
1. [[Stable Baseline3 Docs]](https://stable-baselines3.readthedocs.io/en/master/)
2. [[Stable Baseline3 Repo]](https://github.com/DLR-RM/stable-baselines3?tab=readme-ov-file)


### Installation

```bash
pip install stable-baselines3
```

### An Official Example

```python
# https://github.com/DLR-RM/stable-baselines3?tab=readme-ov-file#example

import gymnasium as gym
from stable_baselines3 import A2C

env = gym.make("CartPole-v1", render_mode="rgb_array")

model = A2C("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10_000)

vec_env = model.get_env()
obs = vec_env.reset()
for i in range(1000):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)
    vec_env.render("human")
    # VecEnv resets automatically
    # if done:
    #   obs = vec_env.reset()

env.close()
```

Note that, `model = A2C("MlpPolicy", env, verbose=1)` means the model takes the environment as input, and it works well with the gym environments.

## Structure

Knowing how to call the developed algorithms is just the first step. I also want to design my own algorithms based on its framework.

My stable_baseline3 package path:
`/opt/anaconda3/envs/rlbasic/lib/python3.8/site-packages/stable_baselines3`


### Inheritance

#### UML of All Classes

```bash
pyreverse -o pdf -p stable_baseline3_diagram /opt/anaconda3/envs/rlbasic/lib/python3.8/site-packages/stable_baselines3
```


<iframe src="{{ site.baseurl }}/assets/img/2023-12-28-Stable-Baseline3/classes_stable_baseline3_diagram.pdf" style="width:100%; height:500px;" frameborder="1"></iframe>

#### UML of A2C

`ABC (Abstract Base Class) -> BaseAlgorithm -> OnPolicyAlgorithm -> A2C`

<!-- Paths:
- `/opt/anaconda3/envs/rlbasic/lib/python3.8/abc.py/ABC`
- `stable_baselines3/common/base_class.py/BaseAlgorithm`
- `stable_baselines3/common/on_policy_algorithm.py/OnPolicyAlgorithm`
- `stable_baselines3/a2c/a2c.py/A2C` -->

For Abstract Base Class, you can check [my blog]({{site.baseurl}}/posts/Python/#abstract-base-class) for details.

Paths:
`/opt/anaconda3/envs/rlbasic/lib/python3.8/site-packages/`
- A2C: `stable_baselines3/a2c`
  - `stable_baselines3/a2c/a2c.py`
  - `stable_baselines3/a2c/policies.py`
- common: `stable_baselines3/common`
  - `stable_baselines3/common/base_class.py`
  - `stable_baselines3/common/on_policy_algorithm.py`
  - `stable_baselines3/common/policies.py`
  - utils
    <!-- - `stable_baselines3/common/torch_layers.py`
    - `stable_baselines3/common/buffers.py`
    - `stable_baselines3/common/type_aliases.py`
    - `stable_baselines3/common/utils.py`
    - `stable_baselines3/common/distributions.py`
    - `stable_baselines3/common/logger.py`
    - `stable_baselines3/common/noise.py` -->


```bash
pyreverse -o pdf -p stable_baseline3_A2C_diagram /opt/anaconda3/envs/rlbasic/lib/python3.8/site-packages/stable_baselines3/a2c /opt/anaconda3/envs/rlbasic/lib/python3.8/site-packages/stable_baselines3/common
```

<iframe src="{{ site.baseurl }}/assets/img/2023-12-28-Stable-Baseline3/classes_stable_baseline3_A2C_diagram.pdf" style="width:100%; height:500px;" frameborder="1"></iframe>

<!-- ```bash
pyreverse -o pdf -p stable_baseline3_A2C_diagram_compact /opt/anaconda3/envs/rlbasic/lib/python3.8/site-packages/stable_baselines3/a2c /opt/anaconda3/envs/rlbasic/lib/python3.8/site-packages/stable_baselines3/common/base_class.py /opt/anaconda3/envs/rlbasic/lib/python3.8/site-packages/stable_baselines3/common/on_policy_algorithm.py /opt/anaconda3/envs/rlbasic/lib/python3.8/site-packages/stable_baselines3/common/policies.py /opt/anaconda3/envs/rlbasic/lib/python3.8/site-packages/stable_baselines3/common/torch_layers.py /opt/anaconda3/envs/rlbasic/lib/python3.8/site-packages/stable_baselines3/common/buffers.py /opt/anaconda3/envs/rlbasic/lib/python3.8/site-packages/stable_baselines3/common/type_aliases.py /opt/anaconda3/envs/rlbasic/lib/python3.8/site-packages/stable_baselines3/common/utils.py /opt/anaconda3/envs/rlbasic/lib/python3.8/site-packages/stable_baselines3/common/distributions.py /opt/anaconda3/envs/rlbasic/lib/python3.8/site-packages/stable_baselines3/common/logger.py /opt/anaconda3/envs/rlbasic/lib/python3.8/site-packages/stable_baselines3/common/noise.py
``` -->



---

> The following part has not been finished yet.
{: .prompt-warning }

It seems that this module is designed for single-agent tasks.