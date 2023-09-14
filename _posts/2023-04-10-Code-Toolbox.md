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

1. [A nice blog cuiqingcai (in Chinese)](http://cuiqingcai.com/categories/Python/爬虫/).


## ML Tricks


### Dropout
> Generated by ChatGPT 4

**Dropout的原理**：

Dropout是一种正则化技巧，用于防止神经网络过拟合。它在训练期间随机地"丢弃"或“关闭”一部分神经元，即将它们的输出设置为0。这样做可以减少神经元之间的相互依赖，从而鼓励每个神经元独立地学习特征。

具体来说，Dropout操作如下：
1. 对于每个训练样本，在前向传播时，每个神经元都有概率`p`被设置为0。
2. 在反向传播时，被设置为0的神经元不会更新其权重。
3. 在测试或验证时，不使用dropout，但为了平衡因dropout导致的输出变化，我们将神经元的输出乘以`(1-p)`进行缩放。

**PyTorch中的Dropout代码示例**：

以下是使用PyTorch的`nn.Dropout`模块的简单示例：

``` python
import torch
import torch.nn as nn

class SimpleNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleNN, self).__init__()
        
        # 定义一个简单的三层神经网络，其中包含一个dropout层
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(p=0.5)  # 设置dropout概率为0.5
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)  # 在隐藏层后应用dropout
        x = self.fc2(x)
        return x

# 创建一个简单的模型实例
model = SimpleNN(input_dim=10, hidden_dim=20, output_dim=2)
input_tensor = torch.randn(5, 10)  # 创建一个5x10的随机输入张量
output = model(input_tensor)
print(output)
```

### Mask


#### As indices
```python
import torch

data = torch.arange(5)  # tensor([0, 1, 2, 3, 4])
mask = data <= 2  # tensor([ True,  True,  True, False, False]); any condition is ok
data[mask] = 0  # tensor([0, 0, 0, 3, 4])
```

#### Retain gradients
```python
import torch

data_shape = 5, 3
data = torch.arange(15, dtype=torch.float64).view(data_shape).requires_grad_(True)

mask = data <= 6  # any condition is ok
data_masked = data * mask

loss = data_masked.sum()
loss.backward()
grad1 = data_masked.grad
grad2 = data.grad

'''
data_masked: 
tensor([[0., 1., 2.],
        [3., 4., 5.],
        [6., 0., 0.],
        [0., 0., 0.],
        [0., 0., 0.]], dtype=torch.float64, grad_fn=<MulBackward0>)

data_masked.grad: None

data.grad:
tensor([[1., 1., 1.],
        [1., 1., 1.],
        [1., 0., 0.],
        [0., 0., 0.],
        [0., 0., 0.]], dtype=torch.float64)
'''
```

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