---
title: RL Toolbox
date: 2023-04-11 02:40:00 +0800
categories: [Artificial Intelligence, Reinforcement Learning]
tags: [Tech, AI, RL, Toolbox]
math: True
---

> This note will be consistently updated.
{: .prompt-info }

---

## PPO Tricks

There are a total of 37 tricks, among which 13 are relatively core.

> - [PPO paper](https://arxiv.org/pdf/1707.06347.pdf)
> - [The 37 Implementation Details of Proximal Policy Optimization](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/).
> - [影响PPO算法性能的10个关键技巧（附PPO算法简洁Pytorch实现） - Beaman的文章 - 知乎](https://zhuanlan.zhihu.com/p/512327050)
{: .prompt-info }

### Adam Optimizer Epsilon Parameter

```python
self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=config.lr_actor, eps=1e-5)
self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=config.lr_critic, eps=1e-5)
```

### Gradient Clip

```python
self.critic_optim.zero_grad()
critic_loss.backward()
torch.nn.utils.clip_grad_norm_(self.critic.parameters(), config.clip_range) # here
self.critic_optim.step()

self.actor_optim.zero_grad()
loss_actor.mean().backward()
torch.nn.utils.clip_grad_norm_(self.actor.parameters(), config.clip_range) # here
self.actor_optim.step()
```

### Tanh Activation Function

```python
# A continuous actor
class Actor(torch.nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(config.state_size, config.mlp_dim), torch.nn.Tanh(),
            torch.nn.LayerNorm(config.mlp_dim),
            torch.nn.Linear(config.mlp_dim, config.action_num),
            torch.nn.Tanh()
        )
        self.log_std = torch.nn.Parameter(torch.zeros(1, config.action_num))  # Gaussian std, learnable

    def forward(self, state):
        mean_raw = self.mlp(state)  # [-1, 1]
        mean = mean_raw * config.action_space_range  # [-max_a, max_a]

        std = torch.exp(self.log_std)  # std=exp(log_std)>0
        distribution = torch.distributions.Normal(mean, std)
        return distribution
```

### Policy Entropy

```python
def choose_action(self, state):
    state_tensor = torch.tensor(state).to(torch.float32).squeeze()
    distribution = self.actor(state_tensor)
    dist_entropy = distribution.entropy()
    action = distribution.sample().squeeze(dim=0)
    log_prob = distribution.log_prob(action)
    return action.detach().numpy(), log_prob, dist_entropy

negative_loss_actor = log_prob * TD_error.detach() + dist_entropy * config.entropy_coe
loss_actor = - negative_loss_actor
self.actor_optim.zero_grad()
loss_actor.mean().backward()
self.actor_optim.step()

```

### Reward Scaling

#### Incremental mean

I have a dataset with $n$ samples $\{x_1, x_2, \ldots, x_n\}$. The expectation of $X$ is calculated as 

$$
\mu_n = \frac{1}{n}\sum\limits_{i=1}^n x_i
$$

Then I get a new sample $x_{n+1}$, then the expectation of $X$ should be updated. And it can be represented by the current expectation: 

$$
\mu_{n+1} = \mu_n + \frac{1}{n+1}\left(x_{n+1} - \mu_n \right)
$$

Derivation:

$$
\begin{aligned}
  \mu_{n+1} =& \frac{1}{n+1}\sum\limits_{i=1}^{n+1} x_i 
  =  \frac{1}{n+1}\left(x_{n+1} + \sum\limits_{i=1}^n x_i \right) \\
  =&  \frac{1}{n+1}x_{n+1} + \frac{n}{n+1} \sum\limits_{i=1}^n x_i \\
  =&  \frac{1}{n+1}x_{n+1} + \left(1 - \frac{1}{n+1}\right) \mu_n \\
  =& \mu_n + \frac{1}{n+1}\left(x_{n+1} - \mu_n \right)
\end{aligned}
$$

To reduce the impact of previous samples, the coefficient is fixed as a constant $\alpha$:

$$
\begin{aligned}
  \mu_{n+1} 
  =& \mu_n + \alpha\left(x_{n+1} - \mu_n \right) \\
  =& \alpha\cdot x_{n+1} - \left(1-\alpha\right)\cdot\mu_n 
\end{aligned}
$$

#### Incremental Variance

$$
s_n^2 = s_{n-1}^2 + \frac{(x_n - \mu_{n-1})(x_n - \mu_n)}{n}
$$

---

$$
s_n^2 = \frac{1}{n} \sum_{i=1}^n (x_i - \mu_n)^2
$$

$$
\mu_n = \mu_{n-1} + \frac{1}{n}(x_n - \mu_{n-1})
$$

$$
\begin{aligned}
s_n^2 =& \frac{1}{n} \sum_{i=1}^n \left(x_i - \mu_{n-1} - \frac{1}{n}(x_n - \mu_{n-1})\right)^2 \\
=& \frac{1}{n} \sum_{i=1}^{n-1}(x_i - \mu_{n-1})^2 + \frac{1}{n}(x_n - \mu_{n-1})^2 - 2\frac{1}{n}(x_n - \mu_{n-1})\sum_{i=1}^{n-1}(x_i - \mu_{n-1}) + \frac{1}{n^2}(x_n - \mu_{n-1})^2\sum_{i=1}^{n-1}1
\end{aligned}
$$


$$
\sum_{i=1}^{n-1}(x_i - \mu_{n-1}) = 0
$$

$$
\begin{aligned}
s_n^2 =& \frac{1}{n} \sum_{i=1}^{n-1}(x_i - \mu_{n-1})^2 + \frac{1}{n}(x_n - \mu_{n-1})^2 - \frac{n-1}{n^2}(x_n - \mu_{n-1})^2 \\
=& \frac{n-1}{n}s_{n-1}^2 + \frac{1}{n}(x_n - \mu_{n-1})^2 - \frac{n-1}{n^2}(x_n - \mu_{n-1})^2 \\
=& s_{n-1}^2 + \frac{1}{n}(x_n - \mu_{n-1})(x_n - \mu_n)
\end{aligned}
$$

<!-- #### Code

```python
class RunningMeanVariance():
    def __init__(self):
        self.reset()

    def update(self, x):
        self.num_sample += 1
        if self.num_sample == 1:
            self.mean = x
        else:
            mean_old = self.mean
            self.mean = mean_old + (x - mean_old) / self.num_sample
            self.variance = (self.variance + (x - mean_old) * (x - self.mean)) / self.num_sample
            self.std_variance = self.variance ** 0.5

    def reset(self):
        self.num_sample, self.mean, self.variance, self.std_variance = 0, 0, 0, 0


class RewardScaling():
    def __init__(self):
        self.running_mean_variance = RunningMeanVariance()
        self.reset()

    def __call__(self, x):
        self.R = config.gamma * self.R + x
        self.running_mean_variance.update(self.R)
        if self.running_mean_variance.num_sample > 1:
            x = x / (self.running_mean_variance.std_variance + 1e-8)
        return x

    def reset(self):
        self.R = 0
        # self.running_mean_variance does not need to be resset.

agent = Agent()
reward_scaling = RewardScaling() # here
for i_episode in range(config.max_num_episode):
    state, _ = env.reset()
    reward_episode = 0
    agent.buffer.reset()
    reward_scaling.reset() # here
    for i_step in range(env_max_episode_steps):
        action, log_prob = agent.choose_action(state)
        state_next, reward_env, done, _, _ = env.step(action)
        reward = reward_scaling(reward_env) # here
        agent.buffer.add(state, state_next, reward, log_prob, done) # here
        reward_episode += reward_env
        state = state_next
        if done:
            break
    rewards.append(reward_episode)
    agent.learn(*agent.buffer.dump(agent.buffer.__len__()))
``` -->


## Embedding for the Q-value Critic

> Check [the implementation of DIAL](https://colab.research.google.com/gist/MJ10/2c0d1972f3dd1edcc3cd17c636aac8d2/dial.ipynb#scrollTo=YnoO2UA5L3pk).
{: .prompt-info }

In my understanding, after going through the embedding, inputs with different ranges can be considered as linearly independent quantities in the same space, so they can be added directly.

```python
# From the CoLab: https://colab.research.google.com/gist/MJ10/2c0d1972f3dd1edcc3cd17c636aac8d2/dial.ipynb#scrollTo=G5e0IeqmIJJj
class CNet(nn.Module):
    def __init__(self, opts):
        """
        Initializes the CNet model
        """
        super(CNet, self).__init__()
        self.opts = opts
        self.comm_size = opts['game_comm_bits']
        self.init_param_range = (-0.08, 0.08)

        ## Lookup tables for the state, action and previous action.
        self.action_lookup = nn.Embedding(opts['game_nagents'], opts['rnn_size'])
        self.state_lookup = nn.Embedding(2, opts['rnn_size'])
        self.prev_action_lookup = nn.Embedding(opts['game_action_space_total'], opts['rnn_size'])

        # Single layer MLP(with batch normalization for improved performance) for producing embeddings for messages.
        self.message = nn.Sequential(
            nn.BatchNorm1d(self.comm_size),
            nn.Linear(self.comm_size, opts['rnn_size']),
            nn.ReLU(inplace=True)
        )

        # RNN to approximate the agent’s action-observation history.
        self.rnn = nn.GRU(input_size=opts['rnn_size'], hidden_size=opts['rnn_size'], num_layers=2, batch_first=True)

        # 2 layer MLP with batch normalization, for producing output from RNN top layer.
        self.output = nn.Sequential(
            nn.Linear(opts['rnn_size'], opts['rnn_size']),
            nn.BatchNorm1d(opts['rnn_size']),
            nn.ReLU(),
            nn.Linear(opts['rnn_size'], opts['game_action_space_total'])
        )

    def forward(self, state, messages, hidden, prev_action, agent):
        """
        Returns the q-values and hidden state for the given step parameters
        """
        state = Variable(torch.LongTensor(state))
        hidden = Variable(torch.FloatTensor(hidden))
        prev_action = Variable(torch.LongTensor(prev_action))
        agent = Variable(torch.LongTensor(agent))

        # Produce embeddings for rnn from input parameters
        z_a = self.action_lookup(agent)
        z_o = self.state_lookup(state)
        z_u = self.prev_action_lookup(prev_action)
        z_m = self.message(messages.view(-1, self.comm_size))

        # Add the input embeddings to calculate final RNN input.
        z = z_a + z_o + z_u + z_m
        z = z.unsqueeze(1)

        rnn_out, h = self.rnn(z, hidden)
        # Produce final CNet output q-values from GRU output.
        out = self.output(rnn_out[:, -1, :].squeeze())

        return h, out
```


## Gumbel-Softmax
- Reparameterization.
- Maintain gradients from the sampled variables.
- Commonly used in communication methods.

### What is gumbel-softmax for?
If $a_t\sim \pi_\theta(\cdot \mid s_t)$, then how to calculate $\nabla_\theta a_t$?

### What is reparameterization?
This trick decouples the deterministic part and the random part of a variable.

This concept can be best illustrated with the example of the Gaussian distribution.

If $z\sim \mathcal{N}(\mu,\sigma^2)$, then $z = \mu + \sigma \cdot \epsilon$, where $\epsilon\sim \mathcal{N}(0,1)$. In this way, $\frac{\partial z}{\partial \mu} = 1$ and $\frac{\partial z}{\partial \sigma} = \epsilon$. Usually $\mu$ and $\sigma$ are estimated by a neural network, and the following gradient can be automatically calculated by deep frameworks.

### What does Gumbel-Softmax do?

We often use neural networks to generate a probability simplex, i.e., a profile of probability where $0\le p_i$ and $\sum\limits_{i} p_i = 1$. Then we will sample an $x$ based on this distribution. 

An example scenario is in RL, where an agent needs to choose an action $a_t$. We output a distribution $\pi(\cdot \mid s_t)$ and then sample an action $a_t\sim \pi(\cdot \mid s_t)$ based on this distribution to execute.

Gumbel-Softmax is used to reparameterization this kind of categorical distribution. This technique allows samples to be drawn according to the original distribution and enables gradient computation.

$$
z\sim \arg\max\limits_i (\log(p_i) + g_i),
$$

where $g_i = -\log(-\log (u_i)), u_i\sim U(0,1)$. 

The argmax is non-differentiable, it can be replaced with softmax. $i = \arg\max\limits_{j} (x_j)$.

$$
\mathrm{softmax}_T (x) = \frac{e^{x_j/T}}{\sum_k e^{x_k/T}}.
$$

If temperature $T$ is small enough, then the output of the softmax can be seen as a one-hot vector which indicates $i$.

### $x\ne \log(\mathrm{softmax}(x))$

$$
\mathrm{softmax}(x_i) = \frac{e^{x_i}}{\sum_{j=1}^{n} e^{x_j}}
$$

$$
\begin{aligned}
    \log(\mathrm{softmax}(x_i)) 
    =& \log\left(\frac{e^{x_i}}{\sum_{j=1}^{n} e^{x_j}}\right) \\
    =& x_i - \log\left(\sum_{j=1}^{n} e^{x_j}\right)
\end{aligned}
$$

```python
import torch

x = torch.rand(5)

x1 = torch.nn.Softmax(dim=0)(x)
x2 = torch.nn.functional.softmax(x, dim=0)
x3 = torch.nn.functional.log_softmax(x, dim=0)

print(x1)
print(x2)
print(torch.log(x1))
print(x3)
```

```bash
tensor([0.1385, 0.1978, 0.2231, 0.2861, 0.1543])
tensor([0.1385, 0.1978, 0.2231, 0.2861, 0.1543])
tensor([-1.9766, -1.6204, -1.4999, -1.2512, -1.8686])
tensor([-1.9766, -1.6204, -1.4999, -1.2512, -1.8686])
```

### Example code

```python
import torch

if __name__ == '__main__':
    batch_size = int(1e7)
    logits_distribution = [2, 3]

    logits_batch = torch.tensor(logits_distribution, dtype=torch.float64) \
        .unsqueeze(dim=0).expand(batch_size, len(logits_distribution))
    softmax = torch.nn.Softmax(dim=-1)
    pi = softmax(logits_batch)

    # -----
    # The standard way.
    temperature = 1
    actions_sampled = torch.nn.functional.gumbel_softmax(logits_batch, tau=temperature, hard=True)

    a0_num = torch.sum(actions_sampled[:, 0])
    a1_num = torch.sum(actions_sampled[:, 1])

    print(pi[0], a0_num, a1_num, sep="\n")

    # -----
    # In RL, the common epsilon-greedy is a operation on the policy space.
    # To sample it, we need to edit the policy first, and then put log(pi) into the gumbel-softmax.
    # See https://stackoverflow.com/questions/64980330/input-for-torch-nn-functional-gumbel-softmax
    print('===============')
    temperature = 1
    actions_sampled = torch.nn.functional.gumbel_softmax(torch.log(pi), tau=temperature, hard=True)

    a0_num = torch.sum(actions_sampled[:, 0])
    a1_num = torch.sum(actions_sampled[:, 1])

    print(pi[0], a0_num, a1_num, sep="\n")
```

```bash
tensor([0.2689, 0.7311], dtype=torch.float64)
tensor(2687766., dtype=torch.float64)
tensor(7312234., dtype=torch.float64)
===============
tensor([0.2689, 0.7311], dtype=torch.float64)
tensor(2690092., dtype=torch.float64)
tensor(7309908., dtype=torch.float64)
```

> Applying Gumbel-Softmax may cause `NaN` during training. Changing the data type of the variable to `float64` seems to have avoided this issue.
{: .prompt-warning }

### Computation graph
Check [my note on computation graph]({{site.baseurl}}/posts/Computation-Graph-Visualization/#example-5-nabla_theta-a-with-gumbel-softmax-reparameterization).


### Advantage

> This section is generated by ChatGPT-4.
{: .prompt-info }

The reparameterization trick reduces the variance of gradient estimates primarily by altering the way stochastic variables are sampled, introducing randomness only through differentiable transformations rather than direct sampling from the policy distribution. This approach offers several key benefits that help understand why it can reduce the variance of gradient estimates:

1. **Direct backpropagation through the stochastic component**: Before reparameterization, the policy's gradient could not be directly backpropagated through the random action sampling process, as this process was non-differentiable. This meant that gradient updates had to rely on external estimates of sampled actions, such as using Monte Carlo methods, typically leading to higher variance. Reparameterization allows gradients to be directly backpropagated through the action generation process, now involving a differentiable transformation (i.e., the output of the policy network plus noise), directly reducing the variance caused by sampling.

2. **Reducing the impact of sampling noise**: By confining randomness to noise from a known distribution and transforming it into actions through the policy network, reparameterization reduces the direct impact of sampling noise on the policy output. The effect of noise is modulated through a differentiable, network-parameterized function, allowing the algorithm to adjust these effects more efficiently through gradient descent rather than relying solely on the outcomes of random sampling.

3. **Smoothing the optimization process**: Since gradients can be directly calculated through the action generation process, each gradient update reflects direct improvements for the current policy parameters, rather than indirect estimates from sampling. This makes the optimization process smoother, reducing the fluctuations in optimization due to high sampling variance.

4. **Improving sample efficiency**: Reducing the variance of gradient estimates means that for the same number of samples, the algorithm can obtain more accurate gradient estimates. This improves sample efficiency because the information provided by each sample is utilized more effectively, accelerating learning speed and enhancing policy performance.

In summary, reparameterization reduces uncertainty and variance caused by sampling by changing the way stochastic variables are sampled and processed, making gradient estimates more stable and accurate. This contributes to improved performance and efficiency of reinforcement learning algorithms.

## Social Influence
- A MARL method.
- An intrinsic reward.
- Agent $i$ chooses the action that has the most impact on others.

$$
\begin{aligned}
    r_t^i 
    =& \sum\limits_{j\ne i} D_{KL}\left[\pi^j(a_t^j \mid s_t, a_t^i) \Big\Vert \sum\limits_{a_t^{i\prime}} \pi^j(a_t^j \mid s_t, a_t^{i\prime})\cdot \pi^i(a_t^{i\prime}\mid s_t) \right] \\
    =& \sum\limits_{j\ne i} D_{KL}\left[\pi^j(a_t^j \mid s_t, a_t^i) \Big\Vert P(a_t^j\mid s_t) \right]
\end{aligned}
$$

In the principal-agent communication:

$$
r^i = D_{KL}\left[ \pi^j(a^j\mid\sigma^i) \Big\Vert \sum\limits_{\sigma'}\varphi^i(\sigma^{i\prime}\mid s)\cdot \pi^j(a^j\mid\sigma^{i\prime})\right]
$$



## Basics



### TD(0)
Resampling techniques are a class of statistical methods that involve creating new samples by repeatedly drawing observations from the original data sample.

Bootstrapping is a method where new "bootstrap samples" are created by drawing observations **with replacement** from the original sample.

In RL, a common example is the Temporal Difference (TD) learning. This method bootstraps from the current estimate of the value function. The value function is defined as

$$
V(s) = \mathbb{E}\left[\sum\limits_{t=0}^\infty \gamma^t \cdot r_t | s_0 = s\right]
$$

But if the trajectory will never end, then we cannot get all the $r_t$ that we need to calculate the expectation.

According to the Bellman equation, the value function can be calculated as

$$
V(s) = \mathbb{E}\left[r_{t+1} + \gamma V(s_{t+1}) | s_t = s\right]
$$

Now I get a new sample of $R_{t+1}$, I can use it to update $V(s_t)$, using the incremental mean trick.

$$
V(s_t) \gets V(s_t) + \alpha\left(x_{n+1} - V(s_t) \right),
$$

where $x_{n+1} = r_{t+1} + \gamma V(s_{t+1}).$ The $V(s_{t+1})$ is not the ground true value, but we can used it. (Proving convergence is another thing to do.) 

So we can say that the value function is updated based on itself. And this method uses $V(s_{t+1})$ instead of $\sum\limits_{k=t}^\infty \gamma^{k-t}\cdot r_{k+2}.$ And that's what bootstrapping means. 