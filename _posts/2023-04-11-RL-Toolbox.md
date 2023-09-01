---
title: RL Toolbox
date: 2023-04-11 02:40:00 +0800
categories: [Code]
tags: [toolbox, reinforcement learning]
math: True
---

> This note will be consistently updated.
{: .prompt-info }

---

## Normalization

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
\text{softmax}_T (x) = \frac{e^{x_j/T}}{\sum_k e^{x_k/T}}.
$$

If temperature $T$ is small enough, then the output of the softmax can be seen as a one-hot vector which indicates $i$.

### Example code
Check [my note](https://yuelin301.github.io/posts/Computation-Graph-Visualization/#example-5-nabla_theta-a-with-gumbel-softmax-reparameterization).



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

## PPO 37 Tricks
> [The 37 Implementation Details of Proximal Policy Optimization](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/).
{: .prompt-info }

