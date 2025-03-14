---
title: "Overcooked: A MARL Task"
date: 2023-12-28 08:00:00 +0800
categories: [Artificial Intelligence, Multi-Agent Reinforcement Learning]
tags: [Tech, AI, Multi_Agents, RL, Environment, Social_Dilemma]
math: True
---

## A Brief Intro

This MARL environment remade the [Overcooked](https://store.steampowered.com/app/728880/Overcooked_2/) game on Steam. Some features (game rules) are cut to simplify the situation.

This game requires two agents to coordinate to cook. If they succeed in cooking a soup recorded on the recipe and delivering the soup, then they both get a reward. The more the ingredients are, the more time to cook, and the more reward the agents can get.

This environment is designed to test the human-ai coordination, or used as a zero-shot coordination task.

## Resources

(Up to Date)
1. [[GitHub Repo: overcooked_ai]](https://github.com/HumanCompatibleAI/overcooked_ai/tree/master)
2. [[Online Game]](https://humancompatibleai.github.io/overcooked-demo/)
3. [[An Environment Wrapper: PantheonRL]](https://github.com/Stanford-ILIAD/PantheonRL)
4. [[Official Blog Post]](https://bair.berkeley.edu/blog/2019/10/21/coordination/)
5. MARLlib
   1. [[GitHub Repo: MARLlib]](https://github.com/Replicable-MARL/MARLlib?tab=readme-ov-file)
   2. [[MARLlib Document about Overcooked]](https://marllib.readthedocs.io/en/latest/handbook/env.html#overcooked-ai)
   3. (But I cannot find any implementation for this task in MARLlib.)

The rest of the post I will show you how to use these resources. It works for the repos of the following version (2023-12-28):
1. **GitHub Repo: overcooked_ai:** `https://github.com/HumanCompatibleAI/overcooked_ai/tree/83f51921869f25ddf5153aa2742b4e1d4da5e6e9`
2. **An Environment Wrapper: PantheonRL:** `https://github.com/Stanford-ILIAD/PantheonRL/tree/d8a0ff99c9c1bbc6bdd2eecde5425b16d77e996c`

## Installation

1. Choose a working directory `/` and enter it. Do it in a terminal.
2. Create a conda environment if you like. `conda create -n overcooked_ai python=3.7`
3. Install overcooked_ai
```bash
pip install overcooked-ai
```
4. Install PantheonRL
```bash
git clone https://github.com/Stanford-ILIAD/PantheonRL.git
cd PantheonRL
pip install -e .
cd ..
```

Then you will have folders `/overcooked_ai` and `/PantheonRL`.

## Rules

There is no official document about the rules. I test all of the following rules by myself by writting a simple interaction python file. The rules are a bit different from the rules of Overcooked on steam.

1. Players' 6 actions:
   1. Go up
   2. Go down
   3. Go left
   4. Go right
   5. Operate (the thing that in front of me)
   6. Do nothing
2. If both players choose to move to the same grid simultaneously, then nobody can move.
3. Orientation
   1. It matters because players can only operate the obejects in front of them.
   2. The player's orientation is determined by the move (or the relative positions before and after taking the move).
   3. If you want to go up and you are facing the left, you can go up directly without turning right.
4. The operate action can
   1. take/drop the plate/onion/tomato/soup
   2. cook (the timer starts to work once a player operate the filled oven)
   3. deliver the soup (by dropping the soup on the grey grid)
   4. (Players cannot drop obejects on the grey grid except the cooked soup.``)
   5. (Players cannot pass the object they have to other player directly.)
5. Reward
   1. The two agents' rewards are always the same. Fully cooperative.
   2. Unlike the overcooked on steam, there is no order here. If the players succeed in make soups listed on the recipes and delivering it, then they can get rewards.
   3. The more the ingredients are, the more time to cook, and the more reward the agents can get.
   4. Recipes and ingredients are recorded in a layout file.
   5. There are bonus recipes. Players succeed in making bonus soup will receive rewards of `bonus_value * soup_value`. Bonus value is usually 2.
   6. The order of ingredients in recipes doesn't matter. E.g., if the recipe reads `"onion, onion, tomato"`, then the players can get rewards of this recipe by making a soup of `"onion, tomato, onion"`.

Details:
1. Agent indexes are random. It will regenerated after `env.reset()`.
2. The recipe icons are respectively: 1 onion; 2 onions; 3 onions; 1 tomato; 1 onion + 1 tomato; 2 onions + 1 tomato; 2 tomatoes; 1 onion + 2 tomatoes; 3 tomatoes.
![](../../../assets/img/2023-12-28-overcooked/img_2023-12-28-10-46-31.png)
3. Players' observations of `Overcooked(gym.Env)` in `/overcooked_ai/src/overcooked_ai_py/mdp/overcooked_env.py` are generated by the true state: `obs = featurize_fn(state)`. And the observations are just gridworld matrices.
4. Layout path: `overcooked_ai/src/overcooked_ai_py/data/layouts/`

A layout file example: `you_shall_not_pass.layout`
```
{
    "grid":  """XXXXXSSSXXXXX
                XXTSSS SSSOXX
                P 1       2 P
                XXDSSSSSSSDXX""",
    "start_bonus_orders": [
        { "ingredients" : ["onion", "onion", "tomato"]},
        { "ingredients" : ["tomato", "tomato", "onion"]}
    ],
    "start_all_orders" : [
        { "ingredients" : [ "onion" ]},
        { "ingredients" : [ "tomato" ]},
        { "ingredients" : ["onion", "onion", "tomato"]},
        { "ingredients" : ["tomato", "tomato", "onion"]},
        { "ingredients" : [ "tomato", "tomato", "tomato"]},
        { "ingredients" : [ "onion", "onion"]}
    ],
    "onion_value" : 21,
    "tomato_value" : 13,
    "onion_time" : 15,
    "tomato_time" : 7
}
```

## featurize_fn

```python
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv, Overcooked
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld

layout_name = "you_shall_not_pass"
mdp = OvercookedGridworld.from_layout_name(layout_name)
base_env = OvercookedEnv.from_mdp(mdp, horizon=400)
env = Overcooked(base_env, featurize_fn=base_env.featurize_state_mdp)
# env = Overcooked(base_env, featurize_fn=base_env.lossless_state_encoding_mdp)

temp_result = env.reset()
print(temp_result.keys())
```

The results of `featurize_state_mdp` and `lossless_state_encoding_mdp` are both:
```
dict_keys(['both_agent_obs', 'overcooked_state', 'other_agent_env_idx'])
```



### featurize_state_mdp

> PantheonRL uses `OvercookedMultiEnv`, whose `featurize_fn` is `featurize_state_mdp`.
{: .prompt-info }

```python
"""
Encode state with some manually designed features. Works for arbitrary number of players

Arguments:
    overcooked_state (OvercookedState): state we wish to featurize
    mlam (MediumLevelActionManager): to be used for distance computations necessary for our higher-level feature encodings
    num_pots (int): Encode the state (ingredients, whether cooking or not, etc) of the 'num_pots' closest pots to each player.
        If i < num_pots pots are reachable by player i, then pots [i+1, num_pots] are encoded as all zeros. Changing this
        impacts the shape of the feature encoding

Returns:
    ordered_features (list[np.Array]): The ith element contains a player-centric featurized view for the ith player

    The encoding for player i is as follows:

        [player_i_features, other_player_features player_i_dist_to_other_players, player_i_position]

        player_{i}_features (length num_pots*10 + 24):
            pi_orientation: length 4 one-hot-encoding of direction currently facing
            pi_obj: length 4 one-hot-encoding of object currently being held (all 0s if no object held)
            pi_wall_{j}: {0, 1} boolean value of whether player i has wall immediately in direction j
            pi_closest_{onion|tomato|dish|soup|serving|empty_counter}: (dx, dy) where dx = x dist to item, dy = y dist to item. (0, 0) if item is currently held
            pi_cloest_soup_n_{onions|tomatoes}: int value for number of this ingredient in closest soup
            pi_closest_pot_{j}_exists: {0, 1} depending on whether jth closest pot found. If 0, then all other pot features are 0. Note: can
                be 0 even if there are more than j pots on layout, if the pot is not reachable by player i
            pi_closest_pot_{j}_{is_empty|is_full|is_cooking|is_ready}: {0, 1} depending on boolean value for jth closest pot
            pi_closest_pot_{j}_{num_onions|num_tomatoes}: int value for number of this ingredient in jth closest pot
            pi_closest_pot_{j}_cook_time: int value for seconds remaining on soup. -1 if no soup is cooking
            pi_closest_pot_{j}: (dx, dy) to jth closest pot from player i location

        other_player_features (length (num_players - 1)*(num_pots*10 + 24)):
            ordered concatenation of player_{j}_features for j != i

        player_i_dist_to_other_players (length (num_players - 1)*2):
            [player_j.pos - player_i.pos for j != i]

        player_i_position (length 2)
"""
```

```python
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv, Overcooked
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld

layout_name = "you_shall_not_pass"
mdp = OvercookedGridworld.from_layout_name(layout_name)
base_env = OvercookedEnv.from_mdp(mdp, horizon=400)
env = Overcooked(base_env, featurize_fn=base_env.featurize_state_mdp)
# env = Overcooked(base_env, featurize_fn=base_env.lossless_state_encoding_mdp)

temp_result = env.reset()
for obs in temp_result['both_agent_obs']:
    print(obs)
    print(obs.shape)
```

```
[  1.   0.   0.   0.   0.   0.   0.   0.   0.  -1.  -8.  -1.   0.   1.
   0.   0.   0.   0.  -1.  -1.   0.   0.   1.   1.   0.   0.   0.   0.
   0.   0.   2.   0.   1.   1.   0.   0.   0.   0.   0.   0. -10.   0.
   1.   1.   0.   0.   1.   0.   0.   0.   0.   0.   0.   0.   8.  -1.
   0.  -1.   0.   1.   0.   0.   0.   0.   1.  -1.   0.   0.   1.   1.
   0.   0.   0.   0.   0.   0.  -2.   0.   1.   1.   0.   0.   0.   0.
   0.   0.  10.   0.   1.   1.   0.   0.  -8.   0.  10.   2.]
(96,)
[  1.   0.   0.   0.   0.   0.   0.   0.   8.  -1.   0.  -1.   0.   1.
   0.   0.   0.   0.   1.  -1.   0.   0.   1.   1.   0.   0.   0.   0.
   0.   0.  -2.   0.   1.   1.   0.   0.   0.   0.   0.   0.  10.   0.
   1.   1.   0.   0.   1.   0.   0.   0.   0.   0.   0.   0.   0.  -1.
  -8.  -1.   0.   1.   0.   0.   0.   0.  -1.  -1.   0.   0.   1.   1.
   0.   0.   0.   0.   0.   0.   2.   0.   1.   1.   0.   0.   0.   0.
   0.   0. -10.   0.   1.   1.   0.   0.   8.   0.   2.   2.]
(96,)
```

The test layout size is $4\times 13$.


### lossless_state_encoding_mdp

Featurizes a OvercookedState object into a stack of **boolean masks** that are easily readable by a **CNN**.

```python
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv, Overcooked
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld

layout_name = "you_shall_not_pass"
mdp = OvercookedGridworld.from_layout_name(layout_name)
base_env = OvercookedEnv.from_mdp(mdp, horizon=400)
# env = Overcooked(base_env, featurize_fn=base_env.featurize_state_mdp)
env = Overcooked(base_env, featurize_fn=base_env.lossless_state_encoding_mdp)

temp_result = env.reset()
for obs in temp_result['both_agent_obs']:
    print(obs.shape, type(obs))
```

```
(13, 4, 26) <class 'numpy.ndarray'>
(13, 4, 26) <class 'numpy.ndarray'>
```

$13$ is the map width, and $4$ is the map height.

$26$ is the number of the boolean mask feature layers:

```
player_0_loc, player_1_loc, 
player_0_orientation_0, player_0_orientation_1, player_0_orientation_2, player_0_orientation_3, 
player_1_orientation_0, player_1_orientation_1, player_1_orientation_2, player_1_orientation_3, 
pot_loc, counter_loc, onion_disp_loc, tomato_disp_loc, dish_disp_loc, serve_loc, 
onions_in_pot, tomatoes_in_pot, onions_in_soup, tomatoes_in_soup, soup_cook_time_remaining, soup_done, 
dishes, onions, tomatoes, urgency
```


```python
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv, Overcooked
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld

layout_name = "you_shall_not_pass"
mdp = OvercookedGridworld.from_layout_name(layout_name)
base_env = OvercookedEnv.from_mdp(mdp, horizon=400)
# env = Overcooked(base_env, featurize_fn=base_env.featurize_state_mdp)
env = Overcooked(base_env, featurize_fn=base_env.lossless_state_encoding_mdp)

temp_result = env.reset()
import numpy as np
are_close  = np.allclose(temp_result['both_agent_obs'][0], temp_result['both_agent_obs'][1], atol=1e-3)
print(f"are_close: {are_close}")
```

```
are_close: False
```



## Useful Files in overcooked_ai

### Layouts
Layouts here are maps of the game. Different layouts have different recipes, ingredients, and objects locations. There are 49 layouts, and I found there are 45 of them are available.

The layouts are in the folder `/overcooked_ai/src/overcooked_ai_py/data/layouts`.

I wrote a code to render the available layouts.
1. Create a folder `layouts_rendered` under `/overcooked/src/overcooked_ai_py/data/`
2. Create a python file `layout_glance` under `/overcooked/src/overcooked_ai_py/data/`

```python
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv, Overcooked
from matplotlib import pyplot as plt
import os

if __name__ == '__main__':

    pwd = os.getcwd()
    layouts_path = os.path.join(pwd, "src/overcooked_ai_py/data/layouts")
    save_path = os.path.join(pwd, "src/overcooked_ai_py/data/layouts_rendered")

    not_working_layouts = ["multiplayer_schelling",
                           "old_dynamics_cook_test",
                           "cramped_room_single",
                           "old_dynamics_put_test"]

    
    layout_name_list = []
    for filename in os.listdir(layouts_path): 
        assert filename.endswith(".layout")
        layout_name = filename[:-7]
        if layout_name in not_working_layouts:
            continue
        print(f"layout_name: {layout_name}")
        layout_name_list.append(layout_name)
        
        mdp = OvercookedGridworld.from_layout_name(layout_name)
        base_env = OvercookedEnv.from_mdp(mdp, horizon=500)
        env = Overcooked(base_env, featurize_fn=base_env.featurize_state_mdp)

        env.reset()
        image = env.render()
        plt.imshow(image)
        
        plt.savefig(os.path.join(save_path, layout_name+".png"), dpi=300)
        plt.clf()

        del(mdp)
        del(base_env)
        del(env)
```
Then you can run it in a terminal. The working directory is `/overcooked_ai`.


### Run Episodes by Yourself

`/overcooked_ai/src/lytest/hardcoded_episode.py`

```python
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv, Overcooked
from matplotlib import pyplot as plt
import os

if __name__ == '__main__':
    save_path = pwd = os.getcwd()
    action_map = "wsdap "
    layout_name = "you_shall_not_pass"

    mdp = OvercookedGridworld.from_layout_name(layout_name)
    base_env = OvercookedEnv.from_mdp(mdp, horizon=500)
    env = Overcooked(base_env, featurize_fn=base_env.featurize_state_mdp)


    def saveimg(time, img, r=0.0):
        plt.imshow(img)
        plt.savefig(os.path.join(save_path, f"t={time},r={r}.png"), dpi=300)
        plt.clf()


    t = 0
    env.reset()
    print(f't={t}')
    image = env.render()
    saveimg(t, image)

    while True:
        action0 = input("Enter action 0 (w,s,a,d,p,space):")
        action1 = input("Enter action 1 (w,s,a,d,p,space):")

        actions = [action_map.index(action0), action_map.index(action1)]
        obs, reward, done, info = env.step(actions)
        t = t + 1
        print(f't={t}, r={reward}')
        print(f'obs={obs}')
        print(f'overcooked_state={obs["overcooked_state"]}')
        # print(t, obs,reward, done, info)

        image = env.render()
        saveimg(t, image, r=reward)
```

You can enter:
   1. `w`: go up
   2. `s`: go down
   3. `a`: go left
   4. `d`: go right
   5. `(space)`: operate
   6. `p`: do nothing


## Fixng Bugs in the PantheonRL Example

`/PantheonRL/overcookedgym/overcooked.py`

The original file
- called the deprecated function `MediumLevelPlanner`,
- used a deprecated way to create the `self.base_env` and `self.featurize_fn`,
- used a wrong index of `info`,
and thus failed to run.

```python
import gym
import numpy as np
from overcooked_ai_py.mdp.actions import Action
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
# from overcooked_ai_py.planning.planners import MediumLevelPlanner, NO_COUNTERS_PARAMS
from overcooked_ai_py.planning.planners import NO_COUNTERS_PARAMS

from pantheonrl.common.multiagentenv import SimultaneousEnv

class OvercookedMultiEnv(SimultaneousEnv):
    def __init__(self, layout_name, ego_agent_idx=0, baselines=False):
        """
        base_env: OvercookedEnv
        featurize_fn: what function is used to featurize states returned in the 'both_agent_obs' field
        """
        super(OvercookedMultiEnv, self).__init__()

        DEFAULT_ENV_PARAMS = {
            "horizon": 400
        }
        rew_shaping_params = {
            "PLACEMENT_IN_POT_REW": 3,
            "DISH_PICKUP_REWARD": 3,
            "SOUP_PICKUP_REWARD": 5,
            "DISH_DISP_DISTANCE_REW": 0,
            "POT_DISTANCE_REW": 0,
            "SOUP_DISTANCE_REW": 0,
        }

        self.mdp = OvercookedGridworld.from_layout_name(layout_name=layout_name, rew_shaping_params=rew_shaping_params)
        # mlp = MediumLevelPlanner.from_pickle_or_compute(self.mdp, NO_COUNTERS_PARAMS, force_compute=False)

        # self.base_env = OvercookedEnv(self.mdp, **DEFAULT_ENV_PARAMS)
        self.base_env = OvercookedEnv.from_mdp(self.mdp, horizon=DEFAULT_ENV_PARAMS["horizon"])
        # self.featurize_fn = lambda x: self.mdp.featurize_state(x, mlp)
        self.featurize_fn = self.base_env.featurize_state_mdp

        if baselines: np.random.seed(0)

        self.observation_space = self._setup_observation_space()
        self.lA = len(Action.ALL_ACTIONS)
        self.action_space  = gym.spaces.Discrete( self.lA )
        self.ego_agent_idx = ego_agent_idx
        self.multi_reset()

    def _setup_observation_space(self):
        dummy_state = self.mdp.get_standard_start_state()
        obs_shape = self.featurize_fn(dummy_state)[0].shape
        high = np.ones(obs_shape, dtype=np.float32) * np.inf  # max(self.mdp.soup_cooking_time, self.mdp.num_items_for_soup, 5)

        return gym.spaces.Box(-high, high, dtype=np.float64)

    def multi_step(self, ego_action, alt_action):
        """
        action:
            (agent with index self.agent_idx action, other agent action)
            is a tuple with the joint action of the primary and secondary agents in index format
            encoded as an int

        returns:
            observation: formatted to be standard input for self.agent_idx's policy
        """
        ego_action, alt_action = Action.INDEX_TO_ACTION[ego_action], Action.INDEX_TO_ACTION[alt_action]
        if self.ego_agent_idx == 0:
            joint_action = (ego_action, alt_action)
        else:
            joint_action = (alt_action, ego_action)

        next_state, reward, done, info = self.base_env.step(joint_action)

        # reward shaping
        # rew_shape = info['shaped_r']
        rew_shape = info['shaped_r_by_agent'][0]
        reward = reward + rew_shape

        #print(self.base_env.mdp.state_string(next_state))
        ob_p0, ob_p1 = self.featurize_fn(next_state)
        if self.ego_agent_idx == 0:
            ego_obs, alt_obs = ob_p0, ob_p1
        else:
            ego_obs, alt_obs = ob_p1, ob_p0

        return (ego_obs, alt_obs), (reward, reward), done, {}#info

    def multi_reset(self):
        """
        When training on individual maps, we want to randomize which agent is assigned to which
        starting location, in order to make sure that the agents are trained to be able to
        complete the task starting at either of the hardcoded positions.

        NOTE: a nicer way to do this would be to just randomize starting positions, and not
        have to deal with randomizing indices.
        """
        self.base_env.reset()
        ob_p0, ob_p1 = self.featurize_fn(self.base_env.state)
        if self.ego_agent_idx == 0:
            ego_obs, alt_obs = ob_p0, ob_p1
        else:
            ego_obs, alt_obs = ob_p1, ob_p0

        return (ego_obs, alt_obs)

    def render(self, mode='human', close=False):
        pass
```


`/PantheonRL/overcookedgym/overcooked_utils.py`

You can add any 45 available layouts here.

```python
LAYOUT_LIST = ['corridor', 'five_by_five', 'mdp_test', 'multiplayer_schelling',
               'random0', 'random1', 'random2', 'random3', 'scenario1_s',
               'scenario2', 'scenario2_s', 'scenario3', 'scenario4',
               'schelling', 'schelling_s', 'simple', 'simple_single',
               'simple_tomato', 'small_corridor', 'unident', 'unident_s', # 
               'you_shall_not_pass']

NAME_TRANSLATION = {
    "cramped_room": "simple",
    "asymmetric_advantages": "unident_s",
    "coordination_ring": "random1",
    "forced_coordination": "random0",
    "counter_circuit": "random3",
}
```


`/PantheonRL/examples/env_lytest.py`

```python
"""
This is a simple example training script for PantheonRL.

To run this script, remember to first install overcooked
via the instructions in the README.md
"""

import gym
from stable_baselines3 import PPO

from pantheonrl.common.agents import OnPolicyAgent
from overcookedgym.overcooked_utils import LAYOUT_LIST

# layout = 'simple'
# assert layout in LAYOUT_LIST

layout = 'cramped_room'
# layout = 'you_shall_not_pass'

# Since pantheonrl's MultiAgentEnv is a subclass of the gym Env, you can
# register an environment and construct it using gym.make.
env = gym.make('OvercookedMultiEnv-v0', layout_name=layout)

# Before training your ego agent, you first need to add your partner agents
# to the environment. You can create adaptive partner agents using
# OnPolicyAgent (for PPO/A2C) or OffPolicyAgent (for DQN/SAC). If you set
# verbose to true for these agents, you can also see their learning progress
partner = OnPolicyAgent(PPO('MlpPolicy', env, verbose=1))
env.add_partner_agent(partner)

# Finally, you can construct an ego agent and train it in the environment
ego = PPO('MlpPolicy', env, verbose=1)
# ego.learn(total_timesteps=10000)
ego.learn(total_timesteps=100)
```