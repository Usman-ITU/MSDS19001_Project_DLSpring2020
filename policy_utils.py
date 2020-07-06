# References:
# https://github.com/Khrylx/PyTorch-RL/blob/d94e1479403b2b918294d6e9b0dc7869e893aa1b/core/agent.py#L8
# https://github.com/MarvineGothic/AtariGAIL/blob/master/gailtf/agent/run_atari.py
# https://github.com/pytorch/examples/blob/master/reinforcement_learning/reinforce.py
# https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
# https://github.com/DanielTakeshi/rl_algorithms/blob/master/vpg/main.py
# https://github.com/DanielTakeshi/rl_algorithms/blob/7a43fa485137dd5e0bbebecbbbc390cdb6dbde0b/trpo/fxn_approx.py#L23

import time
import random
from itertools import count
from collections import namedtuple
import gym
import numpy as np

import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from atari_wrapper import make_atari

from utils import list_of_list_into_tensor

eps = np.finfo(np.float32).eps.item() # smallest number


item = namedtuple('Item', ['state', 'action', 'next_state', 
                            'reward', 'returns', 'advantage'])

class Memory:
    # ! INCOMPLETE
    """
    Prposed usage:
    
    """
    def __init__(self):
        self.batch_list = []

    def sample(self, size=None):
        if size is None:
            return full_batch 
        else:
            shuffle(full_batch)[:size, ...]



class DataCollector:
    def __init__(self, 
                env,
                memory,
                reward_to_return_fn,
                mode='expert'):
        if format not in ['GAE', 
                            'baseline', 
                            'return']:
            print("Format not recognizied.")
        self.env = env 

    def collect_expert_data(self, n_samples):
        for t in range(n_samples):
            action = Agent.choose_action()
            next_obs, reward, done, info = env.step(action)
            if done:
                obs = env.reset()

    def run_a_policy(self, policy, n_samples):
        pass 
    



def collect_samples(env, 
                    policy, 
                    n_episodes,
                    render=False,
                    max_time_steps=10000):
    """
    Runs the policy in the environment till max time steps 
    or till environment returns the done flag.

    Returns a tuple containing:
        Rewards:      torch tensor of shape [n_episodes,
                                              max_episode_length]
                      corresponds to the rewards returned by the 
                      policy.
        
        Masks:        torch tensor of shape [n_episodes,
                                              max_episode_length]
                      corresponds to the binary mask whose 
                      (i,j) entry is 1 if episode i lasted at 
                      least j steps else 0.
        
        Log Probabilities: torch tensor of shape [n_episodes,
                                              max_episode_length]
                      corresponds to the log probability of 
                      action taken by the policy.

        Observations: list of list of length n_episodes containing 
                       observations returned by environment.

        Actions:      list of list of length n_episodes containing 
                       actions chosen by policy.
                
        Episode_lengths: A list of ints corresponding to length
                      of different episodes contained in the batch.
                            
    """
    observations_list, actions_list, \
        rewards_list, episode_lengths, \
        log_prob_list, masks_list = [], [], [], [], [], []

    for _ in range(n_episodes):
        observations, actions, \
            rewards, log_probs = [], [], [], []
        ob = env.reset()
        done_t = 0
        for t in range(max_time_steps):
            observations.append(ob)
            # get action
            action, log_prob = policy.sample_action(ob)
            log_probs.append(log_prob)
            # execute action in env
            ob, reward, done, _ = env.step(action)

            if render:
                env.render()
            actions.append(action)
            rewards.append(reward)
            if done:
                done_t = t+1
                break
        else:
            done_t = t+1
    
        observations_list.append(observations)
        actions_list.append(actions)
        rewards_list.append(rewards)
        episode_lengths.append(done_t)
        log_prob_list.append(torch.stack(log_probs))

    max_episode_length = max(episode_lengths)

    for i,t in enumerate(episode_lengths):
        rewards_list[i].extend(
                        [rewards_list[i][-1]]*
                        (max_episode_length-t))
        log_prob_list[i] = F.pad(log_prob_list[i],
                            (0, 0, 0, max_episode_length - t))
        masks_list.append([1.]*t + 
                         [0.]*(max_episode_length - t))
        

    rewards_tensor = torch.tensor(rewards_list)
    log_prob_tensor = torch.stack(log_prob_list)[...,0]
    masks_tensor = torch.tensor(masks_list)

    return (rewards_tensor,
            masks_tensor,
            log_prob_tensor,
            observations_list,
            actions_list,
            episode_lengths)


def discount_rewards(rewards, mask, values=None, gamma=0.99):
    if values is None:
        returns = torch.zeros(*rewards.size())
        returns[:,-1] = rewards[:,-1]
        for i in reversed(range(rewards.shape[1] - 1)):
            returns[:,i] = (rewards[:,i] + \
                            gamma*returns[:,i+1])*mask[:,i]
        return returns
    else:
        returns = torch.zeros(*rewards.size())
        advantage = torch.zeros(*rewards.size())
        returns[:,-1] = rewards[:,-1]
        for i in reversed(range(rewards.shape[1] - 1)):
            returns[:,i] = (rewards[:,i] + \
                            gamma*returns[:,i+1])*mask[:,i]
        advantage = returns - values
        return advantage, returns



def generalized_advantage(rewards, 
                            masks, 
                            values, 
                            gamma=0.99, 
                            lmbda=1.0, 
                            normalize_advantage=True):
    """
    Estimates generalized advantage of Schulman et al.
    """
    advantage = torch.zeros(*rewards.size())
    deltas = torch.zeros(*rewards.size())
    deltas[:,-1] = (rewards[:,-1] - values[:,-1])*masks[:,-1]
    advantage[:, -1] = deltas[:, -1]
    for i in reversed(range(rewards.shape[1] -1)):
        deltas[:, i] = (rewards[:,i] + gamma*values[:,i+1] 
                        - values[:, i])*masks[:,i]
        advantage[:, i] = (deltas[:, i] + 
                        gamma*lmbda*advantage[:,i+1])*masks[:,i]

    returns = values + advantage

    # debugging information
    # calculate return
    returns2 = torch.zeros(*rewards.size())
    returns2[:,-1] = rewards[:,-1]
    for i in reversed(range(rewards.shape[1] - 1)):
        returns2[:,i] = (rewards[:,i] + \
                        gamma*returns2[:,i+1])*masks[:,i]

    if normalize_advantage:
        advantage = (advantage - advantage.mean())/(advantage.std()+eps)

    return advantage, returns





class Policy(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Policy, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(state_dim, 400))
        self.layers.append(nn.Dropout(p=0.6))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(400, action_dim))

    def forward(self, x):
        for l in self.layers:
            x = l(x)
        return F.softmax(x, dim=1)

    def sample_action(self, x):
        x = torch.from_numpy(x).float().unsqueeze(0)
        probs = self.forward(x)
        m = Categorical(probs)
        action = m.sample()
        log_prob = m.log_prob(action)
        return action.item(), log_prob
    
    def log_prob(self, state, action):
        """
        Args:
            x (torch tensor of shape [batch_size, state_dim])
        """
        if type(state) is list:
            state = list_of_list_into_tensor(state).float()
        if type(action) is list:
            action = list_of_list_into_tensor(action)[...,None].float()
        
        assert state.shape[:-1] == action.shape[:-1]
        if len(state.shape) > 2:
            state_shape = state.shape 
            action_shape = action.shape
        state = state.view(-1, state_shape[-1])
        action = action.view(-1, action_shape[-1])
        dist = Categorical(self.forward(state))
        log_probs = dist.log_prob(action.flatten()).reshape(action_shape)
        if log_probs.shape[-1] == 1:
            return log_probs[...,0]
        else:
            log_probs





class Value(nn.Module):
    def __init__(self, state_dim=4):
        super(Value, self).__init__()
        self.affine1 = nn.Linear(state_dim, 128)
        self.dropout = nn.Dropout(p=0.6)
        self.affine2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.affine1(x)
        x = self.dropout(x)
        x = F.relu(x)
        x = self.affine2(x)
        return x

    def __call__(self, x):
        if type(x) is list:
            max_length = max([len(obs) for obs in x])
            outs = []
            for obs in x:
                y = torch.stack([torch.from_numpy(ob).float()
                            for ob in obs])
                out = self.forward(y)
                outs.append(F.pad(out, (0, 0, 0, max_length - len(obs))))
            values = torch.stack(outs)[...,0]
            return values
        else:
            y = torch.from_numpy(x).float().unsqueeze(0)
            value = self.forward(y)
            return value


def ppo(env, policy_optim_steps=4, clip_ratio=0.2):
    start_time = time.time()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    policy = Policy(state_dim, action_dim)
    value_net = Value(state_dim)
    running_reward = 10 # from torch tutorial
    alpha = 0.95 #weight for running sum
    optimizer_p = optim.Adam(policy.parameters(), lr=1e-2)
    optimizer_v = optim.Adam(value_net.parameters(), lr=1e-2)
    for i in count(1):
        rewards, masks, log_probs, obs_list, action_list, _ = \
                    collect_samples(env, policy, 8, max_time_steps=128)
        values = value_net(obs_list)
        advantage, returns = generalized_advantage(rewards, masks, values,
                                                    gamma=0.99,
                                                    lmbda=0.95)
        fixed_log_probs = log_probs
        fixed_values = values

        optimizer_p.zero_grad()
        log_probs = policy.log_prob(obs_list, action_list)
        rt = torch.exp(log_probs - fixed_log_probs)*masks
        clip_adv = torch.clamp(rt, 1-clip_ratio, 1+clip_ratio)
        loss_p = torch.min(rt*advantage, clip_adv*advantage).sum(axis=1).mean()
        loss_p.backward(retain_graph=True)
        optimizer_p.step()
        # Take from here
        # https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_atari.py#L620
            # I am not sure about the logic of this. 

        optimizer_v.zero_grad()
        new_values = value_net(obs_list)
        v_loss_unclipped = ((new_values - returns) ** 2)*masks
        v_clipped = fixed_values + torch.clamp(new_values - fixed_values, -clip_ratio, clip_ratio)
        v_loss_clipped = ((v_clipped - returns)**2)*masks
        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
        v_loss = 0.5*v_loss_max.sum(axis=1).mean()
        v_loss.backward()
        optimizer_v.zero_grad()

        mean_episode_reward = (rewards*masks).sum(axis=1).mean().item() 
        running_reward = (1-alpha)*mean_episode_reward \
                    + alpha*running_reward
        if i%10 == 0:
            print("Episode:{}\t Mean Episode Reward:{} \t\
                        Running Reward:{}".format(i, 
                                        mean_episode_reward, 
                                        running_reward))
        if running_reward > 400:
            print("Model converged in", time.time()-start_time)
            _, _, rewards, masks, _, log_probs = \
                        collect_samples(env, policy, 1, render=True)
            break
    return policy
        



def vanilla_pg_with_GAE(env):
    start_time = time.time()
    state_dim = env.observation_space.shape
    action_dim = env.action_space.n
    print("State Dimension ", state_dim)
    print("Action Dimension ", action_dim)
    policy = Policy(state_dim, action_dim)
    value_net = Value(state_dim)
    running_reward = 10 # from torch tutorial
    alpha = 0.95 #weight for running sum
    optimizer_p = optim.Adam(policy.parameters(), lr=1e-2)
    optimizer_v = optim.Adam(value_net.parameters(), lr=1e-2)
    for i in count(1):
        rewards, masks, log_probs, obs_list, _, _ = \
                    collect_samples(env, policy, 2)
        values = value_net(obs_list)
        advantage, returns = generalized_advantage(rewards, masks, values)
        # normalize retuns
        optimizer_p.zero_grad()
        loss_p = (-log_probs*advantage).sum(axis=1).mean()
        loss_p.backward(retain_graph=True)
        optimizer_p.step()

        # Update value net
        optimizer_v.zero_grad()
        loss_v = F.mse_loss(values, returns, reduction='none').sum(axis=1).mean()
        loss_v.backward()
        optimizer_v.step()


        mean_episode_reward = (rewards*masks).sum(axis=1).mean().item() 
        running_reward = (1-alpha)*mean_episode_reward \
                    + alpha*running_reward
        if i%10:
            print("Episode:{}\t Mean Episode Reward:{} \t\
                        Running Reward:{}".format(i, 
                                        mean_episode_reward, 
                                        running_reward))

        if np.abs(running_reward) > np.abs(env.spec.reward_threshold):
            print("Model converged in", time.time()-start_time)
            _, _, rewards, masks, _, log_probs = \
                        collect_samples(env, policy, 1, render=True)
            break
    return policy


def vanilla_pg_with_baseline(env):
    start_time = time.time()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    policy = Policy(state_dim, action_dim)
    value_net = Value(state_dim)
    running_reward = 10 # from torch tutorial
    alpha = 0.95 #weight for running sum
    optimizer_p = optim.Adam(policy.parameters(), lr=1e-2)
    optimizer_v = optim.Adam(value_net.parameters(), lr=1e-2)
    for i in count(1):
        rewards, masks, log_probs, obs_list, _, _ = \
                    collect_samples(env, policy, 2)
        values = value_net(obs_list)
        advantage, returns = discount_rewards(rewards, masks, values)
        # normalize retuns
        optimizer_p.zero_grad()
        loss_p = (-log_probs*advantage).sum(axis=1).mean()
        loss_p.backward(retain_graph=True)
        optimizer_p.step()

        # Update value net
        optimizer_v.zero_grad()
        loss_v = F.mse_loss(values, returns, reduction='none').sum(axis=1).mean()
        loss_v.backward()
        optimizer_v.step()

        mean_episode_reward = (rewards*masks).sum(axis=1).mean().item() 
        running_reward = (1-alpha)*mean_episode_reward \
                    + alpha*running_reward
        if i%10 == 0:
            print("Episode:{}\t Mean Episode Reward:{} \t\
                        Running Reward:{}".format(i, 
                                        mean_episode_reward, 
                                        running_reward))
        if running_reward > env.spec.reward_threshold:
            print("Model converged in", time.time()-start_time)
            _, _, rewards, masks, _, log_probs = \
                        collect_samples(env, policy, 1, render=True)
            break
    return policy

def vanilla_pg(env):
    start_time = time.time()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    policy = Policy(state_dim, action_dim)
    running_reward = 10 # from torch tutorial
    alpha = 0.95 #weight for running sum
    optimizer = optim.Adam(policy.parameters(), lr=1e-2)
    render = False
    for i in count(1):
        # if random.uniform(0,1) > 0.7:
        #     render = True
        rewards, masks, log_probs, _, _, _ = \
                    collect_samples(env, policy, 5, render=render)
        returns = discount_rewards(rewards, masks)
        # normalize retuns
        returns = (returns - returns.mean())/(returns.std() - eps)
        optimizer.zero_grad()
        loss = (-log_probs*returns).sum(axis=1).mean()
        loss.backward()
        optimizer.step()
        mean_episode_reward = (rewards*masks).sum(axis=1).mean().item() 
        running_reward = (1-alpha)*mean_episode_reward \
                    + alpha*running_reward
        if i%10 == 0:
            print("Episode:{}\t Mean Episode Reward:{} \t\
                        Running Reward:{}".format(i, 
                                        mean_episode_reward, 
                                        running_reward))
        if running_reward > 400:
            print("Model converged in", time.time()-start_time)
            _, _, rewards, masks, _, log_probs = \
                        collect_samples(env, policy, 1, render=True)
            break
    return policy
        
if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    ppo(env)
    # env = make_atari('PongNoFrameskip-v4')
    # vanilla_pg_with_GAE(env)
    


