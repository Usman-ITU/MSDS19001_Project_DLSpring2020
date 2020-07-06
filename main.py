from itertools import count
import pickle, time
import gym
import torch 
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim

from utils import list_of_list_into_tensor
from policy_utils import collect_samples
from policy_utils import generalized_advantage
from policy_utils import Policy, Value
from policy_utils import vanilla_pg_with_GAE
from atari_wrapper import make_atari, wrap_deepmind
from networks import Buffer, ConvPolicyAtari

import matplotlib.pyplot as plt

def TrainExpertPolicy(env):
    return vanilla_pg_with_GAE(env)

class Discriminator(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Discriminator, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(state_dim+action_dim, 400))
        self.layers.append(nn.Dropout(p=0.6))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(400, 1))
        self.layers.append(nn.Sigmoid())

    def forward(self, state, action):
        if type(state) is list:
            state = list_of_list_into_tensor(state).float()
        if type(action) is list:
            action = list_of_list_into_tensor(action)[...,None].float()
        x = torch.cat([state, action], axis=2)
        x_shape = x.shape 
        x = x.view(-1, self.state_dim+self.action_dim)
        for l in self.layers:
            x = l(x)
        return x.view(x_shape[0], -1)
    

def evaluate_reward(disc, state_trajs, action_trajs):
    """
    Args: 
        traj: list containing list of states/observations
                and list of actions
    """
    return torch.log(disc(state_trajs, action_trajs))


def gail(config):
    env = gym.make(config['env'])
    # collect trajectories 
    if config['train_expert']:
        expert_policy = TrainExpertPolicy(env)
        _, expert_masks,_, expert_obs, expert_actions,_ = collect_samples(env, 
                                                        expert_policy,
                                                        config['expert_trajs_num'])
        # save
        with open(config['env'] + '_expert_trajs.pkl', 'wb') as f:
            pickle.dump([expert_masks, expert_obs, expert_actions], f)
    else:
        with open(config['env'] + '_expert_trajs.pkl', 'rb') as f:
            expert_masks, expert_obs, expert_actions = pickle.load(f)

    # make expert trajs into tensors
    expert_obs_tensor = list_of_list_into_tensor(expert_obs).float()
    expert_actions_tensor = list_of_list_into_tensor(expert_actions)[...,None].float()

    # Initialize Networks
    disc = Discriminator(env.observation_space.shape[0],
                            1)
    policy = Policy(env.observation_space.shape[0],
                        env.action_space.n)
    value_net = Value(env.observation_space.shape[0])

    # Initialize loss modules
    disc_loss_fn = nn.BCELoss(reduction='none')

    # initialize optimizer
    optimizer_disc = optim.Adam(disc.parameters(), lr=1e-2)
    optimizer_p = optim.Adam(policy.parameters(), lr=1e-2)
    optimizer_v = optim.Adam(value_net.parameters(), lr=1e-2)

    start_time = time.time()
    # Training loop
    for i in count(1):
        # run policy and get samples
        _, gen_masks, gen_log_probs, gen_obs, \
            gen_actions, _ = collect_samples(env, 
                                            policy,
                                            n_episodes=config['gen_trajs_num'])
        # discriminator loss
        g = disc(gen_obs, gen_actions)*gen_masks
        e = disc(expert_obs_tensor, expert_actions_tensor)
        disc_loss = (disc_loss_fn(g, torch.ones(*g.size()))*gen_masks).sum(axis=1).mean() +\
                    (disc_loss_fn(e, torch.zeros(*e.size()))*expert_masks).sum(axis=1).mean()
        optimizer_disc.zero_grad()
        disc_loss.backward()

        # evaluate reward under current discriminator
        reward = evaluate_reward(disc, gen_obs, gen_actions)

        # update policy net
        values = value_net(gen_obs)
        advantage, returns = generalized_advantage(reward, gen_masks, values)
        optimizer_p.zero_grad()
        loss_p = (gen_log_probs*advantage).sum(axis=1).mean() # no minus sign
        loss_p.backward(retain_graph=True)
        optimizer_p.step()

        # Update value net
        optimizer_v.zero_grad()
        loss_v = F.mse_loss(values, returns, reduction='none').sum(axis=1).mean()
        loss_v.backward()
        optimizer_v.step()
        print(i)
        # evaluate current policy under the true reward function
        if i%5 == 0:
            eval_rewards, _, _, _, _, \
                gen_actions = collect_samples(env, 
                                                policy,
                                                n_episodes=5)
            eval_rewards = eval_rewards.sum(axis=1)
            print(eval_rewards.detach().numpy())
            if eval_rewards.mean().item() > env.spec.reward_threshold:
                print("Model converged in", time.time()-start_time)
                _, _, rewards, masks, _, log_probs = \
                            collect_samples(env, policy, 1, render=True)
                break
            

if __name__ == '__main__':
    config = {
        'env': 'CartPole-v1',
        
        'train_expert': False,
        'expert_trajs_num': 50,

        'gen_trajs_num': 50
    }
    # gail(config)
    env = wrap_deepmind(make_atari('PongNoFrameskip-v4'), scale=True)
    action_dim = env.action_space.n
    buffer = Buffer(4)
    convnet = ConvPolicyAtari(action_dim)
    obs = env.reset()
    start = time.time()
    for i in range(2000):
        obs, reward, done, _ = env.step(env.action_space.sample())
        action, log_probs, values = convnet(buffer(obs))
    print(time.time()-start)
