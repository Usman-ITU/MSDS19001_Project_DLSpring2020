from itertools import count
import time
import numpy as np 

import gym 
from networks import ConvPolicyAtari, Buffer
from atari_wrapper import make_atari, wrap_deepmind

import torch
import torch.optim as optim
import torch.nn.functional as F 

eps = np.finfo(np.float32).eps.item() # smallest number


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
    rewards_list, episode_lengths, \
        log_prob_list, masks_list, \
            values_list = [], [], [], [], []
    
    buf = Buffer(4)   # buffer to hold observations
    
    for _ in range(n_episodes):
        observations, actions, \
            rewards, log_probs, values \
                 = [], [], [], [], []
        ob = env.reset()
        buf.reset()    # flush the buffer 
        done_t = 0
        for t in range(max_time_steps):
            # get action
            action, log_prob, value = policy(buf(ob))
            log_probs.append(log_prob)
            values.append(value)
            # execute action in env
            ob, reward, done, _ = env.step(action)

            if render:
                env.render()
            actions.append(action)
            rewards.append(reward)
            if done:
                done_t = t+1
                break
    
        rewards_list.append(rewards)
        episode_lengths.append(done_t)
        log_prob_list.append(torch.stack(log_probs))
        values_list.append(torch.cat(values,dim=1))

    max_episode_length = max(episode_lengths)

    for i,t in enumerate(episode_lengths):
        rewards_list[i].extend(
                        [rewards_list[i][-1]]*
                        (max_episode_length-t))
        log_prob_list[i] = F.pad(log_prob_list[i],
                            (0, 0, 0, max_episode_length - t))
        values_list[i] = F.pad(values_list[i],
                            (0, 0, 0, max_episode_length - t))
        masks_list.append([1.]*t + 
                         [0.]*(max_episode_length - t))
        

    rewards_tensor = torch.tensor(rewards_list)
    log_prob_tensor = torch.stack(log_prob_list)[...,0]
    values_tensor = torch.stack(values_list)[0,...]
    masks_tensor = torch.tensor(masks_list)

    return (rewards_tensor,
            masks_tensor,
            log_prob_tensor,
            values_tensor,
            episode_lengths)

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


def vanilla_pg_with_GAE(env):
    start_time = time.time()
    state_dim = env.observation_space.shape
    action_dim = env.action_space.n
    print("State Dimension ", state_dim)
    print("Action Dimension ", action_dim)
    actorCritic = ConvPolicyAtari(action_dim)
    running_reward = 0 # from torch tutorial
    alpha = 0.95 #weight for running sum
    optimizer = optim.Adam(actorCritic.parameters(), lr=1e-2)
    for i in count(1):
        rewards, masks, log_probs, values, _ = \
                    collect_samples(env, actorCritic, 1)
        advantage, returns = generalized_advantage(rewards, masks, values)

        # normalize retuns
        optimizer.zero_grad()
        loss_p = (-log_probs*advantage).sum(axis=1).mean()
        loss_v = F.mse_loss(values, returns, reduction='none').sum(axis=1).mean()
        loss = loss_p + loss_v
        loss.backward()
        optimizer.step()


        mean_episode_reward = (rewards*masks).sum(axis=1).mean().item() 
        running_reward = (1-alpha)*mean_episode_reward \
                    + alpha*running_reward
        if i%10:
            print("Episode:{}\t Mean Episode Reward:{} \t\
                        Running Reward:{}".format(i, 
                                        mean_episode_reward, 
                                        running_reward))

        if np.abs(running_reward) > 20:
            print("Model converged in", time.time()-start_time)
            rewards, _, _, _, _ = \
                        collect_samples(env, policy, 1, render=True)
            break
    return policy

if __name__ == '__main__':
    env = wrap_deepmind(make_atari('PongNoFrameskip-v4'), scale=True)
    vanilla_pg_with_GAE(env)
