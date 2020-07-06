import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch.distributions import Categorical
import numpy as np 

class Buffer:
    """
    Buffer class for storing last n-1 observations.
    Calling an instance of this class returns a 
    torch tensor with last n observations.

    This class is exclusive for Atari games with observations
    of form 84*84*1.
    """
    def __init__(self, size):
        self.size = size
        self.initialize()

    def insert(self, obs):
        self.buffer.append(obs)
        if len(self.buffer) > self.size:
            del(self.buffer[0])

    def reset(self):
        del(self.buffer)
        self.initialize()
    
    def initialize(self):
        self.buffer = []
        for i in range(self.size):
            self.buffer.append(np.zeros((84,84), dtype=np.float32))


    def __call__(self, obs):
        self.insert(obs[...,0])
        return torch.tensor(self.buffer)[None,...]   # adding batch dimension


class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)


class ConvPolicyAtari(nn.Module):
    def __init__(self, action_dim):
        super(ConvPolicyAtari, self).__init__()
        self.action_dim = action_dim
        self.common_layers = nn.ModuleList([
        nn.Conv2d(in_channels=4, 
                  out_channels=32,
                  kernel_size=(8,8),
                  stride=(4,4)),
        nn.ReLU(),
        nn.Conv2d(in_channels=32, 
                  out_channels=64,
                  kernel_size=(4,4),
                  stride=(2,2)),
        nn.ReLU(),
        nn.Conv2d(in_channels=64, 
                  out_channels=64,
                  kernel_size=(3,3),
                  stride=(1,1)),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(3136, 512)
        ])

        self.policy_layer = nn.Linear(512,self.action_dim)
        self.value_layer = nn.Linear(512,1)

    def forward(self, x):
        for l in self.common_layers:
            x = l(x)
        action_probs = F.softmax(self.policy_layer(x), dim=1)
        value = self.value_layer(x)
        m = Categorical(action_probs)
        action = m.sample()
        log_prob = m.log_prob(action)
        return action.item(), log_prob, value

