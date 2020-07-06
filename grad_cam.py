from atari_wrapper import wrap_deepmind, wrap_pytorch, make_atari as wrap_atari
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 
import gym
import matplotlib.pyplot as plt
import random
from torch.distributions import Categorical 

class Scale(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        return x * self.scale

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def make_env(gym_id, seed, idx):
# def thunk():
    # env = gym.make(gym_id)
    env = wrap_atari(gym_id)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    # if args.capture_video:
        # if idx == 0:
        #     env = Monitor(env, f'videos/{experiment_name}')
    env = wrap_pytorch(
        wrap_deepmind(
            env,
            clip_rewards=True,
            frame_stack=True,
            scale=False,
        )
    )
    env.seed(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return env
    # return thunk

device = 'cpu'
gym_id = 'PongNoFrameskip-v4'
seed = 0
envs = make_env(gym_id, seed, 0)

class Agent(nn.Module):
    def __init__(self, frames=4):
        super(Agent, self).__init__()
        self.network = nn.Sequential(
            Scale(1/255),
            layer_init(nn.Conv2d(frames, 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(3136, 512)),
            nn.ReLU()
        )
        self.actor = layer_init(nn.Linear(512, envs.action_space.n), std=0.01)
        self.critic = layer_init(nn.Linear(512, 1), std=1)
        self.discriminator = nn.Sequential(
                    layer_init(nn.Linear(512+1, 400)),    # 1 is for action
                    nn.Dropout(p=0.6),
                    nn.ReLU(),
                    layer_init(nn.Linear(400, 1)),
                    nn.Sigmoid())

    def forward(self, x):
        return self.network(x)

    def get_action(self, x, action=None):
        logits = self.actor(self.forward(x))
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy()

    def get_value(self, x):
        return self.critic(self.forward(x))

    def discriminate_fn(self, obs, actions):
        if len(obs.shape) > 4:
            states = self.network(obs.view((-1,)+envs.observation_space.shape))
        else:
            states = self.network(obs)

        acts = actions.view(-1, 1)   # assuming discrete actions
        x = torch.cat([states, acts], dim=1)
        return self.discriminator(x)

    def evaluate_reward(self, obs, actions):
        reward = self.discriminate_fn(obs, actions).view(obs.shape[0], obs.shape[1])
        return torch.log(reward)



class Agent2(nn.Module):
    def __init__(self):
        super(Agent2, self).__init__()
        
        # get the pretrained VGG19 network
        self.agent = Agent()
        self.agent.load_state_dict(torch.load('PongNoFrameskip-v4__cleanrl_ppo__1__1592777189__agent__', 
                                                        map_location=torch.device('cpu')))

        # disect the network to access its last convolutional layer
        self.features_conv = self.agent.network[:6]
        
        # get the max pool of the features stem
        # self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        self.block2 = self.agent.network[6:]
        
        # get the classifier of the vgg19
        self.classifier = self.agent.actor
        
        # placeholder for the gradients
        self.gradients = None
    
    # hook for the gradients of the activations
    def activations_hook(self, grad):
        self.gradients = grad
        
    def forward(self, x):
        # print(x.requires_grad)
        x = self.features_conv(x)
        print(x.requires_grad)
        # register the hook
        h = x.register_hook(self.activations_hook)
        
        # apply the remaining pooling
        # x = self.max_pool(x)
        # x = x.view((1, -1))
        x = self.block2(x)
        x = self.classifier(x)
        return x
    
    # method for the gradient extraction
    def get_activations_gradient(self):
        return self.gradients
    
    # method for the activation exctraction
    def get_activations(self, x):
        return self.features_conv(x)


agent = Agent2()
print(agent.agent.network)
next_obs = envs.reset()
next_done = torch.zeros(1).to(device)

obs = np.zeros((128, 1) + envs.observation_space.shape)
dones = np.zeros((128, 1))


for step in range(0, 128):
    obs[step] = next_obs
    dones[step] = next_done

    # ALGO LOGIC: put action logic here
    # plt.imshow(obs[step].squeeze().transpose(1,2,0))
    # plt.imshow(obs[step][0,0,:,:])
    # plt.show()
    logits = agent(torch.from_numpy(obs[step]).float())
    with torch.no_grad():
        action, _, _ = agent.agent.get_action(torch.from_numpy(obs[step]).float())
    print(action)
    next_obs, _, _, _ = envs.step(action)
    print(type(next_obs))
    if random.random() > 0.90:
        logits[:, np.argmax(logits.clone().detach().numpy())].backward()

        gradients = agent.get_activations_gradient()

        # pool the gradients across the channels
        pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])

        # get the activations of the last convolutional layer
        activations = agent.get_activations(torch.from_numpy(obs[step]).float()).detach()

        # weight the channels by corresponding gradients
        for i in range(activations.shape[1]):
            activations[:, i, :, :] *= pooled_gradients[i]
            
        # average the channels of the activations
        heatmap = torch.mean(activations, dim=1).squeeze()

        heatmap = np.maximum(heatmap, 0)

        # normalize the heatmap
        heatmap /= torch.max(heatmap)
        heatmap = heatmap.detach().numpy()
        # draw the heatmap
        # plt.matshow(heatmap.squeeze())
        # plt.show()

        import cv2
        print(type(heatmap))
        print(heatmap.dtype)
        print(heatmap.shape)
        img = obs[0][0,1:,...].transpose(1,2,0)
        # img = obs[0][0,1,...]
        heatmap = cv2.resize(heatmap, (84, 84))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        superimposed_img = heatmap * 0.4 + img
        cv2.imwrite('./map'+str(step)+'.jpg', superimposed_img)