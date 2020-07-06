import torch
import os 
import pickle
import multiprocessing 
import glob 
import time 
from random import shuffle, sample

def list_of_list_into_tensor(x):
    """
    Takes a list of list of tensors and makes it 
    into a pytorch tensor. 
    All inner lists are padded on right in mode 'reflecion'
    so that all inner lists have same length equal
    to the maximum length inner list.
    """
    max_length = max([len(i) for i in x])
    for i in range(len(x)):
        x[i].extend([x[i][-1]]*(max_length - len(x[i])))

    return torch.tensor(x)


def process_file(filepath):
    with open(filepath, 'rb') as f:
        y = pickle.load(f)
        return y


def callback(loaded_data):
    obs_all.append(torch.tensor(loaded_data[0]).squeeze().permute(0,3,1,2))
    actions_all.append(torch.tensor(loaded_data[1]))

def error_callback(e):
    print(e)

def load_expert_data_atari(path_to_dir, num_of_trajs='all'):
    start = time.time()
    global obs_all, actions_all
    obs_all = []
    actions_all = []
    if num_of_trajs == 'all':
        files = glob.glob(path_to_dir + "/*.pkl")
    else:
        files = glob.glob(path_to_dir + "/*.pkl")[:num_of_trajs]
    print(files)
    pool = multiprocessing.Pool()
    for file in files:
        pool.apply_async(process_file, [file], callback=callback, error_callback=error_callback)
    pool.close()
    pool.join()
    obs_all_t = torch.cat(obs_all, dim=0)
    actions_all_t = torch.cat(actions_all, dim=0)
    print(time.time()-start)
    return obs_all_t.float(), actions_all_t.float()            

def load_expert_data_atari_no_multi(path_to_dir, num_of_trajs='all'):
    start = time.time()
    obs_all = []
    actions_all = []
    if num_of_trajs == 'all':
        files = glob.glob(path_to_dir + "/*.pkl")
    else:
        files = glob.glob(path_to_dir + "/*.pkl")[:num_of_trajs]
    print(files)
    for file in files:
        with open(file, 'rb') as f:
            loaded_data = pickle.load(f)
        obs_all.append(torch.tensor(loaded_data[0]).squeeze().permute(0,3,1,2))
        actions_all.append(torch.tensor(loaded_data[1]))
    obs_all_t = torch.cat(obs_all, dim=0)
    actions_all_t = torch.cat(actions_all, dim=0)
    print(time.time()-start)
    return obs_all_t.float(), actions_all_t.float()            


class DataLoader:
    def __init__(self, path_to_dir, num_of_trajs='all'):
        if num_of_trajs == 'all':
            self.files = (glob.glob(path_to_dir + "/*.pkl"))
        else:
            self.files = (glob.glob(path_to_dir + "/*.pkl"))[:num_of_trajs]
        shuffle(self.files)
        self.idx = 0
        self.buffer = []

    def sample(self, device):
        with open(self.files[self.idx], 'rb') as f:
            obs_list, acts_list = pickle.load(f)
        self.idx += 1
        if self.idx == len(self.files):
            self.idx = 0
            shuffle(self.files)
            
        return torch.tensor(obs_list).squeeze().permute(0,3,1,2).to(device), \
                torch.tensor(acts_list).float().to(device)



if __name__ == '__main__':
    dataloader = DataLoader('/home/usman/repos/rl_zoo/rl-baselines-zoo/ExpertData/ppo2_BreakoutNoFrameskip-v4', num_of_trajs='all')
    a, b = dataloader.sample('cpu')
    print(torch.max(a))
