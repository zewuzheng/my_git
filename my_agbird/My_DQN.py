import numpy as np 
import random
import torch
import torch.optim as opt
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import os
#from torch.utils.tensorboard import SummaryWriter

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(9, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        ) 

    def Forward(self, state):
        return self.fc(state)


class Replay_buffer():
    def __init__(self, state_size):
        self.count = 0
        self.size = 300000
        self.data_type = np.dtype([('s', np.float64, state_size), 
                                   ('r', np.float64), ('n', np.float64)])
        self.buffer = [] #np.empty(0, dtype=self.data_type)

    def add_sample(self, s, r, n):
        if self.count == 0:
            self.buffer = []
        self.buffer.append((s, r, n))
        self.count += 1

    def is_ready(self):
        if self.count == self.size:
            self.buffer = np.array(self.buffer, dtype=self.data_type)
            return True
        else:
            return False


class Agent():
    def __init__(self):
        self.net= Net().float()
        self.buffer = Replay_buffer(9)
        self.optimizer = opt.Adam(self.net.parameters(), lr=1e-3)
        self.gamma = 0.995

    @torch.no_grad()
    def get_act(self, state):
        state = torch.from_numpy(state).float()
        q = self.net.Forward(state)
        print('q value is: ', q)
        return 0 if q <= 0 else 1

    def get_value(self, state):
     #   state = torch.from_numpy(state).float()
        q = self.net.Forward(state)
        return q 

    def train(self):
        for i in range(self.buffer.buffer.size // 20):
            print(f'iteration {i}...................................................')
            sample_index = np.random.choice(self.buffer.buffer.size,256)
            s = torch.tensor(self.buffer.buffer['s'], dtype=torch.float)[sample_index]
            r = torch.tensor(self.buffer.buffer['r'], dtype=torch.float).view(-1, 1)[sample_index]   ## n * 1
            n = torch.tensor(self.buffer.buffer['n'], dtype=torch.float).view(-1, 1)[sample_index]
            print(r)
            q_eval = self.get_value(s)
            loss = F.mse_loss(q_eval, (self.gamma ** n) * r)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def save_model(self, t):
        torch.save(self.net.state_dict(), f'models/agent_{t}.pt')

    def load_model(self, path):
        self.net.load_state_dict(torch.load(path, map_location=torch.device('cpu')))

    def store_buffer(self):
        data = np.array(self.buffer.buffer)
        np.save('models/buffer.npy', data)
    
    def get_buffer(self):
        data = np.load('models/buffer.npy', allow_pickle=True)
        temp_buffer = data.tolist()
        self.buffer.buffer = [tuple(x) for x in temp_buffer]
        self.buffer.count = len(self.buffer.buffer)

    def pre_data(self):
        buf = np.array(self.buffer.buffer, dtype = self.buffer.data_type) if evals == True else self.buffer.buffer
        r_buf = buf['r']
        size_r = len(r_buf)
        r_ind = np.where(r_buf>0)[0]
        ratio = int((size_r // len(r_ind)) * 0.7)
        print('ratio is ', ratio / 0.7)
        for ind in r_ind:
            s1 = np.array([buf[ind] for _ in range(ratio)] , dtype = self.buffer.data_type)
            buf = np.concatenate((buf,s1), axis = 0)
        np.random.shuffle(buf)
        self.buffer.buffer = buf
        buf = []

    


if __name__ == '__main__':
    ############################## test for buffer ##########################
    # state = torch.tensor([1,2,3,4,5,6,7,8,9]).float()
    # action = torch.tensor([1,-1]).float()
    # r = 10
    # n = 100
    # buffer = Replay_buffer(9, 2)
    # buffer.add_sample(state, action, r, 90)
    # buffer.add_sample(state, action, r, 10)
    # test_buffer = np.array(buffer.buffer, dtype = buffer.data_type)
    # print(buffer.buffer)
    # print(buffer.count)
    # print(test_buffer['n'])
    # test_buffer['n'] += 1
    # print(test_buffer['n'])
    # test_buffer['n'] += 3
    # print(test_buffer['n'])
    #########################################################################
    ######################### test for network
    # state = torch.tensor([[1,2,3,4,5,6,7,8,9, 10, 11], [3,4,5,6,7,8,9,0,1,2,3]]).float()
    # print(state)
    # net =Net().float()
    # output = net.Forward(state)
    # print(output)
    # a = torch.LongTensor([1, 0]).view(-1, 1)
    # print(output.gather(1, a))
    ################################################
    # state = torch.tensor([1,2,3,4,5,6,7,8,9]).float()
    # action = torch.max(state, 0)
    # print(action)
    ################################### test save model ############################
    a = Agent()
    a.get_buffer()
    a.pre_data()
    a.train()
    a.save_model(10)
