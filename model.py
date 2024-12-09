
import torch
import torch.nn as nn

class Env:
    def __init__(self, data, history_back=90):
        self.data = data
        self.history_back = history_back
        self.init()
        
    def init(self):
        self.time_step = 0
        self.done = False
        self.profits = 0
        self.positions = []
        self.position_value = 0
        self.history = [0 for _ in range(self.history_back)]
        return [self.position_value] + self.history # obs
    
    def step(self, action):
        reward = 0
        
        # action = 0: stay, 1: buy, 2: sell
        if action == 1:
            self.positions.append(self.data.iloc[self.time_step, :]['Close'])
        elif action == 2: # sell
            if len(self.positions) == 0:
                reward = -1
            else:
                profits = 0
                for p in self.positions:
                    profits += (self.data.iloc[self.time_step, :]['Close'] - p)
                reward += profits
                self.profits += profits
                self.positions = []
        
        # set next time
        self.time_step += 1
        
        self.position_value = 0
        for p in self.positions:
            self.position_value += (self.data.iloc[self.time_step, :]['Close'] - p)
        self.history.pop(0)
        self.history.append(self.data.iloc[self.time_step, :]['Close'] - self.data.iloc[(self.time_step-1), :]['Close'])
        if self.time_step == len(self.data)-1:
            self.done=True
        # clipping reward
        if reward > 0:
            reward = 1
        elif reward < 0:
            reward = -1
        return [self.position_value] + self.history, reward, self.done # obs, reward, done
        
class Network(nn.Module):
        
    def __init__(self, input_size,hidden_size,output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        super(Network,self).__init__()
            
        self.fc_val = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
        
        
    def forward(self,x):
        return (self.fc_val(x))
