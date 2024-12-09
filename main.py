# _*_ coding: utf-8 _*_
'''
requires numpy 
'''
import time
import copy
import numpy as np
import pandas as pd
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import yfinance as yf
import sys

import os, glob

from model import Env, Network

#import more APIs here

data_dir = "data"
model_dir = "models"

default_model_name = r"rl.net"
LR = 0.001
SHOW_ACTION = True

def save_model(net: Network, model_name: str) -> None:
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    with open(os.path.join(model_dir, model_name), "wb") as f:
        pickle.dump(net, f)

def load_model(model_name):
    if not os.path.exists(model_dir):
        print(f"{model_dir} directory does not exists")
    else:
        with open(os.path.join(model_dir, model_name), "rb") as f:
            return pickle.load(f)
            
def train_helper(model_name=default_model_name, env: Env = None, net: Network = None, 
    epoch_num = 50):
    loss_function = nn.MSELoss()
    optimizer = optim.Adam(list(net.parameters()), lr=LR)
    
    step_max = len(env.data)-1
    memory_size = 200
    batch_size = 50
    gamma = 0.97
    memory = []
    total_step = 0
    total_rewards = []
    total_losses = []
    epsilon = 1.0
    epsilon_decrease = 1e-3
    epsilon_min = 0.1
    start_reduce_epsilon = 200
    train_freq = 10
    update_q_freq = 20
    gamma = 0.97
    show_log_freq = 5
    net_back = copy.deepcopy(net)

    start = time.time()
    for epoch in range(epoch_num):
        pobs = env.init()
        step = 0
        done = False
        total_reward = 0
        total_loss = 0

        while not done and step < step_max:
            # select act
            pact = np.random.randint(3)
            if np.random.rand() > epsilon:
                pact = net(torch.from_numpy(np.array(pobs, dtype=np.float32).reshape(1, -1)))
                pact = np.argmax(pact.data)
                pact = pact.numpy()

            # act
            obs, reward, done = env.step(pact)

            # add memory
            memory.append((pobs, pact, reward, obs, done))
            if len(memory) > memory_size:
                memory.pop(0)

            # train or update q
            if len(memory) == memory_size:
                if total_step % train_freq == 0:
                    shuffled_memory = np.random.permutation(memory)
                    memory_idx = range(len(shuffled_memory))
                    for i in memory_idx[::batch_size]:
                        batch = np.array(shuffled_memory[i:i+batch_size])
                        b_pobs = np.array(batch[:, 0].tolist(), dtype=np.float32).reshape(batch_size, -1)
                        b_pact = np.array(batch[:, 1].tolist(), dtype=np.int32)
                        b_reward = np.array(batch[:, 2].tolist(), dtype=np.int32)
                        b_obs = np.array(batch[:, 3].tolist(), dtype=np.float32).reshape(batch_size, -1)
                        b_done = np.array(batch[:, 4].tolist(), dtype=np.bool)

                        q = net(torch.from_numpy(b_pobs))
                        q_ = net_back(torch.from_numpy(b_obs))
                        maxq = np.max(q_.data.numpy(),axis=1)
                        target = copy.deepcopy(q.data)
                        for j in range(batch_size):
                            target[j, b_pact[j]] = b_reward[j]+gamma*maxq[j]*(not b_done[j])
                        net.zero_grad()
                        loss = loss_function(q, target)
                        total_loss += loss.data.item()
                        loss.backward()
                        optimizer.step()
                        
                if total_step % update_q_freq == 0:
                    net_back = copy.deepcopy(net)
                    
                # epsilon
                if epsilon > epsilon_min and total_step > start_reduce_epsilon:
                    epsilon -= epsilon_decrease

                # next step
                total_reward += reward
                pobs = obs
                step += 1
                total_step += 1

            total_rewards.append(total_reward)
            total_losses.append(total_loss)

            if (epoch+1) % show_log_freq == 0:
                log_reward = sum(total_rewards[((epoch+1)-show_log_freq):])/show_log_freq
                log_loss = sum(total_losses[((epoch+1)-show_log_freq):])/show_log_freq
                elapsed_time = time.time()-start
                print('\t'.join(map(str, [epoch+1, epsilon, total_step, log_reward, log_loss, elapsed_time])))
                start = time.time()

        save_model(net, model_name)


def test_helper(test_data, net) -> float:
    test_env = Env(test_data)
    pobs = test_env.init()
    test_acts = []
    test_rewards = []
    for i in range(len(test_env.data)):
        pact = net(torch.from_numpy(np.array(pobs, dtype=np.float32).reshape(1, -1)))
        pact = np.argmax(pact.data)
        if SHOW_ACTION:
            action = pact.numpy()
            date_str = str(test_env.data.index[i]).split()[0]
            if action == 0 and i == len(test_env.data) - 1:
                print(f"On {date_str}, the action you should do now is: Stay")
            elif action == 1 and i == len(test_env.data) - 1:
                print(f"On {date_str}, the action you should do now is: Buy")
            elif action == 2 and i == len(test_env.data) - 1:
                print(f"On {date_str}, the action you should do now is: Sell")
        if i <  len(test_env.data) - 1:
            test_acts.append(pact.item())
            obs, reward, done = test_env.step(pact.numpy())
            test_rewards.append(reward)
            pobs = obs
        
    return test_env.profits, action

def train(file_name, split_date, hidden_size, num_epoch = 50, model_name=default_model_name):
    data = pd.read_csv(os.path.join(data_dir, file_name))
    data['Date'] = pd.to_datetime(data['Date'])
    data = data.set_index('Date')
    train_data = data[:split_date]
    env = Env(train_data)
    net = Network(env.history_back + 1, hidden_size, 3)
    train_helper(model_name, env, net, num_epoch)

def test(file_name=None, model_name=default_model_name):
    net = load_model(model_name)
    if file_name is None:
        files = glob.glob(os.path.join(data_dir, "*.csv"))
        num_win = 0
        num_loss = 0
        num_draw = 0
        for i in files:
            p = test_one_file(i, net)
            if p is None:
                continue
            if p > 0:
                num_win += 1
            elif p < 0:
                num_loss += 1
            else:
                num_draw += 1
        print(f"win: {num_win} loss: {num_loss} draw: {num_draw} win percent: {num_win / (num_win + num_loss + num_draw)}")
    else:
        test_one_file(os.path.join(data_dir, file_name),net)

def test_one_file(data, net):
    try:
        #data = pd.read_csv(file_path)
        data['Date'] = pd.to_datetime(data['Date'])
        data = data.set_index('Date')
        profit, naction = test_helper(data, net)
        #print(f"The simulation profit is: {profit}")
        #print(naction)
        return naction
    except Exception as e:
        print(f"{data} error:{str(e)}")

if __name__ == "__main__":

    #data = sys.argv[1]
    # train("AMD.csv", "2019-09-01", 1000, 50)
    # test()
    use = input("Which stock do you want to trade today? ")
    #tick = yf.Ticker(data)
    tick = yf.Ticker(use)
    df = tick.history(period="24mo")
    df['Date'] = df.index

    action = test_one_file(df, load_model(default_model_name))


    #Use APIs here
    #if(action == 0):
        #Do nothing
    #elif(action == 1):
        #Buy
    #else:
        #Sell
    
    

