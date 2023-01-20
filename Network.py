# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 15:08:12 2022

@author: Yugantar Prakash, Shrayani Mondal
"""

import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import config
from Replay import ReplayBuffer

BUFFER_SIZE = config.BUFFER_SIZE
BATCH_SIZE = config.BATCH_SIZE
GAMMA = config.GAMMA
TAU = config.TAU
LR = config.LR
UPDATE_EVERY = config.UPDATE_EVERY

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class SoftQNetwork(nn.Module):
    def __init__(self, state_size, action_size, seed, alpha=4):
        """
        Build a fully connected neural network
        
        Parameters
        ----------
        state_size (int): State dimension
        action_size (int): Action dimension
        seed (int): random seed
        """
        super(SoftQNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.alpha = alpha
        self.fc1 = nn.Linear(state_size, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, action_size)  
        
    def forward(self, x):
        """Forward pass"""
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x
    
    def getV(self,q_value):
        """Get Value function from q-value"""
        v = self.alpha * torch.log(torch.sum(torch.exp(q_value / self.alpha), dim=1, keepdim=True))
        return v
    
class SoftDQNAgent:
    def __init__(self, state_size, action_size, seed, algo = "benchmark", alpha=4):
        """
        DQN Agent interacts with the environment, 
        stores the experience and learns from it
        
        Parameters
        ----------
        state_size (int): Dimension of state
        action_size (int): Dimension of action
        seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.alpha = alpha
        # Initialize Q and Fixed Q networks
        self.q_network = SoftQNetwork(state_size, action_size, seed, alpha).to(device)
        self.fixed_network = SoftQNetwork(state_size, action_size, seed, alpha).to(device)
        self.optimizer = optim.Adam(self.q_network.parameters())
        # Initiliase memory 
        self.memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE, seed)
        self.timestep = 0
        self.algo = algo
    
    def step(self, state, action, reward, next_state, done):
        """
        Update Agent's knowledge
        
        Parameters
        ----------
        state (array_like): Current state of environment
        action (int): Action taken in current state
        reward (float): Reward received after taking action 
        next_state (array_like): Next state returned by the environment after taking action
        done (bool): whether the episode ended after taking action
        """
        self.memory.add(state, action, reward, next_state, done)
        self.timestep += 1
        if self.timestep % UPDATE_EVERY == 0:
            #if we at least BATCH_SIZE num of elements in replay, we can learn
            if len(self.memory) > BATCH_SIZE:
                if self.algo == "benchmark":
                    sampled_experiences = self.memory.sample() #take a sample from replay
                elif self.algo == "ours":
                    sampled_experiences = self.memory.sampleOurAlgo() 
                self.learn(sampled_experiences)
        
    def learn(self, experiences):
        """
        Learn from experience by training the q_network 
        
        Parameters
        ----------
        experiences (array_like): List of experiences sampled from agent's memory
        """
        states, actions, rewards, next_states, dones = experiences
        # Get the action values of fixed Network (target network), and get next v values
        action_values = self.fixed_network(next_states).detach()
        next_v = self.fixed_network.getV(action_values).detach()
        # Notes
        # tensor.max(1)[0] returns the values, tensor.max(1)[1] will return indices
        # unsqueeze operation --> np.reshape
        # Here, we make it from torch.Size([64]) -> torch.Size([64, 1])
        # max_action_values = action_values.max(1)[0].unsqueeze(1)
        
        # If done just use reward, else update Q_target with discounted action values
        Q_target = rewards + (GAMMA * next_v * (1 - dones))
        Q_expected = self.q_network(states).gather(1, actions)
        
        # Calculate loss
        loss = F.mse_loss(Q_expected, Q_target)
        self.optimizer.zero_grad()
        # backward pass
        loss.backward()
        # update weights
        self.optimizer.step()
        
        # Update fixed weights
        self.update_fixed_network(self.q_network, self.fixed_network)
        
    def update_fixed_network(self, q_network, fixed_network):
        """
        Update fixed network by copying weights from Q network using TAU param
        
        Parameters
        ----------
        q_network (PyTorch model): Q network
        fixed_network (PyTorch model): Fixed target network
        """
        for source_parameters, target_parameters in zip(q_network.parameters(), fixed_network.parameters()):
            target_parameters.data.copy_(TAU * source_parameters.data + (1.0 - TAU) * target_parameters.data)
        
        
    def act(self, state):
        """
        Choose the action using max entropy
        
        Parameters
        ----------
        state (array_like): current state of environment
        """
        
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        # set the network into evaluation mode 
        self.q_network.eval()
        with torch.no_grad():
            action_values = self.q_network(state)
        v = self.q_network.getV(action_values).squeeze()
        # Back to training mode
        self.q_network.train()

        dist = torch.exp((action_values-v)/self.alpha)
        if torch.sum(torch.isnan(dist)) != 0:
            print('distribution is nan after exp')
        throwdist = dist/torch.sum(dist) #normalizing the distribution
        # print(throwdist)
        if torch.sum(torch.isnan(throwdist)) != 0:
            print('distribution is nan after normalization')
            print(dist)
            throwdist = torch.tensor([[0.25,0.25,0.25,0.25]])
        c = torch.distributions.Categorical(throwdist)
        action = c.sample() #sampling from distribution
        # action = np.argmax(action_values.cpu().data.numpy())
        return action.item()
    
    def demonstrate(self, state):
        """
        Choose the action using max entropy
        
        Parameters
        ----------
        state (array_like): current state of environment
        """
        
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        # set the network into evaluation mode 
        self.q_network.eval()
        with torch.no_grad():
            action_values = self.q_network(state)
        v = self.q_network.getV(action_values).squeeze()
        # Back to training mode
        self.q_network.train()

        dist = torch.exp((action_values-v)/self.alpha)
        if torch.sum(torch.isnan(dist)) != 0:
            print('distribution is nan after exp')
        throwdist = dist/torch.sum(dist) #normalizing the distribution
        # print(throwdist)
        if torch.sum(torch.isnan(throwdist)) != 0:
            print('distribution is nan after normalization')
            print(dist)
            throwdist = torch.tensor([[0.25,0.25,0.25,0.25]])
        c = torch.distributions.Categorical(throwdist)
        action = c.sample() #sampling from distribution
        # action = np.argmax(action_values.cpu().data.numpy())
        return action.item(), action_values
        
    def checkpoint(self, filename):
        torch.save(self.q_network.state_dict(), filename)
        
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, seed):
        """
        Build a fully connected neural network
        
        Parameters
        ----------
        state_size (int): State dimension
        action_size (int): Action dimension
        seed (int): random seed
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, action_size)  
        
    def forward(self, x):
        """Forward pass"""
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x