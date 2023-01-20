# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 15:59:49 2022

@author: Yugantar Prakash, Shrayani Mondal
"""

import random
from collections import deque, namedtuple
import numpy as np
import torch

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class ReplayBuffer:
    def __init__(self, buffer_size, batch_size, seed):
        """
        Replay memory allow agent to record experiences and learn from them
        
        Parametes
        ---------
        buffer_size (int): maximum size of internal memory
        batch_size (int): sample size from experience
        seed (int): random seed
        """
        self.batch_size = batch_size
        self.seed = random.seed(seed)
        self.memory = deque(maxlen=buffer_size)
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
    
    def add(self, state, action, reward, next_state, done):
        """Add experience"""
        experience = self.experience(state, action, reward, next_state, done)
        self.memory.append(experience)
          
    #sampling algo used in benchmark
    def sample(self):
        """ 
        Sample randomly and return (state, action, reward, next_state, done) tuple as torch tensors 
        """
        experiences = random.sample(self.memory, k=self.batch_size)
        print('replay buffer shapes:',[experience.state for experience in experiences if experience is not None])
        # Convert to torch tensors
        states = torch.from_numpy(np.vstack([experience.state for experience in experiences if experience is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([experience.action for experience in experiences if experience is not None])).long().to(device)        
        rewards = torch.from_numpy(np.vstack([experience.reward for experience in experiences if experience is not None])).float().to(device)        
        next_states = torch.from_numpy(np.vstack([experience.next_state for experience in experiences if experience is not None])).float().to(device)  
        # Convert done from boolean to int
        dones = torch.from_numpy(np.vstack([experience.done for experience in experiences if experience is not None]).astype(np.uint8)).float().to(device)        
        
        return (states, actions, rewards, next_states, dones)
    
    #sample algo used in our algorithm
    def sampleOurAlgo(self, overrideSize=None):
        """ 
        Sample randomly and return (state, action, reward, next_state, done) tuple as torch tensors 
        Parameters
        ---------
        overrideSize (int): size of samples if different from self.batchSize
        """
        if not overrideSize:
            batchSize = np.max(self.batch_size, overrideSize)
        else:
            batchSize = overrideSize
        # print('batch size is:', batchSize, 'and override size is:', overrideSize)
        if len(self.memory) >= batchSize:
            experiences = random.sample(self.memory, k=batchSize)
            
            # Convert to torch tensors
            states = torch.from_numpy(np.vstack([experience.state for experience in experiences if experience is not None])).float().to(device)
            actions = torch.from_numpy(np.vstack([experience.action for experience in experiences if experience is not None])).long().to(device)        
            rewards = torch.from_numpy(np.vstack([experience.reward for experience in experiences if experience is not None])).float().to(device)        
            next_states = torch.from_numpy(np.vstack([experience.next_state for experience in experiences if experience is not None])).float().to(device)  
            # Convert done from boolean to int
            dones = torch.from_numpy(np.vstack([experience.done for experience in experiences if experience is not None]).astype(np.uint8)).float().to(device)        
            
            return (states, actions, rewards, next_states, dones)
        else:
            return None
    
    def __len__(self):
        return len(self.memory)
    
class ExpertReplayBuffer:
    def __init__(self, buffer_size, batch_size, seed):
        """
        Replay memory allow agent to record experiences and learn from them
        
        Parametes
        ---------
        buffer_size (int): maximum size of internal memory
        batch_size (int): sample size from experience
        seed (int): random seed
        """
        self.batch_size = batch_size
        self.seed = random.seed(seed)
        self.memory = deque(maxlen=buffer_size)
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done", "actionVals"])
    
    def add(self, state, action, reward, next_state, done, action_values=None):
        """Add experience"""
        experience = self.experience(state, action, reward, next_state, done, action_values)
        self.memory.append(experience)

    def sample(self):
        """ 
        Sample randomly and return (state, action, reward, next_state, done) tuple as torch tensors 
        """
        experiences = random.sample(self.memory, k=self.batch_size)
        
        # Convert to torch tensors
        states = torch.from_numpy(np.vstack([experience.state for experience in experiences if experience is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([experience.action for experience in experiences if experience is not None])).long().to(device)        
        rewards = torch.from_numpy(np.vstack([experience.reward for experience in experiences if experience is not None])).float().to(device)        
        next_states = torch.from_numpy(np.vstack([experience.next_state for experience in experiences if experience is not None])).float().to(device)  
        # Convert done from boolean to int
        dones = torch.from_numpy(np.vstack([experience.done for experience in experiences if experience is not None]).astype(np.uint8)).float().to(device)        
        action_values = torch.from_numpy(np.vstack([experience.actionVals for experience in experiences if experience is not None])).float().to(device)
        return (states, actions, rewards, next_states, dones, action_values)
        
    def __len__(self):
        return len(self.memory)