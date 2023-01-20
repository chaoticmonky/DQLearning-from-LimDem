# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 19:27:04 2022

@author: Lenovo
"""

BUFFER_SIZE = int(1e5)  # Replay memory size
BATCH_SIZE = 64         # Number of experiences to sample from memory
GAMMA = 0.99            # Discount factor
TAU = 1e-3              # Soft update parameter for updating fixed q network
LR = 1e-4               # Q Network learning rate
UPDATE_EVERY = 4        # How often to update Q network

MAX_EPISODES = 2000  # Max number of episodes to play
MAX_STEPS = 1000     # Max steps allowed in a single episode/play
ENV_SOLVED = 200     # MAX score at which we consider environment to be solved
PRINT_EVERY = 100    # How often to print the progress

#used in our algorithm
MAX_EXPERT_EPISODES = 5   # Max number of episodes to call an expert for
MAX_EXPERT_ITERS = 100    # Max number of iterations to call an expert for

# Epsilon schedule

EPS_START = 1.0      # Default/starting value of eps
EPS_DECAY = 0.999    # Epsilon decay rate
EPS_MIN = 0.01       # Minimum epsilon 

EXPERT_MEMORY = int(1e4) #Expert Replay Buffer Size
STD_THRESHOLD = 20