# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 16:07:52 2022

@author: Yugantar Prakash, Shrayani Mondal
"""
import random
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import config
from Network import QNetwork, SoftQNetwork
from Replay import ReplayBuffer, ExpertReplayBuffer

BUFFER_SIZE = config.BUFFER_SIZE
BATCH_SIZE = config.BATCH_SIZE
GAMMA = config.GAMMA
TAU = config.TAU
LR = config.LR
UPDATE_EVERY = config.UPDATE_EVERY
EXPERT_MEMORY = config.EXPERT_MEMORY
MAX_EPISODES = config.MAX_EPISODES
MAX_STEPS = config.MAX_STEPS
MAX_EXPERT_ITERS = config.MAX_EXPERT_ITERS

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class DQNAgent:
    def __init__(self, state_size, action_size, seed):
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
        # Initialize Q and Fixed Q networks
        self.q_network = QNetwork(state_size, action_size, seed).to(device)
        self.fixed_network = QNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.q_network.parameters())
        # Initiliase memory 
        self.memory = ReplayBuffer(BUFFER_SIZE, int(BATCH_SIZE * 0.75), seed)
        self.expertMemory = ExpertReplayBuffer(EXPERT_MEMORY, int(BATCH_SIZE * 0.25), seed)
        self.timestep = 0
        
    
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
            if len(self.memory) > BATCH_SIZE:
                sampled_experiences = self.memory.sample()
                expert_sampled_experiences = self.expertMemory.sample()
                # print(sampled_experiences[0].shape, expert_sampled_experiences[0].shape)
                teStates = torch.cat((sampled_experiences[0], expert_sampled_experiences[0]), dim=0)
                teAction = torch.cat((sampled_experiences[1], expert_sampled_experiences[1]), dim=0)
                teReward = torch.cat((sampled_experiences[2], expert_sampled_experiences[2]), dim=0)
                teNState = torch.cat((sampled_experiences[3], expert_sampled_experiences[3]), dim=0)
                teDones  = torch.cat((sampled_experiences[4], expert_sampled_experiences[4]), dim=0)
                #TO CHECK IF IT MATTERS
                # indexes = torch.randperm(total_experiences.shape[0])
                # total_experiences = total_experiences[indexes]
                expertActionVals = expert_sampled_experiences[-1]
                total_experiences = (teStates, teAction, teReward, teNState, teDones, expertActionVals)
                self.learn(total_experiences)

    def expertStore(self, expertAgent, env):
        """
        Populate expert buffer for the use of BC during training
        
        Parameters
        ----------
        expertAgent (SoftDQNAgent): the expert agent that provides trajectories for BC
        """
        for episode in range(1, MAX_EPISODES + 1):
            state = env.reset()
            score = 0
            for t in range(MAX_STEPS):
                action = expertAgent.act(state)
                next_state, reward, done, truncated, info = env.step(action)
                self.expertMemory.add(state, action, reward, next_state, done, action_values)
                state = next_state
                if len(self.expertMemory) == EXPERT_MEMORY: #we stop learning if expert replay filled
                    break
                if done: #if the episode is over, we move on to next episode
                    break
            if len(self.expertMemory) == EXPERT_MEMORY: #we stop learning if expert replay filled
                break
        env.close()
    
    def expertStoreKL(self, expertAgent, env):
        """
        Populate expert buffer for the use of BC during training
        
        Parameters
        ----------
        expertAgent (SoftDQNAgent): the expert agent that provides trajectories for BC
        """
        for episode in range(1, MAX_EPISODES + 1):
            state = env.reset()[0]
            score = 0
            for t in range(MAX_STEPS):
                # print('populating expert:', type(state[0]), state)
                action, action_values = expertAgent.demonstrate(state)
                # print('populating:', env.step(action))
                next_state, reward, done, truncated, info = env.step(action)
                self.expertMemory.add(state, action, reward, next_state, done, action_values)
                state = next_state
                if len(self.expertMemory) == EXPERT_MEMORY: #we stop learning if expert replay filled
                    break
                if done: #if the episode is over, we move on to next episode
                    break
            if len(self.expertMemory) == EXPERT_MEMORY: #we stop learning if expert replay filled
                break

    def learn(self, experiences):
        """
        Learn from experience by training the q_network 
        
        Parameters
        ----------
        experiences (array_like): List of experiences sampled from agent's memory
        """
        states, actions, rewards, next_states, dones, expertActionVals = experiences
        eStates = states[int(BATCH_SIZE*0.75):,:]
        eActions = states[int(BATCH_SIZE*0.75):,:]

        prediction_actions = self.q_network(eStates).detach()
        prediction_actions = prediction_actions.max(1)[1].unsqueeze(1)

        # Get the action with max Q value
        action_values = self.fixed_network(next_states).detach()
        # Notes
        # tensor.max(1)[0] returns the values, tensor.max(1)[1] will return indices
        # unsqueeze operation --> np.reshape
        # Here, we make it from torch.Size([64]) -> torch.Size([64, 1])
        max_action_values = action_values.max(1)[0].unsqueeze(1)
        
        
        # If done just use reward, else update Q_target with discounted action values
        Q_target = rewards + (GAMMA * max_action_values * (1 - dones))
        Q_expected = self.q_network(states).gather(1, actions)
        # print('states have size', states.shape, 'expert states', eStates.shape)                 #64x8 vs 16x8
        # print('action values from fixed network is:', action_values.shape, actions.shape)
        # print('prediction actions', prediction_actions.shape, 'expert actions', eActions.shape) #16x8 vs 16x4
        # Calculate loss
        q_loss = F.mse_loss(Q_expected, Q_target)
        # bc_loss = F.mse_loss(eActions, prediction_actions)
        bc_loss = F.kl_div(prediction_actions, expertActionVals)
        loss = q_loss + bc_loss
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
        
        
    def act(self, state, eps=0.0):
        """
        Choose the action
        
        Parameters
        ----------
        state (array_like): current state of environment
        eps (float): epsilon for epsilon-greedy action selection
        """
        rnd = random.random()
        if rnd < eps:
            return np.random.randint(self.action_size)
        else:
            state = torch.from_numpy(state).float().unsqueeze(0).to(device)
            # set the network into evaluation mode 
            self.q_network.eval()
            with torch.no_grad():
                action_values = self.q_network(state)
            # Back to training mode
            self.q_network.train()
            action = np.argmax(action_values.cpu().data.numpy())
            return action    
        
    def checkpoint(self, filename):
        torch.save(self.q_network.state_dict(), filename)
  
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
                if self.algo in ["benchmark","expert"]:
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
        
        
class DQNAgentOurAlgo:
    def __init__(self, state_size, action_size, seed):
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
        # Initialize Q and Fixed Q networks
        self.q_network = QNetwork(state_size, action_size, seed).to(device)
        self.fixed_network = QNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.q_network.parameters())
        # Initiliase memory 
        self.memory = ReplayBuffer(BUFFER_SIZE, int(BATCH_SIZE * 0.75), seed)
        self.expertMemory = ExpertReplayBuffer(EXPERT_MEMORY, int(BATCH_SIZE * 0.25), seed)
        self.timestep = 0
        
    
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
            if len(self.memory) > BATCH_SIZE:
                if len(self.expertMemory)  > int( BATCH_SIZE*0.25):
                    sampled_experiences = self.memory.sampleOurAlgo()
                    expert_sampled_experiences = self.expertMemory.sample()
                    teStates = torch.cat((sampled_experiences[0], expert_sampled_experiences[0]), dim=0)
                    teAction = torch.cat((sampled_experiences[1], expert_sampled_experiences[1]), dim=0)
                    teReward = torch.cat((sampled_experiences[2], expert_sampled_experiences[2]), dim=0)
                    teNState = torch.cat((sampled_experiences[3], expert_sampled_experiences[3]), dim=0)
                    teDones  = torch.cat((sampled_experiences[4], expert_sampled_experiences[4]), dim=0)

                    expertActionVals = expert_sampled_experiences[-1]
                    total_experiences = (teStates, teAction, teReward, teNState, teDones, expertActionVals)
                    self.learn(total_experiences)
                else:
                    sampled_experiences = self.memory.sampleOurAlgo(overrideSize = BATCH_SIZE)
                    self.learn(sampled_experiences)
    
    def learn(self, experiences):
        """
        Learn from experience by training the q_network 
        
        Parameters
        ----------
        experiences (array_like): List of experiences sampled from agent's memory
        """
        if len(experiences) == 5:
            states, actions, rewards, next_states, dones = experiences
            expertCalled = None
        else:
            states, actions, rewards, next_states, dones, expertActionVals  = experiences
            expertCalled = 1
        

        ####BC LOSS = KL div between Expert's Action vector and Agent's action vector#### 
        eStates = states[int(BATCH_SIZE*0.75):,:] 
        prediction_actions = self.q_network(eStates).detach() if expertCalled else None
        bc_loss = F.kl_div(prediction_actions, expertActionVals) if expertCalled else None
		
        """
        #if expertActionVals:
        # print('q_network output:', prediction_actions.shape, 'max', prediction_actions.max(1)[1].unsqueeze(1).shape)
        # max_prediction_actions = prediction_actions.max(1)[1].unsqueeze(1) if expertActionVals else None
        """
        
        ####1-step Q LOSS####
        # Get the action with max Q value
        action_values = self.fixed_network(next_states).detach()
        # Here, we make it from torch.Size([64]) -> torch.Size([64, 1])
        max_action_values = action_values.max(1)[0].unsqueeze(1)
        
        # If done just use reward, else update Q_target with discounted action values
        Q_target = rewards + (GAMMA * max_action_values * (1 - dones))
        Q_expected = self.q_network(states).gather(1, actions)
        q_loss = F.mse_loss(Q_expected, Q_target)
        
        loss = q_loss + bc_loss if expertCalled else q_loss
        self.optimizer.zero_grad()
        # backward passlen(self.expertMemory)  > int( BATCH_SIZE*0.25)
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
        

    def expertStore(self, expertAgent, env):
        """
        Populate expert buffer for the use of BC during training
        
        Parameters
        ----------
        expertAgent (SoftDQNAgent): the expert agent that provides trajectories for BC
        """
        for episode in range(1, MAX_EPISODES + 1):
            state = env.reset()
            score = 0
            for t in range(MAX_STEPS):
                action = expertAgent.act(state)
                next_state, reward, done, truncated, info = env.step(action)
                self.expertMemory.add(state, action, reward, next_state, done)
                state = next_state
                if len(self.expertMemory) == EXPERT_MEMORY: #we stop learning if expert replay filled
                    break
                if done: #if the episode is over, we move on to next episode
                    break
            if len(self.expertMemory) == EXPERT_MEMORY: #we stop learning if expert replay filled
                break
    
    def _call_expert(self, expertAgent, env, MAX_EXPERT_EPISODES):
        """
        Populate expert buffer when required
        
        Parameters
        ----------
        expertAgent (SoftDQNAgent): the expert agent that provides trajectories for BC
        """
        print('before called expert:', len(self.expertMemory))
        iters_added = 0
        for episode in range(1, MAX_EXPERT_EPISODES + 1):
            state = env.reset()
            score = 0
            for t in range(MAX_STEPS):
                action = expertAgent.act(state)
                next_state, reward, done, truncated, info = env.step(action)
                self.expertMemory.add(state, action, reward, next_state, done)
                state = next_state
                iters_added += 1
                if len(self.expertMemory) == EXPERT_MEMORY or iters_added == MAX_EXPERT_ITERS: #we stop learning if expert replay filled
                    break
                if done: #if the episode is over, we move on to next episode
                    break
            if len(self.expertMemory) == EXPERT_MEMORY or iters_added == MAX_EXPERT_ITERS: #we stop learning if expert replay filled
                break
        print('after called expert:', len(self.expertMemory))

    def _call_expert_KL(self, expertAgent, env, MAX_EXPERT_EPISODES):
        """
        Populate expert buffer when required
        
        Parameters
        ----------
        expertAgent (SoftDQNAgent): the expert agent that provides trajectories for BC
        """
        print('before called expert:', len(self.expertMemory))
        iters_added = 0
        for episode in range(1, MAX_EXPERT_EPISODES + 1):
            state = env.reset()
            score = 0
            for t in range(MAX_STEPS):
                action, action_values = expertAgent.demonstrate(state)
                next_state, reward, done, truncated, info = env.step(action)
                self.expertMemory.add(state, action, reward, next_state, done, action_values)
                state = next_state
                iters_added += 1
                if len(self.expertMemory) == EXPERT_MEMORY or iters_added == MAX_EXPERT_ITERS: #we stop learning if expert replay filled
                    break
                if done: #if the episode is over, we move on to next episode
                    break
            if len(self.expertMemory) == EXPERT_MEMORY or iters_added == MAX_EXPERT_ITERS: #we stop learning if expert replay filled
                break
        print('after called expert:', len(self.expertMemory))

    
    def act(self, state, eps=0.0):
        """
        Choose the action
        
        Parameters
        ----------
        state (array_like): current state of environment
        eps (float): epsilon for epsilon-greedy action selection
        """
        rnd = random.random()
        if rnd < eps:
            return np.random.randint(self.action_size)
        else:
            state = torch.from_numpy(state).float().unsqueeze(0).to(device)
            # set the network into evaluation mode 
            self.q_network.eval()
            with torch.no_grad():
                action_values = self.q_network(state)
            # Back to training mode
            self.q_network.train()
            action = np.argmax(action_values.cpu().data.numpy())
            return action    
        
    def checkpoint(self, filename):
        torch.save(self.q_network.state_dict(), filename)
        
class OgDQNAgent:
    def __init__(self, state_size, action_size, seed):
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
        # Initialize Q and Fixed Q networks
        self.q_network = QNetwork(state_size, action_size, seed).to(device)
        self.fixed_network = QNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.q_network.parameters())
        # Initiliase memory 
        self.memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE, seed)
        self.timestep = 0
        
    
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
            if len(self.memory) > BATCH_SIZE:
                sampled_experiences = self.memory.sample()
                self.learn(sampled_experiences)
        
    def learn(self, experiences):
        """
        Learn from experience by training the q_network 
        
        Parameters
        ----------
        experiences (array_like): List of experiences sampled from agent's memory
        """
        states, actions, rewards, next_states, dones = experiences
        # Get the action with max Q value
        action_values = self.fixed_network(next_states).detach()
        # Notes
        # tensor.max(1)[0] returns the values, tensor.max(1)[1] will return indices
        # unsqueeze operation --> np.reshape
        # Here, we make it from torch.Size([64]) -> torch.Size([64, 1])
        max_action_values = action_values.max(1)[0].unsqueeze(1)
        
        # If done just use reward, else update Q_target with discounted action values
        Q_target = rewards + (GAMMA * max_action_values * (1 - dones))
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
        
        
    def act(self, state, eps=0.0):
        """
        Choose the action
        
        Parameters
        ----------
        state (array_like): current state of environment
        eps (float): epsilon for epsilon-greedy action selection
        """
        rnd = random.random()
        if rnd < eps:
            return np.random.randint(self.action_size)
        else:
            state = torch.from_numpy(state).float().unsqueeze(0).to(device)
            # set the network into evaluation mode 
            self.q_network.eval()
            with torch.no_grad():
                action_values = self.q_network(state)
            # Back to training mode
            self.q_network.train()
            action = np.argmax(action_values.cpu().data.numpy())
            return action    
        
    def checkpoint(self, filename):
        torch.save(self.q_network.state_dict(), filename)