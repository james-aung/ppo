import os
import numpy as np
import torch as T
import touch.nn as nn
import touch.optim as optim
from torch.distributions.categorical import Categorical

class PPOMemory:
    def __init__(self, batch_size):
        self.states = []
        self.probs = [] #log probs
        self.vals = [] #state values
        self.actions = []
        self.rewards = []
        self.dones = []

        self.batch_size = batch_size

    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size) # np.arrange creates an array of evenly spaced values, thus we are creating an array of batch start indices
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices) # shuffle the indices for stochastic gradient ascent
        batches = [indices[i:i+self.batch_size] for i in batch_start] # for each batch start index, we create a batch of indices

        return np.array(self.states), np.array(self.actions), np.array(self.probs), np.array(self.vals), np.array(self.rewards), np.array(self.dones), batches
    
    def store_memory(self, state, action, probs, vals, reward, done):
        self.states.append(state)
        self.probs.append(probs)
        self.vals.append(vals)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear_memory(self):
        # clear the memory after each trajectory
        self.states = []
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []

