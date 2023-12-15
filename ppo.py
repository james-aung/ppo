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

class ActorNetwork(nn.Module): # The nn.Module class is the base class for all neural network modules in PyTorch
    def __init__(self, n_actions, input_dims, alpha, fc1_dims=256, fc2_dims=256, chkpt_dir='tmp/ppo'): # checkpoint directory is where we will save the model
        super(ActorNetwork, self).__init__()

        self.checkpoint_file = os.path.join(chkpt_dir, 'actor_torch_ppo')
        self.actor = nn.Sequential(
            nn.Linear(*input_dims, fc1_dims), # TODO Why do we unpack input_dims? How many dimensions does input_dims have?
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims, n_actions),
            nn.Softmax(dim=-1) #dim=-1 means we are applying softmax across the action dimension and not the batch dimension
        )

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        distribution = self.actor(state)
        distribution = Categorical(distribution)
        return distribution
    
    def save_checkpoint(self):
        # save the model parameters
        T.save(self.state_dict(), self.checkpoint_file) # state_dict() returns a dictionary containing a whole state of the module

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))

