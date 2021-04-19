# Based on : https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
from collections import namedtuple

import numpy as np
import torch

Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'done', 'next_state'))


class ReplayMemory(object):
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.buffer_size:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.buffer_size

    def sample(self, batch_size, device):
        indices = np.random.choice(len(self), batch_size, replace=False)
        states, actions, rewards, dones, next_states = zip(
            *[self.memory[idx] for idx in indices])
        states = torch.from_numpy(np.array(states)).to(device)
        actions = torch.from_numpy(np.array(actions)).to(device)
        rewards = torch.from_numpy(np.array(rewards,
                                            dtype=np.float32)).to(device)
        dones = torch.from_numpy(np.array(dones, dtype=np.int32)).to(device)
        next_states = torch.from_numpy(np.array(next_states)).to(device)
        return states, actions, rewards, dones, next_states

    def __len__(self):
        return len(self.memory)
