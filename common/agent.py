# Based on : https://github.com/djbyrne/core_rl/blob/master/algos/dqn/model.py

import numpy as np
import torch


class Agent:
    def __init__(self,
                 net,
                 action_space=None,
                 exploration_initial_eps=None,
                 exploration_decay=None,
                 exploration_final_eps=None):

        self.net = net
        self.action_space = action_space
        self.exploration_initial_eps = exploration_initial_eps
        self.exploration_decay = exploration_decay
        self.exploration_final_eps = exploration_final_eps
        self.epsilon = 0.

    def __call__(self, state, device=torch.device('cpu')):
        if np.random.random() < self.epsilon:
            action = self.get_random_action()
        else:
            action = self.get_action(state, device)

        return action

    def get_random_action(self):
        action = self.action_space.sample()
        return action

    def get_action(self, state, device=torch.device('cpu')):
        if not isinstance(state, torch.Tensor):
            state = torch.tensor([state])

        if device.type != 'cpu':
            state = state.cuda(device)

        q_values = self.net.eval()(state)
        _, action = torch.max(q_values, dim=1)
        return int(action.item())

    def update_epsilon(self, step):
        self.epsilon = max(
            self.exploration_final_eps, self.exploration_final_eps +
            (self.exploration_initial_eps - self.exploration_final_eps) *
            self.exploration_decay**step)
        return self.epsilon
