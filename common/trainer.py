import os
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm.autonotebook import tqdm

from .agent import Agent
from .memory import ReplayMemory


class Trainer:
    def __init__(self,
                 env,
                 net,
                 target_net,
                 gamma,
                 learning_rate,
                 batch_size,
                 exploration_initial_eps,
                 exploration_decay,
                 exploration_final_eps,
                 train_freq,
                 target_update_interval,
                 buffer_size,
                 learning_rate_input=None,
                 learning_rate_output=None,
                 loss_func='MSE',
                 optim_class='RMSprop',
                 device='auto',
                 logging=False):

        assert loss_func in ['MSE', 'L1', 'SmoothL1'
                             ], "Supported losses : ['MSE', 'L1', 'SmoothL1']"
        assert optim_class in [
            'SGD', 'RMSprop', 'Adam', 'Adagrad', 'Adadelta'
        ], "Supported optimizers : ['SGD', 'RMSprop', 'Adam', 'Adagrad', 'Adadelta']"
        assert device in ['auto', 'cpu', 'cuda:0'
                          ], "Supported devices : ['auto', 'cpu', 'cuda:0']"

        self.env = env
        self.net = net
        self.target_net = target_net
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.exploration_initial_eps = exploration_initial_eps
        self.exploration_decay = exploration_decay
        self.exploration_final_eps = exploration_final_eps
        self.train_freq = train_freq
        self.target_update_interval = target_update_interval
        self.buffer_size = buffer_size
        self.learning_rate_input = learning_rate_input
        self.learning_rate_output = learning_rate_output
        self.loss_func = loss_func
        self.optim_class = optim_class
        self.device = device
        self.logging = logging

        self.build()
        self.reset()

    def build(self):

        # set networks
        if self.device == "auto":
            self.device = torch.device(
                "cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(self.device)
        self.net = self.net.to(self.device)
        self.target_net = self.target_net.to(self.device)

        # set loss
        self.loss_func = getattr(nn, self.loss_func + 'Loss')()

        # set optimizer
        optim_class = getattr(optim, self.optim_class)
        params = []
        params.append({'params': self.net.q_layers.parameters()})
        if hasattr(self.net, 'w_input') and self.net.w_input is not None:
            lr_input = self.learning_rate_input if self.learning_rate_input is not None else self.learning_rate
            params.append({'params': self.net.w_input, 'lr': lr_input})
        if hasattr(self.net, 'w_output') and self.net.w_output is not None:
            lr_output = self.learning_rate_output if self.learning_rate_output is not None else self.learning_rate
            params.append({'params': self.net.w_output, 'lr': lr_output})
        self.opt = optim_class(params, lr=self.learning_rate)

        # set agent
        self.agent = Agent(self.net, self.env.action_space,
                           self.exploration_initial_eps,
                           self.exploration_decay, self.exploration_final_eps)

        # set memory
        self.memory = ReplayMemory(self.buffer_size)

        # set loggers

        if self.logging:
            exp_name = datetime.now().strftime("DQN-%d_%m_%Y-%H_%M_%S")
            if not os.path.exists('./logs/'):
                os.makedirs('./logs/')
            self.log_dir = './logs/{}/'.format(exp_name)
            os.makedirs(self.log_dir)
            self.writer = SummaryWriter(log_dir=self.log_dir)

    def reset(self):
        self.global_step = 0
        self.episode_count = 0
        self.env.seed(123)
        self.n_actions = self.env.action_space.n
        state = self.env.reset()
        while len(self.memory) < self.buffer_size:
            action = self.agent.get_random_action()
            next_state, reward, done, _ = self.env.step(action)
            self.memory.push(state, action, reward, done, next_state)
            if done:
                state = self.env.reset()
            else:
                state = next_state

    def update_net(self):

        self.net.train()
        self.opt.zero_grad()

        # sample transitions
        states, actions, rewards, dones, next_states = self.memory.sample(
            self.batch_size, self.device)

        # compute q-values
        state_action_values = self.net(states)
        state_action_values = state_action_values.gather(
            1, actions.unsqueeze(-1)).squeeze(-1)

        # compute target q-values
        with torch.no_grad():
            next_state_values = self.target_net(next_states)
            next_state_values = next_state_values.max(1)[0].detach()

        expected_state_action_values = (1 - dones) * next_state_values.to(
            self.device) * self.gamma + rewards

        # compute loss
        loss = self.loss_func(state_action_values,
                              expected_state_action_values)
        loss.backward()
        self.opt.step()

        return loss.item()

    def update_target_net(self):
        self.target_net.load_state_dict(self.net.state_dict())

    def train_step(self):
        episode_epsilon = self.agent.update_epsilon(self.episode_count)
        episode_steps = 0
        episode_reward = 0
        episode_loss = []
        state = self.env.reset()
        done = False

        while not done:

            # take action
            action = self.agent(state, self.device)
            next_state, reward, done, _ = self.env.step(action)

            # update memory
            self.memory.push(state, action, reward, done, next_state)

            # update state
            state = next_state

            # optimize net
            if self.global_step % self.train_freq == 0:
                loss = self.update_net()
                episode_loss.append(loss)

            # update target net
            if self.global_step % self.target_update_interval == 0:
                self.update_target_net()

            self.global_step += 1
            episode_reward += reward
            episode_steps += 1

        self.episode_count += 1
        if len(episode_loss) > 0:
            episode_loss = np.mean(episode_loss)
        else:
            episode_loss = 0.
        return {
            'steps': episode_steps,
            'loss': episode_loss,
            'reward': episode_reward,
            'epsilon': episode_epsilon
        }

    def test_step(self, n_eval_episodes):
        episode_steps = []
        episode_reward = []

        for _ in range(n_eval_episodes):
            state = self.env.reset()
            done = False
            episode_steps.append(0)
            episode_reward.append(0)
            while not done:
                action = self.agent.get_action(state, self.device)
                next_state, reward, done, _ = self.env.step(action)
                state = next_state
                episode_steps[-1] += 1
                episode_reward[-1] += reward

        episode_steps = np.mean(episode_steps)
        episode_reward = np.mean(episode_reward)
        return {'steps': episode_steps, 'reward': episode_reward}

    def learn(self,
              total_episodes,
              n_eval_episodes=5,
              log_train_freq=-1,
              log_eval_freq=-1,
              log_ckp_freq=-1):

        # Stats
        postfix_stats = {}
        with tqdm(range(total_episodes), desc="DQN",
                  unit="episode") as tepisodes:

            for t in tepisodes:

                # train dqn
                train_stats = self.train_step()

                # update train stats
                postfix_stats['train/reward'] = train_stats['reward']
                postfix_stats['train/steps'] = train_stats['steps']

                if t % log_eval_freq == 0:

                    # test dqn
                    test_stats = self.test_step(n_eval_episodes)

                    # update test stats
                    postfix_stats['test/reward'] = test_stats['reward']
                    postfix_stats['test/steps'] = test_stats['steps']

                if self.logging and (t % log_train_freq == 0):
                    for key, item in train_stats.items():
                        self.writer.add_scalar('train/' + key, item, t)

                if self.logging and (t % log_eval_freq == 0):
                    for key, item in test_stats.items():
                        self.writer.add_scalar('test/' + key, item, t)

                if self.logging and (t % log_ckp_freq == 0):
                    torch.save(self.net.state_dict(),
                               self.log_dir + 'episode_{}.pt'.format(t))

                # update progress bar
                tepisodes.set_postfix(postfix_stats)

            if self.logging and (log_ckp_freq > 0):
                torch.save(self.net.state_dict(),
                           self.log_dir + 'episode_final.pt')


def test_step(env, agent, n_eval_episodes):
    episode_steps = []
    episode_reward = []

    for _ in range(n_eval_episodes):
        state = env.reset()
        done = False
        episode_steps.append(0)
        episode_reward.append(0)
        while not done:
            action = agent.get_action(state, device)
            next_state, reward, done, _ = env.step(action)
            state = next_state
            episode_steps[-1] += 1
            episode_reward[-1] += reward

    episode_steps = np.mean(episode_steps)
    episode_reward = np.mean(episode_reward)
    return {'steps': episode_steps, 'reward': episode_reward}
