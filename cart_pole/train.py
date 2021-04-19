import argparse
import sys

# add common dqn utils
sys.path.append('..')

import gym
import numpy as np
import pennylane as qml
import torch
import torch.nn as nn
import yaml
from common.trainer import Trainer
from torch.nn.parameter import Parameter

parser = argparse.ArgumentParser()

parser.add_argument("--n_layers", default=5, type=int)
parser.add_argument("--gamma", default=0.99, type=float)
parser.add_argument("--w_input", default=True, type=bool)
parser.add_argument("--w_output", default=False, type=bool)
parser.add_argument("--lr", default=0.001, type=float)
parser.add_argument("--lr_input", default=0.001, type=float)
parser.add_argument("--lr_output", default=0.1, type=float)
parser.add_argument("--batch_size", default=2, type=int)
parser.add_argument("--eps_init", default=1., type=float)
parser.add_argument("--eps_decay", default=0.99, type=int)
parser.add_argument("--eps_min", default=0.01, type=float)
parser.add_argument("--train_freq", default=10, type=int)
parser.add_argument("--target_freq", default=30, type=int)
parser.add_argument("--memory", default=10, type=int)
parser.add_argument("--data_reupload", default=True, type=bool)
parser.add_argument("--loss", default='SmoothL1', type=str)
parser.add_argument("--optimizer", default='RMSprop', type=str)
parser.add_argument("--total_episodes", default=10, type=int)
parser.add_argument("--n_eval_episodes", default=5, type=int)
parser.add_argument("--logging", default=True, type=bool)
parser.add_argument("--log_train_freq", default=1, type=int)
parser.add_argument("--log_eval_freq", default=1, type=int)
parser.add_argument("--log_ckp_freq", default=1, type=int)
parser.add_argument("--device", default='auto', type=str)
args = parser.parse_args()


def encode(n_qubits, inputs):
    for wire in range(n_qubits):
        qml.RX(inputs[wire], wires=wire)


def layer(n_qubits, y_weight, z_weight):
    for wire, y_weight in enumerate(y_weight):
        qml.RY(y_weight, wires=wire)
    for wire, z_weight in enumerate(z_weight):
        qml.RZ(z_weight, wires=wire)
    for wire in range(n_qubits):
        qml.CZ(wires=[wire, (wire + 1) % n_qubits])


def measure(n_qubits):
    return [
        qml.expval(qml.PauliZ(0) @ qml.PauliZ(1)),
        qml.expval(qml.PauliZ(2) @ qml.PauliZ(3))
    ]


def get_model(n_qubits, n_layers, data_reupload):
    dev = qml.device("default.qubit", wires=n_qubits)
    shapes = {
        "y_weights": (n_layers, n_qubits),
        "z_weights": (n_layers, n_qubits)
    }

    @qml.qnode(dev, interface='torch')
    def circuit(inputs, y_weights, z_weights):
        for layer_idx in range(n_layers):
            if (layer_idx == 0) or data_reupload:
                encode(n_qubits, inputs)
            layer(n_qubits, y_weights[layer_idx], z_weights[layer_idx])
        return measure(n_qubits)

    model = qml.qnn.TorchLayer(circuit, shapes)

    return model


class QuantumNet(nn.Module):
    def __init__(self, n_layers, w_input, w_output, data_reupload):
        super(QuantumNet, self).__init__()
        self.n_qubits = 4
        self.n_actions = 2
        self.data_reupload = data_reupload
        self.q_layers = get_model(n_qubits=self.n_qubits,
                                  n_layers=n_layers,
                                  data_reupload=data_reupload)
        if w_input:
            self.w_input = Parameter(torch.Tensor(self.n_qubits))
            nn.init.normal_(self.w_input)
        else:
            self.register_parameter('w_input', None)
        if w_output:
            self.w_output = Parameter(torch.Tensor(self.n_actions))
            nn.init.normal_(self.w_output, mean=90.)
        else:
            self.register_parameter('w_output', None)

    def forward(self, inputs):
        if self.w_input is not None:
            inputs = inputs * self.w_input
        inputs = torch.atan(inputs)
        outputs = self.q_layers(inputs)
        outputs = (1 + outputs) / 2
        if self.w_output is not None:
            outputs = outputs * self.w_output
        else:
            outputs = 90 * outputs
        return outputs


def main():

    # Environment
    env_name = 'CartPole-v0'
    env = gym.make(env_name)

    # Networks
    net = QuantumNet(args.n_layers, args.w_input, args.w_output,
                     args.data_reupload)
    target_net = QuantumNet(args.n_layers, args.w_input, args.w_output,
                            args.data_reupload)

    # Trainer
    trainer = Trainer(env,
                      net,
                      target_net,
                      gamma=args.gamma,
                      learning_rate=args.lr,
                      batch_size=args.batch_size,
                      exploration_initial_eps=args.eps_init,
                      exploration_decay=args.eps_decay,
                      exploration_final_eps=args.eps_min,
                      train_freq=args.train_freq,
                      target_update_interval=args.target_freq,
                      buffer_size=args.memory,
                      learning_rate_input=args.lr_input,
                      learning_rate_output=args.lr_output,
                      loss_func=args.loss,
                      optim_class=args.optimizer,
                      device=args.device,
                      logging=args.logging)

    if args.logging:
        with open(trainer.log_dir + 'config.yaml', 'w') as f:
            yaml.safe_dump(args.__dict__, f, indent=2)

    trainer.learn(args.total_episodes,
                  n_eval_episodes=args.n_eval_episodes,
                  log_train_freq=args.log_train_freq,
                  log_eval_freq=args.log_eval_freq,
                  log_ckp_freq=args.log_ckp_freq)


if __name__ == '__main__':
    main()
