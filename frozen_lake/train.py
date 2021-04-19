import argparse

import gym
import numpy as np
import pennylane as qml
import torch
import torch.nn as nn
from common.trainer import Trainer
from common.wrappers import BinaryWrapper
from torch.nn.parameter import Parameter
from tqdm import tqdm

parser = argparse.ArgumentParser()

parser.add_argument("--n_layers", default=5, type=int)
parser.add_argument("--gamma", default=0.8, type=float)
parser.add_argument("--lr", default=0.001, type=float)
parser.add_argument("--batch_size", default=11, type=int)
parser.add_argument("--eps_init", default=1., type=float)
parser.add_argument("--eps_decay", default=0.99, type=int)
parser.add_argument("--eps_min", default=0.01, type=float)
parser.add_argument("--train_freq", default=5, type=int)
parser.add_argument("--target_freq", default=10, type=int)
parser.add_argument("--memory", default=10000, type=int)
parser.add_argument("--loss", default='SmoothL1Loss', type=str)
parser.add_argument("--optimizer", default='RMSprop', type=str)
parser.add_argument("--total_episodes", default=3500, type=int)
parser.add_argument("--n_eval_episodes", default=5, type=int)
parser.add_argument("--log_dir", default='./logs/frozen_lake/', type=str)
parser.add_argument("--log_train_freq", default=1, type=int)
parser.add_argument("--log_eval_freq", default=10, type=int)
parser.add_argument("--log_ckp_freq", default=100, type=int)
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
    return [qml.expval(qml.PauliZ(wire)) for wire in range(n_qubits)]


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
    def __init__(self, n_layers):
        super(QuantumNet, self).__init__()
        self.q_layers = get_model(n_qubits=4,
                                  n_layers=n_layers,
                                  data_reupload=False)

    def forward(self, inputs):
        inputs = np.pi * inputs
        outputs = self.q_layers(inputs)
        outputs = (1 + outputs) / 2
        return outputs


def main():
    # Environment
    env_name = 'FrozenLake-v0'
    env = gym.make(env_name)
    env = BinaryWrapper(env)

    # Networks
    net = QuantumNet(args.n_layers)
    target_net = QuantumNet(args.n_layers)

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
                      loss_func=args.loss,
                      optim_class=args.optimizer,
                      device=args.device,
                      tensorboard_log=args.log_dir)

    trainer.learn(args.total_episodes,
                  n_eval_episodes=args.n_eval_episodes,
                  log_train_freq=args.log_train_freq,
                  log_eval_freq=args.log_eval_freq,
                  log_ckp_freq=args.log_ckp_freq)


if __name__ == '__main__':
    main()
