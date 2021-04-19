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
from common.agent import Agent
from common.evaluator import evaluate_agent
from common.wrappers import BinaryWrapper
from torch.nn.parameter import Parameter

parser = argparse.ArgumentParser()
parser.add_argument("--n_eval_episodes", default=20, type=int)
parser.add_argument("--path", default='logs/your_exp_name/', type=str)
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
        self.n_qubits = 4
        self.n_actions = 4
        self.q_layers = get_model(n_qubits=self.n_qubits,
                                  n_layers=n_layers,
                                  data_reupload=False)

    def forward(self, inputs):
        inputs = inputs * np.pi
        outputs = self.q_layers(inputs)
        outputs = (1 + outputs) / 2
        return outputs


def main():

    # Environment
    env_name = 'FrozenLake-v0'
    env = gym.make(env_name)
    env = BinaryWrapper(env)

    # Network
    with open(args.path + 'config.yaml', 'r') as f:
        hparams = yaml.safe_load(f)

    net = QuantumNet(hparams['n_layers'])
    state_dict = torch.load(args.path + 'episode_final.pt')
    net.load_state_dict(state_dict)

    # Agent
    agent = Agent(net)

    # Evaluation
    result = evaluate_agent(env, agent, args.n_eval_episodes)
    print('Average reward : {}'.format(result['reward']))


if __name__ == '__main__':
    main()
