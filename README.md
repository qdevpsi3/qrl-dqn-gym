<h1 align="center" style="margin-top: 0px;"> <b>Quantum agents in the Gym: a variational quantum algorithm for deep Q-learning</b></h1>
<div align="center" >

[![paper](https://img.shields.io/static/v1.svg?label=Paper&message=arXiv:2103.15084&color=b31b1b)](https://arxiv.org/abs/2103.15084)
[![framework](https://img.shields.io/static/v1.svg?label=Framework&message=PyTorch&color=ee4c2d)](https://pytorch.org)
[![packages](https://img.shields.io/static/v1.svg?label=Made%20with&message=PennyLane&color=649ea1)](https://pennylane.ai)
[![license](https://img.shields.io/static/v1.svg?label=License&message=GPL%20v3.0&color=green)](https://www.gnu.org/licenses/gpl-3.0.html)
[![exp](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/qdevpsi3/qrl-dqn-gym/blob/main/cart_pole/colab_notebook.ipynb)
</div>

## **Description**
This repository contains an implementation of the <ins>Quantum Deep Q-learning algorithm</ins> and its application to the  <ins>FrozenLake</ins> and <ins>CartPole</ins> environments as in :

- Paper : **Quantum agents in the Gym: a variational quantum algorithm for deep Q-learning**
- Authors : **Skolik, Jerbi and Dunjko**
- Date : **2021**

## **Hyperparameters**
| Hyperparameters | Frozen-Lake | Cart-Pole | Explanation |
|:-:|:-:|:-:|-|
| n_layers | 5,10,15 | 5 | number of layers |
| gamma | 0.8 | 0.99 | discount factor for Q-learning |
| w_input |  | True, False | train weights on the model input |
| w_output |  | True, False | train weights on the model output |
| lr | 0.001 | 0.001 | model parameter learning rate |
| lr_input |  | 0.001 | input weight learning rate |
| lr_output |  | 0.1 | output weight learning rate |
| batch_size | 11 | 16 | number of samples shown to optimizer at each update |
| eps_init | 1. | 1. | initial value for ε-greedy policy |
| eps_decay | 0.99 | 0.99 | decay of ε for ε -greedy policy |
| eps_min | 0.01 | 0.01 | minimal value of ε for ε-greedy policy |
| train_freq | 5 | 10 | steps in episode after which model is updated |
| target_freq | 10 | 30 | steps in episode after which target is updated |
| memory | 10000 | 10000 | size of memory for experience replay |
| data_reupload |  | True, False | use data re-uploading |
| loss | SmoothL1 | SmoothL1 | loss type : *MSE*, *L1* or *SmoothL1*  |
| optimizer | RMSprop | RMSprop | optimizer type : *SGD*, *RMSprop*, *Adam*, ...  |
| total_episodes | 3500 | 5000 | total training episodes |
| n_eval_episodes | 5 | 5 | number episodes to evaluate the agent |


## **Experiments**
The <ins>experiments</ins> in the paper are reproduced using *PyTorch* for optimization, *PennyLane* for quantum circuits and *Gym* for the environments. 
### **Training**
- Option 1 : Open in [Colab](https://colab.research.google.com/github/qdevpsi3/qrl-dqn-gym/blob/main/cart_pole/colab_notebook.ipynb). You can activate the <ins>GPU</ins> in *Notebook Settings*.
- Option 2 : Run on local machine. First, you need to install :
```
$ pip install gym torch torchvision pennylane tensorboard
```
You can run an experiment using the following command :
```
$ cd cart_pole/
$ python train.py 
```
You can set your own hyperparameters : 
```
$ cd cart_pole/
$ python train.py --batch_size=32
```
The list of hyperparameters is given above and accessible via : 
```
$ cd cart_pole/
$ python train.py --help
```
To monitor the training process using tensorboard : 
```
$ cd cart_pole/
$ python train.py
$ tensorboard --logdir logs/
```
The hyperparameters, checkpoints, training and evaluation metrics are saved in the *logs/* folder.
### **Testing**
You can test your agent by passing the path to your logged model. 
```
$ cd cart_pole/
$ python test.py --path=logs/exp_name/ --n_eval_episodes=10
```
<ins>Trained agents</ins> are also provided in the logs folder. 
```
$ cd cart_pole/
$ python test.py --path=logs/input_only/ --n_eval_episodes=10
```
## **Results**
### ***Cart-Pole*** 
The circuit output is multiplied by 90 if no output weight is available.
| Setting | Average Reward | Hyperparameters and Checkpoints
|:-:|:-:|:-:|
| No Weights | 181  |cart_pole/logs/no_weights/|
| Input Weights | 200  |cart_pole/logs/input_only/|
| Output Weights | 101 |cart_pole/logs/output_only/|
| Input and Output Weights | 199 |cart_pole/logs/input_output/|