from stable_baselines import DQN

from stable_baselines import PPO2


model = PPO2.load("ppo2_cartpole_50000step")


weight1=model.get_parameters()["model/pi_fc0/w:0"]
bias1=model.get_parameters()["model/pi_fc0/b:0"]
weight2=model.get_parameters()["model/pi_fc1/w:0"]
bias2=model.get_parameters()["model/pi_fc1/b:0"]
weight3=model.get_parameters()["model/pi/w:0"]
bias3=model.get_parameters()["model/pi/b:0"]


# weight1=model.get_parameters()["deepq/model/action_value/fully_connected/weights:0"]
# bias1=model.get_parameters()["deepq/model/action_value/fully_connected/biases:0"]
# weight2=model.get_parameters()["deepq/model/action_value/fully_connected_1/weights:0"]
# bias2=model.get_parameters()["deepq/model/action_value/fully_connected_1/biases:0"]
# weight3=model.get_parameters()["deepq/model/action_value/fully_connected_2/weights:0"]
# bias3=model.get_parameters()["deepq/model/action_value/fully_connected_2/biases:0"]

# print(weight1.T)
# print(bias1)
# print(weight2)
# print(bias2)
# print(weight3)
# print(bias3)

def get_weights(net):
    """ Extract parameters from net, and return a list of tensors"""
    return [p.data for p in net.parameters()]

import torch
import torch.nn as nn
import torch.nn.functional as F

class PPO_net(nn.Module):
    def __init__(self):
        super(PPO_net, self).__init__()
        self.fc1 = nn.Linear(4, 10)
        self.fc2 = nn.Linear(10, 10)
        self.out = nn.Linear(10, 2)
        self.fc1.weight = torch.nn.Parameter(torch.tensor(weight1.T))
        self.fc1.bias = torch.nn.Parameter(torch.tensor(bias1))
        self.fc2.weight = torch.nn.Parameter(torch.tensor(weight2.T))
        self.fc2.bias = torch.nn.Parameter(torch.tensor(bias2))
        self.out.weight = torch.nn.Parameter(torch.tensor(weight3.T))
        self.out.bias = torch.nn.Parameter(torch.tensor(bias3))

net=PPO_net()

print(get_weights(net))

torch.save(net, 'ppo_net_50000step')