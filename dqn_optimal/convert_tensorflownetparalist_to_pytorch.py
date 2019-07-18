import json
import numpy as np
import torch

with open('dqn_optimal_parameter_150epi','r') as fp:
    dict=json.load(fp)

weight1=np.array(dict['layer1_weight'],dtype=np.float32)
bias1=np.array(dict['layer1_bias'],dtype=np.float32)
weight2=np.array(dict['layer2_weight'],dtype=np.float32)
bias2=np.array(dict['layer2_bias'],dtype=np.float32)
weight3=np.array(dict['layer3_weight'],dtype=np.float32)
bias3=np.array(dict['layer3_bias'],dtype=np.float32)

#
#
# print(weight1)
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

class DQN_net(nn.Module):
    def __init__(self):
        super(DQN_net, self).__init__()
        self.fc1 = nn.Linear(4, 32)
        self.fc2 = nn.Linear(32, 64)
        self.out = nn.Linear(64, 2)
        self.fc1.weight = torch.nn.Parameter(torch.tensor(weight1.T).double())
        self.fc1.bias = torch.nn.Parameter(torch.tensor(bias1).double())
        self.fc2.weight = torch.nn.Parameter(torch.tensor(weight2.T).double())
        self.fc2.bias = torch.nn.Parameter(torch.tensor(bias2).double())
        self.out.weight = torch.nn.Parameter(torch.tensor(weight3.T).double())
        self.out.bias = torch.nn.Parameter(torch.tensor(bias3).double())

net=DQN_net()
net=net.float()

print(get_weights(net))

torch.save(net, 'float32_dqn_net_150epi')