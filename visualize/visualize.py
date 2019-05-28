import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import copy
import h5py
import time
import socket
import os
import sys
import numpy as np
import torchvision
from matplotlib import pyplot as plt
import time

import mountCar_playground

class Net(nn.Module):
    def __init__(self, ):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(2, 50)
        self.fc1.weight.data.normal_(0, 0.1)   # initialization
        self.out = nn.Linear(50, 3)
        self.out.weight.data.normal_(0, 0.1)   # initialization

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        actions_value = self.out(x)
        return actions_value

def get_weights(net):
    """ Extract parameters from net, and return a list of tensors"""
    return [p.data for p in net.parameters()]

def generate_random_dir(weights):
    """
        Produce a random direction that is a list of random Gaussian tensors
        with the same shape as the network's weights, so one direction entry per weight.
    """
    return [torch.randn(w.size()) for w in weights]

def normalize_directions_for_weights(direction, weights, norm='filter', ignore='biasbn'):
    """
        The normalization scales the direction entries according to the entries of weights.
    """
    assert(len(direction) == len(weights))
    for d, w in zip(direction, weights):
        if d.dim() <= 1:
            if ignore == 'biasbn':
                d.fill_(0) # ignore directions for weights with 1 dimension
            else:
                d.copy_(w) # keep directions for weights/bias that are only 1 per node
        else:
            normalize_direction(d, w, norm)

def normalize_direction(direction, weights, norm='filter'):
    """
        Rescale the direction so that it has similar norm as their corresponding
        model in different levels.

        Args:
          direction: a variables of the random direction for one layer
          weights: a variable of the original model for one layer
          norm: normalization method, 'filter' | 'layer' | 'weight'
    """
    if norm == 'filter':
        # Rescale the filters (weights in group) in 'direction' so that each
        # filter has the same norm as its corresponding filter in 'weights'.
        for d, w in zip(direction, weights):
            d.mul_(w.norm()/(d.norm() + 1e-10))


def plot_1D(x, value, env_name, algo_name):
    print('------------------------------------------------------------------')
    print('plot_1d_value_function')
    print('------------------------------------------------------------------')
    plt.figure()
    plt.plot(x, value, 'b-', label='Value', linewidth=1)
    plt.ylabel('Value', fontsize='xx-large')
    plt.xlim(min(x), max(x))
    plt.ylim(min(value), max(value))
    plt.savefig(env_name + '_' + algo_name + '_1d_value_func' + str(min(x)) + '-' + str(max(x)) + '.pdf',
               dpi=300, bbox_inches='tight', format='pdf')


def plot_2D_contour(x, y, value, env_name, algo_name):
    print('------------------------------------------------------------------')
    print('plot_2d_contour')
    print('------------------------------------------------------------------')
    fig = plt.figure()
    CS = plt.contour(x, y, value, cmap='summer', levels=np.arange(value.min(), value.max(), 15))  # can modify here
    plt.clabel(CS, inline=1, fontsize=8)
    fig.savefig(env_name + '_' + algo_name + '_2dcontour' + '.pdf', dpi=300,
                bbox_inches='tight', format='pdf')


if __name__=="__main__":
    algo_name ="dqn"
    env_name="mountainCar"
    model_name=env_name+"_"+algo_name+"_v2"
    model = Net()
    model = torch.load(model_name)
    weights = get_weights(model)
    print(weights)
    num_of_dir = 1    # can modify here

    value_output_file = env_name + '_' + algo_name + 'value'



    if num_of_dir ==2:
        dir_x = generate_random_dir(weights)
        dir_y = generate_random_dir(weights)
        normalize_directions_for_weights(dir_x, weights, norm='filter')
        normalize_directions_for_weights(dir_y, weights, norm='filter')
        print(dir_x, dir_y)
    else:
        dir = generate_random_dir(weights)
        normalize_directions_for_weights(dir, weights, norm='filter')
    # print(dir)
    # print(dir)



    # start calculating value function under different theta


    if num_of_dir ==1:
        x = list(np.linspace(-0.05,0.05,11))        # can modify here
        print(x)
        value = []

        for step in x:
            changes = [d*step for d in dir]

            for (p, w, d) in zip(model.parameters(), weights, changes):
                p.data = w + torch.Tensor(d).type(type(w))

            t_s=time.time()
            model_value=mountCar_playground.playground(model, 10, False)  # can modify here
            value.append(model_value)
            print(model_value)
            print('time: ',time.time()-t_s)

        print(value)

    else:
        x = list(np.linspace(-0.05,0.05,11))  # can modify here
        y = list(np.linspace(-0.05,0.05,11))  # can modify here
        value = np.zeros(shape=(len(x), len(y)))
        for i in range(len(x)):
            for j in range(len(y)):
                print(i,j)
                changes = [dx * x[i] + dy * y[i] for dx,dy in zip(dir_x,dir_y)]
                for (p, w, d) in zip(model.parameters(), weights, changes):
                    p.data = w + torch.Tensor(d).type(type(w))


                model_value = mountCar_playground.playground(model, 20, True)  # can modify here
                value[i][j]=model_value

        print(value)



    if num_of_dir == 2:
        plot_2D_contour(x, y, value, env_name, algo_name)
    else:
        plot_1D(x,value,env_name,algo_name)


    import json

    with open(value_output_file,'w') as fp:
        json.dump(x, fp)
        if num_of_dir == 2:
            json.dump(y, fp)
        json.dump(value, fp)
