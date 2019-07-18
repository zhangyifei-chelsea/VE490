import torch
import torch.nn as nn
import torch.nn.functional as F
import h5py
import time
import socket
import os
import sys
import numpy as np
import torchvision
from matplotlib import pyplot as plt
import time


import PPO.Cartpole_playground_policy_net as Cartpole_playground

class PPO_net(nn.Module):   # can modify the classname to fit in the pytorch pkl name
    def __init__(self, ):
        # cartpole policy net#
        super(PPO_net, self).__init__()
        self.fc1 = nn.Linear(4, 32)
        self.fc1.weight.data.normal_(0, 0.1)   # initialization
        self.fc2 = nn.Linear(32, 64)
        self.fc2.weight.data.normal_(0, 0.1)  # initialization
        # self.fc3 = nn.Linear(20, 20)
        # self.fc3.weight.data.normal_(0, 0.1)  # initialization
        self.out = nn.Linear(64, 2)
        self.out.weight.data.normal_(0, 0.1)   # initialization

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        # residue = x
        x = self.fc2(x)
        x = F.relu(x)
        # x = self.fc3(x)
        # x += residue
        # x = F.relu(x)
        x = self.out(x)
        actions_prob = F.softmax(x)
        return actions_prob


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

def angle_between_two_dirctions(dir_x,dir_y):
    prod=0
    norm_x=0
    norm_y=0
    for x,y in zip(dir_x,dir_y):
        prod+=torch.sum(x*y)
    for x in dir_x:
        norm_x+=torch.sum(x*x)

    for y in dir_y:
        norm_y += torch.sum(y * y)

    angle=prod/torch.sqrt(norm_x)/torch.sqrt(norm_y)
    return angle


def plot_1D(x, value, model_name):
    print('------------------------------------------------------------------')
    print('plot_1d_value_function')
    print('------------------------------------------------------------------')
    plt.figure()
    plt.plot(x, value, 'b-', label='Value', linewidth=1)
    plt.ylabel('Value', fontsize='xx-large')
    plt.xlim(min(x), max(x))
    plt.ylim(min(value), max(value))
    plt.savefig(model_name + '_1d_value_func' + str(min(x)) + '-' + str(max(x)) + '.pdf',
               dpi=300, bbox_inches='tight', format='pdf')    # can modify here


def plot_2D_contour(x, y, value, model_name):
    print('------------------------------------------------------------------')
    print('plot_2d_contour')
    print('------------------------------------------------------------------')
    fig = plt.figure()

    level=np.arange(0,500,50)   # can modify here
    level= [10,20,30,40,50,60,70, 80,100,120,150,199]
    CS = plt.contour(x, y, value, cmap='summer', levels=level)  # can modify here
    plt.xlim(min(x), max(x))
    plt.ylim(min(y), max(y))
    plt.clabel(CS, inline=1, fontsize=8)
    fig.savefig(model_name + '_2dcontour' + str(value.min())+ '_' + str(value.max()) + '.pdf', dpi=300,
                bbox_inches='tight', format='pdf')


if __name__=="__main__":

    # fix seeds

    # env.seed(1)  # not needed here, since we are choosing average over different init state
    # torch.manual_seed(123)
    # np.random.seed(1)

    dir=""
    model_name="ppo_net_50000step"   # can modify here
    model = PPO_net()
    model = torch.load(model_name)
    weights = get_weights(model)
    print(weights)
    num_of_dir = 2    # can modify here

    value_output_file =dir + model_name + '_' + 'value'


    dir_angle=1
    if num_of_dir ==2:
        while abs(dir_angle)>0.0001:
            dir_x = generate_random_dir(weights)
            dir_y = generate_random_dir(weights)
            normalize_directions_for_weights(dir_x, weights, norm='filter')
            normalize_directions_for_weights(dir_y, weights, norm='filter')
            dir_angle=angle_between_two_dirctions(dir_x,dir_y)
            print(dir_angle)
        print(dir_x, dir_y)
    else:
        dir = generate_random_dir(weights)
        normalize_directions_for_weights(dir, weights, norm='filter')
    # print(dir)
    # print(dir)



    # start calculating value function under different theta


    if num_of_dir ==1:
        x = list(np.linspace(-0.01,0.01,401))        # can modify here
        print(x)
        value = []

        for step in x:
            changes = [d*step for d in dir]

            for (p, w, d) in zip(model.parameters(), weights, changes):
                p.data = w + torch.Tensor(d).type(type(w))

            t_s=time.time()
            model_value=Cartpole_playground.playground(model, 100, True)  # can modify here
            value.append(model_value)
            print(model_value)
            print('time: ',time.time()-t_s)

        print(value)

    else:
        x = list(np.linspace(-3,3,51))  # can modify here
        y = list(np.linspace(-3,3,51))  # can modify here
        value = np.zeros(shape=(len(x), len(y)))
        for i in range(len(x)):
            for j in range(len(y)):
                print(i,j)
                changes = [dx * x[i] + dy * y[j] for dx,dy in zip(dir_x,dir_y)]
                for (p, w, d) in zip(model.parameters(), weights, changes):
                    p.data = w + torch.Tensor(d).type(type(w))

                t_s = time.time()
                model_value = Cartpole_playground.playground(model, 30, True)  # can modify here
                value[i][j]=model_value
                print(model_value)
                print('time: ', time.time() - t_s)

        print(value)



    if num_of_dir == 2:
        plot_2D_contour(x, y, value, dir+model_name)
    else:
        plot_1D(x,value, model_name)


    import json

    with open(value_output_file,'w') as fp:
        if num_of_dir ==2:
            value=value.tolist()
        json.dump(value, fp)
