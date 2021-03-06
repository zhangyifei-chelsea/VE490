import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym


def get_weights(net):
    """ Extract parameters from net, and return a list of tensors"""
    return [p.data for p in net.parameters()]


def playground_choose_action(model, x):
    x = torch.unsqueeze(torch.FloatTensor(x), 0)
    # input only one sample
    actions_value = model.forward(x)
    action = torch.max(actions_value, 1)[1].data.numpy()
    action = action[0]
    return action


def playground(model, iteration, use_env_reward=True):
    GAMMA = 0.9                                             # can modify here
    env = gym.make('CartPole-v0')
    env = env.unwrapped
    dqn = model

    accumulated_reward = 0
    # print(get_weights(dqn))

    for i_episode in range(iteration):
        s = env.reset()
        episode_reward = 0
        for i in range(200):    # can modify here
            a = playground_choose_action(dqn, s)

            # take action
            s_, r, done, info = env.step(a)
            s = s_

            if use_env_reward:
                episode_reward += r #* (GAMMA**i)

            if done:
                break
        accumulated_reward += episode_reward

    env.close()
    return accumulated_reward / iteration
