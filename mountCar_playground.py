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
    GAMMA = 1                                             # can modify here
    torch.manual_seed(1)
    np.random.seed(1)
    env = gym.make('MountainCar-v0')
    env = env.unwrapped
    dqn = model

    accumulated_reward = 0
    # print(get_weights(dqn))

    for i_episode in range(iteration):
        s = env.reset()
        episode_reward = 0
        for i in range(500):
            a = playground_choose_action(dqn, s)

            # take action
            s_, r, done, info = env.step(a)
            position = s_[0]
            if position < -0.6:
                modified_reward = abs(position + 0.5) / 3
            elif position > -0.4:
                modified_reward = position + 0.5
            else:
                modified_reward = 0

            s = s_

            if use_env_reward:
                episode_reward += r* (GAMMA**i)
            else:
                episode_reward += modified_reward * (GAMMA ** i)

            if done:
                break
        accumulated_reward += episode_reward

    env.close()
    return accumulated_reward / iteration
