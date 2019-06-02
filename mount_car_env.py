import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym


# Hyper Parameters
BATCH_SIZE = 64
learningRate = 0.005                   # learning rate
EPSILON = 0.0               # greedy policy
EPSILON_THRESHOLD = 0.9
EPSILON_INCRESMENT = 0.01
GAMMA = 0.9                 # reward discount
TARGET_REPLACE_ITER = 100   # target update frequency
MEMORY_CAPACITY = 5000
env = gym.make('MountainCar-v0')
env = env.unwrapped
N_ACTIONS = env.action_space.n
N_STATES = env.observation_space.shape[0]
ENV_A_SHAPE = 0 if isinstance(env.action_space.sample(), int) else env.action_space.sample().shape     # to confirm the shape


class Net(nn.Module):
    def __init__(self, ):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(N_STATES, 10)  # can modify here
        self.fc1.weight.data.normal_(0, 0.1)   # initialization
        # self.bn1 = nn.BatchNorm1d(num_features=50)  # bn
        self.out = nn.Linear(10, N_ACTIONS)
        self.out.weight.data.normal_(0, 0.1)   # initialization

    def forward(self, x):
        x = self.fc1(x)
        # x = self.bn1(x)
        x = F.relu(x)
        actions_value = self.out(x)
        return actions_value

    # def get_value(self, x):
    #     x = self.fc1(x)
    #     x = F.relu(x)
    #     actions_value = self.out(x)
    #     return actions_value


class DQN(object):
    def __init__(self):
        self.online_net = Net()
        self.target_net = Net()
        self.learn_step_counter = 0                                     # for target updating
        self.memory_counter = 0                                         # for storing memory
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2))     # initialize memory
        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=learningRate)
        self.loss_func = nn.MSELoss()

    def choose_action(self, x):
        x = torch.unsqueeze(torch.FloatTensor(x), 0)
        # input only one sample
        if np.random.uniform() < EPSILON:   # greedy
            actions_value = self.online_net.forward(x)
            action = torch.max(actions_value, 1)[1].data.numpy()
            action = action[0] if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)  # return the argmax index
        else:   # random
            action = np.random.randint(0, N_ACTIONS)
            action = action if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)
        return action

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))
        # replace the old memory with new memory
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        # target parameter update
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())
        self.learn_step_counter += 1

        # sample batch transitions
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        b_memory = self.memory[sample_index, :]
        b_s = torch.FloatTensor(b_memory[:, :N_STATES])
        b_a = torch.LongTensor(b_memory[:, N_STATES:N_STATES+1].astype(int))
        b_r = torch.FloatTensor(b_memory[:, N_STATES+1:N_STATES+2])
        b_s_ = torch.FloatTensor(b_memory[:, -N_STATES:])

        # q_eval w.r.t the action in experience
        q_eval = self.online_net(b_s).gather(1, b_a)  # shape (batch, 1)
        q_next = self.target_net(b_s_).detach()     # detach from graph, don't backpropagate
        q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)   # shape (batch, 1)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def printPara(self):
        for p in self.online_net.parameters():
            if p.requires_grad:
                print(p.name, p.data)

    def getOnlineNet(self):
        return self.online_net


env.seed(1)
torch.manual_seed(1)
np.random.seed(1)

dqn = DQN()

print('\nCollecting experience...')
for i_episode in range(1000):
    if EPSILON<0.9:
        EPSILON += EPSILON_INCRESMENT
    s = env.reset()
    episode_reward = 0
    for i in range(5000):
        env.render()
        a = dqn.choose_action(s)

        # take action
        s_, r, done, info = env.step(a)
        # position = s_[0]
        # if position < -0.6:
        #     modified_reward = abs(position + 0.5)/3
        # elif position > -0.4:
        #     modified_reward = position + 0.5
        # else:
        #     modified_reward = 0
        dqn.store_transition(s, a, r, s_)
        s = s_

        episode_reward += r
        if dqn.memory_counter > MEMORY_CAPACITY:
            dqn.learn()
            if done:
                print('Episode: ', i_episode,
                      '| Episode_reward: ', episode_reward)

        if done:
            break
    print('Episode: ', i_episode,
              '| Episode_reward: ', episode_reward)

env.close()

print([p.data for p in dqn.getOnlineNet().parameters()])


torch.save(dqn.getOnlineNet(), 'mountainCar_dqn_origin')