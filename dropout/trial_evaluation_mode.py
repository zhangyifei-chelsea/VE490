import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym

# Hyper Parameters
BATCH_SIZE = 128
learningRate = 0.001  # learning rate

EPSILON = 0  # greedy policy parameters
EPSILON_INCRESMENT = 0.01
EPSILON_THRES = 0.99
GAMMA = 0.99  # reward discount
TARGET_REPLACE_ITER = 100  # target update frequency
MEMORY_CAPACITY = 50000
env = gym.make('CartPole-v0')
# env=env.unwrapped
N_ACTIONS = env.action_space.n
N_STATES = env.observation_space.shape[0]
ENV_A_SHAPE = 0 if isinstance(env.action_space.sample(), int) else env.action_space.sample().shape


class Net(nn.Module):
    def __init__(self, ):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(N_STATES, 128)
        self.fc1.weight.data.normal_(0, 0.1)  # initialization
        self.fc2 = nn.Linear(128, 128)
        self.fc2.weight.data.normal_(0, 0.1)  # initialization
        self.fc3 = nn.Linear(128, 128)
        self.fc3.weight.data.normal_(0, 0.1)  # initialization
        self.fc4 = nn.Linear(128, 128)
        self.fc4.weight.data.normal_(0, 0.1)  # initialization
        self.fc5 = nn.Linear(128, 128)
        self.fc5.weight.data.normal_(0, 0.1)  # initialization
        self.out = nn.Linear(128, N_ACTIONS)
        self.out.weight.data.normal_(0, 0.1)  # initialization
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.dropout(F.relu(self.fc3(x)))
        x = self.dropout(F.relu(self.fc4(x)))
        x = self.dropout(F.relu(self.fc5(x)))
        actions_value = self.out(x)
        return actions_value

    def agent_forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        actions_value = self.out(x)
        # print(self.parameters())
        return actions_value


def playground_choose_action(model, x):
    x = torch.unsqueeze(torch.FloatTensor(x), 0)
    # input only one sample
    actions_value = model.forward(x)
    action = torch.max(actions_value, 1)[1].data.numpy()
    action = action[0]
    return action


model = torch.load("Cartpole_dqn_with_dropout_new_trial1")
model.eval()
print([p.data for p in model.parameters()])
eval_num = 500
# start evaluation
sum_of_performance = 0
for i in range(eval_num):
    s = env.reset()
    reward = 0
    while True:
        a = playground_choose_action(model,s)
        s_, r, done, info = env.step(a)
        s = s_
        reward += r
        if done:
            break
    sum_of_performance+=reward

print(sum_of_performance/eval_num)
