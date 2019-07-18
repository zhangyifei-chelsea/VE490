import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym

# Hyper Parameters
BATCH_SIZE = 128
learningRate = 0.001                   # learning rate

EPSILON = 0              # greedy policy parameters
EPSILON_INCRESMENT = 0.01
EPSILON_THRES = 0.99
GAMMA = 0.99                # reward discount
TARGET_REPLACE_ITER = 100   # target update frequency
MEMORY_CAPACITY = 50000
env = gym.make('CartPole-v0')
# env=env.unwrapped
N_ACTIONS = env.action_space.n
N_STATES = env.observation_space.shape[0]
ENV_A_SHAPE = 0 if isinstance(env.action_space.sample(), int) else env.action_space.sample().shape     # to confirm the shape


class Net(nn.Module):
    def __init__(self, ):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(N_STATES, 128)
        self.fc1.weight.data.normal_(0, 0.1)   # initialization
        self.fc2 = nn.Linear(128, 128)
        self.fc2.weight.data.normal_(0, 0.1)  # initialization
        self.fc3 = nn.Linear(128, 128)
        self.fc3.weight.data.normal_(0, 0.1)  # initialization
        self.fc4 = nn.Linear(128, 128)
        self.fc4.weight.data.normal_(0, 0.1)  # initialization
        self.fc5 = nn.Linear(128, 128)
        self.fc5.weight.data.normal_(0, 0.1)  # initialization
        self.out = nn.Linear(128, N_ACTIONS)
        self.out.weight.data.normal_(0, 0.1)   # initialization
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.dropout(F.relu(self.fc3(x)))
        x = self.dropout(F.relu(self.fc4(x)))
        x = self.dropout(F.relu(self.fc5(x)))
        actions_value = self.out(x)
        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        # x = F.relu(self.fc3(x))
        # x = F.relu(self.fc4(x))
        # x = F.relu(self.fc5(x))
        # actions_value = self.out(x)
        return actions_value




class DQN(object):
    def __init__(self):
        self.online_net = Net()
        self.target_net = Net()
        self.learn_step_counter = 0                                     # for target updating
        self.memory_counter = 0                                         # for storing memory
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 3))     # initialize memory
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

    def store_transition(self, s, a, r, s_, done):
        transition = np.hstack((s, [a, r, done], s_))
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
        if self.memory.shape[0] == MEMORY_CAPACITY:
            sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        else:
            sample_index = np.random.choice(self.memory.shape[0], BATCH_SIZE)
        b_memory = self.memory[sample_index, :]
        b_s = torch.FloatTensor(b_memory[:, :N_STATES])
        b_a = torch.LongTensor(b_memory[:, N_STATES:N_STATES+1].astype(int))
        b_r = torch.FloatTensor(b_memory[:, N_STATES+1:N_STATES+2])
        b_s_ = torch.FloatTensor(b_memory[:, -N_STATES:])
        b_done = torch.FloatTensor(b_memory[:, N_STATES+2:N_STATES+3])

        # q_eval w.r.t the action in experience

        q_eval = self.online_net(b_s).gather(1, b_a)  # shape (batch, 1)
        q_next = self.target_net(b_s_).detach()     # detach from graph, don't backpropagate
        q_target = b_r + GAMMA * (1-b_done)* q_next.max(1)[0].view(BATCH_SIZE, 1)   # shape (batch, 1)




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

#
env.seed(11)
torch.manual_seed(34513)
np.random.seed(251)

dqn = DQN()

print('\nCollecting experience...')

num_episode=630

for i_episode in range(num_episode):
    # if i_episode%5==0:
    #     torch.save(dqn.getOnlineNet(), 'trajectory_visualize/Cartpole_dqn_origin_'+str(num_episode)+'ep_2layer_10neu_'+str(learningRate)+'lr_'+str(MEMORY_CAPACITY)+'memory_'+str(BATCH_SIZE)+"batchsize"+str(int(i_episode)))
    if EPSILON<EPSILON_THRES:
        EPSILON += EPSILON_INCRESMENT
    s = env.reset()
    episode_reward = 0

    substep = 0
    while substep<5000:
        substep += 1
        # env.render()
        a = dqn.choose_action(s)

        # take action
        s_, r, done, info = env.step(a)


        if done:
            r = -r
        dqn.store_transition(s, a, r, s_, done)
        s = s_

        episode_reward += r
        if dqn.memory_counter > BATCH_SIZE:
            dqn.learn()

        if done:
            break

    print('Episode: ', i_episode,
                  '| Episode_reward: ', episode_reward)

env.close()

print([p.data for p in dqn.getOnlineNet().parameters()])


torch.save(dqn.getOnlineNet(), 'Cartpole_dqn_with_dropout_600ep_5layer_128neu_0.001lr')