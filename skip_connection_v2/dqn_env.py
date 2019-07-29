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
    def __init__(self, skip):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(N_STATES, 16)
        self.fc1.weight.data.normal_(0, 0.1)   # initialization
        self.fc2 = nn.Linear(16, 16)
        self.fc2.weight.data.normal_(0, 0.1)  # initialization
        self.fc3 = nn.Linear(16, 16)
        self.fc3.weight.data.normal_(0, 0.1)  # initialization
        self.fc4 = nn.Linear(16, 16)
        self.fc4.weight.data.normal_(0, 0.1)  # initialization
        self.fc5 = nn.Linear(16, 16)
        self.fc5.weight.data.normal_(0, 0.1)  # initialization
        self.fc6 = nn.Linear(16, 16)
        self.fc6.weight.data.normal_(0, 0.1)  # initialization
        self.fc7 = nn.Linear(16, 16)
        self.fc7.weight.data.normal_(0, 0.1)  # initialization
        self.fc8 = nn.Linear(16, 16)
        self.fc8.weight.data.normal_(0, 0.1)  # initialization
        self.fc9 = nn.Linear(16, 16)
        self.fc9.weight.data.normal_(0, 0.1)  # initialization
        self.fc10 = nn.Linear(16, 16)
        self.fc10.weight.data.normal_(0, 0.1)  # initialization
        self.fc11 = nn.Linear(16, 16)
        self.fc11.weight.data.normal_(0, 0.1)  # initialization
        self.fc12 = nn.Linear(16, 16)
        self.fc12.weight.data.normal_(0, 0.1)  # initialization
        self.fc13 = nn.Linear(16, 16)
        self.fc13.weight.data.normal_(0, 0.1)  # initialization
        self.fc14 = nn.Linear(16, 16)
        self.fc14.weight.data.normal_(0, 0.1)  # initialization
        self.fc15 = nn.Linear(16, 16)
        self.fc15.weight.data.normal_(0, 0.1)  # initialization
        self.out = nn.Linear(16, N_ACTIONS)
        self.out.weight.data.normal_(0, 0.1)   # initialization
        self.dropout = nn.Dropout(p=0.5)
        self.skip=skip

    def forward(self, x):
        if self.skip:
            x = F.relu(self.fc1(x))
            res = x
            x = F.relu(self.fc2(x))
            x = F.relu(self.fc3(x)+res)
            res = x
            x = F.relu(self.fc4(x))
            x = F.relu(self.fc5(x)+res)
            res = x
            x = F.relu(self.fc6(x))
            x = F.relu(self.fc7(x)+res)
            res = x
            x = F.relu(self.fc8(x))
            x = F.relu(self.fc9(x)+res)
            res = x
            x = F.relu(self.fc10(x))
            x = F.relu(self.fc11(x) + res)
            # res = x
            # x = F.relu(self.fc12(x))
            # x = F.relu(self.fc13(x) + res)
            # res = x
            # x = F.relu(self.fc14(x))
            # x = F.relu(self.fc15(x) + res)
            actions_value = self.out(x)
            return actions_value
        else:
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = F.relu(self.fc3(x))
            x = F.relu(self.fc4(x))
            x = F.relu(self.fc5(x))
            x = F.relu(self.fc6(x))
            x = F.relu(self.fc7(x))
            x = F.relu(self.fc8(x))
            x = F.relu(self.fc9(x))
            x = F.relu(self.fc10(x))
            x = F.relu(self.fc11(x))
            # x = F.relu(self.fc12(x))
            # x = F.relu(self.fc13(x))
            # x = F.relu(self.fc14(x))
            # x = F.relu(self.fc15(x))
            actions_value = self.out(x)
            return actions_value



class DQN(object):
    def __init__(self,skip):
        self.online_net = Net(skip)
        self.target_net = Net(skip)
        self.learn_step_counter = 0                                     # for target updating
        self.memory_counter = 0                                         # for storing memory
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 3))     # initialize memory
        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=learningRate)
        self.loss_func = nn.MSELoss()

    def choose_action(self, x, eval=False):
        x = torch.unsqueeze(torch.FloatTensor(x), 0)
        # input only one sample
        if np.random.uniform() < EPSILON or eval:   # greedy
            # try to use evaluate
            self.online_net.eval()
            actions_value = self.online_net.forward(x)
            # self.online_net.train()
            # actions_value = self.online_net.agent_forward(x)
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
        self.online_net.train()
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


env.seed(1111)
torch.manual_seed(1345413)
np.random.seed(2151)

dqn = DQN(skip=False)

print('\nCollecting experience...')

num_episode = 4000

best_in_eval=dqn.getOnlineNet()
best_avg_reward=0

for i_episode in range(num_episode):
    # if i_episode%5==0:
    #     torch.save(dqn.getOnlineNet(), 'trajectory_visualize/Cartpole_dqn_origin_'+str(num_episode)+'ep_2layer_10neu_'+str(learningRate)+'lr_'+str(MEMORY_CAPACITY)+'memory_'+str(BATCH_SIZE)+"batchsize"+str(int(i_episode)))
    if EPSILON<EPSILON_THRES:
        EPSILON += EPSILON_INCRESMENT
    s = env.reset()
    episode_reward = 0

    substep = 0
    while True:

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

    eval_num=70
    if i_episode % 5 == 0 and i_episode>2000:
        # start evaluation
        sum_of_performance=0
        for i in range(eval_num):
            s = env.reset()
            reward=0
            while True:
                a = dqn.choose_action(s,True)
                s_, r, done, info = env.step(a)
                s=s_
                reward += r
                if done:
                    break

            sum_of_performance+=reward
        print("current performance: ", sum_of_performance/eval_num)
        if sum_of_performance/eval_num>=best_avg_reward:
            best_avg_reward=sum_of_performance/eval_num
            print("best performance:", sum_of_performance/eval_num)
            best_in_eval=dqn.getOnlineNet()
            if best_avg_reward<200:
                torch.save(best_in_eval, 'Cartpole_dqn_skip_new_trial')
            else:
                torch.save(best_in_eval, 'Cartpole_dqn_skip_new_v3_episode'+str(i_episode))

env.close()

print([p.data for p in dqn.getOnlineNet().parameters()])


# torch.save(best_in_eval, 'Cartpole_dqn_with_dropout_new_trial1')