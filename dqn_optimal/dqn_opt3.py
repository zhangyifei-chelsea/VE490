import gym
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# import numpy as np
from collections import deque
import random


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size=int(1e5), random_seed=1234):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size: maximum size of buffer
            The right side of the deque contains the most recent experiences.
        """
        self.buffer_size = buffer_size
        self.buffer = deque(maxlen=buffer_size)
        random.seed(random_seed)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.buffer)

    def add(self, s, a, r, done, s2):
        """Add a new experience to buffer.
        Params
        ======
        s: one state sample, numpy array shape (s_dim,)
        a: one action sample, scalar (for DQN)
        r: one reward sample, scalar
        done: True/False, scalar
        s2: one state sample, numpy array shape (s_dim,)
        """
        e = (s, a, r, done, s2)
        self.buffer.append(e)

    def sample_batch(self, batch_size):
        """Randomly sample a batch of experiences from buffer."""

        # ensure the buffer is large enough for sampleling
        assert (len(self.buffer) >= batch_size)

        # sample a batch
        batch = random.sample(self.buffer, batch_size)

        # Convert experience tuples to separate arrays for each element (states, actions, rewards, etc.)
        states, actions, rewards, dones, next_states = zip(*batch)
        states = np.asarray(states).reshape(batch_size, -1)  # shape (batch_size, s_dim)
        next_states = np.asarray(next_states).reshape(batch_size, -1)  # shape (batch_size, s_dim)
        actions = np.asarray(actions)  # shape (batch_size,), for DQN, action is an int
        rewards = np.asarray(rewards)  # shape (batch_size,)
        dones = np.asarray(dones, dtype=np.uint8)  # shape (batch_size,)
        return states, actions, rewards, dones, next_states


def build_summaries():
    """
    tensorboard summary for monitoring training process
    """

    # performance per episode
    ph_reward = tf.placeholder(tf.float32)
    tf.summary.scalar("Reward_ep", ph_reward)
    ph_Qmax = tf.placeholder(tf.float32)
    tf.summary.scalar("Qmax_ep", ph_Qmax)

    # merge all summary op (must be done at the last step)
    summary_op = tf.summary.merge_all()

    return summary_op, ph_reward, ph_Qmax


import time
from keras import layers, initializers, regularizers
from functools import partial


def build_net(model_name, state, a_dim, args, trainable):
    """
    neural network model
    model input: state
    model output: Qhat
    """
    h1 = int(args['h1'])
    h2 = int(args['h2'])

    my_dense = partial(layers.Dense, trainable=trainable)
    with tf.variable_scope(model_name):
        net = my_dense(h1, name="l1-dense-{}".format(h1))(state)
        net = layers.Activation('relu', name="relu1")(net)
        net = my_dense(h2, name="l2-dense-{}".format(h2))(net)
        net = layers.Activation('relu', name="relu2")(net)
        net = my_dense(a_dim, name="l3-dense-{}".format(a_dim))(net)
    Qhat = layers.Activation('linear', name="Qhat")(net)
    nn_params = tf.trainable_variables(scope=model_name)
    return Qhat, nn_params


class DeepQNetwork:
    def __init__(self, sess, a_dim, s_dim, args):
        self.a_dim = a_dim
        self.s_dim = s_dim
        self.h1 = args["h1"]
        self.h2 = args["h2"]
        self.lr = args["learning_rate"]
        self.gamma = args["gamma"]
        self.epsilon_start = args["epsilon_start"]
        self.epsilon_stop = args["epsilon_stop"]
        self.epsilon_decay = args["epsilon_decay"]
        self.epsilon = self.epsilon_start  # current exploration probability
        self.update_target_C = args["update_target_C"]
        self.update_target_tau = args['update_target_tau']
        self.learn_step_counter = 0

        # initialize replay buffer
        self.replay_buffer = ReplayBuffer(int(args['buffer_size']), int(args['random_seed']))
        self.minibatch_size = int(args['minibatch_size'])

        self.s = tf.placeholder(tf.float32, [None, self.s_dim], name='state')  # input State
        self.s_ = tf.placeholder(tf.float32, [None, self.s_dim], name='state_next')  # input Next State
        self.r = tf.placeholder(tf.float32, [None, ], name='reward')  # input Reward
        self.a = tf.placeholder(tf.int32, [None, ], name='action')  # input Action
        self.done = tf.placeholder(tf.float32, [None, ], name='done')

        # initialize NN, self.q shape (batch_size, a_dim)
        self.q, self.nn_params = build_net("DQN", self.s, a_dim, args, trainable=True)
        self.q_, self.nn_params_ = build_net("target_DQN", self.s_, a_dim, args, trainable=False)
        for var in self.nn_params:
            vname = var.name.replace("kernel:0", "W").replace("bias:0", "b")
            tf.summary.histogram(vname, var)

        with tf.variable_scope("Qmax"):
            self.Qmax = tf.reduce_max(self.q_, axis=1)  # shape (batch_size,)

        with tf.variable_scope("yi"):
            self.yi = self.r + self.gamma * self.Qmax * (1 - self.done)  # shape (batch_size,)

        with tf.variable_scope("Qa_all"):
            Qa = tf.Variable(tf.zeros([self.minibatch_size, self.a_dim]))
            for aval in np.arange(self.a_dim):
                tf.summary.histogram("Qa{}".format(aval), Qa[:, aval])
            self.Qa_op = Qa.assign(self.q)

        with tf.variable_scope("Q_at_a"):
            # select the Q value corresponding to the action
            one_hot_actions = tf.one_hot(self.a, self.a_dim)  # shape (batch_size, a_dim)
            q_all = tf.multiply(self.q, one_hot_actions)  # shape (batch_size, a_dim)
            self.q_at_a = tf.reduce_sum(q_all, axis=1)  # shape (batch_size,)

        with tf.variable_scope("loss_MSE"):
            self.loss = tf.losses.mean_squared_error(labels=self.yi, predictions=self.q_at_a)

        with tf.variable_scope("train_DQN"):
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(loss=self.loss, var_list=self.nn_params)

        with tf.variable_scope("soft_update"):
            TAU = self.update_target_tau
            self.update_op = [tf.assign(t, (1 - TAU) * t + TAU * e) for t, e in zip(self.nn_params_, self.nn_params)]

    def choose_action(self, sess, observation):
        # Explore or Exploit
        explore_p = self.epsilon  # exploration probability

        if np.random.uniform() <= explore_p:
            # Explore: make a random action
            action = np.random.randint(0, self.a_dim)
        else:
            # Exploit: Get action from Q-network
            observation = np.reshape(observation, (1, self.s_dim))
            Qs = sess.run(self.q, feed_dict={self.s: observation})  # shape (1, a_dim)
            action = np.argmax(Qs[0])
        return action

    def learn_a_batch(self, sess):
        # update target every C learning steps
        if self.learn_step_counter % self.update_target_C == 0:
            sess.run(self.update_op)

        # Sample a batch
        s_batch, a_batch, r_batch, done_batch, s2_batch = self.replay_buffer.sample_batch(self.minibatch_size)

        # Train
        _, _, Qhat, loss = sess.run([self.train_op, self.Qa_op, self.q_at_a, self.loss], feed_dict={
            self.s: s_batch, self.a: a_batch, self.r: r_batch, self.done: done_batch, self.s_: s2_batch})

        # count learning steps
        self.learn_step_counter += 1

        # decay exploration probability after each learning step
        if self.epsilon > self.epsilon_stop:
            self.epsilon *= self.epsilon_decay

        return np.max(Qhat)


args = {"env": 'CartPole-v0',
        "random_seed": 1234,
        "max_episodes": 150,         # number of episodes
        "max_episode_len": 200,      # time steps per episode, 200 for CartPole-v0
        ## NN params
        "h1": 32,                   # 32
        "h2": 64,                   # 64
        "learning_rate": 0.001,     # 1e-3
        "gamma": 0.9,               # 0.9 (32), 0.95 (34) better than 0.99
        "update_target_C": 1,       # update every C learning steps (C=1 if soft update, C=100 if hard update)
        "update_target_tau": 8e-2,  # soft update (tau=8e-2), hard update (tau=1)
        ## exploration prob
        "epsilon_start": 1.0,
        "epsilon_stop": 0.01,       # 0.01
        "epsilon_decay": 0.999,     # 0.999
        ## replay buffer
        "buffer_size": 1e5,
        "minibatch_size": 32,       # 32
        ## tensorboard logs
        "summary_dir": './results/dqn',
        }



sess = tf.InteractiveSession()
tf.set_random_seed(int(args['random_seed']))

# initialize numpy seed
np.random.seed(int(args['random_seed']))

# initialize gym env
env = gym.make(args['env'])
env.seed(int(args['random_seed']))
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
print("states:", env.observation_space)
print("actions:", env.action_space)

# initialize DQN agent
agent = DeepQNetwork(sess, action_size, state_size, args)

# saver=tf.train.Saver()

# initialize summary (for visualization in tensorboard)
summary_op, ph_reward, ph_Qmax = build_summaries()
subdir = time.strftime("%Y%m%d-%H%M%S", time.localtime())  # a sub folder, e.g., yyyymmdd-HHMMSS
logdir = args['summary_dir'] + '/' + subdir
writer = tf.summary.FileWriter(logdir, sess.graph)  # must be done after graph is constructed

# initialize variables existed in the graph
sess.run(tf.global_variables_initializer())

# training DQN agent
rewards_list = []
loss = -999
num_ep = args['max_episodes']
max_t = args['max_episode_len']
for ep in range(num_ep):
    state = env.reset()  # shape (s_dim,)
    ep_reward = 0  # total reward per episode
    ep_qmax = 0
    t_step = 0
    done = False
    while (t_step < max_t) and (not done):

        # choose an action
        action = agent.choose_action(sess, state)

        # interact with the env
        next_state, reward, done, _ = env.step(action)

        # add the experience to replay buffer
        agent.replay_buffer.add(state, action, reward, done, next_state)

        # learn from a batch of experiences
        if len(agent.replay_buffer) > 3 * agent.minibatch_size:
            qmax = agent.learn_a_batch(sess)
            ep_qmax = max(ep_qmax, qmax)

        # next time step
        t_step += 1
        ep_reward += reward
        state = next_state

    # end of an episode
    rewards_list.append((ep, ep_reward))

    # write to tensorboard summary
    summary_str = sess.run(summary_op, feed_dict={ph_reward: ep_reward, ph_Qmax: ep_qmax})
    writer.add_summary(summary_str, ep)
    writer.flush()

    if ep % 10 == 0:
        print("episode: {}/{}, steps: {}, explore_prob: {:.2f}, total reward: {}". \
              format(ep, num_ep, t_step, agent.epsilon, ep_reward))


# saver.save(sess,'output')

parameter_dict={}
parameter_dict['layer1_weight']=sess.run(agent.nn_params)[0].tolist()
parameter_dict['layer1_bias']=sess.run(agent.nn_params)[1].tolist()
parameter_dict['layer2_weight']=sess.run(agent.nn_params)[2].tolist()
parameter_dict['layer2_bias']=sess.run(agent.nn_params)[3].tolist()
parameter_dict['layer3_weight']=sess.run(agent.nn_params)[4].tolist()
parameter_dict['layer3_bias']=sess.run(agent.nn_params)[5].tolist()
# print(sess.run(agent.nn_params)[0])

print(parameter_dict)

import json

with open('dqn_optimal_parameter_trial','w') as output:
    json.dump(parameter_dict,output)