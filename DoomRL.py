import threading
import multiprocessing
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.rnn as rnn
from scipy import misc
from vizdoom import *
import os

from random import choice
from time import sleep
from time import time


def update_params(scope_from, scope_to):
    vars_from = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope_from)
    vars_to = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope_to)

    ops = []
    for from_var, to_var in zip(vars_from, vars_to):
        ops.append(to_var.assign(from_var))
    return ops


def discounted_return(r, gamma):
    r = r.astype(float)
    r_out = np.zeros_like(r)
    val = 0
    for i in reversed(xrange(r.shape[0])):
        r_out[i] = r[i] + gamma * val
        val = r_out[i]
    return r_out


# Processes Doom screen image to produce cropped and resized image.
def process_frame(frame):
    s = frame[10:-10, 30:-30]
    s = misc.imresize(s, [120, 120])
    s = np.reshape(s, [np.prod(s.shape)]) / 255.0
    return s


# Used to initialize weights for policy and value output layers
def normalized_columns_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer


class DoomNetwork:
    def __init__(self, frame_size, a_dim, scope, trainer, update_interval=30, rgb=False):
        var_scope = tf.variable_scope(scope)
        var_scope.__enter__()
        # with tf.variable_scope(scope):
        s_dim = frame_size[0] * frame_size[1] * (3 if rgb else 1)

        # convolution neural network for image input with 120*120 gray image
        self.input = tf.placeholder(shape=[None, s_dim], dtype=tf.float32)
        self.image = tf.reshape(self.input, shape=[-1, 120, 120, 1])
        self.conv1 = slim.conv2d(activation_fn=tf.nn.relu, inputs=self.image, num_outputs=32,
                                 kernel_size=[7, 7], stride=[2, 2], padding='VALID')
        self.conv2 = slim.conv2d(activation_fn=tf.nn.relu, inputs=self.conv1, num_outputs=64,
                                 kernel_size=[7, 7], stride=[2, 2], padding='VALID')
        self.maxpool3 = slim.max_pool2d(inputs=self.conv2, kernel_size=[3, 3], stride=2)
        self.conv4 = slim.conv2d(activation_fn=tf.nn.relu, inputs=self.maxpool3, num_outputs=128,
                                 kernel_size=[3, 3], stride=[1, 1], padding='VALID')
        self.maxpool5 = slim.max_pool2d(inputs=self.conv4, kernel_size=[3, 3], stride=2)
        self.conv6 = slim.conv2d(activation_fn=tf.nn.relu, inputs=self.maxpool5, num_outputs=192,
                                 kernel_size=[3, 3], stride=[1, 1], padding='VALID')
        self.fc7 = slim.fully_connected(inputs=slim.flatten(self.conv6), num_outputs=1024, activation_fn=None)

        self.lstm = rnn.BasicLSTMCell(num_units=256, state_is_tuple=True)

        # batch size 1 for every frame
        c_input = tf.placeholder(tf.float32, [1, self.lstm.state_size.c])
        h_input = tf.placeholder(tf.float32, [1, self.lstm.state_size.h])

        step_cnt = tf.shape(self.input)[:1]
        self.lstm_state_input = rnn.LSTMStateTuple(c_input, h_input)
        lstm_input = tf.expand_dims(self.fc7, 0)
        lstm_output, self.lstm_state_output = tf.nn.dynamic_rnn(self.lstm, lstm_input,
                                                                initial_state=self.lstm_state_input,
                                                                sequence_length=step_cnt)
        # lstm_c_output, lstm_h_output = lstm_state_output
        self.lstm_output = tf.reshape(lstm_output, [-1, 256])
        self.policy_pred = slim.fully_connected(inputs=self.lstm_output, num_outputs=a_dim,
                                                activation_fn=tf.nn.softmax,
                                                weights_initializer=normalized_columns_initializer(0.01),
                                                biases_initializer=None)
        self.val_pred = slim.fully_connected(inputs=self.lstm_output, num_outputs=1,
                                             activation_fn=None,
                                             weights_initializer=normalized_columns_initializer(1.0),
                                             biases_initializer=None)

        if scope != 'global':
            self.action = tf.placeholder(shape=[None], dtype=tf.int32)
            self.action_onehot = tf.one_hot(self.action, a_dim, dtype=tf.float32)
            self.target_val = tf.placeholder(shape=[None], dtype=tf.float32)
            self.advantages = tf.placeholder(shape=[None], dtype=tf.float32)

            self.pi_stoch = tf.reduce_sum(self.policy_pred * self.action_onehot, [1])

            # Loss functions
            self.value_loss = tf.reduce_sum(tf.square(self.target_val - tf.reshape(self.val_pred, [-1])))
            self.action_entropy = -tf.reduce_sum(self.policy_pred * tf.log(self.policy_pred))
            self.policy_loss = -tf.reduce_sum(tf.log(self.pi_stoch) * self.advantages)
            self.loss = 0.2 * self.value_loss + self.policy_loss - 0.01 * self.action_entropy

            local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
            self.gradients = tf.gradients(self.loss, local_vars)
            self.var_norms = tf.global_norm(local_vars)
            grads, self.grad_norms = tf.clip_by_global_norm(self.gradients, 40.0)

            # Apply local gradients to global network
            global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
            self.apply_grads = trainer.apply_gradients(zip(grads, global_vars))
        var_scope.__exit__(None, None, None)


class Agent:
    def __init__(self, game, name, frame_size, a_dim, trainer, model_path, global_episodes):
        self.name = "agent_" + str(name)
        self.number = name
        self.model_path = model_path
        self.trainer = trainer
        self.global_episodes = global_episodes
        self.increment = self.global_episodes.assign_add(1)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_mean_values = []
        self.summary_writer = tf.summary.FileWriter("train_" + str(self.number))

        # Create the local copy of the network and the tensorflow op to copy global paramters to local network
        self.local_network = DoomNetwork(frame_size, a_dim, self.name, trainer)
        self.update_local_from_global_ops = update_params('global', self.name)

        # The Below code is related to setting up the Doom environment
        game.set_doom_scenario_path("basic.wad")  # This corresponds to the simple task we will pose our agent
        game.set_doom_map("map01")
        game.set_screen_resolution(ScreenResolution.RES_160X120)
        game.set_screen_format(ScreenFormat.GRAY8)
        game.set_render_hud(False)
        game.set_render_crosshair(False)
        game.set_render_weapon(True)
        game.set_render_decals(False)
        game.set_render_particles(False)
        game.add_available_button(Button.MOVE_LEFT)
        game.add_available_button(Button.MOVE_RIGHT)
        game.add_available_button(Button.ATTACK)
        game.add_available_game_variable(GameVariable.AMMO2)
        game.add_available_game_variable(GameVariable.POSITION_X)
        game.add_available_game_variable(GameVariable.POSITION_Y)
        game.set_episode_timeout(300)
        game.set_episode_start_time(10)
        game.set_window_visible(False)
        game.set_sound_enabled(False)
        game.set_living_reward(-1)
        game.set_mode(Mode.PLAYER)
        game.init()
        self.actions = list(np.identity(a_dim, dtype=bool).tolist())
        # End Doom set-up
        self.env = game

    def train_batch(self, lstm_state_backup, replay, sess, gamma, val_last_state):
        replay = np.array(replay)
        states = replay[:, 0]
        actions = replay[:, 1]
        rewards = replay[:, 2]
        values = replay[:, 4]

        # Here we take the rewards and values from the rollout, and use them to
        # generate the advantage and discounted returns.
        # The advantage function uses "Generalized Advantage Estimation"
        self.rewards_plus = np.append(rewards, val_last_state)
        target_val = discounted_return(self.rewards_plus, gamma)[:-1]
        self.value_plus = np.append(values, val_last_state)

        # this is GAE(lambda) with lambda = 1
        advantages = rewards + gamma * self.value_plus[1:] - self.value_plus[:-1]
        advantages = discounted_return(advantages, gamma)

        # Update the global network using gradients from loss
        # Generate network statistics to periodically save

        feed_dict = {self.local_network.target_val: target_val,
                     self.local_network.input: np.vstack(states),
                     self.local_network.action: actions,
                     self.local_network.advantages: advantages,
                     self.local_network.lstm_state_input[0]: lstm_state_backup[0],
                     self.local_network.lstm_state_input[1]: lstm_state_backup[1]}
        v_l, p_l, e_l, g_n, v_n, _ = sess.run([self.local_network.value_loss,
                                               self.local_network.policy_loss,
                                               self.local_network.action_entropy,
                                               self.local_network.grad_norms,
                                               self.local_network.var_norms,
                                               self.local_network.apply_grads],
                                              feed_dict=feed_dict)
        return v_l / len(replay), p_l / len(replay), e_l / len(replay), g_n, v_n

    def run(self, max_episode_length, gamma, sess, coord, saver):
        episode_count = sess.run(self.global_episodes)
        total_steps = 0
        print ("Starting agent " + str(self.number))
        with sess.as_default(), sess.graph.as_default():
            while not coord.should_stop():
                sess.run(self.update_local_from_global_ops)
                episode_buffer = []
                episode_values = []
                episode_frames = []
                episode_reward = 0
                episode_step_count = 0

                self.env.new_episode()
                s = self.env.get_state().screen_buffer
                episode_frames.append(s)
                s = process_frame(s)

                # init LSTM state
                c_init = np.zeros((1, self.local_network.lstm.state_size.c), np.float32)
                h_init = np.zeros((1, self.local_network.lstm.state_size.h), np.float32)
                lstm_state = [c_init, h_init]
                lstm_state_backup = lstm_state

                while not self.env.is_episode_finished():
                    # Take an action using probabilities from policy network output.
                    a_distribution, v, lstm_state = sess.run(
                        [self.local_network.policy_pred, self.local_network.val_pred, self.local_network.lstm_state_output],
                        feed_dict={self.local_network.input: [s],
                                   self.local_network.lstm_state_input[0]: lstm_state[0],
                                   self.local_network.lstm_state_input[1]: lstm_state[1]})
                    a = np.random.choice(a_dim, p=a_distribution[0])
                    # a = np.argmax(a_distribution == a)

                    r = self.env.make_action(self.actions[a]) / 100.0
                    if self.env.is_episode_finished() or episode_step_count == max_episode_length - 1:
                        # terminal state with value 0
                        if len(episode_buffer) != 0:
                            v_l, p_l, e_l, g_n, v_n = self.train_batch(lstm_state_backup, episode_buffer, sess, gamma, 0.)
                        break
                    s_prime = self.env.get_state().screen_buffer
                    episode_frames.append(s_prime)
                    s_prime = process_frame(s_prime)

                    episode_buffer.append([s, a, r, s_prime, v[0, 0]])
                    episode_values.append(v[0, 0])
                    episode_reward += r

                    s = s_prime
                    total_steps += 1
                    episode_step_count += 1

                    # If the episode hasn't ended, but the experience buffer is full, then we
                    # make an update step using that experience rollout.
                    if len(episode_buffer) == 30:
                        # Since we don't know what the true final return is, we "bootstrap" from our current
                        # value estimation.

                        val_last_state = sess.run(self.local_network.val_pred,
                                                  feed_dict={self.local_network.input: [s],
                                                             self.local_network.lstm_state_input[0]: lstm_state[0],
                                                             self.local_network.lstm_state_input[1]: lstm_state[1]})[0, 0]
                        v_l, p_l, e_l, g_n, v_n = self.train_batch(lstm_state_backup, episode_buffer,
                                                                   sess, gamma, val_last_state)
                        episode_buffer = []

                        # backup our state
                        lstm_state_backup = lstm_state

                        # update local network
                        sess.run(self.update_local_from_global_ops)

                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_step_count)
                self.episode_mean_values.append(np.mean(episode_values))

                # Periodically save gifs of episodes, model parameters, and summary statistics.
                if episode_count % 5 == 0 and episode_count != 0:
                    if self.name == 'agent_0' and episode_count % 25 == 0:
                        time_per_step = 0.05
                        images = np.array(episode_frames)
                        # make_gif(images, './frames/image' + str(episode_count) + '.gif',
                        #          duration=len(images) * time_per_step, true_image=True, salience=False)
                    if episode_count % 250 == 0 and self.name == 'agent_0':
                        saver.save(sess, self.model_path + '/model-' + str(episode_count) + '.cptk')
                        print ("Saved Model")

                    mean_reward = np.mean(self.episode_rewards[-5:])
                    mean_length = np.mean(self.episode_lengths[-5:])
                    mean_value = np.mean(self.episode_mean_values[-5:])
                    summary = tf.Summary()
                    summary.value.add(tag='Perf/Reward', simple_value=float(mean_reward))
                    summary.value.add(tag='Perf/Length', simple_value=float(mean_length))
                    summary.value.add(tag='Perf/Value', simple_value=float(mean_value))
                    summary.value.add(tag='Losses/Value Loss', simple_value=float(v_l))
                    summary.value.add(tag='Losses/Policy Loss', simple_value=float(p_l))
                    summary.value.add(tag='Losses/Entropy', simple_value=float(e_l))
                    summary.value.add(tag='Losses/Grad Norm', simple_value=float(g_n))
                    summary.value.add(tag='Losses/Var Norm', simple_value=float(v_n))
                    self.summary_writer.add_summary(summary, episode_count)

                    self.summary_writer.flush()
                if self.name == 'agent_0':
                    sess.run(self.increment)
                episode_count += 1


def runGame(sess, network):
    a_dim = 3
    fps = 30

    game = DoomGame()
    game.set_doom_scenario_path("basic.wad")  # This corresponds to the simple task we will pose our agent
    game.set_doom_map("map01")
    game.set_screen_resolution(ScreenResolution.RES_640X480)
    game.set_screen_format(ScreenFormat.GRAY8)
    game.set_render_hud(False)
    game.set_render_crosshair(False)
    game.set_render_weapon(True)
    game.set_render_decals(False)
    game.set_render_particles(False)
    game.add_available_button(Button.MOVE_LEFT)
    game.add_available_button(Button.MOVE_RIGHT)
    game.add_available_button(Button.ATTACK)
    game.add_available_game_variable(GameVariable.AMMO2)
    game.add_available_game_variable(GameVariable.POSITION_X)
    game.add_available_game_variable(GameVariable.POSITION_Y)
    game.set_episode_timeout(300)
    game.set_episode_start_time(10)
    game.set_window_visible(True)
    game.set_sound_enabled(True)
    game.set_living_reward(-1)
    game.set_mode(Mode.PLAYER)
    game.init()

    actions = list(np.identity(a_dim, dtype=bool).tolist())

    for i in xrange(20):
        game.new_episode()
        s = game.get_state().screen_buffer
        s = misc.imresize(s, [120, 160])
        s = process_frame(s)

        # init LSTM state
        c_init = np.zeros((1, network.lstm.state_size.c), np.float32)
        h_init = np.zeros((1, network.lstm.state_size.h), np.float32)
        lstm_state = [c_init, h_init]
        while not game.is_episode_finished():
            sleep(1./30)
            # Take an action using probabilities from policy network output.
            a_distribution, v, lstm_state = sess.run(
                [network.policy_pred, network.val_pred, network.lstm_state_output],
                feed_dict={network.input: [s],
                           network.lstm_state_input[0]: lstm_state[0],
                           network.lstm_state_input[1]: lstm_state[1]})
            a = np.random.choice(a_dim, p=a_distribution[0])
            # a = np.argmax(a_distribution == a)

            r = game.make_action(actions[a]) / 100.0
            if game.is_episode_finished():
                break
            s_prime = game.get_state().screen_buffer
            s_prime = process_frame(s_prime)

            s = s_prime


if __name__ == '__main__':
    max_episode_length = 60
    gamma = .99  # discount rate for advantage estimation and reward discounting
    a_dim = 3  # Agent can move Left, Right, or Fire
    load_model = True
    model_path = './model'

    tf.reset_default_graph()

    if not os.path.exists(model_path):
        os.makedirs(model_path)

    # Create a directory to save episode playback gifs to
    # if not os.path.exists('./frames'):
    #     os.makedirs('./frames')

    with tf.device("/cpu:0"):
        # global_episodes = tf.Variable(0, dtype=tf.int32, name='global_episodes', trainable=False)
        # trainer = tf.train.AdamOptimizer(learning_rate=1e-4)
        network = DoomNetwork([120, 120], a_dim, 'global', None)  # Generate global network
        # # num_agents = multiprocessing.cpu_count()  # Set workers ot number of available CPU threads
        # num_agents = 4
        # agents = []
        # # Create worker classes
        # for i in range(num_agents):
        #     agents.append(Agent(DoomGame(), i, [120, 120], a_dim, trainer, model_path, global_episodes))
        saver = tf.train.Saver(max_to_keep=5)

    with tf.Session() as sess:
        # coord = tf.train.Coordinator()
        if load_model:
            print ('Loading Model...')
            ckpt = tf.train.get_checkpoint_state(model_path)
            saver.restore(sess, ckpt.model_checkpoint_path)
            runGame(sess, network)
        else:
            sess.run(tf.global_variables_initializer())

        # agents[0].run(max_episode_length, gamma, sess, coord, saver)

        # This is where the asynchronous magic happens.
        # Start the "work" process for each worker in a separate threat.
        # agent_threads = []
        # for agent in agents:
        #     agent_run = lambda: agent.run(max_episode_length, gamma, sess, coord, saver)
        #     t = threading.Thread(target=(agent_run))
        #     t.start()
        #     sleep(0.5)
        #     agent_threads.append(t)
        # coord.join(agent_threads)
