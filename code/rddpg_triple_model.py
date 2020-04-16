'''
Author: Sunghoon Hong
Title: rddpg.py
Description:
    Recurrent Deep Deterministic Policy Gradient Agent for Airsim
Detail:
    - not use join()
    - reset for zero-image error
    - tensorflow v1 + keras
    - soft update for target model

'''


import os
import csv
import time
import random
import argparse
from copy import deepcopy
from collections import deque
from datetime import datetime as dt
import numpy as np
import tensorflow as tf
import keras.backend as K
from keras.layers import TimeDistributed, BatchNormalization, Flatten, Add, Lambda, Concatenate
from keras.layers import Conv2D, MaxPooling2D, Dense, GRU, Input, ELU, Activation
from keras.optimizers import Adam
from keras.models import Model
from PIL import Image
import cv2
from airsim_env import Env

np.set_printoptions(suppress=True, precision=4)
agent_name1 = 'rddpg1'
agent_name2 = 'rddpg2'
agent_name3 = 'rddpg3'


class RDDPGAgent(object):

    def __init__(self, state_size, action_size, actor_lr, critic_lr, tau,
                gamma, lambd, batch_size, memory_size,
                epsilon, epsilon_end, decay_step, load_model, agent_name):
        self.state_size = state_size
        self.vel_size = 1
        self.action_size = action_size
        self.action_high = 1.5
        self.action_low = -self.action_high
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.tau = tau
        self.gamma = gamma
        self.lambd = lambd
        self.batch_size = batch_size
        self.memory_size = memory_size
        self.epsilon = epsilon
        self.epsilon_end = epsilon_end
        self.decay_step = decay_step
        self.epsilon_decay = (epsilon - epsilon_end) / decay_step

        self.sess = tf.Session()
        K.set_session(self.sess)

        self.actor, self.critic = self.build_model()
        self.target_actor, self.target_critic = self.build_model()
        self.actor_update = self.build_actor_optimizer()
        self.critic_update = self.build_critic_optimizer()
        self.sess.run(tf.global_variables_initializer())
        if load_model:
            self.load_model('./save_model/'+ agent_name)

        self.target_actor.set_weights(self.actor.get_weights())
        self.target_critic.set_weights(self.critic.get_weights())

        self.memory = deque(maxlen=self.memory_size)

    def build_model(self):
        # shared network
        # image process
        image = Input(shape=self.state_size)
        image_process = BatchNormalization()(image)
        image_process = TimeDistributed(
            Conv2D(16, (3, 3), activation='elu', padding='same', kernel_initializer='he_normal'))(image_process)
        #72 128
        image_process = TimeDistributed(Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal'))(
            image_process)
        #70 126
        image_process = TimeDistributed(Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal'))(
            image_process)
        #68 124
        image_process = TimeDistributed(MaxPooling2D((2, 2)))(image_process)
        #34 62
        image_process = TimeDistributed(Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal'))(
            image_process)
        #32 60
        image_process = TimeDistributed(Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal'))(
            image_process)
        #30 58
        image_process = TimeDistributed(MaxPooling2D((2, 2)))(image_process)
        #15 29
        image_process = TimeDistributed(Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal'))(
            image_process)
        #13 27
        image_process = TimeDistributed(Conv2D(32, (4, 4), activation='elu', kernel_initializer='he_normal'))(
            image_process)
        #10 24
        image_process = TimeDistributed(MaxPooling2D((2, 2)))(image_process)
        #5 12
        image_process = TimeDistributed(Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal'))(
            image_process)
        #3 10
        image_process = TimeDistributed(Conv2D(8, (1, 1), activation='elu', kernel_initializer='he_normal'))(
            image_process)
        image_process = TimeDistributed(Flatten())(image_process)
        image_process = GRU(48, kernel_initializer='he_normal', use_bias=False)(image_process)
        image_process = BatchNormalization()(image_process)
        image_process = Activation('tanh')(image_process)

        # vel process
        vel = Input(shape=[self.vel_size])
        vel_process = Dense(48, kernel_initializer='he_normal', use_bias=False)(vel)
        vel_process = BatchNormalization()(vel_process)
        vel_process = Activation('tanh')(vel_process)

        # state process
        state_process = Add()([image_process, vel_process])

        # Actor
        policy = Dense(32, kernel_initializer='he_normal', use_bias=False)(state_process)
        policy = BatchNormalization()(policy)
        policy = ELU()(policy)
        policy = Dense(32, kernel_initializer='he_normal', use_bias=False)(policy)
        policy = BatchNormalization()(policy)
        policy = ELU()(policy)
        policy = Dense(self.action_size, kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))(policy)
        policy = Lambda(lambda x: K.clip(x, self.action_low, self.action_high))(policy)
        actor = Model(inputs=[image, vel], outputs=policy)

        # Critic
        action = Input(shape=[self.action_size])
        action_process = Dense(48, kernel_initializer='he_normal', use_bias=False)(action)
        action_process = BatchNormalization()(action_process)
        action_process = Activation('tanh')(action_process)
        state_action = Add()([state_process, action_process])

        Qvalue = Dense(32, kernel_initializer='he_normal', use_bias=False)(state_action)
        Qvalue = BatchNormalization()(Qvalue)
        Qvalue = ELU()(Qvalue)
        Qvalue = Dense(32, kernel_initializer='he_normal', use_bias=False)(Qvalue)
        Qvalue = BatchNormalization()(Qvalue)
        Qvalue = ELU()(Qvalue)
        Qvalue = Dense(1, kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))(Qvalue)
        critic = Model(inputs=[image, vel, action], outputs=Qvalue)

        actor._make_predict_function()
        critic._make_predict_function()

        return actor, critic

    def build_actor_optimizer(self):
        pred_Q = self.critic.output
        action_grad = tf.gradients(pred_Q, self.critic.input[2])
        target = -action_grad[0] / self.batch_size
        params_grad = tf.gradients(
            self.actor.output, self.actor.trainable_weights, target)
        params_grad, global_norm = tf.clip_by_global_norm(params_grad, 5.0)
        grads = zip(params_grad, self.actor.trainable_weights)
        optimizer = tf.train.AdamOptimizer(self.actor_lr)
        updates = optimizer.apply_gradients(grads)
        train = K.function(
            [self.actor.input[0], self.actor.input[1], self.critic.input[2]],
            [global_norm],
            updates=[updates]
        )
        return train

    def build_critic_optimizer(self):
        y = K.placeholder(shape=(None, 1), dtype='float32')
        pred = self.critic.output

        loss = K.mean(K.square(pred - y))
        # Huber Loss
        # error = K.abs(y - pred)
        # quadratic = K.clip(error, 0.0, 1.0)
        # linear = error - quadratic
        # loss = K.mean(0.5 * K.square(quadratic) + linear)

        optimizer = Adam(lr=self.critic_lr)
        updates = optimizer.get_updates(self.critic.trainable_weights, [], loss)
        train = K.function(
            [self.critic.input[0], self.critic.input[1], self.critic.input[2], y],
            [loss],
            updates=updates
        )
        return train

    def get_action(self, state):
        policy = self.actor.predict(state)[0]
        noise = np.random.normal(0, self.epsilon, self.action_size)
        action = np.clip(policy + noise, self.action_low, self.action_high)
        return action, policy

    def train_model(self):
        batch = random.sample(self.memory, self.batch_size)

        images = np.zeros([self.batch_size] + self.state_size)
        vels = np.zeros([self.batch_size, self.vel_size])
        actions = np.zeros((self.batch_size, self.action_size))
        rewards = np.zeros((self.batch_size, 1))
        next_images = np.zeros([self.batch_size] + self.state_size)
        next_vels = np.zeros([self.batch_size, self.vel_size])
        dones = np.zeros((self.batch_size, 1))

        targets = np.zeros((self.batch_size, 1))

        for i, sample in enumerate(batch):
            images[i], vels[i] = sample[0]
            actions[i] = sample[1]
            rewards[i] = sample[2]
            next_images[i], next_vels[i] = sample[3]
            dones[i] = sample[4]
        states = [images, vels]
        next_states = [next_images, next_vels]
        policy = self.actor.predict(states)
        target_actions = self.target_actor.predict(next_states)
        target_next_Qs = self.target_critic.predict(next_states + [target_actions])
        targets = rewards + self.gamma * (1 - dones) * target_next_Qs

        actor_loss = self.actor_update(states + [policy])
        critic_loss = self.critic_update(states + [actions, targets])
        return actor_loss[0], critic_loss[0]

    def append_memory(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def load_model(self, name):
        if os.path.exists(name + '_actor.h5'):
            self.actor.load_weights(name + '_actor.h5')
            print('Actor loaded')
        if os.path.exists(name + '_critic.h5'):
            self.critic.load_weights(name + '_critic.h5')
            print('Critic loaded')

    def save_model(self, name):
        self.actor.save_weights(name + '_actor.h5')
        self.critic.save_weights(name + '_critic.h5')

    def update_target_model(self):
        self.target_actor.set_weights(
            self.tau * np.array(self.actor.get_weights()) \
            + (1 - self.tau) * np.array(self.target_actor.get_weights())
        )
        self.target_critic.set_weights(
            self.tau * np.array(self.critic.get_weights()) \
            + (1 - self.tau) * np.array(self.target_critic.get_weights())
        )


'''
Environment interaction
'''

def transform_input(responses, img_height, img_width):
    img = np.array(cv2.cvtColor(responses[:,:,:3], cv2.COLOR_BGR2GRAY))
    img_norm = np.zeros((img_height, img_width))
    img_norm = cv2.normalize(img, img_norm, 0, 255, cv2.NORM_MINMAX)
    image = img_norm.reshape(1, img_height, img_width, 1)
    return image

def transform_action(action):
    real_action = np.array(action)
    real_action /= 10 # Scale down due to camera FOV limitation
    real_action[0] += 0.1 if np.random.rand() > 0.5 else -0.1 # At each time step randomise to move left or right
    real_action[1] += 0.1 if np.random.rand() > 0.5 else -0.1 # At each time step randomise to move front or back
    return real_action

def transform_action_forplay(action):
    real_action = np.array(action)
    real_action /= 10
    return real_action

if __name__ == '__main__':
    # CUDA config
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True

    # argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--verbose',    action='store_true')
    parser.add_argument('--load_model', action='store_true')
    parser.add_argument('--play',       action='store_true')
    parser.add_argument('--img_height', type=int,   default=224)
    parser.add_argument('--img_width',  type=int,   default=352)
    parser.add_argument('--actor_lr',   type=float, default=1e-4)
    parser.add_argument('--critic_lr',  type=float, default=5e-4)
    parser.add_argument('--tau',        type=float, default=5e-3)
    parser.add_argument('--gamma',      type=float, default=0.99)
    parser.add_argument('--lambd',      type=float, default=0.90)
    parser.add_argument('--seqsize',    type=int,   default=5)
    parser.add_argument('--epoch',      type=int,   default=1)
    parser.add_argument('--batch_size', type=int,   default=64)
    parser.add_argument('--memory_size',type=int,   default=10000)
    parser.add_argument('--train_start',type=int,   default=1000)
    parser.add_argument('--train_rate', type=int,   default=4)
    parser.add_argument('--epsilon',    type=float, default=1)
    parser.add_argument('--epsilon_end',type=float, default=0.01)
    parser.add_argument('--decay_step', type=int,   default=20000)

    args = parser.parse_args()

    if not os.path.exists('save_graph/'+ agent_name1):
        os.makedirs('save_graph/'+ agent_name1)
    if not os.path.exists('save_graph/'+ agent_name2):
        os.makedirs('save_graph/'+ agent_name2)
    if not os.path.exists('save_graph/'+ agent_name3):
        os.makedirs('save_graph/'+ agent_name3)
    if not os.path.exists('save_stat'):
        os.makedirs('save_stat')
    if not os.path.exists('save_model'):
        os.makedirs('save_model')

    # Make RL agent
    state_size = [args.seqsize, args.img_height, args.img_width, 1]
    action_size = 3
    agent1 = RDDPGAgent(
        state_size=state_size,
        action_size=action_size,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        tau=args.tau,
        gamma=args.gamma,
        lambd=args.lambd,
        batch_size=args.batch_size,
        memory_size=args.memory_size,
        epsilon=args.epsilon,
        epsilon_end=args.epsilon_end,
        decay_step=args.decay_step,
        load_model=args.load_model,
        agent_name=agent_name1
    )
    agent2 = RDDPGAgent(
        state_size=state_size,
        action_size=action_size,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        tau=args.tau,
        gamma=args.gamma,
        lambd=args.lambd,
        batch_size=args.batch_size,
        memory_size=args.memory_size,
        epsilon=args.epsilon,
        epsilon_end=args.epsilon_end,
        decay_step=args.decay_step,
        load_model=args.load_model,
        agent_name=agent_name2
    )
    agent3 = RDDPGAgent(
        state_size=state_size,
        action_size=action_size,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        tau=args.tau,
        gamma=args.gamma,
        lambd=args.lambd,
        batch_size=args.batch_size,
        memory_size=args.memory_size,
        epsilon=args.epsilon,
        epsilon_end=args.epsilon_end,
        decay_step=args.decay_step,
        load_model=args.load_model,
        agent_name=agent_name3
    )

    episode = 0
    env = Env()

    if args.play:
        while True:
            try:
                done = False
                bug = False

                # stats
                bestY, timestep, score, avgvel, avgQ = 0., 0, 0., 0., 0.

                observe = env.reset()
                image, vel = observe
                vel = np.array(vel)
                try:
                    image1 = transform_input(image[0], args.img_height, args.img_width)
                    image2 = transform_input(image[1], args.img_height, args.img_width)
                    image3 = transform_input(image[2], args.img_height, args.img_width)
                except:
                    print('BUG')
                    continue
                history1 = np.stack([image1] * args.seqsize, axis=1)
                history2 = np.stack([image2] * args.seqsize, axis=1)
                history3 = np.stack([image3] * args.seqsize, axis=1)

                vel1 = vel[0].reshape(1, -1)
                vel2 = vel[1].reshape(1, -1)
                vel3 = vel[2].reshape(1, -1)

                state1 = [history1, vel1]
                state2 = [history2, vel2]
                state3 = [history3, vel3]

                while not done:
                    timestep += 1
                    
                    action1 = agent1.actor.predict(state1)[0]
                    action2 = agent2.actor.predict(state2)[0]
                    action3 = agent3.actor.predict(state3)[0]

                    noise = [np.random.normal(scale=args.epsilon) for _ in range(action_size)]
                    noise = np.array(noise, dtype=np.float32)

                    action1 = np.clip(action1 + noise, -1, 1)
                    action2 = np.clip(action2 + noise, -1, 1)
                    action3 = np.clip(action3 + noise, -1, 1)
                    
                    real_action1 = transform_action(action1)
                    real_action2 = transform_action(action2)
                    real_action3 = transform_action(action3)
                    
                    observe, reward, done, info = env.step(transform_action([real_action1, real_action2, real_action3]))
                    image, vel = observe
                    vel = np.array(vel)
                    try:
                        image1 = transform_input(image[0], args.img_height, args.img_width)
                        image2 = transform_input(image[1], args.img_height, args.img_width)
                        image3 = transform_input(image[2], args.img_height, args.img_width)
                    except:
                        print('BUG')
                        bug = True
                        break

                    history1 = np.append(history1[:, 1:], [image], axis=1)
                    history2 = np.append(history2[:, 1:], [image], axis=1)
                    history3 = np.append(history3[:, 1:], [image], axis=1)

                    vel1 = vel[0].reshape(1, -1)
                    vel2 = vel[1].reshape(1, -1)
                    vel3 = vel[2].reshape(1, -1)

                    next_state1 = [history1, vel1]
                    next_state2 = [history2, vel2]
                    next_state3 = [history3, vel3]

                    # stats
                    avgQ += float(agent1.critic.predict([state1[0], state1[1], action1.reshape(1, -1)])[0][0])
                    avgQ += float(agent2.critic.predict([state2[0], state2[1], action2.reshape(1, -1)])[0][0])
                    avgQ += float(agent3.critic.predict([state3[0], state3[1], action3.reshape(1, -1)])[0][0])
                    avgQ /= 3

                    avgvel += float(np.linalg.norm(real_action1))
                    avgvel += float(np.linalg.norm(real_action2))
                    avgvel += float(np.linalg.norm(real_action3))
                    avgvel /= 3

                    score += reward
                    if float(reward) > bestY:
                        bestY = float(reward)
                    print('%s' % (real_action1), end='\r', flush=True)
                    print('%s' % (real_action2), end='\r', flush=True)
                    print('%s' % (real_action3), end='\r', flush=True)

                    if args.verbose:
                        print('Step %d Action1 %s Action2 %s Action3 %s Reward %.2f Info1 %s Info2 %s Info3 %s:' % (timestep, real_action1, real_action2, real_action3, reward, info[0]['status'], info[1]['status'], info[2]['status']))

                    state1 = next_state1
                    state2 = next_state2
                    state3 = next_state3

                if bug:
                    continue

                avgQ /= timestep
                avgvel /= timestep

                # done
                print('Ep %d: BestY %.3f Step %d Score %.2f AvgQ %.2f AvgVel %.2f'
                        % (episode, bestY, timestep, score, avgQ, avgvel))

                episode += 1
            except KeyboardInterrupt:
                env.disconnect()
                break
    else:
        # Train
        time_limit = 600
        highscoreY = -9999999999.
        if os.path.exists('save_stat/'+ agent_name1 + '_stat.csv'):
            with open('save_stat/'+ agent_name1 + '_stat.csv', 'r') as f:
                read = csv.reader(f)
                episode = int(float(next(reversed(list(read)))[0]))
                print('Last episode:', episode)
                episode += 1
        if os.path.exists('save_stat/'+ agent_name2 + '_stat.csv'):
            with open('save_stat/'+ agent_name2 + '_stat.csv', 'r') as f:
                read = csv.reader(f)
                episode = int(float(next(reversed(list(read)))[0]))
                print('Last episode:', episode)
                episode += 1
        if os.path.exists('save_stat/'+ agent_name3 + '_stat.csv'):
            with open('save_stat/'+ agent_name3 + '_stat.csv', 'r') as f:
                read = csv.reader(f)
                episode = int(float(next(reversed(list(read)))[0]))
                print('Last episode:', episode)
                episode += 1

        if os.path.exists('save_stat/'+ agent_name1 + '_highscore.csv'):
            with open('save_stat/'+ agent_name1 + '_highscore.csv', 'r') as f:
                read = csv.reader(f)
                highscoreY = float(next(reversed(list(read)))[0])
                print('Best Y:', highscoreY)
        if os.path.exists('save_stat/'+ agent_name2 + '_highscore.csv'):
            with open('save_stat/'+ agent_name2 + '_highscore.csv', 'r') as f:
                read = csv.reader(f)
                highscoreY = float(next(reversed(list(read)))[0])
                print('Best Y:', highscoreY)
        if os.path.exists('save_stat/'+ agent_name3 + '_highscore.csv'):
            with open('save_stat/'+ agent_name3 + '_highscore.csv', 'r') as f:
                read = csv.reader(f)
                highscoreY = float(next(reversed(list(read)))[0])
                print('Best Y:', highscoreY)

        global_step = 0
        while True:
            try:
                done = False
                bug = False

                # stats
                bestY, timestep, score, avgvel, avgQ, avgAct = 0., 0, 0., 0., 0., 0.
                train_num, actor_loss1, critic_loss1 = 0, 0., 0.
                actor_loss2, critic_loss2 = 0, 0.
                actor_loss3, critic_loss3 = 0, 0.

                observe = env.reset()
                image, vel = observe
                vel = np.array(vel)
                try:
                    image1 = transform_input(image[0], args.img_height, args.img_width)
                    image2 = transform_input(image[1], args.img_height, args.img_width)
                    image3 = transform_input(image[2], args.img_height, args.img_width)
                except:
                    continue
                history1 = np.stack([image1] * args.seqsize, axis=1)
                history2 = np.stack([image2] * args.seqsize, axis=1)
                history3 = np.stack([image3] * args.seqsize, axis=1)

                vel1 = vel[0].reshape(1, -1)
                vel2 = vel[1].reshape(1, -1)
                vel3 = vel[2].reshape(1, -1)

                state1 = [history1, vel1]
                state2 = [history2, vel2]
                state3 = [history3, vel3]

                while not done and timestep < time_limit:
                    timestep += 1
                    global_step += 1
                    if len(agent1.memory) >= args.train_start and global_step >= args.train_rate:
                        if args.verbose:
                            print(f'training models, memory length:{len(agent1.memory)}')
                        for _ in range(args.epoch):
                            a_loss1, c_loss1 = agent1.train_model()
                            actor_loss1 += float(a_loss1)
                            critic_loss1 += float(c_loss1)

                            a_loss2, c_loss2 = agent2.train_model()
                            actor_loss2 += float(a_loss2)
                            critic_loss2 += float(c_loss2)

                            a_loss3, c_loss3 = agent3.train_model()
                            actor_loss3 += float(a_loss3)
                            critic_loss3 += float(c_loss3)

                            train_num += 1
                        if args.verbose:
                            print(f'update target models')
                        agent1.update_target_model()
                        agent2.update_target_model()
                        agent3.update_target_model()
                        global_step = 0
                    action1, policy1 = agent1.get_action(state1)
                    action2, policy2 = agent2.get_action(state2)
                    action3, policy3 = agent3.get_action(state3)
                    real_action1, real_policy1 = transform_action(action1), transform_action(policy1)
                    real_action2, real_policy2 = transform_action(action2), transform_action(policy2)
                    real_action3, real_policy3 = transform_action(action3), transform_action(policy3)
                    observe, reward, done, info = env.step([real_action1,real_action2,real_action3])
                    image, vel = observe
                    vel = np.array(vel)
                    try:
                        image1 = transform_input(image[0], args.img_height, args.img_width)
                        image2 = transform_input(image[1], args.img_height, args.img_width)
                        image3 = transform_input(image[2], args.img_height, args.img_width)
                    except:
                        bug = True
                        break
                    history1 = np.append(history1[:, 1:], [image1], axis=1)
                    history2 = np.append(history2[:, 1:], [image2], axis=1)
                    history3 = np.append(history3[:, 1:], [image3], axis=1)

                    vel1 = vel[0].reshape(1, -1)
                    vel2 = vel[1].reshape(1, -1)
                    vel3 = vel[2].reshape(1, -1)

                    next_state1 = [history1, vel1]
                    next_state2 = [history2, vel2]
                    next_state3 = [history3, vel3]

                    reward = np.sum(np.array(reward))

                    agent1.append_memory(state1, action1, reward, next_state1, done)
                    agent2.append_memory(state2, action2, reward, next_state2, done)
                    agent3.append_memory(state3, action3, reward, next_state3, done)

                    # stats
                    avgQ += float(agent1.critic.predict([state1[0], state1[1], action1.reshape(1, -1)])[0][0])
                    avgQ += float(agent2.critic.predict([state2[0], state2[1], action2.reshape(1, -1)])[0][0])
                    avgQ += float(agent3.critic.predict([state3[0], state3[1], action3.reshape(1, -1)])[0][0])
                    avgQ /= 3
                    avgvel += float(np.linalg.norm(real_policy1))
                    avgvel += float(np.linalg.norm(real_policy2))
                    avgvel += float(np.linalg.norm(real_policy3))
                    avgvel /= 3
                    avgAct += float(np.linalg.norm(real_action1))
                    avgAct += float(np.linalg.norm(real_action2))
                    avgAct += float(np.linalg.norm(real_action3))
                    avgAct /= 3

                    score += reward
                    if float(reward) > bestY:
                        bestY = float(reward)

                    print('%s | %s' % (real_action1, real_policy1), end='\r', flush=True)
                    print('%s | %s' % (real_action2, real_policy2), end='\r', flush=True)
                    print('%s | %s' % (real_action3, real_policy3), end='\r', flush=True)


                    if args.verbose:
                        print('Step %d Action1 %s Reward %.2f Info1 %s:' % (timestep, real_action1, reward, info[0]['status']))
                        print('Step %d Action2 %s Reward %.2f Info2 %s:' % (timestep, real_action2, reward, info[1]['status']))
                        print('Step %d Action3 %s Reward %.2f Info3 %s:' % (timestep, real_action3, reward, info[2]['status']))

                    state1 = next_state1
                    state2 = next_state2
                    state3 = next_state3

                    if agent1.epsilon > agent1.epsilon_end:
                        agent1.epsilon -= agent1.epsilon_decay
                    if agent2.epsilon > agent2.epsilon_end:
                        agent2.epsilon -= agent2.epsilon_decay
                    if agent3.epsilon > agent3.epsilon_end:
                        agent3.epsilon -= agent3.epsilon_decay

                if bug:
                    continue
                if train_num:
                    actor_loss1 /= train_num
                    critic_loss1 /= train_num
                    actor_loss2 /= train_num
                    critic_loss2 /= train_num
                    actor_loss3 /= train_num
                    critic_loss3 /= train_num
                avgQ /= timestep
                avgvel /= timestep
                avgAct /= timestep

                # done
                if args.verbose or episode % 10 == 0:
                    print('Ep %d: BestY %.3f Step %d Score %.2f AvgQ %.2f AvgVel %.2f AvgAct %.2f'
                            % (episode, bestY, timestep, score, avgQ, avgvel, avgAct))
                stats1 = [
                    episode, timestep, score, bestY, avgvel, \
                    actor_loss1, critic_loss1, info[0]['level'], avgQ, avgAct, info[0]['status']
                ]
                stats2 = [
                    episode, timestep, score, bestY, avgvel, \
                    actor_loss2, critic_loss2, info[1]['level'], avgQ, avgAct, info[1]['status']
                ]
                stats3 = [
                    episode, timestep, score, bestY, avgvel, \
                    actor_loss3, critic_loss3, info[2]['level'], avgQ, avgAct, info[2]['status']
                ]
                # log stats
                with open('save_stat/'+ agent_name1 + '_stat.csv', 'a', encoding='utf-8', newline='') as f:
                    wr = csv.writer(f)
                    wr.writerow(['%.4f' % s if type(s) is float else s for s in stats1])
                with open('save_stat/'+ agent_name2 + '_stat.csv', 'a', encoding='utf-8', newline='') as f:
                    wr = csv.writer(f)
                    wr.writerow(['%.4f' % s if type(s) is float else s for s in stats2])
                with open('save_stat/'+ agent_name3 + '_stat.csv', 'a', encoding='utf-8', newline='') as f:
                    wr = csv.writer(f)
                    wr.writerow(['%.4f' % s if type(s) is float else s for s in stats3])

                if highscoreY < score:
                    highscoreY = score
                    with open('save_stat/'+ agent_name1 + '_highscore.csv', 'w', encoding='utf-8', newline='') as f:
                        wr = csv.writer(f)
                        wr.writerow('%.4f' % s if type(s) is float else s for s in [highscoreY, episode, score, dt.now().strftime('%Y-%m-%d %H:%M:%S')])

                    with open('save_stat/'+ agent_name2 + '_highscore.csv', 'w', encoding='utf-8', newline='') as f:
                        wr = csv.writer(f)
                        wr.writerow('%.4f' % s if type(s) is float else s for s in [highscoreY, episode, score, dt.now().strftime('%Y-%m-%d %H:%M:%S')])

                    with open('save_stat/'+ agent_name3 + '_highscore.csv', 'w', encoding='utf-8', newline='') as f:
                        wr = csv.writer(f)
                        wr.writerow('%.4f' % s if type(s) is float else s for s in [highscoreY, episode, score, dt.now().strftime('%Y-%m-%d %H:%M:%S')])
                    agent1.save_model('./save_model/'+ agent_name1 + '_best')
                    agent2.save_model('./save_model/'+ agent_name2 + '_best')
                    agent3.save_model('./save_model/'+ agent_name3 + '_best')

                agent1.save_model('./save_model/'+ agent_name1)
                agent2.save_model('./save_model/'+ agent_name2)
                agent3.save_model('./save_model/'+ agent_name3)
                episode += 1
            except KeyboardInterrupt:
                env.disconnect()
                break
