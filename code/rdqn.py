# Based on: https://github.com/sunghoonhong/AirsimDRL
"""
Date: 1/2/2020
Team: Kenneth Goh (A0198544N) Raymond Ng (A0198543R) Wong Yoke Keong (A0195365U)

Intelligent Robotic Systems Practice Module
"""

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
from keras.layers import TimeDistributed, BatchNormalization, Flatten, Lambda, Concatenate
from keras.layers import Conv2D, MaxPooling2D, Dense, GRU, Input, ELU, Activation
from keras.optimizers import Adam
from keras.models import Model
from PIL import Image
import cv2
from airsim_env import Env, ACTION

np.set_printoptions(suppress=True, precision=4)
agent_name = 'rdqn'


class RDQNAgent(object):
    
    def __init__(self, state_size, action_size, lr,
                gamma, batch_size, memory_size, 
                epsilon, epsilon_end, decay_step, load_model):
        self.state_size = state_size
        self.vel_size = 3
        self.action_size = action_size
        self.lr = lr
        self.gamma = gamma
        self.batch_size = batch_size
        self.memory_size = memory_size
        self.epsilon = epsilon
        self.epsilon_end = epsilon_end
        self.decay_step = decay_step
        self.epsilon_decay = (epsilon - epsilon_end) / decay_step

        self.sess = tf.Session()
        K.set_session(self.sess)

        self.critic = self.build_model()
        self.target_critic = self.build_model()
        self.critic_update = self.build_critic_optimizer()
        self.sess.run(tf.global_variables_initializer())
        if load_model:
            self.load_model('./save_model/'+ agent_name)
        
        self.target_critic.set_weights(self.critic.get_weights())

        self.memory = deque(maxlen=self.memory_size)

    def build_model(self):
        # image process
        image = Input(shape=self.state_size)
        image_process = BatchNormalization()(image)
        image_process = TimeDistributed(Conv2D(32, (8, 8), activation='elu', padding='same', kernel_initializer='he_normal'))(image_process)
        image_process = TimeDistributed(MaxPooling2D((2, 2)))(image_process)
        image_process = TimeDistributed(Conv2D(32, (5, 5), activation='elu', kernel_initializer='he_normal'))(image_process)
        image_process = TimeDistributed(MaxPooling2D((2, 2)))(image_process)
        image_process = TimeDistributed(Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal'))(image_process)
        image_process = TimeDistributed(MaxPooling2D((2, 2)))(image_process)
        image_process = TimeDistributed(Conv2D(8, (1, 1), activation='elu', kernel_initializer='he_normal'))(image_process)
        image_process = TimeDistributed(Flatten())(image_process)
        image_process = GRU(64, kernel_initializer='he_normal', use_bias=False)(image_process)
        image_process = BatchNormalization()(image_process)
        image_process = Activation('tanh')(image_process)
        
        # vel process
        vel = Input(shape=[self.vel_size])
        vel_process = Dense(6, kernel_initializer='he_normal', use_bias=False)(vel)
        vel_process = BatchNormalization()(vel_process)
        vel_process = Activation('tanh')(vel_process)

        # state process
        state_process = Concatenate()([image_process, vel_process])

        # Critic
        Qvalue1 = Dense(128, kernel_initializer='he_normal', use_bias=False)(state_process)
        Qvalue1 = BatchNormalization()(Qvalue1)
        Qvalue1 = ELU()(Qvalue1)
        Qvalue1 = Dense(128, kernel_initializer='he_normal', use_bias=False)(Qvalue1)
        Qvalue1 = BatchNormalization()(Qvalue1)
        Qvalue1 = ELU()(Qvalue1)
        Qvalue1 = Dense(self.action_size, kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))(Qvalue1)

        Qvalue2 = Dense(128, kernel_initializer='he_normal', use_bias=False)(state_process)
        Qvalue2 = BatchNormalization()(Qvalue2)
        Qvalue2 = ELU()(Qvalue2)
        Qvalue2 = Dense(128, kernel_initializer='he_normal', use_bias=False)(Qvalue2)
        Qvalue2 = BatchNormalization()(Qvalue2)
        Qvalue2 = ELU()(Qvalue2)
        Qvalue2 = Dense(self.action_size, kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))(Qvalue2)

        Qvalue3 = Dense(128, kernel_initializer='he_normal', use_bias=False)(state_process)
        Qvalue3 = BatchNormalization()(Qvalue3)
        Qvalue3 = ELU()(Qvalue3)
        Qvalue3 = Dense(128, kernel_initializer='he_normal', use_bias=False)(Qvalue3)
        Qvalue3 = BatchNormalization()(Qvalue3)
        Qvalue3 = ELU()(Qvalue3)
        Qvalue3 = Dense(self.action_size, kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))(Qvalue3)

        critic = Model(inputs=[image, vel], outputs=[Qvalue1, Qvalue2, Qvalue3])

        critic._make_predict_function()
        
        return critic

    def build_critic_optimizer(self):
        action1 = K.placeholder(shape=(None, ), dtype='int32')
        action2 = K.placeholder(shape=(None, ), dtype='int32')
        action3 = K.placeholder(shape=(None, ), dtype='int32')
        y1 = K.placeholder(shape=(None, ), dtype='float32')
        y2 = K.placeholder(shape=(None, ), dtype='float32')
        y3 = K.placeholder(shape=(None, ), dtype='float32')

        pred1, pred2, pred3 = self.critic.output
        
        # loss = K.mean(K.square(pred - y))
        # Huber Loss
        action_vec1 = K.one_hot(action1, self.action_size)
        action_vec2 = K.one_hot(action2, self.action_size)
        action_vec3 = K.one_hot(action3, self.action_size)

        Q1 = K.sum(pred1 * action_vec1, axis=1)
        Q2 = K.sum(pred2 * action_vec2, axis=1)
        Q3 = K.sum(pred3 * action_vec3, axis=1)

        error1 = K.abs(y1 - Q1)
        error2 = K.abs(y2 - Q2)
        error3 = K.abs(y3 - Q3)

        quadratic1 = K.clip(error1, 0.0, 1.0)
        quadratic2 = K.clip(error2, 0.0, 1.0)
        quadratic3 = K.clip(error3, 0.0, 1.0)

        linear1 = error1 - quadratic1
        linear2 = error2 - quadratic2
        linear3 = error3 - quadratic3

        preloss1 = K.mean(0.5 * K.square(quadratic1) + linear1)
        preloss2 = K.mean(0.5 * K.square(quadratic2) + linear2)
        preloss3 = K.mean(0.5 * K.square(quadratic3) + linear3)

        concatpreloss = tf.stack([preloss1, preloss2, preloss3], axis=0)
        loss = K.mean(concatpreloss)

        optimizer = Adam(lr=self.lr)
        updates = optimizer.get_updates(self.critic.trainable_weights, [], loss)
        train = K.function(
            [self.critic.input[0], self.critic.input[1], action1, action2, action3, y1, y2, y3],
            [loss],
            updates=updates
        )
        return train

    def get_action(self, state):
        Qs1, Qs2, Qs3 = self.critic.predict(state)
        Qmax1 = np.amax(Qs1)
        Qmax2 = np.amax(Qs2)
        Qmax3 = np.amax(Qs3)
        if np.random.random() < self.epsilon:
            return [(np.random.choice(self.action_size), np.argmax(Qs1), Qmax1),(np.random.choice(self.action_size), np.argmax(Qs2), Qmax2),(np.random.choice(self.action_size), np.argmax(Qs3), Qmax3)]
        return [(np.argmax(Qs1), np.argmax(Qs1), Qmax1), (np.argmax(Qs2), np.argmax(Qs2), Qmax2), (np.argmax(Qs3), np.argmax(Qs3), Qmax3)]

    def train_model(self):
        print(f'lem mem: {len(self.memory)}, batch: {self.batch_size}')
        batch = random.sample(self.memory, self.batch_size)

        images = np.zeros([self.batch_size] + self.state_size)
        vels = np.zeros([self.batch_size, self.vel_size])

        actions1 = np.zeros((self.batch_size))
        actions2 = np.zeros((self.batch_size))
        actions3 = np.zeros((self.batch_size))

        rewards = np.zeros((self.batch_size))

        next_images = np.zeros([self.batch_size] + self.state_size)
        next_vels = np.zeros([self.batch_size, self.vel_size])

        dones = np.zeros((self.batch_size))

        targets = np.zeros((self.batch_size, 1))
        for i, sample in enumerate(batch):
            images[i], vels[i] = sample[0]
            actions1[i] = sample[1]
            actions2[i] = sample[2]
            actions3[i] = sample[3]
            rewards[i] = sample[4]
            next_images[i], next_vels[i] = sample[5]
            dones[i] = sample[6]
        states = [images, vels]
        next_states = [next_images, next_vels]
        target_next_Qs = self.target_critic.predict(next_states)
        targets1 = rewards + self.gamma * (1 - dones) * np.amax(target_next_Qs[0], axis=1)
        targets2 = rewards + self.gamma * (1 - dones) * np.amax(target_next_Qs[1], axis=1)
        targets3 = rewards + self.gamma * (1 - dones) * np.amax(target_next_Qs[2], axis=1)
        critic_loss = self.critic_update(states + [actions1, actions2, actions3, targets1, targets2, targets3])
        return critic_loss[0]

    def append_memory(self, state, action1, action2, action3, reward, next_state, done):        
        self.memory.append((state, action1, action2, action3, reward, next_state, done))
        
    def load_model(self, name):
        if os.path.exists(name + '.h5'):
            self.critic.load_weights(name + '.h5')
            print('Model loaded')

    def save_model(self, name):
        self.critic.save_weights(name + '.h5')

    def update_target_model(self):
        self.target_critic.set_weights(self.critic.get_weights())

'''
Environment interaction
'''

def transform_input(responses, img_height, img_width):
    d1img = np.array(cv2.cvtColor(responses[0][:,:,:3], cv2.COLOR_BGR2GRAY))
    d2img = np.array(cv2.cvtColor(responses[1][:,:,:3], cv2.COLOR_BGR2GRAY))
    d3img = np.array(cv2.cvtColor(responses[2][:,:,:3], cv2.COLOR_BGR2GRAY))
    d1norm = np.zeros((img_height, img_width))
    d2norm = np.zeros((img_height, img_width))
    d3norm = np.zeros((img_height, img_width))
    d1norm = cv2.normalize(d1img, d1norm, 0, 255, cv2.NORM_MINMAX)
    d2norm = cv2.normalize(d2img, d2norm, 0, 255, cv2.NORM_MINMAX)
    d3norm = cv2.normalize(d3img, d3norm, 0, 255, cv2.NORM_MINMAX)
    dimg = np.array([d1norm, d2norm, d3norm])
    image = dimg.reshape(1, img_height, img_width, 3)
    return image

def interpret_action(action):
    scaling_factor = 0.1
    if action == 0:
        quad_offset = (0, 0, 0)
    elif action == 1:
        quad_offset = (scaling_factor, 0, 0)
    elif action == 2:
        quad_offset = (0, scaling_factor, 0)
    elif action == 3:
        quad_offset = (0, 0, scaling_factor)
    elif action == 4:
        quad_offset = (-scaling_factor, 0, 0)    
    elif action == 5:
        quad_offset = (0, -scaling_factor, 0)
    elif action == 6:
        quad_offset = (0, 0, -scaling_factor)
    
    return quad_offset

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
    parser.add_argument('--lr',         type=float, default=1e-4)
    parser.add_argument('--gamma',      type=float, default=0.99)
    parser.add_argument('--seqsize',    type=int,   default=5)
    parser.add_argument('--epoch',      type=int,   default=5)
    parser.add_argument('--batch_size', type=int,   default=32)
    parser.add_argument('--memory_size',type=int,   default=50000)
    parser.add_argument('--train_start',type=int,   default=5000)
    parser.add_argument('--train_rate', type=int,   default=5)
    parser.add_argument('--target_rate',type=int,   default=1000)
    parser.add_argument('--epsilon',    type=float, default=1)
    parser.add_argument('--epsilon_end',type=float, default=0.05)
    parser.add_argument('--decay_step', type=int,   default=20000)

    args = parser.parse_args()

    if not os.path.exists('save_graph/'+ agent_name):
        os.makedirs('save_graph/'+ agent_name)
    if not os.path.exists('save_stat'):
        os.makedirs('save_stat')
    if not os.path.exists('save_model'):
        os.makedirs('save_model')

    # Make RL agent
    state_size = [args.seqsize, args.img_height, args.img_width, 3]
    action_size = 7
    agent = RDQNAgent(
        state_size=state_size,
        action_size=action_size,
        lr=args.lr,
        gamma=args.gamma,
        batch_size=args.batch_size,
        memory_size=args.memory_size,
        epsilon=args.epsilon,
        epsilon_end=args.epsilon_end,
        decay_step=args.decay_step,
        load_model=args.load_model
    )  

    episode = 0
    env = Env()
    if args.play:
        while True:
            try:
                done = False
                bug = False

                # stats
                bestY, timestep, score, avgQ = 0., 0, 0., 0.
                observe = env.reset()
                image, vel = observe
                vel = np.array(vel)
                try:
                    image = transform_input(image, args.img_height, args.img_width)
                except:
                    continue
                history = np.stack([image] * args.seqsize, axis=1)
                vel = vel.reshape(1, -1)
                state = [history, vel]
                while not done:
                    timestep += 1
                    
                    Qs1, Qs2, Qs3 = agent.critic.predict(state)
                    action1, action2, action3 = np.argmax(Qs1), np.argmax(Qs2), np.argmax(Qs3)
                    Qmax1, Qmax2, Qmax3 = np.amax(Qs1), np.amax(Qs2), np.amax(Qs3)
                    real_action1, real_action2, real_action3 = interpret_action(action1), interpret_action(action2), interpret_action(action3)
                    observe, reward, done, info = env.step([real_action1, real_action2, real_action3])
                    image, vel = observe
                    vel = np.array(vel)
                    try:
                        image = transform_input(image, args.img_height, args.img_width)
                    except:
                        print('BUG')
                        bug = True
                        break
                    history = np.append(history[:, 1:], [image], axis=1)
                    vel = vel.reshape(1, -1)
                    next_state = [history, vel]
                    reward = np.sum(np.array(reward))
                    info1, info2, info3 = info[0]['status'], info[1]['status'], info[2]['status']

                    # stats
                    avgQ += float(Qmax1 + Qmax2 + Qmax3)
                    score += float(reward)
                    if timestep > bestY:
                        bestY = timestep
                    print('%s' % (ACTION[action1]), end='\r', flush=True)
                    print('%s' % (ACTION[action2]), end='\r', flush=True)
                    print('%s' % (ACTION[action3]), end='\r', flush=True)

                    if args.verbose:
                        print('Step %d Action1 %s Action2 %s Action3 %s Reward %.2f Info1 %s Info2 %s Info3 %s:' % (timestep, real_action1, real_action2, real_action3, reward, info1, info2, info3))

                    state = next_state

                if bug:
                    continue
                
                avgQ /= timestep

                # done
                print('Ep %d: BestY %.3f Step %d Score %.2f AvgQ %.2f Info1 %s Info2 %s Info3 %s'
                        % (episode, bestY, timestep, score, avgQ, info1, info2, info3))

                episode += 1
            except KeyboardInterrupt:
                env.disconnect()
                break
    else:
        # Train
        time_limit = 600
        highscoreY = 0.
        if os.path.exists('save_stat/'+ agent_name + '_stat.csv'):
            with open('save_stat/'+ agent_name + '_stat.csv', 'r') as f:
                read = csv.reader(f)
                episode = int(float(next(reversed(list(read)))[0]))
                print('Last episode:', episode)
                episode += 1
        if os.path.exists('save_stat/'+ agent_name + '_highscore.csv'):
            with open('save_stat/'+ agent_name + '_highscore.csv', 'r') as f:
                read = csv.reader(f)
                highscoreY = float(next(reversed(list(read)))[0])
                print('Best Y:', highscoreY)
        global_step = 0
        global_train_num = 0
        while True:
            try:
                done = False
                bug = False

                # stats
                bestY, timestep, score, avgQ = 0., 0, 0., 0.
                train_num, loss = 0, 0.
                
                observe = env.reset()
                image, vel = observe
                vel = np.array(vel)
                try:
                    image = transform_input(image, args.img_height, args.img_width)
                except:
                    continue
                history = np.stack([image] * args.seqsize, axis=1)                
                vel = vel.reshape(1, -1)
                state = [history, vel]
                print(f'Main Loop: done: {done}, timestep: {timestep}, time_limit: {time_limit}')
                while not done and timestep < time_limit:
                    print(f'Sub Loop: timestep: {timestep}, global_step: {global_step}')
                    timestep += 1
                    global_step += 1
                    if len(agent.memory) >= args.train_start and global_step >= args.train_rate:
                        for _ in range(args.epoch):
                            c_loss = agent.train_model()
                            loss += float(c_loss)
                            train_num += 1
                            global_train_num += 1
                        global_step = 0
                    if global_train_num >= args.target_rate:
                        agent.update_target_model()
                        global_train_num = 0

                    (action1, policy1, Qmax1), (action2, policy2, Qmax2), (action3, policy3, Qmax3) = agent.get_action(state)
                    real_action1, real_action2, real_action3 = interpret_action(action1), interpret_action(action2), interpret_action(action3)
                    observe, reward, done, info = env.step([real_action1,real_action2,real_action3])
                    image, vel = observe
                    vel = np.array(vel)
                    info1, info2, info3 = info[0]['status'], info[1]['status'], info[2]['status']

                    try:
                        if timestep < 3 and info[0]['status'] == 'landed' and info[1]['status'] == 'landed' and info[2]['status'] == 'landed':
                            raise Exception
                        image = transform_input(image, args.img_height, args.img_width)
                    except:
                        print('BUG')
                        bug = True
                        break
                    history = np.append(history[:, 1:], [image], axis=1)
                    vel = vel.reshape(1, -1)
                    next_state = [history, vel]
                    reward = np.sum(np.array(reward))
                    agent.append_memory(state, action1, action2, action3, reward, next_state, done)

                    # stats
                    avgQ += float(Qmax1 + Qmax2 + Qmax3)
                    score += float(reward)
                    if float(reward) > bestY:
                        bestY = float(reward)

                    print('%s | %s' % (ACTION[action1], ACTION[policy1]), end='\r', flush=True)
                    print('%s | %s' % (ACTION[action2], ACTION[policy2]), end='\r', flush=True)
                    print('%s | %s' % (ACTION[action3], ACTION[policy3]), end='\r', flush=True)

                    if args.verbose:
                        print('Step %d Action1 %s Action2 %s Action3 %s Reward %.2f Info1 %s Info2 %s Info3 %s:' % (timestep, real_action1, real_action2, real_action3, reward, info1, info2, info3))

                    state = next_state

                    if agent.epsilon > agent.epsilon_end:
                        agent.epsilon -= agent.epsilon_decay

                if bug:
                    continue
                if train_num:
                    loss /= train_num
                avgQ /= timestep

                # done
                if args.verbose or episode % 10 == 0:
                    print('Ep %d: BestY %.3f Step %d Score %.2f AvgQ %.2f'
                            % (episode, bestY, timestep, score, avgQ))
                stats = [
                    episode, timestep, score, bestY, \
                    loss, info[0]['level'], info[1]['level'], info[2]['level'], avgQ, info[0]['status'], info[1]['status'], info[2]['status']
                ]
                # log stats
                with open('save_stat/'+ agent_name + '_stat.csv', 'a', encoding='utf-8', newline='') as f:
                    wr = csv.writer(f)
                    wr.writerow(['%.4f' % s if type(s) is float else s for s in stats])
                if highscoreY < bestY:
                    highscoreY = bestY
                    with open('save_stat/'+ agent_name + '_highscore.csv', 'w', encoding='utf-8', newline='') as f:
                        wr = csv.writer(f)
                        wr.writerow('%.4f' % s if type(s) is float else s for s in [highscoreY, episode, score, dt.now().strftime('%Y-%m-%d %H:%M:%S')])
                    agent.save_model('./save_model/'+ agent_name + '_best')
                agent.save_model('./save_model/'+ agent_name)
                episode += 1
            except KeyboardInterrupt:
                env.disconnect()
                break
            except Exception as e:
                print(f'{e}')
                break