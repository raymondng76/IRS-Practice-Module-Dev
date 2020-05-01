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
agent_name1 = 'rdqn1'
agent_name2 = 'rdqn2'
agent_name3 = 'rdqn3'

class RDQNAgent(object):
    
    def __init__(self, state_size, action_size, lr,
                gamma, batch_size, memory_size, 
                epsilon, epsilon_end, decay_step, load_model, agent_name):
        self.state_size = state_size
        self.vel_size = 1
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
        Qvalue = Dense(128, kernel_initializer='he_normal', use_bias=False)(state_process)
        Qvalue = BatchNormalization()(Qvalue)
        Qvalue = ELU()(Qvalue)
        Qvalue = Dense(128, kernel_initializer='he_normal', use_bias=False)(Qvalue)
        Qvalue = BatchNormalization()(Qvalue)
        Qvalue = ELU()(Qvalue)
        Qvalue = Dense(self.action_size, kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))(Qvalue)
        critic = Model(inputs=[image, vel], outputs=Qvalue)

        critic._make_predict_function()
        
        return critic

    def build_critic_optimizer(self):
        action = K.placeholder(shape=(None, ), dtype='int32')
        y = K.placeholder(shape=(None, ), dtype='float32')
        pred = self.critic.output
        
        # loss = K.mean(K.square(pred - y))
        # Huber Loss
        action_vec = K.one_hot(action, self.action_size)
        Q = K.sum(pred * action_vec, axis=1)
        error = K.abs(y - Q)
        quadratic = K.clip(error, 0.0, 1.0)
        linear = error - quadratic
        loss = K.mean(0.5 * K.square(quadratic) + linear)

        optimizer = Adam(lr=self.lr)
        updates = optimizer.get_updates(self.critic.trainable_weights, [], loss)
        train = K.function(
            [self.critic.input[0], self.critic.input[1], action, y],
            [loss],
            updates=updates
        )
        return train

    def get_action(self, state):
        Qs = self.critic.predict(state)[0]
        Qmax = np.amax(Qs)
        if np.random.random() < self.epsilon:
            return np.random.choice(self.action_size), np.argmax(Qs), Qmax
        return np.argmax(Qs), np.argmax(Qs), Qmax

    def train_model(self):
        batch = random.sample(self.memory, self.batch_size)

        images = np.zeros([self.batch_size] + self.state_size)
        vels = np.zeros([self.batch_size, self.vel_size])
        actions = np.zeros((self.batch_size))
        rewards = np.zeros((self.batch_size))
        next_images = np.zeros([self.batch_size] + self.state_size)
        next_vels = np.zeros([self.batch_size, self.vel_size])
        dones = np.zeros((self.batch_size))

        targets = np.zeros((self.batch_size, 1))
        
        for i, sample in enumerate(batch):
            images[i], vels[i] = sample[0]
            actions[i] = sample[1]
            rewards[i] = sample[2]
            next_images[i], next_vels[i] = sample[3]
            dones[i] = sample[4]
        states = [images, vels]
        next_states = [next_images, next_vels]
        target_next_Qs = self.target_critic.predict(next_states)
        targets = rewards + self.gamma * (1 - dones) * np.amax(target_next_Qs, axis=1)
        critic_loss = self.critic_update(states + [actions, targets])
        return critic_loss[0]

    def append_memory(self, state, action, reward, next_state, done):        
        self.memory.append((state, action, reward, next_state, done))
        
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
    img = np.array(cv2.cvtColor(responses[:,:,:3], cv2.COLOR_BGR2GRAY))
    img_norm = np.zeros((img_height, img_width))
    img_norm = cv2.normalize(img, img_norm, 0, 255, cv2.NORM_MINMAX)
    image = img_norm.reshape(1, img_height, img_width, 1)
    return image

def interpret_action(action):
    scaling_factor = 0.01
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
    parser.add_argument('--memory_size',type=int,   default=10000)
    parser.add_argument('--train_start',type=int,   default=1000)
    parser.add_argument('--train_rate', type=int,   default=5)
    parser.add_argument('--target_rate',type=int,   default=1000)
    parser.add_argument('--epsilon',    type=float, default=1)
    parser.add_argument('--epsilon_end',type=float, default=0.05)
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
    action_size = 7
    agent1 = RDQNAgent(
        state_size=state_size,
        action_size=action_size,
        lr=args.lr,
        gamma=args.gamma,
        batch_size=args.batch_size,
        memory_size=args.memory_size,
        epsilon=args.epsilon,
        epsilon_end=args.epsilon_end,
        decay_step=args.decay_step,
        load_model=args.load_model,
        agent_name=agent_name1
    )
    agent2 = RDQNAgent(
        state_size=state_size,
        action_size=action_size,
        lr=args.lr,
        gamma=args.gamma,
        batch_size=args.batch_size,
        memory_size=args.memory_size,
        epsilon=args.epsilon,
        epsilon_end=args.epsilon_end,
        decay_step=args.decay_step,
        load_model=args.load_model,
        agent_name=agent_name2
    )
    agent3 = RDQNAgent(
        state_size=state_size,
        action_size=action_size,
        lr=args.lr,
        gamma=args.gamma,
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
                bestReward, timestep, score, avgQ = 0., 0, 0., 0.

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

                while not done:
                    timestep += 1

                    Qs1 = agent1.critic.predict(state1)[0]
                    Qs2 = agent2.critic.predict(state2)[0]
                    Qs3 = agent3.critic.predict(state3)[0]

                    action1 = np.argmax(Qs1)
                    action2 = np.argmax(Qs2)
                    action3 = np.argmax(Qs3)

                    Qmax1 = np.amax(Qs1)
                    Qmax2 = np.amax(Qs2)
                    Qmax3 = np.amax(Qs3)

                    real_action1 = interpret_action(action1)
                    real_action2 = interpret_action(action2)
                    real_action3 = interpret_action(action3)

                    observe, reward, done, info = env.step([real_action1, real_action2, real_action3])
                    image, vel = observe
                    vel = np.array(vel)
                    info1, info2, info3 = info[0]['status'], info[1]['status'], info[2]['status']

                    try:
                        image1 = transform_input(image[0], args.img_height, args.img_width)
                        image2 = transform_input(image[1], args.img_height, args.img_width)
                        image3 = transform_input(image[2], args.img_height, args.img_width)
                    except:
                        print('BUG')
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
                    # stats
                    totalQmax = Qmax1 + Qmax2 + Qmax3
                    avgQ += float(totalQmax)
                    score += float(reward)
                    if float(reward) > bestReward:
                        bestReward = float(reward)
                    print('%s' % (ACTION[action1]), end='\r', flush=True)
                    print('%s' % (ACTION[action2]), end='\r', flush=True)
                    print('%s' % (ACTION[action3]), end='\r', flush=True)

                    if args.verbose:
                        print('Step %d Action1 %s Action2 %s Action3 %s Reward %.2f Info1 %s Info2 %s Info3 %s:' % (timestep, real_action1, real_action2, real_action3, reward, info1, info2, info3))

                    state1 = next_state1
                    state2 = next_state2
                    state3 = next_state3

                if bug:
                    continue
                
                avgQ /= timestep

                # done
                print('Ep %d: BestReward %.3f Step %d Score %.2f AvgQ %.2f Info1 %s Info2 %s Info3 %s'
                        % (episode, bestReward, timestep, score, avgQ, info1, info2, info3))

                episode += 1
            except KeyboardInterrupt:
                env.disconnect()
                break
    else:
        # Train
        time_limit = 999
        highscore = -9999999999.
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
                highscore = float(next(reversed(list(read)))[0])
                print('Best Y:', highscore)
        if os.path.exists('save_stat/'+ agent_name2 + '_highscore.csv'):
            with open('save_stat/'+ agent_name2 + '_highscore.csv', 'r') as f:
                read = csv.reader(f)
                highscore = float(next(reversed(list(read)))[0])
                print('Best Y:', highscore)
        if os.path.exists('save_stat/'+ agent_name3 + '_highscore.csv'):
            with open('save_stat/'+ agent_name3 + '_highscore.csv', 'r') as f:
                read = csv.reader(f)
                highscore = float(next(reversed(list(read)))[0])
                print('Best Y:', highscore)

        global_step = 0
        global_train_num = 0
        while True:
            try:
                done = False
                bug = False

                # stats
                bestReward, timestep, score, avgQ = 0., 0, 0., 0.
                train_num, loss1, loss2, loss3 = 0, 0., 0., 0.

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
                print(f'Main Loop: done: {done}, timestep: {timestep}, time_limit: {time_limit}')
                while not done and timestep < time_limit:
                    print(f'Sub Loop: timestep: {timestep}, global_step: {global_step}')
                    timestep += 1
                    global_step += 1
                    if len(agent1.memory) >= args.train_start and global_step >= args.train_rate:
                        print('Training model')
                        for _ in range(args.epoch):
                            c_loss1 = agent1.train_model()
                            loss1 += float(c_loss1)
                            
                            c_loss2 = agent2.train_model()
                            loss2 += float(c_loss2)

                            c_loss3 = agent3.train_model()
                            loss3 += float(c_loss3)

                            train_num += 1
                            global_train_num += 1
                        global_step = 0
                    if global_train_num >= args.target_rate:
                        print('Updating target model')
                        agent1.update_target_model()
                        agent2.update_target_model()
                        agent3.update_target_model()

                        global_train_num = 0

                    action1, policy1, Qmax1 = agent1.get_action(state1)
                    action2, policy2, Qmax2 = agent2.get_action(state2)
                    action3, policy3, Qmax3 = agent3.get_action(state3)

                    real_action1 = interpret_action(action1)
                    real_action2 = interpret_action(action2)
                    real_action3 = interpret_action(action3)

                    observe, reward, done, info = env.step([real_action1, real_action2, real_action3])
                    image, vel = observe
                    vel = np.array(vel)
                    info1, info2, info3 = info[0]['status'], info[1]['status'], info[2]['status']

                    try:
                        if timestep < 3 and info1 == 'landed' and info2 == 'landed' and info3 == 'landed':
                            raise Exception
                        image1 = transform_input(image[0], args.img_height, args.img_width)
                        image2 = transform_input(image[1], args.img_height, args.img_width)
                        image3 = transform_input(image[2], args.img_height, args.img_width)
                    except:
                        print('BUG')
                        bug = True
                        break

                    history1 = np.stack([image1] * args.seqsize, axis=1)
                    history2 = np.stack([image2] * args.seqsize, axis=1)
                    history3 = np.stack([image3] * args.seqsize, axis=1)

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
                    totalQmax = Qmax1 + Qmax2 + Qmax3
                    avgQ += float(totalQmax)
                    score += float(reward)
                    if float(reward) > bestReward:
                        bestReward = float(reward)

                    print('%s | %s' % (ACTION[action1], ACTION[policy1]), end='\r', flush=True)
                    print('%s | %s' % (ACTION[action2], ACTION[policy2]), end='\r', flush=True)
                    print('%s | %s' % (ACTION[action3], ACTION[policy3]), end='\r', flush=True)


                    if args.verbose:
                        print('Step %d Action1 %s Action2 %s Action3 %s Reward %.2f Info1 %s Info2 %s Info3 %s:' % (timestep, real_action1, real_action2, real_action3, reward, info1, info2, info3))

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
                    loss1 /= train_num
                    loss2 /= train_num
                    loss3 /= train_num
                avgQ /= timestep

                # done
                if args.verbose or episode % 10 == 0:
                    print('Ep %d: BestReward %.3f Step %d Score %.2f AvgQ %.2f'
                            % (episode, bestReward, timestep, score, avgQ))
                stats1 = [
                    episode, timestep, score, bestReward, \
                    loss1, info[0]['level'], avgQ, info[0]['status']
                ]
                stats2 = [
                    episode, timestep, score, bestReward, \
                    loss2, info[1]['level'], avgQ, info[1]['status']
                ]
                stats3 = [
                    episode, timestep, score, bestReward, \
                    loss3, info[2]['level'], avgQ, info[2]['status']
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

                if highscore < score:
                    highscore = score
                    with open('save_stat/'+ agent_name1 + '_highscore.csv', 'w', encoding='utf-8', newline='') as f:
                        wr = csv.writer(f)
                        wr.writerow('%.4f' % s if type(s) is float else s for s in [highscore, episode, score, dt.now().strftime('%Y-%m-%d %H:%M:%S')])
                    with open('save_stat/'+ agent_name2 + '_highscore.csv', 'w', encoding='utf-8', newline='') as f:
                        wr = csv.writer(f)
                        wr.writerow('%.4f' % s if type(s) is float else s for s in [highscore, episode, score, dt.now().strftime('%Y-%m-%d %H:%M:%S')])
                    with open('save_stat/'+ agent_name3 + '_highscore.csv', 'w', encoding='utf-8', newline='') as f:
                        wr = csv.writer(f)
                        wr.writerow('%.4f' % s if type(s) is float else s for s in [highscore, episode, score, dt.now().strftime('%Y-%m-%d %H:%M:%S')])

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