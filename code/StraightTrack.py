# Based on: https://github.com/sunghoonhong/AirsimDRL/blob/master/randomly.py
# Test Script that only moves forward

import time
import csv
import math
import argparse
import numpy as np
from airsim_env import Env

class StraightMultiAgentDiscrete(object):
    def __init__(self, action_size):
        self.action_size = action_size

    def get_action(self):
        # Always go straight during testing for all 3 drones
        actions = [1,1,1]
        return actions
    
def interpret_action(action_list):
    scaling_factor = 0.5
    quad_offset_list = []
    for action in action_list:
        if action == 0:
            quad_offset_list.append((0, 0, 0))
        elif action == 1:
            quad_offset_list.append((scaling_factor, 0, 0))
        elif action == 2:
            quad_offset_list.append((0, scaling_factor, 0))
        elif action == 3:
            quad_offset_list.append((0, 0, scaling_factor))
        elif action == 4:
            quad_offset_list.append((-scaling_factor, 0, 0))
        elif action == 5:
            quad_offset_list.append((0, -scaling_factor, 0))
        elif action == 6:
            quad_offset_list.append((0, 0, -scaling_factor))
    
    return quad_offset_list

agent = StraightMultiAgentDiscrete(1)

env = Env()
episode = 0
while True:
    done = False
    timestep = 0
    score = 0
    _ = env.reset()
    
    print('env reset!')
    
    while not done:
        print('start episode')
        timestep += 1
        action = agent.get_action()
        action = interpret_action(action)
        observation, reward, done, info = env.step(action)
        
        responses, gps_dist = observation
        print(reward)
        score += sum(reward)

        # stack history here
        # print('Step %d Action %s Reward %.2f Info %s:' % (timestep, action, reward, info))
        # print(f'gps dist from target for drones 1,2,3: {gps_dist}')
    # done
    print('Ep %d: Step %d Score %.3f' % (episode, timestep, score))
    episode += 1
    
#env.disconnect()