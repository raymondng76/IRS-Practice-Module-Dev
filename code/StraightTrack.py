# Based on: https://github.com/sunghoonhong/AirsimDRL/blob/master/randomly.py
# Test Script that only moves forward

import time
import csv
import math
import argparse
import numpy as np
from airsim_env import Env

class StraightAgentDiscrete(object):
    def __init__(self, action_size):
        self.action_size = action_size

    def get_action(self):
        action = 1 # Always go straight for testing
        return action
    
def interpret_action(action):
    scaling_factor = 0.5
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

agent = StraightAgentDiscrete(1)

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
        
        responses, gps_dist, = observation
        
        score += reward

        # stack history here
        print('Step %d Action %s Reward %.2f Info %s:' % (timestep, action, reward, info))
        print(f'gps dist from target for drones 1,2,3: {gps_dist}')
    # done
    print('Ep %d: Step %d Score %.3f' % (episode, timestep, score))
    episode += 1
    
#env.disconnect()