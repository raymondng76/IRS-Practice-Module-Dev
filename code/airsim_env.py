# Based on: https://github.com/sunghoonhong/AirsimDRL/blob/master/airsim_env.py

import time
import numpy as np
import airsim
import config

from DroneControlAPI import DroneControl
from yolov3_inference import *
from geopy import distance


clockspeed = 1
timeslice = 0.5 / clockspeed
goalY = 57
outY = -0.5
floorZ = 1.18
goals = [7, 17, 27.5, 45, goalY]
speed_limit = 0.2
ACTION = ['00', '+x', '+y', '+z', '-x', '-y', '-z']

droneList = ['Drone1', 'Drone2', 'Drone3', 'DroneTarget']
yolo_weights = 'weights\drone.h5'

class Env:
    def __init__(self):
        # connect to the AirSim simulator
        self.dc = DroneControl(droneList)
        # Load the inference model
        self.infer_model = YoloPredictor(yolo_weights)
        self.action_size = 3
        self.level = 0

    def reset(self):
        self.level = 0
        self.dc.resetAndRearm_Drones()

        # all drones takeoff
        self.dc.simPause(False)
        for drone in droneList:
            print(f'{drone} taking off...')
            self.dc.moveDrone(drone, [0,0,-1], 2 * timeslice)
            self.dc.moveDrone(drone, [0,0,0], 0.1 * timeslice)
            self.dc.hoverAsync(drone).join()
        
        # calibrate drone2 and drone3 camera heading
        self.dc.setCameraHeading(-100, droneList[1])
        self.dc.setCameraHeading(100, droneList[2])

        self.dc.simPause(True)
        quad_vel = self.dc.getMultirotorState("Drone1").kinematics_estimated.linear_velocity
        #responses = self.client.simGetImages([airsim.ImageRequest(1, airsim.ImageType.DepthVis, True)])
        responses = []
        quad_vel = np.array([quad_vel.x_val, quad_vel.y_val, quad_vel.z_val])
        observation = [responses, quad_vel]
        return observation

    def step(self, quad_offset):
        # move with given velocity
        quad_offset = [float(i) for i in quad_offset]
        quad_vel = self.dc.getMultirotorState("Drone1").kinematics_estimated.linear_velocity
        #print(f'quad_vel: {quad_vel} \n offset: {quad_offset}')
        self.dc.simPause(False)
        # Target follow fixed path now..
        self.dc.moveDrone("DroneTarget", [1,0,0], 2 * timeslice)

        # Calculations for Drone1
        has_collided = False
        landed = False
        #self.dc.moveDrone('Drone1', [quad_offset[0], quad_offset[1], quad_offset[2]], timeslice)
        self.dc.moveDrone('Drone1', [quad_vel.x_val+quad_offset[0], quad_vel.y_val+quad_offset[1], quad_vel.z_val+quad_offset[2]], timeslice)
        collision_count = 0
        start_time = time.time()
        while time.time() - start_time < timeslice:
            # get quadrotor states
            quad_pos = self.dc.getMultirotorState("Drone1").kinematics_estimated.position
            quad_vel = self.dc.getMultirotorState("Drone1").kinematics_estimated.linear_velocity

            # decide whether collision occured
            collided = self.dc.simGetCollisionInfo("Drone1").has_collided
            landed = (quad_vel.x_val == 0 and quad_vel.y_val == 0 and quad_vel.z_val == 0)
            landed = landed or quad_pos.z_val > floorZ
            #print(f'collided var: {collided}, landed var: {landed}')
            collision = collided or landed
            if collision:
                collision_count += 1
            if collision_count > 10:
                has_collided = True
                break
        self.dc.simPause(True)

        # observe with depth camera
        #responses = self.client.simGetImages([airsim.ImageRequest(1, airsim.ImageType.DepthVis, True)])
        drone1_img = self.dc.getImage('Drone1')
        drone2_img = self.dc.getImage('Drone2')
        drone3_img = self.dc.getImage('Drone3')

        # responses = []
        # get distance between follower and target
        gps_drone1 = self.dc.getGpsData('Drone1')
        gps_drone2 = self.dc.getGpsData('Drone2')
        gps_drone3 = self.dc.getGpsData('Drone3')
        gps_droneTarget = self.dc.getGpsData('DroneTarget')
        
        gps_dist = []
        target = (gps_droneTarget.latitude, gps_droneTarget.longitude)
        for x in [gps_drone1, gps_drone2, gps_drone3]:
            source = (x.latitude, x.longitude)
            gps_dist.append(distance.distance(source, target).m)
        
        # get quadrotor states
        quad_pos = self.dc.getMultirotorState("Drone1").kinematics_estimated.position
        quad_vel = self.dc.getMultirotorState("Drone1").kinematics_estimated.linear_velocity

        # decide whether done
        dead = has_collided or quad_pos.y_val <= outY
        done = dead or quad_pos.y_val >= goalY

        # compute reward
        #reward = self.compute_reward(quad_pos, quad_vel, dead)
        reward=-1

        # log info
        info = {}
        info['Y'] = quad_pos.y_val
        info['level'] = self.level
        if landed:
            info['status'] = 'landed'
        elif has_collided:
            info['status'] = 'collision'
        elif quad_pos.y_val <= outY:
            info['status'] = 'out'
        elif quad_pos.y_val >= goalY:
            info['status'] = 'goal'
        else:
            info['status'] = 'going'
        quad_vel = np.array([quad_vel.x_val, quad_vel.y_val, quad_vel.z_val])
        observation = [responses, gps_dist, quad_vel]
        return observation, reward, done, info

    def compute_reward(self, quad_pos, quad_vel, dead):
        vel = np.array([quad_vel.x_val, quad_vel.y_val, quad_vel.z_val], dtype=np.float)
        speed = np.linalg.norm(vel)
        if dead:
            reward = config.reward['dead']
        elif quad_pos.y_val >= goals[self.level]:
            self.level += 1
            # reward = config.reward['forward'] * (1 + self.level / len(goals))
            reward = config.reward['goal'] * (1 + self.level / len(goals))
        elif speed < speed_limit:
            reward = config.reward['slow']
        else:
            reward = float(vel[1]) * 0.1
        # elif vel[1] > 0:
        #     reward = config.reward['forward'] * (1 + self.level / len(goals))
        # else:
        #     reward = config.reward['normal']
        return reward
    
    
    def lineseg_dists(self, p, a, b):
        # Ref: https://stackoverflow.com/questions/54442057/calculate-the-euclidian-distance-between-an-array-of-points-to-a-line-segment-in/54442561#54442561
        # Gets distance between point p and line ab
        
        if np.all(a - b):
            return np.linalg.norm(p - a, axis=1)

        # normalized tangent vector
        d = np.divide(b - a, np.linalg.norm(b - a))

        # signed parallel distance components
        s = np.dot(a - p, d)
        t = np.dot(p - b, d)

        # clamped parallel distance
        h = np.maximum.reduce([s, t, np.zeros(len(p))])

        # perpendicular distance component, as before
        # note that for the 3D case these will be vectors
        c = np.cross(p - a, d)

        # use hypot for Pythagoras to improve accuracy
        return np.hypot(h, c)
    
    def disconnect(self):
        #self.client.enableApiControl(False)
        #self.client.armDisarm(False)
        self.dc.shutdown_AirSim()
        print('Disconnected.')