# Based on: https://github.com/sunghoonhong/AirsimDRL/blob/master/airsim_env.py

import time
import numpy as np
import airsim
import config
import math
from pathlib import Path
from DroneControlAPI import DroneControl
from yolov3_inference import *
from geopy import distance
import os

clockspeed = 1
timeslice = 0.5 / clockspeed
goalY = 57
outY = -0.5
floorZ = 1.18
goals = [7, 17, 27.5, 45, goalY]
speed_limit = 0.2
ACTION = ['00', '+x', '+y', '+z', '-x', '-y', '-z']

droneList = ['Drone1', 'Drone2', 'Drone3', 'DroneTarget']
dir = os.path.abspath(os.getcwd())
yolo_weights = os.path.join(dir, 'weights\\drone.h5')
#print(yolo_weights)
# base_dir = Path(__file__)
# yolo_weights = base_dir/'weights'/'drone.h5'

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
        
        # move drone to initial position
        for drone in droneList[:-1]:
            # print('other loop')
            pos = self.dc.getMultirotorState(drone).kinematics_estimated.position
            self.dc.moveDrone(drone, [pos.x_val, pos.y_val, -2.5], 0.5)
            # adjust drone1, drone2 and drone3 camera angle

        self.dc.setCameraAngle(-30, 'Drone1', 0)
        self.dc.client.simSetCameraOrientation('4', airsim.to_quaternion(-30 * math.pi/180,0,62 * math.pi/180), vehicle_name='Drone2')
        self.dc.client.simSetCameraOrientation('4', airsim.to_quaternion(-30 * math.pi/180,0,-62 * math.pi/180), vehicle_name='Drone3')

        responses = []
        for drone in droneList[:-1]:
            img = self.dc.getImage(drone)
            # cv2.imwrite(f'{drone}.png', img)
            responses.append(self.dc.getImage(drone))

        gps_drone1 = self.dc.getGpsData('Drone1')
        gps_drone2 = self.dc.getGpsData('Drone2')
        gps_drone3 = self.dc.getGpsData('Drone3')
        gps_droneTarget = self.dc.getGpsData('DroneTarget')

        gps_dist = []
        target = (gps_droneTarget.latitude, gps_droneTarget.longitude)
        for x in [gps_drone1, gps_drone2, gps_drone3]:
            source = (x.latitude, x.longitude)
            gps_dist.append(distance.distance(source, target).m)

        # quad_vel = np.array([quad_vel.x_val, quad_vel.y_val, quad_vel.z_val])
        print(gps_dist)
        observation = [responses, gps_dist]
        height = responses[0].shape[0]
        width = responses[0].shape[1]
        channels = responses[0].shape[2]
        print(f"the height is {height}, the width is {width}, the no. of channels is {channels}")
        return observation

    def step(self, quad_offset_list):
        # move with given velocity
        quad_offset = []
        for qoffset in quad_offset_list: # [(xyz),(xyz),(xyz)]
            quad_offset.append([float(i) for i in qoffset])

        quad_vel = []
        for drone in droneList[:-1]:
            quad_vel.append(self.dc.getMultirotorState(drone).kinematics_estimated.linear_velocity)
        #print(f'quad_vel: {quad_vel} \n offset: {quad_offset}')
        self.dc.simPause(False)
        # Target follow fixed path now..
        self.dc.moveDrone("DroneTarget", [1,0,0], 2 * timeslice)

        # Calculations for Drone1
        has_collided = [False, False, False]
        landed = [False, False, False]
        #self.dc.moveDrone('Drone1', [quad_offset[0], quad_offset[1], quad_offset[2]], timeslice)
        for droneidx in range(len(droneList[:-1])):
            print(f'Drone{droneidx}: X:{quad_vel[droneidx].x_val+quad_offset[droneidx][0]}, Y:{quad_vel[droneidx].y_val+quad_offset[droneidx][1]}, Z:{quad_vel[droneidx].z_val+quad_offset[droneidx][2]}')
            self.dc.moveDrone(droneList[droneidx], [quad_offset[droneidx][0], quad_offset[droneidx][1], quad_offset[droneidx][2]], timeslice)
            # self.dc.moveDrone(droneList[droneidx], [quad_vel[droneidx].x_val+quad_offset[droneidx][0], quad_vel[droneidx].y_val+quad_offset[droneidx][1], quad_vel[droneidx].z_val+quad_offset[droneidx][2]], timeslice)

        collision_count = [0, 0, 0]
        start_time = time.time()
        while time.time() - start_time < timeslice:
            # get quadrotor states
            quad_pos = []
            quad_vel = []
            for drone in droneList[:-1]:
                quad_pos.append(self.dc.getMultirotorState(drone).kinematics_estimated.position)
                quad_vel.append(self.dc.getMultirotorState(drone).kinematics_estimated.linear_velocity)

            # decide whether collision occured
            collided = [False, False, False]
            landed = [False, False, False]
            for droneidx in range(len(droneList[:-1])):
                collided[droneidx] = self.dc.simGetCollisionInfo(drone).has_collided
                land = (quad_vel[droneidx].x_val == 0 and quad_vel[droneidx].y_val == 0 and quad_vel[droneidx].z_val == 0)
                landed[droneidx] = land or quad_pos[droneidx].z_val > floorZ
            #print(f'collided var: {collided}, landed var: {landed}')
            has_collided = [False, False, False]
            for droneidx in range(len(droneList[:-1])):
                collision = collided[droneidx] or landed[droneidx]
                if collision:
                    collision_count[droneidx] += 1
                if collision_count[droneidx] > 10:
                    has_collided[droneidx] = True
                    break
        self.dc.simPause(True)

        # observe with depth camera
        #responses = self.client.simGetImages([airsim.ImageRequest(1, airsim.ImageType.DepthVis, True)])
        responses = []
        for drone in droneList[:-1]:
            responses.append(self.dc.getImage(drone))

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
        for droneidx in range(len(droneList[:-1])):
            quad_pos[droneidx] = self.dc.getMultirotorState(droneList[droneidx]).kinematics_estimated.position
            quad_vel[droneidx] = self.dc.getMultirotorState(droneList[droneidx]).kinematics_estimated.linear_velocity

        # decide whether done
        done = False
        for droneidx in range(len(droneList[:-1])):
            print(f'Drone{droneidx}: collided? {has_collided[droneidx]}, outY? {quad_pos[droneidx].y_val <= outY}')
            dead = has_collided[droneidx] or quad_pos[droneidx].y_val <= outY
            # done[droneidx] = dead
            if dead:
                done = True
            print(f'Drone{droneidx}: dead? {dead}')

        # compute reward
        # reward = self.compute_reward(responses, quad_pos, quad_vel, dead)
        reward = self.compute_reward(responses, gps_dist, done)
        # reward=-1

        # log info
        loginfo = []
        for droneidx in range(len(droneList[:-1])):
            info = {}
            info['Y'] = quad_pos[droneidx].y_val
            info['level'] = self.level
            if landed[droneidx]:
                info['status'] = 'landed'
            elif has_collided[droneidx]:
                info['status'] = 'collision'
            elif quad_pos[droneidx].y_val <= outY:
                info['status'] = 'out'
            elif quad_pos[droneidx].y_val >= goalY:
                info['status'] = 'goal'
            else:
                info['status'] = 'going'
            loginfo.append(loginfo)
            # quad_vel.append(np.array([quad_vel[droneidx].x_val, quad_vel[droneidx].y_val, quad_vel[droneidx].z_val]))
            # observation = [responses, gps_dist, quad_vel]
            observation = [responses, gps_dist]
        # observation = [responses[D1,D2,D3], gps_dist[D1,D2,D3]]
        # reward = [?]
        # done = [D1,D2,D3]
        # loginfo = [D1{'Y','level','status'},D2{'Y','level','status'},D3{'Y','level','status'}]
        # print(f'done : {done} and all {all(done)}')
        return observation, reward, done, loginfo

    # def compute_reward(self, responses, quad_pos, quad_vel, dead):
    def compute_reward(self, responses, gps_dist, dead):
        reward = [None] * len(droneList[:-1])
        for droneidx in range(len(droneList[:-1])):
            # Calculate image rewards
            img = responses[droneidx]
            try:
                bbox = self.infer_model.get_yolo_boxes(img[:,:,:3])
            except:
                bbox = BoundBox(xmin=0, xmax=0, ymin=0, ymax=0)
            img_status = self.calculate_bbox_zone(bbox, img)

            if dead or img_status == 'dead':
                reward[droneidx] = config.reward['dead']
            # elif quad_pos.y_val >= goals[self.level]:
            #     self.level += 1
            #     # reward = config.reward['forward'] * (1 + self.level / len(goals))
            #     reward = config.reward['goal'] * (1 + self.level / len(goals))
            # elif speed < speed_limit:
            elif img_status == 'slow':
                reward[droneidx] = config.reward['slow']
            elif img_status == 'normal':
                reward[droneidx] = config.reward['normal']
            else:
                reward = [None] * len(droneList[:-1])
                # reward = float(vel[1]) * 0.1
                reward[droneidx] = config.reward['forward']
                
            gps = gps_dist[droneidx]
            if gps > 9 or gps < 2.3:
                reward[droneidx] = reward[droneidx] + config.reward['dead']
            else:
                reward[droneidx] = reward[droneidx] + config.reward['forward']
            # elif vel[1] > 0:
            #     reward = config.reward['forward'] * (1 + self.level / len(goals))
            # else:
            #     reward = config.reward['normal']
        return reward
    
    def _calculate_zone_param(self, img_size):
        # All box in [Xmin, Xmax, Ymin, Ymax]
        img_h, img_w = img_size[0], img_size[1]
        img_cen_x = img_w / 2
        img_cen_y = img_h / 2

        # Inner (forward -> 20% of whole image fron the center)
        forward_bbox = {
            'xmin': img_cen_x - (img_w * 0.2 / 2), # Xmin
            'xmax': img_cen_x + (img_w * 0.2 / 2), # Xmax
            'ymin': img_cen_y - (img_h * 0.2 / 2), # Ymin
            'ymax': img_cen_y + (img_h * 0.2 / 2)  # Ymax
        }
        # Inner (slow 40%)
        slow_bbox = {
            'xmin': img_cen_x - (img_w * 0.4 / 2), # Xmin
            'xmax': img_cen_x + (img_w * 0.4 / 2), # Xmax
            'ymin': img_cen_y - (img_h * 0.4 / 2), # Ymin
            'ymax': img_cen_y + (img_h * 0.4 / 2)  # Ymax
        }
        # Inner (normal 60%)
        normal_bbox = {
            'xmin': img_cen_x - (img_w * 0.6 / 2), # Xmin
            'xmax': img_cen_x + (img_w * 0.6 / 2), # Xmax
            'ymin': img_cen_y - (img_h * 0.6 / 2), # Ymin
            'ymax': img_cen_y + (img_h * 0.6 / 2)  # Ymax
        }

        # Outer (dead 80%)
        dead_bbox = {
            'xmin': img_cen_x - (img_w * 0.8 / 2), # Xmin
            'xmax': img_cen_x + (img_w * 0.8 / 2), # Xmax
            'ymin': img_cen_y - (img_h * 0.8 / 2), # Ymin
            'ymax': img_cen_y + (img_h * 0.8 / 2)  # Ymax
        }
        return forward_bbox, slow_bbox, normal_bbox, dead_bbox
    
    def calculate_bbox_zone(self, bbox, img):
        # Get all tier zone
        fbbox, sbbox, nbbox, dbbox = self._calculate_zone_param([img.shape[0], img.shape[1]])
        # Get bbox dimension and find center point
        xmin, xmax, ymin, ymax = bbox.xmin, bbox.xmax, bbox.ymin, bbox.ymax
        bcen_x = xmin + ((xmax - xmin) / 2)
        bcen_y = ymin + ((ymax - xmin) / 2)
        
        # Check if bbox center point is in dead zone
        if (bcen_x < dbbox['xmin'] or bcen_x > dbbox['xmax']) and (bcen_y < dbbox['ymin'] or bcen_y > dbbox['ymax']):
            status = 'dead'
        # Check if bbox center point is in normal zone
        elif ((bcen_x > dbbox['xmin'] and bcen_x < nbbox['xmin']) or (bcen_x < dbbox['xmax'] and bcen_x > nbbox['xmax'])) and \
            ((bcen_y > dbbox['ymin'] and bcen_y < nbbox['ymin']) or (bcen_y < dbbox['ymax'] and bcen_y > nbbox['ymax'])):
            status = 'normal'
        # Check if bbox center point is in slow zone
        elif ((bcen_x > nbbox['xmin'] and bcen_x < sbbox['xmin']) or (bcen_x > nbbox['xmax'] and bcen_x > sbbox['xmax'])) and \
            ((bcen_y > nbbox['ymin'] and bcen_y < sbbox['ymin']) or (bcen_y < nbbox['ymax'] and bcen_y > sbbox['ymax'])):
            status = 'slow'
        # Bbox center point is in center forward zone
        else:
            status = 'forward'
        return status
        
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