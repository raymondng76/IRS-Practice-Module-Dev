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
        
        # move drone to initial position
        for drone in droneList[:-1]:
            pos = dc.getMultirotorState(drone).kinematics_estimated.position
            self.dc.moveDrone(drone, [pos.x_val, pos.y_val, -0.8], 0.5)
            # adjust drone1, drone2 and drone3 camera angle
            self.setCameraAngle(-15, drone)

        
        # calibrate drone2 and drone3 camera heading
        self.dc.setCameraHeading(-125, droneList[1])
        self.dc.setCameraHeading(125, droneList[2])

        self.dc.simPause(True)
        # quad_vel = self.dc.getMultirotorState("Drone1").kinematics_estimated.linear_velocity
        # responses = self.client.simGetImages([airsim.ImageRequest(1, airsim.ImageType.DepthVis, True)])

        responses = []
        for drone in droneList[:-1]:
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
        observation = [responses, gps_dist]
        return observation

    def step(self, quad_offset_list):
        # move with given velocity
        for quad_offset in quad_offset_list:
            quad_offset = [float(i) for i in quad_offset]

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
            self.dc.moveDrone(drone, [quad_vel[droneidx].x_val+quad_offset[0], quad_vel[droneidx].y_val+quad_offset[1], quad_vel[droneidx].z_val+quad_offset[2]], timeslice)

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
                if collision_count > 10:
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
        done = [False, False, False]
        for droneidx in range(len(droneList[:-1])):
            dead = has_collided[droneidx] or quad_pos[droneidx].y_val <= outY
            done[droneidx] = dead or quad_pos[droneidx].y_val >= goalY

        # compute reward
        #reward = self.compute_reward(responses, quad_pos, quad_vel, dead)
        reward=-1

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
        return observation, reward, done, loginfo

    def compute_reward(self, responses, quad_pos, quad_vel, dead):
        reward = []
        for droneidx in range(len(droneList[:-1])):
            # Calculate image rewards
            # Image size : height=224, width=352, centerpoint=(112,176)
            # Inner red zone (Dead) : 10% from center = YMin=(112-(112*0.1))=101, YMax(112+(112*0.1))=123, XMin=(176-(176*0.1))=159, XMax=(176+(176*0.1))=193
            # Next Inner red zone (Normal) : 20% from center = YMin(112-(112*0.2))=90, YMax(112+(112*0.2))=134, XMin(176-(176*0.2))=141, XMax=(176+(176*0.2))=211
            # Next Inner red zone (Slow) : 30% from center = YMin(112-(112*0.3))=79, YMax(112+(112*0.3))=145, XMin(176-(176*0.3))=123, XMax=(176+(176*0.3))=229
            # Calculate outer, use center points as percentage reference to maintain same ratio
            # Outer red zone (Dead) : 10% from image edge = YMin(0+(112*0.1))=11, YMax(224-(112*0.1))=213, XMin(0+(176*0.1))=18, XMax(352-(172*0.1))=334
            # Next outer red zone (Normal) : 20% from image edge = YMin(0+(112*0.2))=22, YMax(224-(112*0.2))=202, XMin(0+(176*0.2))=36, XMax(352-(172*0.2))=316
            # Next outer red zone (Slow) : 30% from image edge = YMin(0+(112*0.3))=33, YMax(224-(112*0.3))=191, XMin(0+(176*0.3))=54, XMax(352-(172*0.3))=298
            # Else (forward)

            img = responses[droneidx]
            img_h, img_w = img.shape[0], img.shape[1]
            bbox = self.infer_model.predict(img)

            # vel = np.array([quad_vel.x_val, quad_vel.y_val, quad_vel.z_val], dtype=np.float)
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
    
    def _calculate_zone_param(self, param_type, percent, center_value, zone_type, edge_value=None):
        if zone_type == 'inner':
            edge_value = center_value
        val = 0
        if param_type == 'XMin' or param_type == 'YMin':
            val = edge_value + (int(center_value[0] * percent))
        elif param_type == 'XMax' or param_type == 'YMax':
            val = edge_value - (int(center_value[1] * percent))
        else:
            raise ValueError('Unknown param_type')
        return val
    
    def calculate_bbox_zone(self, bbox, img):
        img_h, img_w = img.shape[0], img.shape[1]
        center_x = bbox.xmin + ((bbox.xmax - bbox.xmin)/2)
        center_y = bbox.ymin + ((bbox.ymax - bbox.ymin)/2)
        zone = ['inner', 'outer']
        tol = [0.1, 0.2, 0.3]
        reward_type = ['dead', 'normal', 'slow']
        status = 'forward'
        # Check inner
        for z in zone:
            for t in range(len(tol)):
                if z == 'inner':
                    xmin = self._calculate_zone_param('XMin', t, [center_x, center_y], z)
                    xmax = self._calculate_zone_param('XMax', t, [center_x, center_y], z)
                    ymin = self._calculate_zone_param('YMin', t, [center_x, center_y], z)
                    ymax = self._calculate_zone_param('YMax', t, [center_x, center_y], z)


        
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