# Based on: https://github.com/sunghoonhong/AirsimDRL/blob/master/airsim_env.py
"""
Date: 1/2/2020
Team: Kenneth Goh (A0198544N) Raymond Ng (A0198543R) Wong Yoke Keong (A0195365U)

Intelligent Robotic Systems Practice Module
"""
import time
import numpy as np
import airsim
import config
import math
from pathlib import Path
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
targetdrone_espsilon = 0.35

droneList = ['Drone1', 'Drone2', 'Drone3', 'DroneTarget']
base_dir = Path('..')
yolo_weights = base_dir/'weights'/'drone.h5'

class Env:
    def __init__(self):
        # connect to the AirSim simulator
        self.dc = DroneControl(droneList)
        # Load the inference model
        self.infer_model = YoloPredictor(yolo_weights)
        self.action_size = 3
        self.level = 0

    def reset(self):
        '''
        Method to reset AirSim env to starting position
        '''
        self.level = 0
        self.dc.resetAndRearm_Drones()

        # all drones takeoff
        self.dc.simPause(False)
        for drone in droneList:
            print(f'{drone} taking off...')
            self.dc.moveDrone(drone, [0,0,-1], 2 * timeslice)
            self.dc.moveDrone(drone, [0,0,0], 0.1 * timeslice)
            self.dc.hoverAsync(drone).join()

        # # move drone to initial position
        for drone in droneList[:-1]:
            pos = self.dc.getMultirotorState(drone).kinematics_estimated.position
            self.dc.client.moveByVelocityZAsync(vx=pos.x_val, vy=pos.y_val, z=-1.0, duration=0.5, vehicle_name=drone)

        # adjust drone1, drone2 and drone3 camera angle
        self.dc.setCameraAngle(-30, droneList[0], 0)
        self.dc.client.simSetCameraOrientation('4', airsim.to_quaternion(-30 * math.pi/180,0,62 * math.pi/180), vehicle_name=droneList[1])
        self.dc.client.simSetCameraOrientation('4', airsim.to_quaternion(-30 * math.pi/180,0,-62 * math.pi/180), vehicle_name=droneList[2])
        time.sleep(1)

        # Drone1, Drone2, Drone3 take image
        responses = []
        for drone in droneList[:-1]:
            cam = '0' if drone == droneList[0] else '4'
            img = self.dc.getImage(drone, cam)
            # cv2.imwrite(f'reset_{drone}.png', cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            responses.append(self.dc.getImage(drone))

        # All drone measure GPS distance
        gps_drone1 = self.dc.getGpsData(droneList[0])
        gps_drone2 = self.dc.getGpsData(droneList[1])
        gps_drone3 = self.dc.getGpsData(droneList[2])
        gps_droneTarget = self.dc.getGpsData(droneList[3])

        gps_dist = []
        target = (gps_droneTarget.latitude, gps_droneTarget.longitude)
        for x in [gps_drone1, gps_drone2, gps_drone3]:
            source = (x.latitude, x.longitude)
            gps_dist.append(distance.distance(source, target).m)

        observation = [responses, gps_dist]
        return observation

    def step(self, quad_offset_list):
        # move with given velocity
        quad_offset = []
        for qoffset in quad_offset_list: # [(xyz),(xyz),(xyz)]
            quad_offset.append([float(i) for i in qoffset])

        quad_vel = []
        for drone in droneList[:-1]:
            quad_vel.append(self.dc.getMultirotorState(drone).kinematics_estimated.linear_velocity)

        self.dc.simPause(False)
        
        # Target drone takes random step in X or Y axis for testing purposes
        # if np.random.random() > targetdrone_espsilon: # Higher chance to move in X axis
        #     if np.random.random() > 0.5:
        #         self.dc.moveDrone(droneList[3], [0.1,0,0], 2 * timeslice)
        #     else:
        #         self.dc.moveDrone(droneList[3], [-0.1,0,0], 2 * timeslice)
        # else:
        #     if np.random.random() > 0.5:
        #         self.dc.moveDrone(droneList[3], [0,0.1,0], 2 * timeslice)
        #     else:
        #         self.dc.moveDrone(droneList[3], [0,-0.1,0], 2 * timeslice)

        # Target drone takes small step in X axis
        self.dc.moveDrone(droneList[3], [0.01,0,0], 2 * timeslice)

        # All follower drones take next move
        has_collided = [False, False, False]
        landed = [False, False, False]
        
        for droneidx in range(len(droneList[:-1])):
            self.dc.moveDrone(droneList[droneidx], [quad_offset[droneidx][0], quad_offset[droneidx][1], quad_offset[droneidx][2]], 2* timeslice)

        # Get follower drones position and linear velocity        
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
            
            has_collided = [False, False, False]
            for droneidx in range(len(droneList[:-1])):
                collision = collided[droneidx] or landed[droneidx]
                if collision:
                    collision_count[droneidx] += 1
                if collision_count[droneidx] > 10:
                    has_collided[droneidx] = True
                    break

        self.dc.simPause(True)
        time.sleep(1)

        # All follower drones take image
        responses = []
        for drone in droneList[:-1]:
            cam = '0' if drone == droneList[0] else '4'
            img = self.dc.getImage(drone, cam)
            # cv2.imwrite(f'step_{drone}.png', cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            responses.append(img)

        # get distance between follower and target
        gps_drone1 = self.dc.getGpsData(droneList[0])
        gps_drone2 = self.dc.getGpsData(droneList[1])
        gps_drone3 = self.dc.getGpsData(droneList[2])
        gps_droneTarget = self.dc.getGpsData(droneList[3])
        
        gps_dist = []
        target = (gps_droneTarget.latitude, gps_droneTarget.longitude)
        for x in [gps_drone1, gps_drone2, gps_drone3]:
            source = (x.latitude, x.longitude)
            gps_dist.append(distance.distance(source, target).m)
        
        # get quadrotor states
        for droneidx in range(len(droneList[:-1])):
            quad_pos[droneidx] = self.dc.getMultirotorState(droneList[droneidx]).kinematics_estimated.position
            quad_vel[droneidx] = self.dc.getMultirotorState(droneList[droneidx]).kinematics_estimated.linear_velocity

        # Get each follower drone image reward
        img_reward = {}
        for droneidx in range(len(droneList[:-1])):
            img = responses[droneidx]
            try:
                bbox = self.infer_model.get_yolo_boxes(img[:,:,:3])
                img_status = self.calculate_bbox_zone(bbox, img)
                img_reward[droneidx] = img_status
            except:
                bbox = BoundBox(xmin=0, xmax=0, ymin=0, ymax=0)
                img_status = 'dead'
                img_reward[droneidx] = img_status

            print(f'Drone[{droneidx}] is [{img_status}]')

            # Save each drone image with YOLOv3 bounding box for debugging purposes only
            test_img = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2RGB)
            cv2.rectangle(test_img, (bbox.xmin, bbox.ymin), (bbox.xmax, bbox.ymax), (0,0,255), 2)
            cv2.imwrite(f'Reward_{droneList[droneidx]}.png', test_img)

        # decide whether done
        done = False
        done = has_collided[droneidx] or self.gps_out_bounds(gps_dist) or any(status == 'dead' for status in img_reward.values())      

        # compute reward
        reward = self.compute_reward(responses, gps_dist, img_reward, done)

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
            elif img_reward[droneidx] == 'normal':
                info['status'] = 'forward'
            elif img_reward[droneidx] == 'slow':
                info['status'] = 'slow'    
            elif img_reward[droneidx] == 'dead' or self.gps_out_bounds(gps_dist):
                info['status'] = 'dead'
            else:
                info['status'] = 'going'
            loginfo.append(info)
            observation = [responses, gps_dist]

        # observation = [responses[D1,D2,D3], gps_dist[D1,D2,D3]]
        # reward = [R]
        # done = [D1,D2,D3]
        # loginfo = [D1{'Y','level','status'},D2{'Y','level','status'},D3{'Y','level','status'}]
        # print(f'done : {done} and all {all(done)}')
        return observation, reward, done, loginfo

    def gps_out_bounds(self, gpslist):
        stats = False
        for gps in gpslist:
            if gps > 10 or gps < 1.5:
                stats = True
                break
        return stats

    def compute_reward(self, responses, gps_dist, image_reward, dead):
        reward = [None] * len(droneList[:-1])
        for droneidx in range(len(droneList[:-1])):           
            img = responses[droneidx]
            img_status = image_reward[droneidx]
            
            # Assign reward value based on status
            if dead or img_status == 'dead':
                reward[droneidx] = config.reward['dead']
            elif img_status == 'slow':
                reward[droneidx] = config.reward['slow']
            elif img_status == 'normal':
                reward[droneidx] = config.reward['normal']
            else:
                reward[droneidx] = config.reward['forward']

            # Append GPS rewards
            if img_status != 'dead':            
                gps = gps_dist[droneidx]
                if gps > 9 or gps < 2.3:
                    reward[droneidx] = reward[droneidx] + config.reward['dead']
                else:
                    reward[droneidx] = reward[droneidx] + config.reward['forward']
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