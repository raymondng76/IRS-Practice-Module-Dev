"""
Date: 1/2/2020
Team: Kenneth Goh (A0198544N) Raymond Ng (A0198543R) Wong Yoke Keong (A0195365U)

Intelligent Robotic Systems Practice Module
"""

import airsim
import time
import math
import cv2
import numpy as np

class DroneControl:
    def __init__(self, droneList):
        #self.client = airsim.MultirotorClient()
        self.client = airsim.MultirotorClient('127.0.0.1') #for connection
        self.client.confirmConnection()
        self.droneList = droneList
        self.init_AirSim()
    
    def init_AirSim(self):
        """
        Method to initialize AirSim for a list of drones
        """
        for drone in self.droneList:
            self.client.enableApiControl(True, drone)
            self.client.armDisarm(True, drone)
    
    def shutdown_AirSim(self):
        """
        Method to un-init all drones and quit AirSim
        """
        self.armDisarm(False)
        self.client.reset()
        self.enableApiControl(False)
    
    def resetAndRearm_Drones(self):
        """
        Method to reset all drones to original starting state and rearm
        """
        #self.armDisarm(False)
        self.client.reset()
        #self.enableApiControl(False)
        time.sleep(0.25)
        self.enableApiControl(True)
        self.armDisarm(True)

    def armDisarm(self, status):
        """
        Method to arm or disarm all drones
        """
        for drone in self.droneList:
            self.client.armDisarm(status, drone)
    
    def enableApiControl(self, status):
        """
        Method to enable or disable drones API control
        """
        for drone in self.droneList:
            self.client.enableApiControl(status, drone)

    def takeOff(self):
        """
        Method to take off for all drones
        """
        dronesClient = []
        for drone in self.droneList:
            cli_drone = self.client.takeoffAsync(vehicle_name=drone)
            dronesClient.append(cli_drone)
        for drone in dronesClient:
            drone.join()

    def getMultirotorState(self, drone):
        """
        Method to get current drone states
        """
        if drone in self.droneList:
            return self.client.getMultirotorState(vehicle_name=drone)
        else:
            print('Drone does not exists!')
    
    def getBarometerData(self, barometer, drone):
        """
        Method to get barometer data
        """
        if drone in self.droneList:
            return self.client.getBarometerData(barometer_name=barometer, vehicle_name=drone)
        else:
            print('Drone does not exists!')
    
    def getImuData(self, imu, drone):
        """
        Method to get imu data
        """
        if drone in self.droneList:
            return self.client.getImuData(imu_name=imu, vehicle_name=drone)
        else:
            print('Drone does not exists!')
    
    def getGpsData(self, drone):
        """
        Method to get gps data
        Returns GeoPoint object containing altitude, latitude and longitude
        """
        if drone in self.droneList:
            #return self.client.getGpsData(gps_name=gps, vehicle_name=drone)
            return self.client.getMultirotorState(vehicle_name=drone).gps_location
        else:
            print('Drone does not exists!')
    
    def getMagnetometerData(self, mag, drone):
        """
        Method to get Magnetometer data
        """
        if drone in self.droneList:
            return self.client.getMagnetometerData(magnetometer_name=mag, vehicle_name=drone)
        else:
            print('Drone does not exists!')
    
    def getDistanceData(self, lidar, drone):
        """
        Method to get Distance data
        """
        if drone in self.droneList:
            return self.client.getDistanceSensorData(lidar_name=lidar, vehicle_name=drone)
        else:
            print('Drone does not exists!')

    def getLidarData(self, lidar, drone):
        """
        Method to get lidar data
        """
        if drone in self.droneList:
            return self.client.getLidarData(lidar_name=lidar, vehicle_name=drone)
        else:
            print('Drone does not exists!')
    
    def getDronePos(self, drone):
        """
        Method to get X, Y, Z axis values of drone
        """
        if drone in self.droneList:
            x_val = self.client.simGetGroundTruthKinematics(vehicle_name=drone).position.x_val
            y_val = self.client.simGetGroundTruthKinematics(vehicle_name=drone).position.y_val
            z_val = self.client.simGetGroundTruthKinematics(vehicle_name=drone).position.z_val
            return np.array([x_val, y_val, z_val])
        else:
            print('Drone does not exists!')
    
    def moveDrone(self, drone, position, duration):
        """
        Method to move drone to indicated position
        pos = [x_val, y_val, z_val]
        """
        if drone in self.droneList:
            self.client.moveByVelocityZAsync(vehicle_name=drone, 
                                             vx=position[0], 
                                             vy=position[1], 
                                             z=position[2],
                                             duration=duration).join()
        else:
            print('Drone does not exists!')
            
    def simPause(self,pause):
        """
        Pass-through method to pause simulation
        """
        self.client.simPause(pause)
        
    def simGetCollisionInfo(self, drone):
        """
        Pass-through method to get collision info
        """
        return self.client.simGetCollisionInfo(drone)
    
    def hoverAsync(self, drone):
        """
        Pass-through method for hoverAsync
        """
        return self.client.hoverAsync(drone)

    def setCameraHeading(self, camera_heading, drone):
        """
        Set camera orientation
        """
        pos = self.getMultirotorState(drone).kinematics_estimated.position
        self.client.moveByVelocityZAsync(pos.x_val, pos.y_val, pos.z_val, 1, airsim.DrivetrainType.MaxDegreeOfFreedom, airsim.YawMode(False, camera_heading), vehicle_name=drone)
    
    def setCameraAngle(self, camera_angle, drone):
        """
        Set camera angle
        """
        pos = self.client.simSetCameraOrientation(0, airsim.to_quaternion(
            camera_angle * math.pi / 180, 0, 0),vehicle_name=drone)  # radians
    
    def getImage(self, drone):
        """
        Get image for single drone
        """
        raw_img = self.client.simGetImage("0", airsim.ImageType.Scene, vehicle_name=drone)
        return cv2.imdecode(airsim.string_to_uint8_array(raw_img), cv2.IMREAD_UNCHANGED)