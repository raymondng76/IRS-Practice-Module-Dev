"""
Date: 1/2/2020
Team: Kenneth Goh (A0198544N) Raymond Ng (A0198543R) Wong Yoke Keong (A0195365U)

Intelligent Robotic Systems Practice Module
"""

import airsim
import time
import numpy as np

class DroneControl:
    def __init__(self, droneList):
        self.client = airsim.MultirotorClient()
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
    
    def getGpsData(self, gps, drone):
        """
        Method to get gps data
        """
        if drone in self.droneList:
            return self.client.getGpsData(gps_name=gps, vehicle_name=drone)
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