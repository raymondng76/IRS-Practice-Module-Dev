"""
Date: 1/2/2020
Team: Kenneth Goh (A0198544N) Raymond Ng (A0198543R) Wong Yoke Keong (A0195365U)

Intelligent Robotic Systems Practice Module
"""

import airsim
import time

class DroneControl:
    def __init__(self, droneList):
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.droneList = droneList
        self.Init_AirSim()

    def Init_AirSim(self):
        """
        Method to initialize AirSim for a list of drones
        """
        for drone in self.droneList:
            self.client.enableApiControl(True, drone)
            self.client.armDisarm(True, drone)
    
    def Shutdown_AirSim(self):
        """
        Method to un-init all drones and quit AirSim
        """
        self.ArmDisarm(False)
        self.client.reset()
        self.EnableApiControl(False)
    
    def ResetAndRearm_Drones(self):
        """
        Method to reset all drones to original starting state and rearm
        """
        self.ArmDisarm(False)
        self.client.reset()
        self.EnableApiControl(False)
        time.sleep(0.5)
        self.EnableApiControl(True)
        self.ArmDisarm(True)

    def ArmDisarm(self, status):
        """
        Method to arm or disarm all drones
        """
        for drone in self.droneList:
            self.client.armDisarm(status, drone)
    
    def EnableApiControl(self, status):
        """
        Method to enable or disable drones API control
        """
        for drone in self.droneList:
            self.client.enableApiControl(status, drone)

    def TakeOff(self):
        """
        Method to take off for all drones
        """
        dronesClient = []
        for drone in self.droneList:
            cli_drone = self.client.takeoffAsync(vehicle_name=drone)
            dronesClient.append(cli_drone)
        for drone in dronesClient:
            drone.join()
    
    def GetMultirotorState(self, drone):
        """
        Method to get current drone states
        """
        if drone in self.droneList:
            return self.client.getMultirotorState(vehicle_name=drone)
        else:
            print('Drone does not exists!')
    
    def GetBarometerData(self, drone):
        """
        Method to get barometer data
        """
        if drone in self.droneList:
            return self.client.getBarometerData(vehicle_name=drone)
        else:
            print('Drone does not exists!')
    
