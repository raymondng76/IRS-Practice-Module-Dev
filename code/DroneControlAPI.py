"""
Date: 1/2/2020
Team: Kenneth Goh (A0198544N) Raymond Ng (A0198543R) Wong Yoke Keong (A0195365U)

Intelligent Robotic Systems Practice Module
"""

import airsim

class DroneControl:
    def __init__(self, droneList):
        self.client = airsim.MultirotorClient()
        self.Init_AirSim(droneList=droneList)

    def Init_AirSim(droneList):
        """
        Method to initialize AirSim for a list of drones
        Input: list of Drone names stated in settings.json
        Output: AirSim MultirotorClient object with all requested drone
        """
        self.client.confirmConnection()
        for drone in droneList:
            self.client.enableApiControl(True, drone)
            self.client.armDisarm(True, drone)
    
    def Shutdown_AirSim(client, droneList):
        """
        Method to un-init all drones and quit AirSim
        Input: list of Drone names stated in settings.json
        Output: Nil
        """
        for drone in droneList:
            self.client.armDisarm(False, drone)
        self.client.reset()
        for drone in droneList:
            self.client.enableApiControl(False, drone)

