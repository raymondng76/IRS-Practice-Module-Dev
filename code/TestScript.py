import airsim
import time
from DroneControlAPI import DroneControl

droneList = ['Drone1', 'Drone2', 'Drone3']
dc = DroneControl(droneList)
airsim.wait_key('Press any key to take off')
dc.TakeOff()
airsim.wait_key('Press any key to reset')
dc.ResetAndRearm_Drones()
airsim.wait_key('Press any key to shutdown')

dc.Shutdown_AirSim()