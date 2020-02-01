import airsim
import time
import pprint
from DroneControlAPI import DroneControl

droneList = ['Drone1', 'Drone2', 'Drone3']
dc = DroneControl(droneList)

airsim.wait_key('Press any key to take off')
dc.TakeOff()

airsim.wait_key('Press any key to read state')
state = pprint.pformat(dc.GetMultirotorState('Drone1'))
print(f"State : {state}")

airsim.wait_key('Press any key to read barometer')
baro = pprint.pformat(dc.GetBarometerData('Drone2'))
print(f'Baro : {baro}')

airsim.wait_key('Press any key to reset')
dc.ResetAndRearm_Drones()

airsim.wait_key('Press any key to shutdown')
dc.Shutdown_AirSim()