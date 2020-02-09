import airsim
import time
import pprint
from DroneControlAPI import DroneControl

droneList = ['Drone1', 'Drone2', 'Drone3', 'DroneTarget']
dc = DroneControl(droneList)

airsim.wait_key('Press any key to take off')
dc.takeOff()

airsim.wait_key('Press any key to read state')
state = pprint.pformat(dc.getMultirotorState('Drone1'))
print(f"State : {state}")

airsim.wait_key('Press any key to move drone')
dc.moveDrone('Drone1', [0,0,-10], 0.5)

airsim.wait_key('Press any key to get drone pos')
pos = dc.getDronePos('Drone1')
print(f'pos = {pos}')

# airsim.wait_key('Press any key to read sensors')
# baro = pprint.pformat(dc.GetBarometerData('Barometer1', 'Drone1'))
# print(f'Baro : {baro}')
# imu = pprint.pformat(dc.GetImuData('Imu1', 'Drone1'))
# print(f'Imu : {imu}')
# gps = pprint.pformat(dc.GetGpsData('Gps1', 'Drone1'))
# print(f'Gps : {gps}')
# mag = pprint.pformat(dc.GetMagnetometerData('Magnetometer1', 'Drone1'))
# print(f'Mag : {mag}')
# dis = pprint.pformat(dc.GetDistanceData('Distance1', 'Drone1'))
# print(f'Dis : {dis}')
# lidar = pprint.pformat(dc.GetLidarData('Lidar1', 'Drone1'))
# print(f'Lidar : {lidar}')

airsim.wait_key('Press any key to reset')
dc.resetAndRearm_Drones()

airsim.wait_key('Press any key to shutdown')
dc.shutdown_AirSim()