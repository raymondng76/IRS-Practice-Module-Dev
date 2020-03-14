import airsim
import time
import pprint
import os
from DroneControlAPI import DroneControl
from yolov3_inference import *

droneList = ['Drone1', 'Drone2', 'Drone3', 'DroneTarget']
dc = DroneControl(droneList)

airsim.wait_key('Press any key to take off')
dc.takeOff()

# airsim.wait_key('Press any key to read state')
# state = pprint.pformat(dc.getMultirotorState('Drone1'))
# print(f"State : {state}")

# airsim.wait_key('Press any key to move drone')
# dc.moveDrone('Drone1', [0,0,-10], 0.5)

# airsim.wait_key('Press any key to get drone pos')
# pos = dc.getDronePos('Drone1')
# print(f'pos = {pos}')

airsim.wait_key('Set cam orientation')
dc.moveDrone('DroneTarget', [0,0,0.4], 0.5)
dc.setCameraOrientation(0)

airsim.wait_key('Get images')
img1 = dc.getImage(droneList[0])
img2 = dc.getImage(droneList[1])
img3 = dc.getImage(droneList[2])

img1 = img1[0]
img2 = img2[0]
img3 = img3[0]



airsim.write_file(os.path.normpath('img1.png'), img1.image_data_uint8)
airsim.write_file(os.path.normpath('img2.png'), img2.image_data_uint8)
airsim.write_file(os.path.normpath('img3.png'), img3.image_data_uint8)



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