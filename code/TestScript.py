import airsim
import time
import pprint
import os
import math
import numpy as np
from DroneControlAPI import DroneControl
# from yolov3_inference import *

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
# d2camera_heading = (135 - math.pi) * 180 / math.pi
# d3camera_heading = (225 - math.pi) * 180 / math.pi
# print(f'D2 cam head: {d2camera_heading}')
# print(f'D3 cam head: {d3camera_heading}')
# d2pos = dc.getMultirotorState('Drone2').kinematics_estimated.position
# dc.client.moveByVelocityZAsync(d2pos.x_val, d2pos.y_val, d2pos.z_val, 1, airsim.DrivetrainType.MaxDegreeOfFreedom, airsim.YawMode(False, -120), vehicle_name='Drone2')
# d3pos = dc.getMultirotorState('Drone3').kinematics_estimated.position
# dc.client.moveByVelocityZAsync(d3pos.x_val, d3pos.y_val, d3pos.z_val, 1, airsim.DrivetrainType.MaxDegreeOfFreedom, airsim.YawMode(False, 120), vehicle_name='Drone3')

dc.setCameraHeading(-120, 'Drone2')
dc.setCameraHeading(120, 'Drone3')


airsim.wait_key('Get images')
img1 = dc.getImage(droneList[0])
img2 = dc.getImage(droneList[1])
img3 = dc.getImage(droneList[2])

img1 = img1[0]
img2 = img2[0]
img3 = img3[0]
img1d = np.fromstring(img1.image_data_uint8, dtype=np.uint8) 
img1_rgb = img1d.reshape(img1.height, img1.width, 3)
print(img1_rgb.shape)

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