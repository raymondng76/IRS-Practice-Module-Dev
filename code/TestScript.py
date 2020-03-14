import airsim
import time
import pprint
import os
import math
import numpy as np
import cv2
from DroneControlAPI import DroneControl
from yolov3_inference import *

yolo = YoloPredictor('..\weights\drone.h5')


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
dc.moveDrone('DroneTarget', [0,0,1.4], 0.5)
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
bb1_out = yolo.get_yolo_boxes(img1[:,:,:3])
bb2_out = yolo.get_yolo_boxes(img2[:,:,:3])
bb3_out = yolo.get_yolo_boxes(img3[:,:,:3])
print('BB1')
print(f'XMIN: {bb1_out.xmin}')
print(f'XMAX: {bb1_out.xmax}')
print(f'YMIN: {bb1_out.ymin}')
print(f'YMAX: {bb1_out.ymax}')
print('BB3')
print(f'XMIN: {bb2_out.xmin}')
print(f'XMAX: {bb2_out.xmax}')
print(f'YMIN: {bb2_out.ymin}')
print(f'YMAX: {bb2_out.ymax}')
print('BB2')
print(f'XMIN: {bb3_out.xmin}')
print(f'XMAX: {bb3_out.xmax}')
print(f'YMIN: {bb3_out.ymin}')
print(f'YMAX: {bb3_out.ymax}')

# img1 = cv2.imdecode(airsim.string_to_uint8_array(img1raw), cv2.IMREAD_UNCHANGED)
# img2 = cv2.imdecode(airsim.string_to_uint8_array(img2raw), cv2.IMREAD_UNCHANGED)
# img3 = cv2.imdecode(airsim.string_to_uint8_array(img3raw), cv2.IMREAD_UNCHANGED)
# img1d = np.fromstring(img1.image_data_uint8, dtype=np.uint8) 
# img1_rgb = img1d.reshape(img1.height, img1.width, 3)
# img1_rgb = img1_rgb[:,:,:3]
print(img1.shape)
# cv2.imshow('img1', img1)
# cv2.imshow('img2', img2)
# cv2.imshow('img3', img3)

cv2.imwrite('img1.png', img1)
cv2.imwrite('img2.png', img2)
cv2.imwrite('img3.png', img3)


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