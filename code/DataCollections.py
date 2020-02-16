"""
Date: 1/2/2020
Team: Kenneth Goh (A0198544N) Raymond Ng (A0198543R) Wong Yoke Keong (A0195365U)

Intelligent Robotic Systems Practice Module

Code is adapted from:
https://github.com/microsoft/DroneRescue
"""

import time
import math
import airsim
from code.DroneControlAPI import DroneControl

class OrbiterImager:
    def __init__(self, 
                cx, 
                cy,
                z,
                drone_control,
                camera_angle,
                radius=2,
                altitude=10,
                speed=2,
                iterations=1,
                snapshots_count=30,
                image_dir='./images/'):
        """
        @param cx: X position of starting orbit
        @param cy: Y position of starting orbit
        @param drone_control: drone_control object
        @param camera_angle: angle where the camera is pointing
        @param radius: orbit radius (default=2)
        @param altitude: orbit altitude (default=10)
        @param speed: drone move speed (default=2)
        @param iterations: how many rounds to orbit
        @param snapshots_count: amount of snapshots to take
        @param image_dir: directory to store snapshots        
        """
       
        self.radius = radius
        self.altitude = altitude
        self.speed = speed
        self.iterations = iterations
        self.image_dir = image_dir
        self.drone_control = drone_control

        x = cx - radius
        y = cy

        # Set Camera Angle
        print('Set camera angle.')
        self.drone_control.client.simSetCameraOrientation(0, 
            airsim.to_quaternion(camera_angle * math.pi / 180, 0, 0))

        # Move to start orbit position
        print('Move to start position.')
        self.drone_control.client.moveToPositionAsync(
            x, y, z, 1, 60, 
            drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
            yaw_mode=airsim.YawMode(False, 0)
        ).join()
        pos = self.drone_control.client.getMultirotorState().kinematics_estimated.position

        dx = x - pos.x_val
        dy = y - pos.y_val
        yaw = airsim.to_eularian_angles(
            self.drone_control.client.getMultirotorState().kinematics_estimated.orientation
        )[2]

        # Correct position
        while abs(dx) > 1 or abs(dy) > 1 or abs(yaw) > 0.1:
            self.drone_control.client.moveToPositionAsync(
                x, y, z, 0.25, 60,
                drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
                yaw_mode=airsim.YawMode(False, 0)
            ).join()
            pos = self.drone_control.client.getMultirotorState().kinematics_estimated.position
            dx = x - pos.x_val
            dy = y - pos.y_val
            yaw = airsim.to_eularian_angles(
                self.drone_control.client.getMultirotorState().kinematics_estimated.orientation
            )[2]
            print(f"Yaw is {yaw}")
        
        print(f'Location offset is {dx}, {dy}')
        fyaw = airsim.to_eularian_angles(
            self.drone_control.client.getMultirotorState().kinematics_estimated.orientation
        )[2]
        print(f'Final yaw is {fyaw}')

        cx = float(cx - pos.x_val)
        cy = float(cy - pos.y_val)
        length = math.sqrt((cx * cx) + (cy * cy))
        cx /= length
        cy /= length
        cx *= self.radius
        cy *= self.radius

        self.home = self.drone_control.client.getMultirotorState().kinematics_estimated.position
        # Check home position stability
        start = time.time()
        count = 0
        while count < 100:
            pos = self.home
            if abs(pos.z_val - self.home.z_val) > 1:
                count = 0
                self.home = pos
                if time.time() - start > 10:
                    print("Drone is drifting, waiting for stability.")
                    start = time
            else:
                count += 1

        self.center = self.drone_control.client.getMultirotorState().kinematics_estimated.position
        self.center.x_val += cx
        self.center.y_val += cy

    def start(self):
        pass







if __name__ == '__main__':
    print('Start')
    droneID = ['Drone']
    dc = DroneControl(droneID)

    # Check landed state
    landed_state = dc.client.getMultirotorState().landed_state

    if landed_state == airsim.LandedState.Landed:
        print('Take Off')
        pos = dc.client.getMultirotorState().kinematics_estimated.position
        z = pos.z_val - 1
        dc.takeOff()
    else:
        print('Already flying')
        dc.client.hover()
        pos = dc.client.getMultirotorState().kinematics_estimated.position
        z = pos.z_val
    
    # Create OrbitImager object
    # oi = OrbiterImager()
    