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
import os
from DroneControlAPI import DroneControl

class OrbiterImager:
    def __init__(self, 
                cx, 
                cy,
                z,
                drone_control,
                droneID,
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
        @param droneID: id of shooter drone
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
        self.droneID = droneID
        self.takeoff = True
        self.z = None
        self.snapshot_index = 0
        self.photo_prefix = 'photo_'
        self.snapshots = snapshots_count

        if self.snapshots is not None and self.snapshots > 0:
            self.snapshot_delta = 360 / self.snapshots

        if self.iterations <= 0:
            self.iterations = 1

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
            yaw_mode=airsim.YawMode(False, 0), vehicle_name=self.droneID
        ).join()
        pos = self.drone_control.getMultirotorState(self.droneID).kinematics_estimated.position

        dx = x - pos.x_val
        dy = y - pos.y_val
        yaw = airsim.to_eularian_angles(
            self.drone_control.getMultirotorState(self.droneID).kinematics_estimated.orientation
        )[2]

        # Correct position
        while abs(dx) > 1 or abs(dy) > 1 or abs(yaw) > 0.1:
            self.drone_control.client.moveToPositionAsync(
                x, y, z, 0.25, 60,
                drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
                yaw_mode=airsim.YawMode(False, 0), vehicle_name=self.droneID
            ).join()
            pos = self.drone_control.getMultirotorState(self.droneID).kinematics_estimated.position
            dx = x - pos.x_val
            dy = y - pos.y_val
            yaw = airsim.to_eularian_angles(
                self.drone_control.getMultirotorState(self.droneID).kinematics_estimated.orientation
            )[2]
            print(f"Yaw is {yaw}")
        
        print(f'Location offset is {dx}, {dy}')
        fyaw = airsim.to_eularian_angles(
            self.drone_control.getMultirotorState(self.droneID).kinematics_estimated.orientation
        )[2]
        print(f'Final yaw is {fyaw}')

        cx = float(cx - pos.x_val)
        cy = float(cy - pos.y_val)
        length = math.sqrt((cx * cx) + (cy * cy))
        cx /= length
        cy /= length
        cx *= self.radius
        cy *= self.radius

        self.home = self.drone_control.getMultirotorState(self.droneID).kinematics_estimated.position
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

        self.center = self.drone_control.getMultirotorState(self.droneID).kinematics_estimated.position
        self.center.x_val += cx
        self.center.y_val += cy

    def start(self):
        print('Arm Drone...')
        self.drone_control.client.armDisarm(True)

        start = self.drone_control.getMultirotorState(self.droneID).kinematics_estimated.position
        landed = self.drone_control.getMultirotorState(self.droneID).landed_state
        if not self.takeoff and landed == airsim.LandedState.Landed:
            self.takeoff = True
            print('Taking off...')
            self.drone_control.client.takeoffAsync(self.droneID).join()
            start = self.drone_control.getMultirotorState(self.droneID).kinematics_estimated.position
            z = -self.altitude + self.home.z_val
        else:
            print(f'Already Flown @ {start.z_val}')
            z = start.z_val
        
        print(f'Climbing to pos: {start.x_val}, {start.y_val}, {z}')
        self.drone_control.client.moveToPositionAsync(
            start.x_val, start.y_val, z, self.speed, vehicle_name=self.droneID
        ).join()
        self.z = z

        print('Ramp to spd...')
        count = 0
        self.start_angle = None
        self.next_snapshot = None

        ramptime = self.radius / 10
        self.start_time = time.time()

        while count < self.iterations and self.snapshot_index < self.snapshots:
            now = time.time()
            speed = self.speed
            diff = now - self.start_time
            if diff < ramptime:
                speed = self.speed * diff / ramptime
            elif ramptime > 0:
                print('Reached full speed...')
                ramptime = 0
            
            lookahead_angle = speed / self.radius

            # compute current angle
            pos = self.drone_control.client.getMultirotorState(self.droneID).kinematics_estimated.position
            dx = pos.x_val - self.center.x_val
            dy = pos.y_val - self.center.y_val
            actual_radius = math.sqrt((dx*dx) + (dy*dy))
            angle_to_center = math.atan2(dy, dx)

            camera_heading = (angle_to_center - math.pi) * 180 / math.pi

            # compute lookahead
            lookahead_x = self.center.x_val + self.radius * \
                math.cos(angle_to_center + lookahead_angle)
            lookahead_y = self.center.y_val + self.radius * \
                math.cos(angle_to_center + lookahead_angle)
            
            vx = lookahead_x - pos.x_val
            vy = lookahead_y - pos.y_val

            if self.track_orbits(angle_to_center * 180 / math.pi):
                count += 1
                print(f'Completed {count} orbits')
            
            self.camera_heading = camera_heading
            self.drone_control.client.moveByVelocityZAsync(
                vx, vy, z, 1, airsim.DrivetrainType.MaxDegreeOfFreedom, airsim.YawMode(False, camera_heading), vehicle_name=self.droneID
            )
        self.drone_control.moveDrone(drone=self.droneID, position=[start.x_val, start.y_val, z], duration=2)


    def track_orbits(self, angle):
        if angle < 0:
            angle += 360
        
        if self.start_angle is None:
            self.start_angle = angle
            if self.snapshot_delta:
                self.next_snapshot = angle + self.snapshot_delta
            self.previous_angle = angle
            self.shifted = False
            self.previous_sign = None
            self.previous_diff = None
            self.quarter = False
            return False
        
        # Watch for smooth crossing from negative diff to positive diff
        if self.previous_angle is None:
            self.previous_angle = angle
            return False

        # ignore click over from 360 back to 0
        if self.previous_angle > 350 and angle < 20:
            if self.snapshot_delta and self.next_snapshot >= 360:
                self.next_snapshot -= 360
            return False
        
        diff = self.previous_angle - angle
        crossing = False
        self.previous_angle = angle

        if self.snapshot_delta and angle > self.next_snapshot:
            print(f'Take snapshot at angle {angle}')
            self.take_snapshot()
            self.next_snapshot += self.snapshot_delta
        
        diff = abs(angle - self.start_angle)
        if diff > 45:
            self.quarter = True
        
        if self.quarter and self.previous_diff is not None and diff != self.previous_diff:
            direction = self.sign(self.previous_diff - diff)
            if self.previous_sign is None:
                self.previous_sign = direction
            elif self.previous_sign > 0 and direction < 0:
                if diff < 45:
                    self.quarter = False
                    if self.snapshots <= self.snapshot_index + 1:
                        crossing = True
            self.previous_sign = direction
        self.previous_diff = diff
        return crossing

    def take_snapshot(self):
        if not os.path.exists(self.image_dir):
            os.makedirs(self.image_dir)

        # first hold our current position so drone doesn't try and keep flying while we take the picture.
        pos = self.drone_control.getMultirotorState(self.droneID).kinematics_estimated.position
        self.drone_control.client.moveToPositionAsync(pos.x_val, pos.y_val, self.z, 0.25, 3, airsim.DrivetrainType.MaxDegreeOfFreedom,
                                        airsim.YawMode(False, self.camera_heading), vehicle_name=self.droneID)
        responses = self.drone_control.client.simGetImages([airsim.ImageRequest(
            0, airsim.ImageType.Scene)], vehicle_name=self.droneID)  # scene vision image in png format
        response = responses[0]
        filename = self.photo_prefix + \
            str(self.snapshot_index) + "_" + str(int(time.time()))
        self.snapshot_index += 1
        airsim.write_file(os.path.normpath(
            self.image_dir + filename + '.png'), response.image_data_uint8)
        print("Saved snapshot: {}".format(filename))
        # cause smooth ramp up to happen again after photo is taken.
        self.start_time = time.time()

    def sign(self, s):
        if s < 0:
            return -1
        return 1

if __name__ == '__main__':
    """
    Settings.json
    {
        "SeeDocsAt": "https://github.com/Microsoft/AirSim/blob/master/docs/settings.md",
        "SettingsVersion": 1.2,
        "ClockSpeed": 5,
        "ViewMode": "FlyWithMe",
        "SimMode": "Multirotor",
        "Vehicles": {
            "ShooterDrone": {
                "VehicleType": "SimpleFlight",
                "X": 2, "Y": 0, "Z": -2
            },
            "TargetDrone": {
                "VehicleType": "SimpleFlight",
                "X": 4, "Y": 0, "Z": -2
            }
        }
    }
    """
    print('Start')
    droneID = ['TargetDrone', 'ShooterDrone']
    dc = DroneControl(droneID)

    
    # Check landed state
    landed_state = dc.client.getMultirotorState(droneID[1]).landed_state

    if landed_state == airsim.LandedState.Landed:
        print('Take Off')
        pos = dc.client.getMultirotorState(droneID[1]).kinematics_estimated.position
        z = pos.z_val - 1
        dc.takeOff()
    else:
        print('Already flying')
        dc.client.hover()
        pos = dc.client.getMultirotorState(droneID[1]).kinematics_estimated.position
        z = pos.z_val
    
    dc.moveDrone(droneID[0], (2, 0, -5), 0.5)

    # Create OrbitImager object
    oi = OrbiterImager(
        cx=2, 
        cy=0, 
        z=z, 
        drone_control=dc,
        droneID=droneID[1], 
        camera_angle=-20, 
        radius=0.4, 
        altitude=10, 
        speed=2, 
        iterations=1, 
        snapshots_count=30, 
        image_dir='./images/')
    oi.start()

    