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

class Position:
    def __init__(self, pos):
        self.x = pos.x_val
        self.y = pos.y_val
        self.z = pos.z_val

# Make the drone fly in a circle.


class OrbitNavigator:
    def __init__(self, photo_prefix="photo_", radius=2, altitude=10, speed=2, iterations=1, center=[1, 0], snapshots=None, image_dir="./images/", id='ShooterDrone'):
        self.radius = radius
        self.altitude = altitude
        self.speed = speed
        self.iterations = iterations
        self.snapshots = snapshots
        self.snapshot_delta = None
        self.next_snapshot = None
        self.image_dir = image_dir
        self.z = None
        self.snapshot_index = 0
        self.photo_prefix = photo_prefix
        self.takeoff = True  # whether we did a take off
        self.id = id

        if self.snapshots is not None and self.snapshots > 0:
            self.snapshot_delta = 360 / self.snapshots

        if self.iterations <= 0:
            self.iterations = 1

        if len(center) != 2:
            raise Exception(
                "Expecting '[x,y]' for the center direction vector")

        # center is just a direction vector, so normalize it to compute the actual cx,cy locations.
        cx = float(center[0])
        cy = float(center[1])
        length = math.sqrt((cx*cx)+(cy*cy))
        cx /= length
        cy /= length
        cx *= self.radius
        cy *= self.radius

        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True, vehicle_name=self.id)

        self.home = self.client.getMultirotorState(vehicle_name=self.id).kinematics_estimated.position
        # check that our home position is stable
        start = time.time()
        count = 0
        while count < 100:
            pos = self.home
            if abs(pos.z_val - self.home.z_val) > 1:
                count = 0
                self.home = pos
                if time.time() - start > 10:
                    print(
                        "Drone position is drifting, we are waiting for it to settle down...")
                    start = time
            else:
                count += 1

        self.center = self.client.getMultirotorState(vehicle_name=self.id).kinematics_estimated.position
        self.center.x_val += cx
        self.center.y_val += cy

    def start(self):
        print("arming the drone...")
        self.client.armDisarm(True,vehicle_name=self.id)

        # AirSim uses NED coordinates so negative axis is up.
        start = self.client.getMultirotorState(vehicle_name=self.id).kinematics_estimated.position
        landed = self.client.getMultirotorState(vehicle_name=self.id).landed_state
        if not self.takeoff and landed == airsim.LandedState.Landed:
            self.takeoff = True
            print("taking off...")
            self.client.takeoffAsync(vehicle_name=self.id).join()
            start = self.client.getMultirotorState(vehicle_name=self.id).kinematics_estimated.position
            z = self.altitude + self.home.z_val
        else:
            print("already flying so we will orbit at current altitude {}".format(
                start.z_val))
            z = start.z_val - self.altitude  # use current altitude then

        print("climbing to position: {},{},{}".format(
            start.x_val, start.y_val, z))
        self.client.moveToPositionAsync(
            start.x_val, start.y_val, z, self.speed, vehicle_name=self.id).join()
        self.z = z

        print("ramping up to speed...")
        count = 0
        self.start_angle = None
        self.next_snapshot = None

        # ramp up time
        ramptime = self.radius / 10
        self.start_time = time.time()

        while count < self.iterations and self.snapshot_index < self.snapshots:
            # ramp up to full speed in smooth increments so we don't start too aggressively.
            now = time.time()
            speed = self.speed
            diff = now - self.start_time
            if diff < ramptime:
                speed = self.speed * diff / ramptime
            elif ramptime > 0:
                print("reached full speed...")
                ramptime = 0

            lookahead_angle = speed / self.radius

            # compute current angle
            pos = self.client.getMultirotorState(vehicle_name=self.id).kinematics_estimated.position
            dx = pos.x_val - self.center.x_val
            dy = pos.y_val - self.center.y_val
            actual_radius = math.sqrt((dx*dx) + (dy*dy))
            angle_to_center = math.atan2(dy, dx)

            camera_heading = (angle_to_center - math.pi) * 180 / math.pi

            # compute lookahead
            lookahead_x = self.center.x_val + self.radius * \
                math.cos(angle_to_center + lookahead_angle)
            lookahead_y = self.center.y_val + self.radius * \
                math.sin(angle_to_center + lookahead_angle)

            vx = lookahead_x - pos.x_val
            vy = lookahead_y - pos.y_val

            if self.track_orbits(angle_to_center * 180 / math.pi):
                count += 1
                print("completed {} orbits".format(count))

            self.camera_heading = camera_heading
            self.client.moveByVelocityZAsync(
                vx, vy, z, 1, airsim.DrivetrainType.MaxDegreeOfFreedom, airsim.YawMode(False, camera_heading),vehicle_name=self.id)

        self.client.moveToPositionAsync(start.x_val, start.y_val, z, 2, vehicle_name=self.id).join()

    def track_orbits(self, angle):
        # tracking # of completed orbits is surprisingly tricky to get right in order to handle random wobbles
        # about the starting point.  So we watch for complete 1/2 orbits to avoid that problem.
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

        # now we just have to watch for a smooth crossing from negative diff to positive diff
        if self.previous_angle is None:
            self.previous_angle = angle
            return False

        # ignore the click over from 360 back to 0
        if self.previous_angle > 350 and angle < 20:
            if self.snapshot_delta and self.next_snapshot >= 360:
                self.next_snapshot -= 360
            return False

        diff = self.previous_angle - angle
        crossing = False
        self.previous_angle = angle

        if self.snapshot_delta and angle > self.next_snapshot:
            print("Taking snapshot at angle {}".format(angle))
            self.take_snapshot()
            self.next_snapshot += self.snapshot_delta

        diff = abs(angle - self.start_angle)
        if diff > 45:
            self.quarter = True

        if self.quarter and self.previous_diff is not None and diff != self.previous_diff:
            # watch direction this diff is moving if it switches from shrinking to growing
            # then we passed the starting point.
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
        pos = self.client.getMultirotorState(vehicle_name=self.id).kinematics_estimated.position
        self.client.moveToPositionAsync(pos.x_val, pos.y_val, self.z, 0.25, 3, airsim.DrivetrainType.MaxDegreeOfFreedom,
                                        airsim.YawMode(False, self.camera_heading),vehicle_name=self.id)
        responses = self.client.simGetImages([airsim.ImageRequest(
            0, airsim.ImageType.Scene)],vehicle_name=self.id)  # scene vision image in png format
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

def Orbit(cx, cy, radius, speed, altitude, camera_angle, drone, id, folder):
    """
    @param cx: The x position of our orbit starting location
    @param cy: The x position of our orbit starting location
    @param radius: The radius of the orbit circle
    @param speed: The speed the drone should more, it's hard to take photos when flying fast
    @param altitude: The altidude we want to fly at, dont fly too high!
    @param camera_angle: The angle of the camera
    @param drone: The name of the drone, used to prefix the photos
    """

    x = cx - radius
    y = cy

    # set camera angle
    client.simSetCameraOrientation(0, airsim.to_quaternion(
        camera_angle * math.pi / 180, 0, 0),vehicle_name=id)  # radians

    # move the drone to the requested location
    print("moving to position...")
    client.moveToPositionAsync(
        x, y, z, 1, 60, drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom, yaw_mode=airsim.YawMode(False, 0),vehicle_name=id).join()
    pos = client.getMultirotorState(vehicle_name=id).kinematics_estimated.position

    dx = x - pos.x_val
    dy = y - pos.y_val
    yaw = airsim.to_eularian_angles(
        client.getMultirotorState(vehicle_name=id).kinematics_estimated.orientation)[2]

    # keep the drone on target, it's windy out there!
    print("correcting position and yaw...")
    while abs(dx) > 1 or abs(dy) > 1 or abs(yaw) > 0.1:
        client.moveToPositionAsync(
            x, y, z, 0.25, 60, drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom, yaw_mode=airsim.YawMode(False, 0), vehicle_name=id).join()
        pos = client.getMultirotorState(vehicle_name=id).kinematics_estimated.position
        dx = x - pos.x_val
        dy = y - pos.y_val
        yaw = airsim.to_eularian_angles(
            client.getMultirotorState(vehicle_name=id).kinematics_estimated.orientation)[2]
        print("yaw is {}".format(yaw))

    print("location is off by {},{}".format(dx, dy))

    o = airsim.to_eularian_angles(
        client.getMultirotorState(vehicle_name=id).kinematics_estimated.orientation)
    print("yaw is {}".format(o[2]))

    # let's orbit around the animal and take some photos
    folder = "./drone_images/" + folder
    nav = OrbitNavigator(photo_prefix=drone, radius=radius, altitude=altitude, speed=speed, iterations=1, center=[
                                     cx - pos.x_val, cy - pos.y_val], snapshots=30, image_dir=folder, id=id)
    nav.start()


def land():
    print("landing...")
    client.landAsync(vehicle_name='ShooterDrone').join()
    client.landAsync(vehicle_name='TargetDrone').join()
    print("disarming.")
    client.armDisarm(False)

    client.reset()
    client.enableApiControl(False)

if __name__ == '__main__':

    # Conect with the airsim server
    print("Start")
    sd = 'ShooterDrone'
    td = 'TargetDrone'
    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.enableApiControl(True, vehicle_name=sd)
    client.enableApiControl(True, vehicle_name=td)
    client.armDisarm(True, vehicle_name=sd)
    client.armDisarm(True, vehicle_name=td)
    print("Armed")
    client.takeoffAsync(vehicle_name=td).join()

    # Check State and takeoff if required
    landed = client.getMultirotorState(vehicle_name=sd).landed_state

    if landed == airsim.LandedState.Landed:
        print("taking off...")
        pos = client.getMultirotorState(vehicle_name=sd).kinematics_estimated.position
        z = pos.z_val - 1
        client.takeoffAsync(vehicle_name=sd).join()
    else:
        print("already flying...")
        client.hover()
        pos = client.getMultirotorState(vehicle_name=sd).kinematics_estimated.position
        z = pos.z_val

    # Start the navigation task
    alt = 0.2
    deg = 0
    folder = 'FootballField'
    filename = "Drone_" + "Alt" + str(alt) + "_Deg" + str(deg) + "_"
    Orbit(4, 0, 3, 0.4, alt, deg, filename, id=sd, folder=folder)

    land()

    print("Image capture complete...")