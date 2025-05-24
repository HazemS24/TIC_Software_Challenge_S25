from TMMC_Wrapper import *
import rclpy
import numpy as np
import math
import time
from ultralytics import YOLO

from collections import deque
from simple_pid import PID          # pip install simple-pid

# -------------------- CONSTANTS --------------------
EMA_ALPHA        = 0.30              # smoothing for tag distance
TAG_HISTORY      = deque(maxlen=5)   # last N lidar samples
CENTER_PID       = PID(0.6, 0.0, 0.12, setpoint=0)   # lane-centering PID
CENTER_PID.output_limits = (-1.2, 1.2)

CAMERA_FOCAL_PX  = 620               # <-- put your calibrated focal length here (pixels)
APRIL_H_METERS   = 0.162  

def detect_stop_sign_one(image):
    RED_THRESHOLD = 1000
    RED_MAX = 20000
    stopped = False  # global variable to track if the robot is stopped
    img_data = image.data

    if img_data:
        img = camera.rosImg_to_cv2()  # convert ROS image to OpenCV format
        filtered_img = camera.red_filter(img)  # apply red filter
        # print(f"filtered img: {filtered_img}")
        contoured_img, max_area, (cX, cY) = camera.add_contour(filtered_img)  # add contours
        # print("countoured_img", contoured_img)

        print(f"max_area: {max_area}")
        if max_area > RED_THRESHOLD and max_area < RED_MAX:  # check if the area of the detected contour is large enough
            print("Stop sign detected! Stopping the robot.")
            if not stopped:
                handle_stop_sign()  # handle stop sign behavior
                stopped = True
        else:
            stopped = False
        print("stopped", stopped)

def detect_april_tag(image):
    # 8 very bottom right corner
    # 2 left bottom triangle
    # 4 right top wall
    # 1 right protruding triangle bottom
    # 3 back face tag slightly left
    # 5 top left corner

    print("IN TAGS")
    img_data = image.data
    height = image.height
    width = image.width
    img_3D = np.reshape(img_data, (height, width, 3))
    tags = camera.detect_april_tag_from_img(img_3D)
    print("tags", tags)
    if (tags):
        tag_id = next(iter(tags))
        scan_data = lidar.checkScan()
        # if scan_data:
        min_dist, min_dist_angle = camera.detect_tag_distance(scan_data.ranges)  
        print("min_dist for april tag", min_dist)      
        print("tag id", tag_id)
        if tag_id == 1 and min_dist < 0.50:
            control.rotate(80,-1)
        elif tag_id == 5 and min_dist < 1.25:
            control.rotate(135, -1)
        elif tag_id == 2 and min_dist < 0.5:
            control.rotate(45, -1)
        elif tag_id == 3 and min_dist < 0.5:
            control.rotate(135, -1)
        # elif tag_id == 4 and min_dist < 0.5:
        #     robot.stop_keyboard_control()
        #     robot.destroy_node()
        #     rclpy.shutdown()
        # elif tag_id == 8:
        #     robot.rotate(180, 1)  # Rotate 30 degrees to the left
        # elif tag_id == 2

def handle_stop_sign():
    control.stop_keyboard_input()
    time.sleep(3)
    control.start_keyboard_input()
    print("Resuming control after stopping for stop sign.")

def _median_fan(ranges, center_deg, fan=6):
    """Median of a narrow lidar fan around <center_deg> (deg in lidar frame)."""
    idx = [(center_deg + i) % 360 for i in range(-fan, fan + 1)]
    vals = [r for i, r in enumerate(ranges) if i in idx and r > 0.02]
    return np.median(vals) if vals else None

def _ema_tag_dist(raw):
    last = TAG_HISTORY[-1] if TAG_HISTORY else raw
    smoothed = EMA_ALPHA * raw + (1 - EMA_ALPHA) * last
    TAG_HISTORY.append(smoothed)
    return smoothed

def _fuse_dist(lidar_d, pix_h):
    """Conservative fusion of lidar and camera size estimate."""
    cam_d = (APRIL_H_METERS * CAMERA_FOCAL_PX) / (pix_h + 1e-6)
    return cam_d if lidar_d is None else min(lidar_d, cam_d)

half = None
def align_path(speed=0.30):
    """Center robot between left & right walls; call at ~20 Hz."""
    scan = lidar.checkScan()
    left  = _median_fan(scan.ranges, -45)  # left wall ≈ +135 ° in TurtleBot frame
    right = _median_fan(scan.ranges,  45)  # right wall ≈ –135 °

    if None in (left, right):
        control.send_cmd_vel(speed, 0.0)    # missing wall → just go straight
        return

    err = left - right                      # +ve ⇒ closer to right wall
    w   = CENTER_PID(err)                   # PID gives angular vel (rad/s)
    v   = speed * (1 - abs(w) / 1.2)        # slow slightly while turning
    control.send_cmd_vel(v, w)


# Variable for controlling which level of the challenge to test -- set to 0 for pure keyboard control
challengeLevel = 6

# Set to True if you want to run the simulation, False if you want to run on the real robot
is_SIM = False

# Set to True if you want to run in debug mode with extra print statements, False otherwise
Debug = False

# Initialization    
if not "robot" in globals():
    robot = Robot(IS_SIM=is_SIM, DEBUG=Debug)
    
control = Control(robot)
camera = Camera(robot)
imu = IMU(robot)
logging = Logging(robot)
lidar = Lidar(robot)

def detect_obstacle_ahead(threshold=0.2) -> bool:
    scan = lidar.checkScan()
    dist, _ = lidar.detect_obstacle_in_cone(scan, distance=threshold, center=0, offset_angle=1)
    return dist != -1

def detect_stop_sign() -> bool:
    img = camera.rosImg_to_cv2()
    if img is None:
        return False
    detected, *_ = camera.ML_predict_stop_sign(img)
    return detected

if challengeLevel <= 2:
    control.start_keyboard_control()
    rclpy.spin_once(robot, timeout_sec=0.1)

try:
    if challengeLevel == 0:
        while rclpy.ok():
            rclpy.spin_once(robot, timeout_sec=0.1)
            time.sleep(0.1)
            # Challenge 0 is pure keyboard control, you do not need to change this it is just for your own testing

    if challengeLevel == 1:
        while rclpy.ok():
            rclpy.spin_once(robot, timeout_sec=0.1)

            # Write your solution here for challenge level 1
            # It is recommended you use functions for aspects of the challenge that will be resused in later challenges
            # For example, create a function that will detect if the robot is too close to a wall
            if detect_obstacle_ahead(0.3):
                print("Wall too close! Reversing.")
                control.set_cmd_vel(-1, 0, 1)
                time.sleep(0.2)

            else: 
                control.move_forward()
                time.sleep(0.2)                

    if challengeLevel == 2:
        while rclpy.ok():
            rclpy.spin_once(robot, timeout_sec=0.1)
            time.sleep(0.1)
            # Write your solution here for challenge level 2
            if detect_stop_sign():
                print("Stop sign detected! Pausing...")
                control.send_cmd_vel(0.0, 0.0)
                time.sleep(3)

            if detect_obstacle_ahead():
                print("Obstacle detected. Reversing.")
                control.send_cmd_vel(-0.3, 0.0)
                time.sleep(0.5)
                control.send_cmd_vel(0.0, 0.0)
    if challengeLevel == 3:
        while rclpy.ok():
            rclpy.spin_once(robot, timeout_sec=0.1)
            time.sleep(0.1)
            # Write your solution here for challenge level 3 (or 3.5)
            img = camera.rosImg_to_cv2()
            stop_detected = camera.ML_predict_stop_sign(img)[0] if img is not None else False
            obstacle_detected = detect_obstacle_ahead(0.5)

            if stop_detected:
                print("Stop sign ahead. Pausing.")
                control.send_cmd_vel(0.0, 0.0)
                time.sleep(3)
                continue

            if obstacle_detected:
                print("Obstacle detected. Stopping.")
                control.send_cmd_vel(0.0, 0.0)
                time.sleep(1)
                continue

            control.send_cmd_vel(0.3, 0.0) #move forwardf
            align_path(speed=0.3, kp=0.5, cone_offset=20, max_detect_dist=0.6, desired=0.5)  # Align path with obstacles
            rclpy.spin_once(robot, timeout_sec=0.1)
            time.sleep(0.1)
    if challengeLevel == 4:
        while rclpy.ok():
            rclpy.spin_once(robot, timeout_sec=0.1)
            time.sleep(0.1)
            # Write your solution here for challenge level 4
            detect_obstacle_ahead()  # check for collisions during each iteration
            currTime = robot.get_clock().now()
            # print("time", currTime)
            if (currTime.nanoseconds % 5 == 0):
                image = camera.checkImage()
                detect_stop_sign_one(image)
                detect_april_tag(image)

            control.send_cmd_vel(0.3, 0.0) #move forward
            if (currTime.nanoseconds % 5e8 == 0): #every second
                align_path(speed=0.3, kp=0.3, cone_offset=20, max_detect_dist=0.6)

            # check if conflict between wall and correction
            # add distance to april tag
            # play aroudn with correction params

    if challengeLevel == 5:
        while rclpy.ok():
            rclpy.spin_once(robot, timeout_sec=0.1)
            time.sleep(0.1)
    if challengeLevel == 6:
        while rclpy.ok():
            rclpy.spin_once(robot, timeout_sec=0.1)
            time.sleep(0.1)
            currTime = robot.get_clock().now()
            # print("time", currTime)
            if (currTime.nanoseconds % 5 == 0):
                image = camera.checkImage()
                detect_stop_sign_one(image)
                detect_april_tag(image)

            # 1) forward 3.5 seconds
            control.set_cmd_vel(0.3, 0.0, 4.25)

            # 2) turn 80° left
            control.rotate(76, 1)

            # 3) forward 8 seconds
            control.set_cmd_vel(0.3, 0.0, 4.5)

            # 4) stop for 2 seconds
            control.set_cmd_vel(0.0, 0.0, 3.0)

            # 3) forward 2 seconds
            control.set_cmd_vel(0.3, 0.0, 2.3)


            # 5) turn 45° left
            control.rotate(35, 1)

            # 6) forward 3 seconds
            control.set_cmd_vel(0.3, 0.0, 5.5)

            # 7) turn 135° left
            control.rotate(117, 1)

            # 8) forward 10 seconds
            control.set_cmd_vel(0.3, 0.0, 11.0)

            break

    if challengeLevel == 7:
        while rclpy.ok():
            rclpy.spin_once(robot, timeout_sec=0.1)
            time.sleep(0.1)
            # define the sequence of actions
            sequence = [
                ("move",   0.3, 0.0,  2.0),   # forward 2s
                ("rotate", 90,  1,    None),  # turn L 90°
                ("move",   0.3, 0.0,  5.0),   # forward 5s
                ("move",   0.0, 0.0,  2.0),   # stop 2s
                ("rotate", 45,  1,    None),  # turn L 45°
                ("move",   0.3, 0.0,  3.0),   # forward 3s
                ("rotate", 135, 1,    None),  # turn L 135°
                ("move",   0.3, 0.0, 10.0),   # forward 10s
            ]

            for action in sequence:
                typ = action[0]
                if typ == "move":
                    vx, w, dur = action[1], action[2], action[3]
                    end = time.time() + dur
                    while time.time() < end and rclpy.ok():
                        control.send_cmd_vel(vx, w)
                        img = camera.checkImage()
                        detect_stop_sign_one(img)
                        detect_april_tag(img)
                        rclpy.spin_once(robot, timeout_sec=0.1)
                        time.sleep(0.1)
                    control.send_cmd_vel(0.0, 0.0)

                elif typ == "rotate":
                    angle, direction = action[1], action[2]
                    control.rotate(angle, direction)
                    # give a moment for the rotation to happen and then check once
                    time.sleep(0.5)
                    img = camera.checkImage()
                    detect_stop_sign_one(img)
                    detect_april_tag(img)

            # end of sequence: ensure robot is stopped
            control.send_cmd_vel(0.0, 0.0)
            break  # run sequence just once


            

except KeyboardInterrupt:
    print("Keyboard interrupt received. Stopping...")

finally:
    control.stop_keyboard_control()
    robot.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


