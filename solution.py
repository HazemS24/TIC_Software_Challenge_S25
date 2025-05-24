from TMMC_Wrapper import *
import rclpy
import numpy as np
import math
import time
from ultralytics import YOLO

def detect_stop_sign(image):
    RED_THRESHOLD = 500
    RED_MAX = 20000
    global stopped
    img_data = image.data

    if img_data:
        img = robot.rosImg_to_cv2()  # convert ROS image to OpenCV format
        filtered_img = robot.red_filter(img)  # apply red filter
        # print(f"filtered img: {filtered_img}")
        contoured_img, max_area, (cX, cY) = robot.add_contour(filtered_img)  # add contours
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
    tags = robot.detect_april_tag_from_img(img_3D)
    print("tags", tags)
    if (tags):
        tag_id = next(iter(tags))
        scan_data = robot.checkScan()
        # if scan_data:
        min_dist, min_dist_angle = robot.detect_tag_distance(scan_data.ranges)  
        print("min_dist for april tag", min_dist)      
        print("tag id", tag_id)
        if tag_id == 1 and min_dist < 0.50:
            robot.rotate(80,-1)
        elif tag_id == 5 and min_dist < 1.25:
            robot.rotate(135, -1)
        elif tag_id == 2 and min_dist < 0.5:
            robot.rotate(45, -1)
        elif tag_id == 3 and min_dist < 0.5:
            robot.rotate(135, -1)
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

# Variable for controlling which level of the challenge to test -- set to 0 for pure keyboard control
challengeLevel = 1

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

            control.send_cmd_vel(0.3, 0.0)
            rclpy.spin_once(robot, timeout_sec=0.1)
            time.sleep(0.1)
    if challengeLevel == 4:
        while rclpy.ok():
            rclpy.spin_once(robot, timeout_sec=0.1)
            time.sleep(0.1)
            # Write your solution here for challenge level 4

    if challengeLevel == 5:
        while rclpy.ok():
            rclpy.spin_once(robot, timeout_sec=0.1)
            time.sleep(0.1)
            # Write your solution here for challenge level 5
            

except KeyboardInterrupt:
    print("Keyboard interrupt received. Stopping...")

finally:
    control.stop_keyboard_control()
    robot.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


