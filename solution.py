from TMMC_Wrapper import *
import rclpy
import numpy as np
import math
import time
from ultralytics import YOLO

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
