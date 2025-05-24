import cv2
import apriltag
import numpy as np
import rclpy
from .Robot import Robot
from ultralytics import YOLO
import math

class Camera:
    def __init__(self, robot : Robot):
        self.robot = robot
        self.model = None


    def checkImage(self) -> np.ndarray:
        ''' Waits for the robot\'s image future to complete, then returns the latest image message. '''
        self.robot.image_future = rclpy.Future()
        try:
            self.robot.spin_until_future_completed(self.robot.image_future)
        except Exception as e:
            # Log the exception and return None when the node is shutting down or a KeyboardInterrupt occurs.
            print("Exception in checkImage:", e)
            return None
        return self.robot.last_image_msg

    def checkImageRelease(self): #this one returns an actual image instead of all the data
        ''' Retrieves the latest image message, reshapes its data into a 3D image array, and displays it using OpenCV. '''
        image = self.checkImage()
        height = image.height
        width = image.width
        img_data = image.data
        img_3D = np.reshape(img_data, (height, width, 3))
        cv2.imshow("image", img_3D)
        cv2.waitKey(10)

    def checkCamera(self) -> np.ndarray:
        ''' Waits for and returns the most recent camera info message using the robot\'s camera info future. '''
        self.robot.camera_info_future = rclpy.Future()
        self.robot.spin_until_future_completed(self.robot.camera_info_future)
        return self.robot.last_camera_info_msg 

    def estimate_apriltag_pose(self, image : np.ndarray) -> list[tuple[int, float, float, float]]:
        ''' Converts the image to grayscale, detects AprilTags, and, if successful, estimates the pose. '''
        # Convert image to grayscale for tag detection
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        detections = apriltag.Detector(apriltag.DetectorOptions(families="tag16h5, tag25h9")).detect(img_gray)

        # If camera calibration matrix is not available or no detections, return empty list
        if not detections or self.robot.k is None:
            return []
        
        
        poses = []
        half_size = self.robot.TAG_SIZE / 2.0
        object_points = np.array([
            [-half_size,  half_size, 0],
            [ half_size,  half_size, 0],
            [ half_size, -half_size, 0],
            [-half_size, -half_size, 0]
        ], dtype=np.float32)

        # Process each detected tag
        for detection in detections:
            tag_id = detection.tag_id
            image_points = np.array(detection.corners, dtype=np.float32)

            # Estimate position
            ret, rvec, tvec = cv2.solvePnP(object_points, image_points, self.robot.k, None)
            if not ret:
                continue

            tvec = tvec.flatten()
            range_ = np.linalg.norm(tvec)
            bearing = np.degrees(np.arctan2(tvec[0], tvec[2]))
            elevation = np.degrees(np.arctan2(tvec[1], tvec[2]))

            poses.append((tag_id, range_, bearing, elevation))

        return poses
    
    def rosImg_to_cv2(self) -> np.ndarray:
        ''' Retrieves the current ROS image message, reshapes its raw data into three-channel image, and returns it. '''
        image = self.checkImage()
        if image is None:
            # Gracefully handle the case when no image was received.
            return None
        height = image.height
        width = image.width
        img_data = image.data
        img_3D = np.reshape(img_data, (height, width, 3))
        return img_3D

    def ML_predict_stop_sign(self, img : np.ndarray) -> tuple[bool, int, int, int, int]:
        ''' Uses the provided ML model to predict the presence of a stop sign within the image, draws a bounding box around any detection, displays the result, and returns both the detection flag and bounding box coords. '''
        if self.model == None:
            self.model = YOLO("yolov8n.pt")  # Load the YOLOv8 model
        
        stop_sign_detected = False

        x1 = -1
        y1 = -1 
        x2 = -1 
        y2 = -1

        # Predict stop signs in image using model
        results = self.model.predict(img, classes=[11], conf=0.25, imgsz=640, max_det=1)
        
        # Results is a list containing the results object with all data
        results_obj = results[0]
        
        # Extract bounding boxes
        boxes = results_obj.boxes.xyxy

        try:
            for box in boxes:
                x1, y1, x2, y2 = map(int, box[:4])
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
                stop_sign_detected = True
        except:
            stop_sign_detected = False

        cv2.imshow("Bounding Box", img)

        return stop_sign_detected, x1, y1, x2, y2   

    def red_filter(self, img):
        """
        mask image for red only area, note that the red HSV bound values are tunable and should be adjusted base on evironment
        :param img: list RGB image array
        :return: list RGB image array of binary filtered image
        """
        # Colour Segmentation
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
        # Define lower and upper bounds for red and brown hue
        lower_red_1 = np.array([-3, 100, 0])     # Lower bound for red hue (reddish)
        lower_red_2 = np.array([170, 70, 50])   # Lower bound for red hue (reddish)
        upper_red_1 = np.array([3, 255, 255])  # Upper bound for red hue (reddish)
        upper_red_2 = np.array([180, 255, 255]) # Upper bound for red hue (reddish)
        lower_brown = np.array([10, 60, 30])    # Lower bound for brown hue
        upper_brown = np.array([30, 255, 255])  # Upper bound for brown hue
        
        # Create masks for red and brown
        red_mask_1 = cv2.inRange(hsv_img, lower_red_1, upper_red_1)
        red_mask_2 = cv2.inRange(hsv_img, lower_red_2, upper_red_2)
        brown_mask = cv2.inRange(hsv_img, lower_brown, upper_brown)
    
        # Combine red masks
        red_mask = cv2.bitwise_or(red_mask_1, red_mask_2)
        # Exclude brown by subtracting its mask from the red mask
        red_mask = cv2.subtract(red_mask, brown_mask)
    
        # Apply the red mask to the original image then convert to grayscale
        red_img = cv2.bitwise_and(img, img, mask=red_mask)
        gray = cv2.cvtColor(red_img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (11, 11), 0)
    
        #get binary image with OTSU thresholding
        (T, threshInv) = cv2.threshold(blurred, 0, 255,
        cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        cv2.imshow("Threshold", threshInv)
        # print("[INFO] otsu's thresholding value: {}".format(T))
    
        #Morphological closing
        kernel_dim = (21,21)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_dim)
        filtered_img = cv2.morphologyEx(threshInv, cv2.MORPH_CLOSE, kernel)
        
        return filtered_img

    def add_contour(self, img):
        """
        apply contour detection to the red only masked image
        :param img: list image array
        :return: contoured img, max area and centroid(cy,cx)
        """
        max_area = 0    # stores the largest red area detected
        #edges = cv2.Canny(img, 100, 200)
        contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    
        #loop through and get the area of all contours
        areas_of_contours = [cv2.contourArea(contour) for contour in contours]
        contoured = np.zeros((img.shape[0], img.shape[1], 3), dtype = np.uint8)
        cX = 0
        cY = 0

        try:
            max_poly_indx = np.argmax(areas_of_contours)
            stop_sign = contours[max_poly_indx]
            #approximate contour into a simpler shape
            epsilon = 0.01 * cv2.arcLength(stop_sign, True)
            approx_polygon = cv2.approxPolyDP(stop_sign, epsilon, True)
            area = cv2.contourArea(approx_polygon)
            max_area = max(max_area, area)
            cv2.drawContours(contoured, [approx_polygon], -1, (0, 255, 0), 3)
    
            # compute the center of the contour
            M = cv2.moments(stop_sign)

            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            # draw the contour and center of the shape on the image
            cv2.circle(contoured, (cX, cY), 2, (255, 255, 255), -1)
        except:
            x=1
    
        return contoured, max_area ,(cX,cY)

    def detect_april_tag_from_img(self, img):
        """
            returns the april tag id, translation vector and rotation matrix from
            :param img: image from camera stream, np array
            :return: dict: {int tag_id: tuple (float distance, float angle)}
            """
        # convert image to grayscale
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # cv2.imshow("gray",img_gray)
        options = apriltag.DetectorOptions(families="tag16h5, tag25h9")
        # Apriltag detection
        detector = apriltag.Detector(options)
        detections = detector.detect(img_gray)
        # dictionary for return
        dict = {}
        # process each apriltag found in image to get the id and spacial information
        for detection in detections:
            tag_id = detection.tag_id
            translation_vector, rotation_matrix = self.homography_to_pose(detection.homography)
            dict[int(tag_id)] = (self.translation_vector_to_distance(translation_vector), self.rotation_matrix_to_angles(rotation_matrix))
        return dict

    def detect_tag_distance(self, scan):
        # calculate indices (this reflects reading data from 45 to 135 degrees)
        front_index = 180
        #90 for right and 270 for left
        front_right_index = front_index - 30
        front_left_index = front_index + 30

        # define maximum distance threshold for obstacles
        obstacle_dist = 0.5

        # read lidar scan and extract data in angle range of interest
        data = scan[front_right_index:front_left_index + 1]

        min_dist = min(data)
        min_dist_index = data.index(min_dist)
        min_dist_angle = (min_dist_index-90)/2

        return min_dist, min_dist_angle

    def rotation_matrix_to_angles(self, R):
        """
        Convert a 3x3 rotation matrix to Euler angles (in degrees).
        Assumes the rotation matrix represents a rotation in the XYZ convention.
        :param R, rotation_matrix: list
        :return: list [float angle_x, float angle_y, float angle_z]
        """
        sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
        singular = sy < 1e-6
        if  not singular :
            x = math.atan2(R[2,1] , R[2,2])
            y = math.atan2(-R[2,0], sy)
            z = math.atan2(R[1,0], R[0,0])
        else :
            x = math.atan2(-R[1,2], R[1,1])
            y = math.atan2(-R[2,0], sy)
            z = 0
        return np.array([x, y, z])

    def translation_vector_to_distance(self, translation_vector):
        """
        convert 3D translation vector to distance
        :param translation_vector: list
        :return: float
        """
        # Calculate the distance from the translation vector
        distance = np.linalg.norm(translation_vector)
        return distance


    def homography_to_pose(self, H):
        """
        Convert a homography matrix to rotation matrix and translation vector.
        :param H: list homography matrix
        :return: tuple (list translation_vector, list rotational_matrix)
        """
        # Perform decomposition of the homography matrix
        R, Q, P = np.linalg.svd(H)

        # Ensure rotation matrix has determinant +1
        if np.linalg.det(R) < 0:
            R = -R

        # Extract translation vector
        t = H[:, 2] / np.linalg.norm(H[:, :2], axis=1)

        return t, R