######## Webcam Object Detection Using Tensorflow-trained Classifier #########
#
# Author: Evan Juras
# Date: 10/27/19
# Description:
# This program uses a TensorFlow Lite model to perform object detection on a live webcam
# feed. It draws boxes and scores around the objects of interest in each frame from the
# webcam. To improve FPS, the webcam object runs in a separate thread from the main program.
# This script will work with either a Picamera or regular USB webcam.
#
# This code is based off the TensorFlow Lite image classification example at:
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/examples/python/label_image.py
#
# I added my own method of drawing boxes and labels using OpenCV.
#
# Modified by: Shawn Hymel
# Date: 09/22/20
# Description:
# Added ability to resize cv2 window and added center dot coordinates of each detected object.
# Objects and center coordinates are printed to console.

# Import packages
import os
import argparse
import cv2
import numpy as np
import sys
import time
from threading import Thread
import importlib.util
import random as rng
from math import floor

rng.seed(12345)


# Define VideoStream class to handle streaming of video from webcam in separate processing thread
# Source - Adrian Rosebrock, PyImageSearch: https://www.pyimagesearch.com/2015/12/28/increasing-raspberry-pi-fps-with-python-and-opencv/
class VideoStream:
    """Camera object that controls video streaming from the Picamera"""

    def __init__(self, resolution=(640, 480), framerate=30):
        # Initialize the PiCamera and the camera image stream
        self.stream = cv2.VideoCapture(0)
        ret = self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        ret = self.stream.set(3, resolution[0])
        ret = self.stream.set(4, resolution[1])

        # Read first frame from the stream
        (self.grabbed, self.frame) = self.stream.read()

        # Variable to control when the camera is stopped
        self.stopped = False

    def start(self):
        # Start the thread that reads frames from the video stream
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        # Keep looping indefinitely until the thread is stopped
        while True:
            # If the camera is stopped, stop the thread
            if self.stopped:
                # Close camera resources
                self.stream.release()
                return

            # Otherwise, grab the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        # Return the most recent frame
        return self.frame

    def stop(self):
        # Indicate that the camera and thread should be stopped
        self.stopped = True


MODEL_NAME = "training"
GRAPH_NAME = "model.tflite"
resW, resH = (1000, 1000)
imW, imH = int(resW), int(resH)

# Import TensorFlow libraries
# If tflite_runtime is installed, import interpreter from tflite_runtime, else import from regular tensorflow
# If using Coral Edge TPU, import the load_delegate library
pkg = importlib.util.find_spec('tflite_runtime')
if pkg:
    from tflite_runtime.interpreter import Interpreter
else:
    from tensorflow.lite.python.interpreter import Interpreter

# Get path to current working directory
CWD_PATH = os.getcwd()

# Path to .tflite file, which contains the model that is used for object detection
PATH_TO_CKPT = os.path.join(CWD_PATH, GRAPH_NAME)

# Load the Tensorflow Lite model.
interpreter = Interpreter(model_path=PATH_TO_CKPT)
interpreter.allocate_tensors()

# Get model details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

floating_model = (input_details[0]['dtype'] == np.float32)
input_mean = 127.5
input_std = 127.5
threshold_canny = 100

# Initialize frame rate calculation
frame_rate_calc = 1
freq = cv2.getTickFrequency()

# Initialize video stream
videostream = VideoStream(resolution=(imW, imH), framerate=30).start()
time.sleep(1)

# Create window
cv2.namedWindow('Object detector', cv2.WINDOW_NORMAL)

# for selecting hand
skinColorUpper = np.array([20, 0.8 * 255, 0.8 * 255])
skinColorLower = np.array([5, 0.2 * 255, 0.2 * 255])

# counters
counter_sec = 3.99

# for frame1 in camera.capture_continuous(rawCapture, format="bgr",use_video_port=True):
while True:

    # Start timer (for calculating frame rate)
    t1 = cv2.getTickCount()

    # Grab frame from video stream
    frame = videostream.read()

    # apply mask on hand skin
    # https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_colorspaces/py_colorspaces.html
    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(frame_hsv, skinColorLower, skinColorUpper)
    res = cv2.bitwise_and(frame, frame, mask=mask)

    # smooth mask
    blurred = cv2.GaussianBlur(mask, (21, 21), 8)
    blurred = (blurred > 127) * 255
    blurred = np.array(blurred).astype('uint8')

    # detect edges on skin
    canny_output = cv2.Canny(blurred, threshold_canny, threshold_canny * 2)
    cv2.imshow('Contours', canny_output)

    # Find contours
    contours, _ = cv2.findContours(canny_output, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Find the convex hull object for each contour
    hull_list = []
    contour_idx_list = []
    for i in range(len(contours)):
        hull = cv2.convexHull(contours[i])
        if (np.abs(np.min(hull[:, :, 1]) - np.max(hull[:, :, 1])) > 200 and np.abs(np.min(hull[:, :, 0]) - np.max(hull[:, :, 0])) > 50) or i == 0:
            contour_idx_list.append(i)
            hull_list.append(hull)

    # Draw contours + hull results
    drawing = frame
    for i in range(len(hull_list)):
        color = (rng.randint(0, 256), rng.randint(0, 256), rng.randint(0, 256))
        cv2.drawContours(drawing, contours, contour_idx_list[i], color)
        cv2.drawContours(drawing, hull_list, i, color)

    # add button & text
    drawing = cv2.rectangle(drawing, (0, 0), (150, 100), (58, 134, 47), -1)
    drawing = cv2.putText(drawing, 'Start', (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

    # counter to start gesture recognition
    # if start
    if counter_sec > 0:
        drawing = cv2.putText(drawing, str(floor(counter_sec)), (180, 75), cv2.FONT_HERSHEY_SIMPLEX, 2, (10, 10, 200), 2, cv2.LINE_AA)
        time.sleep(0.05)
        counter_sec -= 0.1
        if counter_sec < 0.1:
            counter_sec = 3.99

    cv2.imshow('res', res)
    cv2.imshow('mask', blurred)
    cv2.imshow('Contours', drawing)
    k = cv2.waitKey(5)

    """# Acquire frame and resize to expected shape [1xHxWx3]
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (width, height))
    input_data = np.expand_dims(frame_resized, axis=0)

    # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
    if floating_model:
        input_data = (np.float32(input_data) - input_mean) / input_std

    # Perform the actual detection by running the model with the image as input
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()"""

cv2.destroyAllWindows()
