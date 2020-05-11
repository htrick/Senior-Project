#!/usr/bin/env python

import sys
import os
print(__file__)
sys.path.append(os.path.dirname(__file__) + '/../../../../src')
sys.path.append(os.path.dirname(__file__) + '/../../../../')
print(sys.version)

import roslib
import rospy
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import rosbag

import cv2

import numpy as np

from ConfigParser import ConfigParser

from math import sqrt, atan, sin, cos, pi
from random import randint

from MobileNet_V3 import Hswish
from trajectory import *

from keras.models import load_model

from keras import backend as K

'''Parse the command line to find what flags were given'''
def parseCommandLine(numArgs, args):
   flags = []

   for i in range(numArgs):
      if args[i] == '-c':
         if '-c' in flags:
            print("Usage: rosrun video_inference_p2 videoinference.py -c <config_file> [-t <bag_file_topics>] [-b <bag_file>] [-w <weights>]")
            return []
         flags.append(args[i])

      if args[i] == '-w':
         if '-w' in flags:
            print("Usage: rosrun video_inference_p2 videoinference.py -c <config_file> [-t <bag_file_topics>] [-b <bag_file>] [-w <weights>]")
            return []
         flags.append(args[i])

      if args[i] == '-b':
         if '-b' in flags:
            print("Usage: rosrun video_inference_p2 videoinference.py -c <config_file> [-t <bag_file_topics>] [-b <bag_file>] [-w <weights>]")
            return []
         flags.append(args[i])

      if args[i] == '-t':
         if '-t' in flags:
            print("Usage: rosrun video_inference_p2 videoinference.py -c <config_file> [-t <bag_file_topics>] [-b <bag_file>] [-w <weights>]")
            return []
         flags.append(args[i])

   return flags

'''Run through the images in the given directory in the config file and determine the
   predictions of the model, save the original image with the prediction overlayed'''
def main():
   config_file = None
   weight_path = None
   bag_file = None
   bag_file_topics = None
   numArgs = len(sys.argv)
   args = sys.argv

   flags = parseCommandLine(numArgs, args);

   for f in flags:
      index = args.index(f)

      #Save the configuration file
      if f == '-c':
         config_file = args[index+1]

      #Use the given weights file instead of the one in the config file
      elif f == '-w':
         weight_path = args[index+1]

      elif f == '-b':
         bag_file = args[index+1]

      elif f == '-t':
         bag_file_topics = args[index+1]

   #Config file flag was not specified
   if config_file is None:
      print("Usage: rosrun video_inference_p2 videoinference.py -c <config_file> [-t <bag_file_topics>] [-b <bag_file>] [-w <weights>]")
      return

   #Predict the free space in the images
   else:
      print("Predict Images")
      predictImages(config_file, weight_path, bag_file, bag_file_topics)

   return

'''Make predictions about the free space for the given images using the model in the 
   config file or given weight_path'''
def predictImages(config_file, weight_path, bag_file, bag_file_topics):
   n = 0

   #Create a publisher
   image_pub = rospy.Publisher("image_inferencer", Image)
   bridge = CvBridge()

   #Initialize a node
   rospy.init_node('video', anonymous=True)

   #Make inferences on the images in the image_dir
   ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

   # ** get configuration
   config_client = ConfigParser()
   config_client.read(config_file)

   # ** set gpu
   os.environ['CUDA_VISIBLE_DEVICES'] = config_client.get('gpu', 'gpu')

   # ** Weights
   if weight_path is None:
      weight_path = config_client.get('inference', 'weight_path')

   # ** Bag File
   if bag_file is None:
      bag_file = config_client.get('video_inference', 'bag_file_path')
   bag = rosbag.Bag(bag_file, "r")

   # ** Topics
   if bag_file_topics is None:
      bag_file_topics = config_client.get('video_inference', 'bag_file_topics')

   # ** Where to store the predictions
   inference_dir = config_client.get('video_inference', 'inference_dir')
   if not os.path.exists(inference_dir):
      os.mkdir(inference_dir)

   # ** Input Sizes
   input_width = config_client.getint('model', 'input_width')
   input_height = config_client.getint('model', 'input_height')

   # ** Number of outputs
   num_outputs = config_client.getint('model', 'num_outputs')

   print("Loading Model")

   model = load_model(weight_path,
                      custom_objects={'_hard_swish':_hard_swish,
                                      '_relu6':_relu6})

   trajectory = Trajectory(input_width, input_height, num_outputs)
   robotCenter = trajectory.robotCenter
   robotWidth = trajectory.robotWidth
   robotCenter = trajectory.robotCenter
   robotLeft = trajectory.robotLeft
   robotRight = trajectory.robotRight
   robotFront = trajectory.robotFront #Front of the robot
   robotCloseUp = trajectory.robotCloseUp #The very front of the robot
   maxTranslation = trajectory.maxTranslation

   prevFrameRotation = None
   jitter = 0

   #Run through the images and predict the free space and trajectory
   stepSize = input_width // num_outputs
   for topic, msg, t in bag.read_messages(topics=[bag_file_topics]):
      #Exit on shutdown
      if rospy.is_shutdown():
         print('Shutting down inferenceer')
         exit()
      #Copy the frame to show processing
      frame = bridge.compressed_imgmsg_to_cv2(msg)
      frame = cv2.resize(frame, (input_width, input_height))
      processed_frame = frame.copy()

      #Normalize the input image
      X = np.empty((1, input_height, input_width, 3), dtype='float32')
      X[0, :, :, :] = image_augmentation(frame)

      #Predict the free space
      print("Predicting Frame: " + str(n))
      prediction = model.predict(X)[0]

      highestPoint = robotCenter
      x = 0
      for i in range(len(prediction)):
         y = int(round(prediction[i] * input_height))

         if y < highestPoint[1]:
            highestPoint = (x, y)

         #Draw circles on the original image to show where the predicted free space occurs
         processed_frame = cv2.circle(processed_frame, (x, y), 1, (0, 255, 0), -1)
         x += stepSize

      #Draw lines representing the sides of the robot
      cv2.line(processed_frame, (robotCenter[0]-robotWidth, robotCenter[1]), 
               (robotCenter[0]-robotWidth, robotFront), (0, 0, 255), 2)
      cv2.line(processed_frame, (robotCenter[0]+robotWidth, robotCenter[1]), 
               (robotCenter[0]+robotWidth, robotFront), (0, 0, 255), 2)

      #Draw a line representing the boundary of the front of the robot
      cv2.line(processed_frame, (robotLeft, robotFront), (robotRight, robotFront), (0, 0, 255), 2)

      #Calculate the trajectory of the robot
      (translation, rotation) = trajectory.calculateTrajectory(prediction)
      #Convert the trajectory percentages to the target point
      (translation_x, translation_y) = trajectory.trajectoryToPoint(translation, rotation)

      #Display the magnitude and direction of the vector the robot should drive along
      cv2.putText(processed_frame, "Rotation: " + str(rotation), (10, 50), 
                  cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
      cv2.putText(processed_frame, "Translation: " + str(translation), (10, 80), 
                  cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

      #Draw an arrowed line indicating the predicted trajectory of the robot
      if rotation == -1 and translation == 0:
         cv2.arrowedLine(processed_frame, (robotCenter[0], robotFront), 
                                              (input_width-1, robotFront), (0, 0, 255), 2)
      elif rotation == 1 and translation == 0:
         cv2.arrowedLine(processed_frame, (robotCenter[0], robotFront), 
                                              (0, robotFront), (0, 0, 255), 2)
      else:
         cv2.arrowedLine(processed_frame, robotCenter, (translation_x, translation_y), (0, 0, 255), 2)

      n += 1
                  
      try:
         image_pub.publish(bridge.cv2_to_imgmsg(processed_frame, "bgr8"))
      except CvBridgeError as e:
         print(e)

      if prevFrameRotation is not None:
         jitter += rotation - prevFrameRotation
      prevFrameRotation = rotation

   print("Jitter: " + str(jitter))

   return

'''Compute the Euclidean distance between 2 points'''
def distance(p1, p2):
   return sqrt(((p1[0]-p2[0])**2) + ((p1[1]-p2[1])**2))

#Turn the image to RGB and normalize it
def image_augmentation(img):
   if img is None:
      raise '** Failed to read image.'
   # to rgb
   img_copy = img.copy()
   img_copy = img_copy[:, :, ::-1]

   img_norm = normalize(img_copy)

   return img_norm

#Normalize the image
def normalize(img):
   return img / 255.0

def _relu6(x):
   """Relu 6
   """
   return K.relu(x, max_value=6.0)

def _hard_swish(x):
   """Hard swish
   """
   return x * K.relu(x + 3.0, max_value=6.0) / 6.0

if __name__ == '__main__':
   main()
