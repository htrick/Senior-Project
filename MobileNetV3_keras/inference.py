from keras.models import load_model

from keras import backend as K

from src.MobileNet_V3 import Hswish
import numpy as np

import sys
import os
import cv2
import configparser


'''Run through the images in the given directory in the config file and determine the
   predictions of the model, save the original image with the prediction overlayed'''
def main():
   numArgs = len(sys.argv)
   config_file = None
   weight_path = None

   args = sys.argv
   if (numArgs == 3):
      if (args[1] == '-c'):
         config_file = args[2]
      else:
         print("Usage: python3 inference.py [-c <config_file>] [<weights>]")
         return
   elif (numArgs == 5):
      if (args[1] == '-c'):
         config_file = args[2]
      else:
         print("Usage: python3 inference.py [-c <config_file>] [-w <weights>]")
         return
      if (args[3] == '-w'):
         weight_path = args[4]
      else:
         print("Usage: python3 inference.py [-c <config_file>] [-w <weights>]")
         return
   else:
      print("Usage: python3 inference.py [-c <config_file>] [-w <weights>]")
      return

   ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

   # ** get configuration
   config_client = configparser.ConfigParser()
   config_client.read(config_file)

   # ** set gpu
   os.environ['CUDA_VISIBLE_DEVICES'] = config_client.get('gpu', 'gpu')

   # ** Weights
   if weight_path is None:
      weight_path = config_client.get('inference', 'weight_path')

   # ** Images
   image_dir = config_client.get('inference', 'image_path')
   if not os.path.exists(image_dir):
      raise 'input dir does not exist'

   # ** Where to store the predictions
   inference_dir = config_client.get('inference', 'inference_dir')
   if not os.path.exists(inference_dir):
      os.mkdir(inference_dir)

   # ** Input Sizes
   input_width = config_client.getint('model', 'input_width')
   input_height = config_client.getint('model', 'input_height')

   print("Loading Model")
   #Load the old model
   '''model = model = load_model(os.path.join(ROOT_DIR, weight_path),
                           custom_objects={'Hswish':Hswish})'''

   #Load the the new model
   model = load_model(os.path.join(ROOT_DIR, weight_path),
                               custom_objects={'_hard_swish':_hard_swish,
                                               '_relu6':_relu6})

   #Run through the images and predict the free space
   n = 1
   for _id in os.listdir(image_dir):
      #Get the image
      _id_path = os.path.join(image_dir, _id)

      #Normalize the input image
      X = np.empty((1, input_height, input_width, 3), dtype='float32')
      X[0, :, :, :] = image_augmentation(cv2.imread(_id_path))

      #Predict the free space
      print("Predicting Image: " + str(n))
      prediction = model.predict(X)[0]

      #Load the image to draw the extracted mask data on for validation
      validationMaskImage = cv2.imread(_id_path)

      #Draw circles on the original image to show where the predicted free pace occurs
      x = 0
      for i in range(len(prediction)):
         y = int(round(prediction[i] * input_height))
         validationMaskImage = cv2.circle(validationMaskImage, (x, y), 1, (0, 255, 0), -1)
         x += 5

      #Save the overlayed image
      cv2.imwrite(inference_dir + "/" + _id.replace(".jpg", "") + "_inference.jpg", validationMaskImage)

      n += 1

   return

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
