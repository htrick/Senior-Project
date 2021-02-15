import labelbox
import scale_ai
import urllib.request
import urllib.error
import csv
import cv2
import os
import sys
import shutil
import ast
import configparser
import json
import argparse
import numpy as np
from PIL import Image
from random import randint

class DataExtractor:
   # Extract input and expected output data from the csv file
   def _main(self):
      args = self.argumentParse()
      
      if args.clean == True:
         self.cleanData()
         return

      validPercent = args.p
      configFile = args.c
      scaleFile = args.scale
      labelboxFile = args.labelbox
      downloadType = '-a' if args.a == True else '-n'

      # Download the images and their associated data
      if labelboxFile is not None:
         self.downloadImageData(downloadType, labelboxFile, configFile, labelbox.Labelbox())

      #Loop through the JSON files given and download data
      if scaleFile is not None:
         for numScaleFiles in range(len(args.scale)):
            scaleFile = args.scale[numScaleFiles]
            self.downloadImageData(downloadType, scaleFile, configFile, scale_ai.Scale_ai())

      # Split images into training and validation directories,
      # Creates new random splits on every call
      print("Splitting images into training and validation")
      self.splitImages(validPercent)
      return

   # Parse the command line to find what flags were given
   def argumentParse(self):
      parser = argparse.ArgumentParser(prog = 'dataExtractor.py',usage = 'Usage: python3 dataExtractor.py [-clean] [-a] [-n] -p [0-1] -labelbox [filename.csv] -scale [filename(s).json] -c [filename]')
      
      parser.add_argument('-clean',action='store_true',help = 'Flag argument to remove all the directories and files containing image data')
      parser.add_argument('-a',action='store_true',help = 'Flag argument to re-download all of the images from the given data file that follows')
      parser.add_argument('-n',action='store_true',help = 'Flag argument to skip already downloaded images and their associated data and download any new images and their associated data and the given data file that follows')
      parser.add_argument('-p',nargs='?',const=0.15,type=float,help = 'Flag argument to use with -a or -n to specify what percentage of the downloaded images to set aside for validation, percentage to be a float between 0 - 1.0. Default percentage is 0.15')
      parser.add_argument('-c',nargs='?',type=str, const='config',help = 'Flag argument to specify the config file to use to determine the height and width of the images to save, and the number of points to extract from the image mask. Default file is "config"')
      parser.add_argument('-labelbox',type=str,help = 'The name of the data file to download and extract image and mask data from, if using a LabelBox file')
      parser.add_argument('-scale',nargs='*',type =str,help = 'The name of the data file(s) to download and extract image(s) and mask data from, if using a scale.ai file(s)')

      args = parser.parse_args()

      #check if the -p flag exists and set to default value if non existent, or out of range
      if(args.p != None):
         if args.p < 0 or args.p > 1:
            print("-p: %d is out of range. Value was set to 0.15" %args.p)
            args.p = 0.15
      elif(args.p == None and args.clean == False):
         args.p = 0.15

      #check if the -c flag is present
      if(args.c == None and args.clean == False):
         print("-c flag was not found.")
         parser.print_usage()
         parser.exit()

      #check if the -c file given exists
      if(args.c != None):
         if (os.path.exists(args.c) == False):
            parser.print_usage()
            print("-c file: '%s' does not exist." %args.c)
            parser.exit()
      
      #check if the CSV file given exists
      if(args.labelbox != None):
         if (os.path.exists(args.labelbox) == False):
            parser.print_usage()
            print("CSV file: '%s' does not exist." %args.labelbox)
            parser.exit()

      #check if the JSON file(s) given exists
      if(args.scale != None):
         for numScaleFiles in range(len(args.scale)):
            if (os.path.exists(args.scale[numScaleFiles]) == False):
               parser.print_usage()
               print("JSON file: '%s' does not exist." %args.scale[numScaleFiles])
               parser.exit()
      
      return args

   # Remove all the directories and files containing data information
   def cleanData(self):
      confirm = "None"
      while (confirm.lower() != 'y' and confirm.lower() != 'n'):
         confirm = input("Are you sure you want to delete all image directories and data? (y/n): ")
      if (confirm == 'n'):
         return

      dirPath = os.getcwd() # Get the current directory path

      # Remove the directories and all the files in them
      try:
         if os.path.isdir(dirPath + '/Input_Images'):
            shutil.rmtree(dirPath + '/Input_Images')
         if os.path.isdir(dirPath + '/Image_Masks'):
            shutil.rmtree(dirPath + '/Image_Masks')
         if os.path.isdir(dirPath + '/Mask_Data'):
            shutil.rmtree(dirPath + '/Mask_Data')
         if os.path.isdir(dirPath + '/Mask_Validation'):
            shutil.rmtree(dirPath + '/Mask_Validation')
         if os.path.isdir(dirPath + '/Blacklist_Masks'):
            shutil.rmtree(dirPath + '/Blacklist_Masks')
         if os.path.isdir(dirPath + '/Whitelist_Masks'):
            shutil.rmtree(dirPath + '/Whitelist_Masks')
         if os.path.isdir(dirPath + '/Training_Images'):
            shutil.rmtree(dirPath + '/Training_Images')
         if os.path.isdir(dirPath + '/Validation_Images'):
            shutil.rmtree(dirPath + '/Validation_Images')
         if os.path.isdir(dirPath + '/Unlabeled'):
            shutil.rmtree(dirPath + '/Unlabeled')
         if os.path.isfile(dirPath + '/Whitelisted_Images.txt'):
            os.remove(dirPath + '/Whitelisted_Images.txt')
         if os.path.isfile(dirPath + '/Blacklisted_Images.txt'):
            os.remove(dirPath + '/Blacklisted_Images.txt')
      except OSError as err:
         print("Error: {0}".format(err))
         return

   # Download the image and mask data from the data file.
   def downloadImageData(self, flag, dataFile, configFile, data):
      try:
         imageFile = open(dataFile, 'r') # Open the data file
      except:
         print("Error opening file: " + dataFile)
         return

      # ** get configuration
      config_client = configparser.ConfigParser()
      config_client.read(configFile)

      # ** Image Sizes
      imgWidth = config_client.getint('model', 'input_width')
      imgHeight = config_client.getint('model', 'input_height')

      # ** Number of outputs
      numOutputs = config_client.getint('model', 'num_outputs')

      '''
      try:
         if (flag == '-a'):
            whiteList = open("Whitelisted_Images.txt", 'w')
            blackList = open("Blacklisted_Images.txt", 'w')
         elif (flag == '-n'):
            whiteList = open("Whitelisted_Images.txt", 'a')
            blackList = open("Blacklisted_Images.txt", 'a')
         else:
            return
      except OSError as err:
         print("Error: {0}".format(err))
         return
      '''

      dirPath = os.getcwd() # Get the current directory path

      # Make the directories to store the image information
      try:
         if not os.path.isdir(dirPath + '/Input_Images'):
            os.mkdir(dirPath + '/Input_Images')
         if not os.path.isdir(dirPath + '/Image_Masks'):
            os.mkdir(dirPath + '/Image_Masks')
         if not os.path.isdir(dirPath + '/Mask_Data'):
            os.mkdir(dirPath + '/Mask_Data')
         if not os.path.isdir(dirPath + '/Mask_Validation'):
            os.mkdir(dirPath + '/Mask_Validation')
         '''
         if not os.path.isdir(dirPath + '/Blacklist_Masks'):
            os.mkdir(dirPath + '/Blacklist_Masks')
         if not os.path.isdir(dirPath + '/Whitelist_Masks'):
            os.mkdir(dirPath + '/Whitelist_Masks')
         '''
         if not os.path.isdir(dirPath + '/Unlabeled'):
            os.mkdir(dirPath + '/Unlabeled')
      except OSError as err:
         print("Error: {0}".format(err))
         return

      # List of image task data
      reader = data.listImageTaskData(imageFile)

      # Download the images and masks from the data file
      imgNum = 0
      for row in reader:
         imgNum += 1
         print(data.getLocationName() + " image: " + str(imgNum), end = '')

         # Get the ID of the task/image
         id = data.getID(row)

         # The name of the original image
         imgName = id + ".jpg"

         # Check if the image has been approved
         if data.isApproved(row) == False:
            print('\nImage ' + id + " has not been approved. Skipping image")

         # Check if current image is already downloaded and only new images need to be download. If it exists, continue to the next image
         if (flag == '-n' and os.path.isfile(dirPath + "/Input_Images/" + imgName)):
            print(" Skipping Image")
            continue

         # Get the original image
         print(" Getting Original, ", end = '')
         imgUrl = data.getImageURL(row)
         orgImg = self.getImageFromURL(imgUrl) # Retrieve the original image
         newImg = Image.open(orgImg[0])
         newImg = newImg.convert("RGB")   # Convert the image to RGB format
         origWidth, origHeight = newImg.size

         # Failed to download the image
         if (orgImg == None):
            print("Downloading the original image " + str(imgNum) + " failed")
            continue

         # Save the original image
         newImg = newImg.resize((imgWidth, imgHeight))  # Resize the image to be 640x360
         newImg.save(dirPath + "/Input_Images/" + imgName)
         newImg.close()

         print("Generating Mask")
         # Create a blank image to draw the mask on
         orgMask = np.zeros([origHeight, origWidth, 3], dtype=np.uint8)

         # Get the polygons for the mask data
         polygons = data.getPolygons(row)

         # Draw the mask and save it
         orgMask = cv2.fillPoly(orgMask, polygons, (255, 255, 255), lineType=cv2.LINE_8)
         newMask = cv2.resize(orgMask, (imgWidth, imgHeight))
         cv2.imwrite(dirPath + "/Image_Masks/" + id + "_mask.png", newMask)

         # Open the mask using PIL
         newMask = Image.open(dirPath + "/Image_Masks/" + id + "_mask.png").convert('L')

         maskDataFile = open(dirPath + "/Mask_Data/" + id + "_mask_data.txt", 'w')
         # Get the pixel array and witdh/height of the original image
         pixels = newMask.load()
         width, height = newMask.size

         # Extract the mask data
         points = self.extractMaskPoints(pixels, width, height, numOutputs)

         # Load the image to draw the extracted mask data on for validation
         validationMaskImage = cv2.imread(dirPath + "/Input_Images/" + imgName)

         # Write the mask data to a file in x,y column format, where y is normalized between 0 and 1 and draw the extracted mask points over the original image
         x = 0
         stepSize = imgWidth // numOutputs
         for y in points:
            # Draw a circle on the original image to validate the correct mask data is extracted
            validationMaskImage = cv2.circle(validationMaskImage, (x, round(y * (height-1))), 1, (0, 255, 0), -1)

            # Write the mask point to the file
            maskDataFile.write(str(x) + ',' + str(y) + '\n')
            x += stepSize

         # Save the overlayed image
         cv2.imwrite(dirPath + "/Mask_Validation/" + id + "_validation_mask.jpg",
                     validationMaskImage)

         '''
         # Check if the mask for the current image can be whitelisted
         inValid = self.checkForBlackEdges(pixels, width, height)
         if not inValid:
            newMask.save(dirPath + "/Whitelist_Masks/" + id + "_mask.png")
            whiteList.write(id + '.png\n')
         else:
            newMask.save(dirPath + "/Blacklist_Masks/" + id + "_mask.png")
            print("Potential labeling error for image: " + id)
            blackList.write(id + '.png\n')
        '''

         maskDataFile.close()
         newMask.close()

      imageFile.close()
      '''
      whiteList.close()
      blackList.close()
      '''
      return

   # Extract 128 points representing the bounds of the image mask between 0-1. Takes in the pixel array representing the mask, and the width & height of the mask
   def extractMaskPoints(self, pixels, width, height, numOutputs):
      found = False
      maskData = []
      stepSize = width // numOutputs

      # Find the numOutputs points along the image that represent the boundary of free space
      # Find the boundary goint from bottom (height - 1) to top (0)
      for x in range(0, width, stepSize):
         for y in range(height-1, -1, -1):
            color = pixels[x,y]
            if color == 0:
               break
         maskData.append(y / (height - 1))
         found = False

      return maskData

   # Download the image from the given URL. Return None if the request fails more than 5 times
   def getImageFromURL(self, url):
      image = None
      trys = 0
      # Attempt to download the image 5 times before quitting
      while (trys < 5):
         try:
            return urllib.request.urlretrieve(url) # Retrieve the image from the URL
         except urllib.error.URLError as err:
            print("Error: {0}".format(err))
            print("Trying again")
         except urllib.error.HTTPError as err:
            print("Error: {0}".format(err))
            print("Trying again")
         except urllib.error.ContentTooShortError as err:
            print("Error: {0}".format(err))
            print("Trying again")
         trys += 1
      return None

   # Return True if there is a black edge along the sides or bottome of the image represented by the pixels array
   def checkForBlackEdges(self, pixels, width, height):
      blackEdge = False

      # Check for black edge along bottom
      for x in range(width):
         if pixels[x, height - 1] < 128:
            blackEdge = True
         else:
            blackEdge = False
            break

      # There is a black border along the bottom of the image
      if blackEdge:
         return True

      # Check for black border on the left side of the image
      for y in range(height):
         if pixels[0, y] < 128:
            blackEdge = True
         else:
            blackEdge = False
            break

      # There is a black border along the left side of the image
      if blackEdge:
         return True

      # Check for black border on the right side of the image
      for y in range(height):
         if pixels[width - 1, y] < 128:
            blackEdge = True
         else:
            blackEdge = False
            break

      return blackEdge

   # Split the newly downloaded images into training and validation directories
   def splitImages(self, validPercent):
      dirPath = os.getcwd() #Get the current directory path

      # Remove any existing training and validation directories and remake them
      try:
         if os.path.isdir(dirPath + '/Training_Images'):
            shutil.rmtree(dirPath + '/Training_Images')
         if os.path.isdir(dirPath + '/Validation_Images'):
            shutil.rmtree(dirPath + '/Validation_Images')
         os.mkdir(dirPath + '/Training_Images')
         os.mkdir(dirPath + '/Validation_Images')
      except OSError as err:
         print("Error: {0}".format(err))
         return

      # List all the images that have been downloaded, now and previously
      if(os.path.exists(dirPath + "/Input_Images")):
         images = os.listdir(dirPath + "/Input_Images")
      else:
         print("No images have been downloaded. Run with the CSV file and JSON file(s) to make sure images are accessible")
         return

      # Determine how many images to use for validation
      numValid = round(len(images) * validPercent)
      numChosen = 0

      # Save images to the validation directory randomly
      while numChosen <= numValid-1:
         index = randint(0, len(images)-1)
         imgName = images.pop(index)
         img = Image.open(dirPath + "/Input_Images/" + imgName)
         img.save(dirPath + "/Validation_Images/" + imgName)
         numChosen += 1

      # Save the rest of the images to the training directory
      for imgName in images:
         img = Image.open(dirPath + "/Input_Images/" + imgName)
         img.save(dirPath + "/Training_Images/" + imgName)

      return

if __name__ == '__main__':
   d = DataExtractor()
   d._main()
