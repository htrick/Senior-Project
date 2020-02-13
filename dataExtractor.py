import urllib.request
import urllib.error
import csv
import cv2
import os
import sys
import shutil
import json
import configparser
from PIL import Image
from random import randint

class DataExtractor:
   '''Extract input and expected output data from the csv file'''
   def _main(self):
      numArgs = len(sys.argv)
      args = sys.argv

      flags = self.parseCommandLine(numArgs, args);
      if flags == "":
         return

      if flags == 'c':
         self.cleanData()
         return

      validPercent = 0.15
      configFile = None
      dataFile = None
      downloadType = None

      for f in flags:
         index = args.index('-'+f)

         #Save the file to download the images from
         if f == 'n' or f == 'a':
            dataFile = args[index+1]
            downloadType = f

         #Save the percentage to use for validation
         elif f == 'p':
            try:
               validPercent = float(args[index+1])
            except:
               print("Percentage used for the validation set must be a float between 0-1")
               return

         #Save the configuration file
         elif f == 'f':
            configFile = args[index+1]
            print(configFile)

      #Not all the required arguments were provided
      if configFile is None or dataFile is None or downloadType is None:
         print("Usage: python3 dataExtractor.py -c | -a <filename.csv> -f <filename> [-p <0-1>] |" +\
               "-n <filename.csv> -f <filename> [-p <0-1>]")
         return

      #Download the images and their associated data
      self.downloadImageData(downloadType, dataFile, configFile)

      #Split images into training and validation directories,
      #Creates new random splits on every call
      print("Splitting images into training and validation")
      self.splitImages(validPercent)
      return

   '''Parse the command line to find what flags were given'''
   def parseCommandLine(self, numArgs, args):
      flags = ""

      for i in range(numArgs):
         if args[i] == '-c':
            if numArgs != 2:
               print("Usage: python3 dataExtractor.py -c | -a <filename.csv> -f <filename> [-p <0-1>] |" +\
                     "-n <filename.csv> -f <filename> [-p <0-1>]")
               return ""
            flags += 'c'
            return flags

         if args[i] == '-n':
            if 'a' in flags or 'n' in flags:
               print("Usage: python3 dataExtractor.py -c | -a <filename.csv> -f <filename> [-p <0-1>] |" +\
                     "-n <filename.csv> -f <filename> [-p <0-1>]")
               return ""
            flags += 'n'

         if args[i] == '-a':
            if 'n' in flags or 'a' in flags:
               print("Usage: python3 dataExtractor.py -c | -a <filename.csv> -f <filename> [-p <0-1>] |" +\
                     "-n <filename.csv> -f <filename> [-p <0-1>]")
               return ""
            flags += 'a'

         if args[i] == '-p':
            if 'p' in flags:
               print("Usage: python3 dataExtractor.py -c | -a <filename.csv> -f <filename> [-p <0-1>] |" +\
                     "-n <filename.csv> -f <filename> [-p <0-1>]")
               return ""
            flags += 'p'

         if args[i] == '-f':
            if 'f' in flags:
               print("Usage: python3 dataExtractor.py -c | -a <filename.csv> -f <filename> [-p <0-1>] |" +\
                     "-n <filename.csv> -f <filename> [-p <0-1>]")
               return ""
            flags += 'f'

      return flags

   '''Remove all the directories and files containing data information'''
   def cleanData(self):
      confirm = "None"
      while (confirm.lower() != 'y' and confirm.lower() != 'n'):
         confirm = input("Are you sure you want to delete all image directories and data? (y/n): ")
      if (confirm == 'n'):
         return

      dirPath = os.getcwd() #Get the current directory path

      #Remove the directories and all the files in them
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

   '''Download the image and mask data from the .csv file.'''
   def downloadImageData(self, flag, csvFile, configFile):
      try:
         imageFile = open(csvFile, 'r') #Open the csv file
      except:
         print("Error opening file: " + csvFile)
         return

      # ** get configuration
      config_client = configparser.ConfigParser()
      config_client.read(configFile)

      # ** Image Sizes
      imgWidth = config_client.getint('model', 'input_width')
      imgHeight = config_client.getint('model', 'input_height')

      # ** Number of outputs
      numOutputs = config_client.getint('model', 'num_outputs')

      reader = csv.DictReader(imageFile)
      try:
         if (flag == 'a'):
            whiteList = open("Whitelisted_Images.txt", 'w')
            blackList = open("Blacklisted_Images.txt", 'w')
         elif (flag == 'n'):
            whiteList = open("Whitelisted_Images.txt", 'a')
            blackList = open("Blacklisted_Images.txt", 'a')
         else:
            return
      except OSError as err:
         print("Error: {0}".format(err))
         return

      dirPath = os.getcwd() #Get the current directory path

      #Make the directories to store the image information
      try:
         if not os.path.isdir(dirPath + '/Input_Images'):
            os.mkdir(dirPath + '/Input_Images')
         if not os.path.isdir(dirPath + '/Image_Masks'):
            os.mkdir(dirPath + '/Image_Masks')
         if not os.path.isdir(dirPath + '/Mask_Data'):
            os.mkdir(dirPath + '/Mask_Data')
         if not os.path.isdir(dirPath + '/Mask_Validation'):
            os.mkdir(dirPath + '/Mask_Validation')
         if not os.path.isdir(dirPath + '/Blacklist_Masks'):
            os.mkdir(dirPath + '/Blacklist_Masks')
         if not os.path.isdir(dirPath + '/Whitelist_Masks'):
            os.mkdir(dirPath + '/Whitelist_Masks')
         if not os.path.isdir(dirPath + '/Unlabeled'):
            os.mkdir(dirPath + '/Unlabeled')
      except OSError as err:
         print("Error: {0}".format(err))
         return

      #Download the images and masks from the csv file
      imgNum = 0
      for row in reader:
         imgNum += 1
         print("Image: " + str(imgNum), end = '')

         #The name of the original image
         imgName = row['ID'] + ".jpg"

         #Get the review score of the image
         review = row['Reviews']
         #Get the dictionary representation from the column
         review = review[1 : len(review) - 1]
         #Load the review column as a dictionary
         review = json.loads(str(review))
         score = review['score']

         #If the image has a negative score, do not download it
         if score < 0:
            print('\nImage ' + row['ID'] + " has a negative review score. Skipping image")
            continue

         '''Check if current image is already downloaded and only new images
            need to be download. If it exists, continue to the next image'''
         if (flag == 'n' and os.path.isfile(dirPath + "/Input_Images/" + imgName)):
            print(" Skipping Image")
            continue

         #Get the original image and mask
         print(" Getting Original, ", end = '')
         imgUrl = row['Labeled Data']
         orgImg = self.getImageFromURL(imgUrl) #Retrieve the original image

         #Failed to download the image
         if (orgImg == None):
            print("Downloading the original image " + str(imgNum) + " failed")
            continue

         #Download the mask
         print("Getting Mask")
         #Get the mask url of the image
         mask = json.loads(row['Masks'])
         maskUrl = mask['Free space']
         orgMask = self.getImageFromURL(maskUrl) #Retrieve the original mask

         #Failed to download the mask
         if (orgMask == None):
            print("Downloading the mask " + str(imgNum) + "  failed")
            continue

         #Save the original image
         #print("Saving original image")
         newImg = Image.open(orgImg[0])
         newImg = newImg.convert("RGB")   #Convert the image to RGB format
         newImg = newImg.resize((imgWidth, imgHeight))  #Resize the image to be 640x360
         newImg.save(dirPath + "/Input_Images/" + imgName)
         newImg.close()

         #Save the mask for the image
         #print("Saving mask")
         newMask = Image.open(orgMask[0])
         newMask = newMask.convert('L')   #Convert the mask to grayscale format
         newMask = newMask.resize((imgWidth, imgHeight))  #Resize the mask to be 640x360
         newMask.save(dirPath + "/Image_Masks/" + row['ID'] + "_mask.jpg")

         maskDataFile = open(dirPath + "/Mask_Data/" + row['ID'] + "_mask_data.txt", 'w')
         #Get the pixel array and witdh/height of the original image
         pixels = newMask.load()
         width, height = newMask.size

         #Extract the mask data
         #print("Extracting points")
         points = self.extractMaskPoints(pixels, width, height, numOutputs)

         #Load the image to draw the extracted mask data on for validation
         validationMaskImage = cv2.imread(dirPath + "/Input_Images/" + imgName)

         '''Write the mask data to a file in x,y column format, where y is normalized between 0 and 1 and
            draw the extracted mask points over the original image'''
         x = 0;
         stepSize = imgWidth // numOutputs
         #print("Drawing points")
         for y in points:
            #Draw a circle on the original image to validate the correct mask data is extracted
            validationMaskImage = cv2.circle(validationMaskImage, (x, round(y * (height-1))), 1, (0, 255, 0), -1)

            #Write the mask point to the file
            maskDataFile.write(str(x) + ',' + str(y) + '\n')
            x += stepSize

         #Save the overlayed image
         cv2.imwrite(dirPath + "/Mask_Validation/" + row['ID'] + "_validation_mask.jpg",
                     validationMaskImage)

         #Check if the mask for the current image can be whitelisted
         #print("Validating mask")
         inValid = self.checkForBlackEdges(pixels, width, height)
         if not inValid:
            newMask.save(dirPath + "/Whitelist_Masks/" + row['ID'] + "_mask.jpg")
            whiteList.write(row['ID'] + '.jpg\n')
         else:
            newMask.save(dirPath + "/Blacklist_Masks/" + row['ID'] + "_mask.jpg")
            print("Potential labeling error for image: " + row['ID'])
            blackList.write(row['ID'] + '.jpg\n')

         maskDataFile.close()
         newMask.close()

      imageFile.close()
      whiteList.close()
      blackList.close()

      return

   '''Extract 128 points representing the bounds of the image mask between 0-1
      Takes in the pixel array representing the mask, and the width & height of the mask'''
   def extractMaskPoints(self, pixels, width, height, numOutputs):
      found = False
      maskData = []
      stepSize = width // numOutputs

      #Find the numOutputs points along the image that represent the boundary of free space
      #Find the boundary goint from bottom (height - 1) to top (0)
      for x in range(0, width, stepSize):
         for y in range(height-1, -1, -1):
            color = pixels[x,y]
            if color == 0:
               break
         maskData.append(y / (height - 1))
         found = False

      return maskData

   '''Download the image from the given URL. Return None if the request fails more than 5 times'''
   def getImageFromURL(self, url):
      image = None
      trys = 0
      #Attempt to download the image 5 times before quitting
      while (trys < 5):
         try:
            return urllib.request.urlretrieve(url) #Retrieve the image from the URL
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

   '''Return True if there is a black edge along the sides or bottome of the image
      represented by the pixels array'''
   def checkForBlackEdges(self, pixels, width, height):
      blackEdge = False

      #Check for black edge along bottom
      for x in range(width):
         if pixels[x, height - 1] < 128:
            blackEdge = True
         else:
            blackEdge = False
            break

      #There is a black border along the bottom of the image
      if blackEdge:
         return True

      #Check for black border on the left side of the image
      for y in range(height):
         if pixels[0, y] < 128:
            blackEdge = True
         else:
            blackEdge = False
            break

      #There is a black border along the left side of the image
      if blackEdge:
         return True

      #Check for black border on the right side of the image
      for y in range(height):
         if pixels[width - 1, y] < 128:
            blackEdge = True
         else:
            blackEdge = False
            break

      return blackEdge

   '''Split the newly downloaded images into training and validation directories'''
   def splitImages(self, validPercent):
      dirPath = os.getcwd() #Get the current directory path

      #Remove any existing training and validation directories and remake them
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

      #List all the images that have been downloaded, now and previously
      images = os.listdir(dirPath + "/Input_Images")

      #Determine how many images to use for validation
      numValid = round(len(images) * validPercent)
      numChosen = 0;

      #Save images to the validation directory randomly
      while numChosen <= numValid-1:
         index = randint(0, len(images)-1)
         imgName = images.pop(index)
         img = Image.open(dirPath + "/Input_Images/" + imgName)
         img.save(dirPath + "/Validation_Images/" + imgName)
         numChosen += 1;

      #Save the rest of the images to the training directory
      for imgName in images:
         img = Image.open(dirPath + "/Input_Images/" + imgName)
         img.save(dirPath + "/Training_Images/" + imgName)

      return

if __name__ == '__main__':
   d = DataExtractor()
   d._main()
