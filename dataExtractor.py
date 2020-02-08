import urllib.request
import urllib.error
import csv
import cv2
import os
import sys
import shutil
from PIL import Image  
from random import randint

'''Extract input and expected output data from the csv file'''
def main():
   numArgs = len(sys.argv)
   validPercent = 0.15
   images = []

   args = sys.argv
   if (numArgs == 2):
      if (args[1] == '-c'):
         cleanData()
         return

   if (numArgs == 3 or numArgs == 5):
      #Download the images and their associate data
      if (args[1] == '-a'):
         downloadImageData(args[2], '-a');
      elif (args[1] == '-n'):
         downloadImageData(args[2], '-n');
      else:
         print("Usage: python3 dataExtractor.py -c | -a <filename.csv> [-p <0-1>] | -n <filename.csv> [-p <0-1>]")
         return

      #Determine new percentage of images to use fro validation
      if (numArgs == 5 and args[3] == '-p'):
         try:
            validPercent = float(args[4])
            if validPercent < 0 or validPercent > 1:
               print("Percentge used for the validation set must be a float between 0-1")
               return
         except:
            print("Percentge used for the validation set must be a float between 0-1")
            return

      #Split images into training and validation directories,
      #Creates new random splits on every call
      print("Splitting images into training and validation")
      splitImages(validPercent)
      return

   print("Usage: python3 dataExtractor.py -c | -a <filename.csv> [-p <0-1>] | -n <filename.csv> [-p <0-1>]")
   return

'''Remove all the directories and files containing data information'''
def cleanData():
   confirm = "None"
   while (confirm.lower() != 'y' and confirm.lower() != 'n'):
      confirm = input("Are you sure you want to delete all image directories and data? (y/n): ")  
   if (confirm == 'n'):
      return

   dirPath = os.getcwd() #Get the current directory path

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
      if os.path.isfile(dirPath + '/Whitelisted_Images.txt'):
         os.remove(dirPath + '/Whitelisted_Images.txt')
      if os.path.isfile(dirPath + '/Blacklisted_Images.txt'):
         os.remove(dirPath + '/Blacklisted_Images.txt')
   except OSError as err:
      print("Error: {0}".format(err))
      return

'''Download the image and mask data from the .csv file. Return the list of
   images that were downloaded'''
def downloadImageData(csvFile, flag):
   try:
      imageFile = open(csvFile, 'r') #Open the csv file
   except:
      print("Error opening file: " + csvFile)
      return

   reader = csv.DictReader(imageFile)
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
   except OSError as err:
      print("Error: {0}".format(err))
      return

   imgNum = 0
   for row in reader:
      imgNum += 1
      print("Image: " + str(imgNum), end = '')

      #The name of the original image
      imgName = row['ID'] + ".jpg"

      '''Check if current image is already downloaded and only new images 
         need to be download. If it exists, continue to the next image'''
      if (flag == '-n' and os.path.isfile(dirPath + "/Input_Images/" + imgName)):
         print(" Skipping Image")
         continue

      #Download the original image
      print(" Getting Original, ", end = '')
      imgUrl = row['Labeled Data']
      orgImg = getImageFromURL(imgUrl) #Retrieve the original image
      newImg = Image.open(orgImg[0])
      newImg = newImg.convert("RGB")   #Convert the image to RGB format
      newImg = newImg.resize((640, 360))  #Resize the image to be 640x360
      newImg.save(dirPath + "/Input_Images/" + imgName)
      newImg.close()

      #Download the mask for the image
      print("Getting Mask")
      maskUrl = row['Masks'].split('\"')[3]
      orgMask = getImageFromURL(maskUrl) #Retrieve the original mask
      newMask = Image.open(orgMask[0])
      newMask = newMask.convert('L')   #Convert the mask to grayscale format
      newMask = newMask.resize((640, 360))  #Resize the mask to be 640x360
      newMask.save(dirPath + "/Image_Masks/" + row['ID'] + "_mask.jpg")
      
      #Extract the mask data
      pixels = newMask.load()
      width, height = newMask.size
      found = False
      maskData = []

      #Load the image to draw the extracted mask data on for validation
      validationMaskImage = cv2.imread(dirPath + "/Input_Images/" + imgName)

      #Find the 128 points along the image that represent the boundary of free space
      for x in range(0, width, 5):
         for y in range(height-1, -1, -1):
            color = pixels[x,y]
            if color == 0:
               break
         #Draw a circle on the original image to validate the correct mask data is extracted
         validationMaskImage = cv2.circle(validationMaskImage, (x, y), 1, (0, 255, 0), -1)
         maskData.append(y)
         found = False

      #Save the overlayed image
      cv2.imwrite(dirPath + "/Mask_Validation/" + row['ID'] + "_validation_mask.jpg",
                  validationMaskImage)

      maskDataFile = open(dirPath + "/Mask_Data/" + row['ID'] + "_mask_data.txt", 'w')

      #Write the mask data to a file in x,y column format, where y is normalized between 0 and 1
      x = 0;
      for i in range(128):
         maskDataFile.write(str(x) + ',' + str(maskData[i]/(height-1)) + '\n')
         x += 5

      #Check if the mask for the current image can be whitelisted
      inValid = checkForBlackEdges(pixels, width, height)
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

'''Download the image from the given URL'''
def getImageFromURL(url):
   image = None
   while (image == None):
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

'''Return True if there is a black edge along the sides or bottome of the image
   represented by the pixels array'''
def checkForBlackEdges(pixels, width, height):
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
def splitImages(validPercent):
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
   main()