import urllib.request
import urllib.error
import csv
import cv2
import os
import sys
from PIL import Image  

'''Extract input and expected output data from the csv file'''
def main():
   numArgs = len(sys.argv)
   if (numArgs < 2 or numArgs > 3):
      print("Usage: python3 dataExtractor.py <filename.csv> [-d]")
      return

   args = sys.argv
   if (numArgs == 3):
      if (args[2] != "-d"):
         print("Unknown flag: " + args[2])
         print("Usage: python3 dataExtractor.py <filename.csv> [-d]")
         return
      downloadImageData(args[1]);

'''Download the image and mask data from the .csv file'''
def downloadImageData(csvFile):
   try:
      imageFile = open(csvFile, 'r') #Open the csv file
   except:
      print("Error opening file: " + csvFile)
      return

   reader = csv.DictReader(imageFile)
   try:
      whiteList = open("Whitelisted_Images.txt", 'w')
      blackList = open("Blacklisted_Images.txt", 'w')
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

   imgNum = 1
   for row in reader:
      print("Image: " + str(imgNum), end = '')
      #Download the original image
      print(" Getting Original, ", end = '')
      imgUrl = row['Labeled Data']
      #orgImg = urllib.request.urlretrieve(imgUrl)  #Retrieve the original image
      orgImg = getImageFromURL(imgUrl) #Retrieve the original image
      newImg = Image.open(orgImg[0])
      newImg = newImg.convert("RGB")   #Convert the image to RGB format
      newImg = newImg.resize((640, 360))  #Resize the image to be 640x360
      newImg.save(dirPath + "/Input_Images/" + row['ID'] + ".jpg")
      newImg.close()

      #Download the mask for the image
      print("Getting Mask")
      maskUrl = row['Masks'].split('\"')[3]
      #orgMask = urllib.request.urlretrieve(maskUrl) #Retrieve the original mask
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
      validationMaskImage = cv2.imread(dirPath + "/Input_Images/" + row['ID'] + ".jpg")

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

      #Write the mask data to a file in x,y column format
      x = 0;
      for i in range(128):
         maskDataFile.write(str(x/(width-1)) + ',' + str(maskData[i]/(height-1)) + '\n')
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

      imgNum += 1

   imageFile.close()
   whiteList.close()
   blackList.close()

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

if __name__ == '__main__':
   main()