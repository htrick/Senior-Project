import urllib.request
import csv
from PIL import Image  

def main():
   imageFile = open("export-2020-01-17T19_04_30.492Z.csv", 'r') #Open the csv file
   reader = csv.DictReader(imageFile)

   for row in reader:
      #Download the original image
      imgUrl = row['Labeled Data']
      orgImg = urllib.request.urlretrieve(imgUrl)  #Retrieve the original image
      newImg = Image.open(orgImg[0])
      newImg = newImg.convert("RGB")   #Convert the image to RGB format
      newImg = newImg.resize((640, 360))  #Resize the image to be 640x360
      newImg.save("C:\Senior Project/Input Images/" + row['ID'] + ".jpg")
      newImg.close()

      #Download the mask for the image
      maskUrl = row['Masks'].split('\"')[3]
      orgMask = urllib.request.urlretrieve(maskUrl) #Retrieve the original mask
      newMask = Image.open(orgMask[0])
      newMask = newMask.convert('1')   #Convert the mask to RGB format
      newMask = newMask.resize((640, 360))  #Resize the mask to be 640x360
      newMask.save("C:\Senior Project/Image Masks/" + row['ID'] + "_mask.jpg")
      
      #Extract the mask data
      pixels = newMask.load() # this is not a list, nor is it list()'able
      width, height = newMask.size
      found = False
      maskData = []

      #Find the 128 points along the image that represent the boundary of free space
      for x in range(0, width, 5):
         for y in range(height):
            color = pixels[x,y]
            if color > 0:
               break
         maskData.append(y)
         found = False

      maskDataFile = open("C:\Senior Project/Mask Data/" + row['ID'] + "_mask_data.txt", 'w')

      #Write the mask data to a file
      for i in range(128):
         maskDataFile.write(str(maskData[i]))
         if i != 127:
            maskDataFile.write(', ')
      maskDataFile.write('\n')

      maskDataFile.close()
      newMask.close()

   imageFile.close()
      

if __name__ == '__main__':
   main()