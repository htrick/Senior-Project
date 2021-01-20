import csv
import ast
import numpy as np

class Labelbox:

   # List of image task data
   def listImageTaskData(self, imageFile):
      return csv.DictReader(imageFile)

   # Get the ID of the image based on its task
   def getID(self, row):
      return row["ID"]

   # Check if the image has been approved and has a positive score
   def isApproved(self, row):
      # Get the review score of the image
      review = row['Reviews']
      review = ast.literal_eval(review)
      runningScore = 0
      numScores = len(review)
      for i in range(numScores):
         # Load the current entry as a dictionary
         entry = ast.literal_eval(str(review[i]))
         # Add the score of the entry to the running total
         runningScore += entry['score']

      # If the image has a non-positive score, do not download it
      if runningScore <= 0:
         return False
      return True
   
   # Get the URL for the original image
   def getImageURL(self, row):
      return row['Labeled Data']
   
   # Get the polygons for the mask data
   def getPolygons(self, row):
      # Get the mask labels
      freeSpace = row['Label']
      freeSpace = ast.literal_eval(freeSpace)
      freeSpace = freeSpace['Free space']

      # Get each polygon in the mask
      polygons = []
      numPolygons = len(freeSpace)
      for i in range(numPolygons):
         # Get the dictionary storing the points for the current polygon
         geometry = ast.literal_eval(str(freeSpace[i]))
         geometry = geometry['geometry']
         numPoints = len(geometry)

         # Form an array of points for the current polygon
         points = []
         for p in range(numPoints):
            point = ast.literal_eval(str(geometry[p]))
            x = point['x']
            y = point['y']
            points.append((x, y))

         # Change the points array to a numpy array
         points = np.array(points)
         polygons.append(points)

      return polygons
