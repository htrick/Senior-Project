import json
import numpy as np

class Scale_ai:

   # Get the name of the location of the image
   def getLocationName(self):
      return "scale.ai"

   # List of image task data
   def listImageTaskData(self, imageFile):
      return json.load(imageFile)

   # Get the ID of the image based on its task
   def getID(self, row):
      return row["task_id"]

   # Check if the image has been approved and has a positive score
   def isApproved(self, row):
      return row["customer_review_status"] == "accepted" or row["customer_review_status"] == "fixed"
   
   # Get the URL for the original image
   def getImageURL(self, row):
      return row["params"]["attachment"]
   
   # Get the polygons for the mask data
   def getPolygons(self, row):
      polygons = []
      for polygon in row["response"]["annotations"]:
         points = []
         for point in polygon["vertices"]:
            points.append((int(round(point["x"])), int(round(point["y"]))))
         polygons.append(np.array(points))

      return polygons
