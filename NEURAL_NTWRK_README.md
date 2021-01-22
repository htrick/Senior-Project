## inference.py
**Functionality**: Using the provided config file, make a prediction of the num_outputs points representing the free space in each image in the directory in the config file. Then overlay the num_outputs points onto the original image and save it. Also display a ray pointing in the direction the robot should travel determined by the driving policy and a line across the image where the highest point is. If the -n flag is specified, limit the number of images to make inferences for. If the -r flag is specified, rank the images on how close the 4 models predict the free space boundary.
```
Usage: python3 inference.py -c <config_file> [-n <int>] [-w <weights>] | [-r]
```

## Comand Line Arguments
* -c: A flag argument to specify the next argument given is the config file to use.
* -w: An optional flag argument to specify a different weights file to use over the file present in the config file
* -n: An optional flag argument to specify the number of images to make predictions for
* -r: An optional flag argument to instead rank images based on their performance in the 4 models used from the weight files.

## trajectory.py
**Functionality**: A class that implements a driving policy. The driving policy selects the point that is furthest away from the robot's current position to drive towards. If there is an obstacle in front of the robot, the policy will have the robot turn left or right based on which side has the point furthest away from the robot. The class needs to be given the width and height of images used, and the number of outputs from the model when created.

