# Video Inference Package
***
## Requirements
```
python2
opencv >= 3.3.1-dev
keras >=  2.3.1
tensorflow >= 2.1.0
numPy >= 1.16.6
ROS >= Kinetic
```

## Setup
```
catkin_make
. ./devel/setup.bash
```

## videoInference.py
**Functionality**: Make inference for the images contained in the given ROS .bag file and publish the results. Depict on the images the boundary of free space and the predicted driving trajectory. Use the -c flag to specify the config file to use. The optional -w flag can be used to specify a weight file different than the one specified in the config file. The optional -b flag can be used to specify a .bag file different than the one specified in the config file. The optional -t flag can be used to specify different topics of the .bag file.
```
Usage: rosrun video_inference_p2 videoInference.py -c <config_file> [-t <bag_file_topics>] [-b <bag_file>] [-w <weights>]
```

## Command Line Arguments
* -c: A flag argument to specify the next argument given is the config file to use.
* -w: An optional flag argument to specify a different weights file to use over the file specified in the config file
* -b: An optional flag argument to specify a different .bag file to use over the file specified in the config file
* -t: An optional flag argument to specify a different topic to use when reading from the .bag file over the one specified in the config file

