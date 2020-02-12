# Senior-Project
***
## Requirements
```
python3
PIL >= 7.0.0
opencv-python >= 4.1.2
```

## dataExtractor.py
**Functionality**: Download the images and extract mask information from the given .csv file. If the -a flag is given, re-download all the images in the file. If the -n flag is given, download only new images in the file. If the -c flag is given, remove all the images and directories. If the -p flag is given, the following float will determine the percentage of images to use for validation. On each run of the script with the -a or -n flag will remove the old training and validation sets and generate a new split of training and validation images based on all the images that are in the Input_Images directory.
```
Usage: python3 dataExtractor.py -c | -a <filename.csv> -f <filename> [-p <0-1>] | -n <filename.csv> -f <filename> [-p <0-1>]
``` 

## Command Line Arguments
* filename.csv: The name of the .csv file to download and extract image and mask data from
* -c: A flag argument to remove all the directories and files containing image data, a way to 'clean' all directory
* -a: A flag argument to re-download all of the images from the given .csv file that follows
* -n: A flag argument to skip already downloaded images and their associated data and download any new images and their associated data from the given .csv file that follows
* -p: An optional flag argument to use with -a or -n to specify what percentage of the downloaded images to set aside for validation, percentage is to be a float between 0-1.0. Default percentage is 0.15
* -f: A flag argument to specify the config file to use to determine the height and width of the images to save, and the number of points to extract from the image masks

## inference.py
**Functionality**: Using the provided config file, make a prediction of the num_outputs points representing the free space in each image in the directory in the config file. Then overlay the num_outputs points onto the original image and save it.
```
Usage: python3 inference.py -c <config_file> [-w <weights>]
```

## Comand Line Arguments
* -c: A flag argument to specify the next argument given is the config file to use.
* -w: An optional flag argument to specify a different weights file to use over the file present in the config file

## Config File

### **model**
Argument|Description|Type|Default
---|---|---|---
input_width|Input width of MobileNet V3 model.|int|640
input_height|Input height of MobileNet V3 model.|int|360
num_outputs|Number of points genreated to define the boundary of free space|int|128

### **gpu**
Argument|Description|Type|Default
---|---|---|---
gpu|Specify a GPU.|str|0

### **inference**
Argument|Description|Type|Default
---|---|---|---
weight_path|Saved weights of MobileNet V3 model.|str|weights/2_5_ep200-loss0.014_1.h5
image_path|Path to the images to make predictions for with the MobileNet V3 model.|str|../Input_Images
inference_dir|Path to save the model predictions|str|Model_Predictions

## Directories
The images and their mask data gathered from the dataExtractor.py script are stored in the following directories and .txt files. All directories and files are stored relative the path where the dataExtractor.py script was called from.
* **Input_Images**: Stores a copy of the original image to be used for input.
* **Image_Masks**: Stores a copy of all the masks associated with each of the input images.
* **Mask_data**: Stores .txt files for all of the image masks with 128 points extracted from the respective image's mask stored in x,y column format. Each y in the data is normalized to between 0 and 1.
* **Mask_Validation**: Stores a copy of each input image with the extracted 128 points from the image's mask overlayed as green circles to validate the correct data is extracted from the mask.
* **Whitelist_Masks**: Stores all of the image masks for images that have been whitelisted to validate no image will be used whose original mask was made incorrectly.
* **Blacklist_Masks**: Stores all the image masks for images that have been blacklisted to validate that ther are no images whose mask was made correctly will not used.
* **Whitelist_Images.txt**: Stores the names of all the images that have been whitelisted to be used as input.
* **Blacklist_Images.txt**: Stores the names of all the images that have been blacklisted to not be used as input because they might have a labeling error.
* **Training_Images**: Stroes a copy of the original input images that are to be uesed for training.
* **Validation_Images**: Stroes a copy of the original input images that are to be used for validation.
* **Model_Prediction**: Stores a copy of the predictions made by the model for each of the images in the "image_path" directory in the config file used.
