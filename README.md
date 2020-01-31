# Senior-Project
***
## Requirments
```
python3
PIL >= 7.0.0
opencv-python >= 4.1.2
```

## dataExtractor.py
Functionality: Download the images and extract mask information from the given .csv file. If the optional -d flag is given, re-download all the images in the file.
```
Usage: python3 dataExtractor.py -c | -a <filename.csv> | -n <filename.csv>
``` 

## Command Line Arguments
* filename.csv: The name of the .csv file to download and extract image and mask data from
* -c: A flag argument to remove all the directories and files containing image data, a way to 'clean' all directory
* -a: A flag argument to re-download all of the images from the given .csv file
* -n: A flag argument to skip already downloaded images and their associated data and download any new images and their associated data

## Directories
The images and their mask data gathered from the dataExtractor.py script are stored in the following directories and .txt files. All directories and files are stored relative the path where the dataExtractor.py script was called from.
* **Input_Images**: Stores a copy of the original image to be used for input.
* **Image_Masks**: Stores a copy of all the masks associated with each of the input images.
* **Mask_data**: Stores .txt files for all of the image masks with 128 points extracted from the respective image's mask stored in x,y column format. Each x,y pair is normalized to between 0 and 1.
* **Mask_Validation**: Stores a copy of each input image with the extracted 128 points from the image's mask overlayed as green circles to validate the correct data is extracted from the mask.
* **Whitelist_Masks**: Stores all of the image masks for images that have been whitelisted to validate no image will be used whose original mask was made incorrectly.
* **Blacklist_Masks**: Stores all the image masks for images that have been blacklisted to validate that ther are no images whose mask was made correctly will not used.
* **Whitelist_Images.txt**: Stores the names of all the images that have been whitelisted to be used as input.
* **Blacklist_Images.txt**: Stores the names of all the images that have been blacklisted to not be used as input because they might have a labeling error.