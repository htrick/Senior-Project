import pretrained_model
import os, sys
import numpy as np
import cv2
import torch
from torchvision import transforms
from PIL import Image
import statistics

def main(imageHeight, imageWidth, numOutputs, inputPath, outputPath):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if not os.path.isdir(outputPath): # make the output inference image directory
        os.mkdir(outputPath)

    # if the model name is passed on the command line, for example:
    # 'python3 inference.py ref.pt'
    # then load the PyTorch model 'ref.pt'
    if len(sys.argv) == 2:
        model = torch.load(sys.argv[1],map_location=device) # load the trained model
        model.eval() #switch the model to inference mode
    else:
        m = pretrained_model.Pretrained_Model(shape=(imageHeight,imageWidth,3), num_outputs=numOutputs)
        model = m.build()
        model.eval() #switch the model to inference mode

    model.to(device) #send the model the GPU if available

    # Convert the image to (C, H, W) Tensor format and divide by 255 to
    # normalize to [0, 1.0]
    preprocess = transforms.Compose([
       transforms.ToTensor(),
       transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
       ])

    for file in os.listdir(inputPath):
        imagePath = os.path.join(inputPath, file)

        input_image = Image.open(imagePath) #use PIL, so it is RGB format
        input_tensor = preprocess(input_image)
        input_tensor = input_tensor.unsqueeze(0) #add 0th dimension

        input_tensor = input_tensor.to(device) #send to the GPU

        with torch.no_grad(): # run a forward pass through the model
            prediction = model(input_tensor)

        p_list = prediction.tolist()[0] # convert tensor to a Python list
        # print (prediction.shape)

        # create the inference image and draw the predicted data points
        validationMaskImage = cv2.imread(imagePath)
        x = 0
        for i in range(len(p_list)):
            y = int(p_list[i] * imageHeight)
            validationMaskImage = cv2.circle(validationMaskImage, (x, y), 1, (0, 255, 0), -1)
            x += imageWidth // numOutputs
        cv2.imwrite('{}/{}_inference.jpg'.format(outputPath, file.split('.')[0]), validationMaskImage)

def compute_variance(imageHeight, imageWidth, numOutputs, inputPath):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load each model in the 'models' directory
    models = []
    for model_file in os.listdir('models'):
        model = torch.load(os.path.join('models', model_file),map_location=device)
        model.eval()
        model.to(device)
        models.append(model)
    
    preprocess = transforms.Compose([
       transforms.ToTensor(),
       transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
       ])
    
    variance_file = open('variance.csv', 'w')

    # For each input file
    for file in os.listdir(inputPath):
        imagePath = os.path.join(inputPath, file)
        input_tensor = preprocess(Image.open(imagePath)).unsqueeze(0).to(device)

        image_variance = 0
        points = [[]] * numOutputs

        # For each model, add the output to the points list
        for model in models:
            with torch.no_grad():
                p_list = model(input_tensor).tolist()[0]
            for i in range(len(p_list)):
                points[i].append(p_list[i])
        
        # Compute the variance at each output point and sum them together
        for output in points:
            image_variance += statistics.variance(output)
        
        variance_file.write('{},{}\n'.format(file, str(image_variance)))
    
    variance_file.close()

if __name__ == '__main__':
    if len(sys.argv) == 2 and sys.argv[1] == '-variance':
        compute_variance(360, 640, 80, '../Unlabeled')
    else:
        #main(360, 640, 128, '../Validation_Images', 'Inference_Images')
        main(360, 640, 80, '../Validation_Images', 'Inference_Images')
