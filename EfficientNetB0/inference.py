import efficientnetb0_pretrained
import os
import numpy as np
import cv2

def main(imageHeight, imageWidth, numOutputs, inputPath, outputPath):
    if not os.path.isdir(outputPath):
        os.mkdir(outputPath)

    m = efficientnetb0_pretrained.EfficientNetB0_Pretrained(shape=(imageHeight,imageWidth,3), num_outputs=numOutputs)
    model = m.build()

    for file in os.listdir(inputPath):
        imagePath = os.path.join(inputPath, file)

        X = np.empty((1, imageHeight, imageWidth, 3), dtype='float32')
        X[0, :, :, :] = cv2.imread(imagePath).copy()[:, :, ::-1] / 255.0

        prediction = model.predict(X)[0]

        validationMaskImage = cv2.imread(imagePath)
        x = 0
        for i in range(len(prediction)):
            y = int(prediction[i] * imageHeight)
            validationMaskImage = cv2.circle(validationMaskImage, (x, y), 1, (0, 255, 0), -1)
            x += imageWidth // numOutputs
        cv2.imwrite('{}/{}_inference.jpg'.format(outputPath, file.split('.')[0]), validationMaskImage)

if __name__ == '__main__':
    main(360, 640, 128, '../Input_Images', 'Inference_Images')
