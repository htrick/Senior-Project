import os
import cv2
import sys
import keras
import numpy as np
from imgaug import augmenters as iaa
from imgaug import parameters as iap
from keras.utils import Sequence, to_categorical
from sklearn.preprocessing import LabelEncoder

class DataGenerator(Sequence):
    def __init__(self, dir_path, batch_size=16, aug_freq=0.5, image_width=640, image_height=360, shuffle=True):
        self.dir_path = dir_path
        self.batch_size = batch_size
        self.image_width = image_width
        self.image_height = image_height
        self.aug_freq = aug_freq
        self.shuffle = shuffle
        self.__get_all_paths()
        #self.__target_encoding()
        #self.__augmentation_operations()
        self.on_epoch_end()

    def on_epoch_end(self):
        self.indices = np.arange(len(self.paths_list))
        if self.shuffle:
            np.random.shuffle(self.indices)

    # def get_num_classes(self):
    #     return self.num_classes

    def __get_all_paths(self):
        if not os.path.exists(self.dir_path):
            raise 'input dir does not exist'
        print('** start getting images/labels path.......')
        self.paths_list = []
        self.label_list = []
        for i, _id in enumerate(os.listdir(self.dir_path)):
            #add the image filename to the list
            _id_path = os.path.join(self.dir_path, _id)
            self.paths_list.append(_id_path)
            #print(_id_path)

            #read the data from the mask file
            f = open("../Mask_Data/"+_id.replace(".jpg","")+"_mask_data.txt","r")
            temp_list = []
            for x in f:
                x = x.split(",")[1]
                x = x.strip()
                temp_list.append(float(x))
            #print(temp_list)
            self.label_list.append(temp_list)

        return None

    # def __target_encoding(self):
    #     self.le = LabelEncoder()
    #     self.le.fit(self.label_list)

    def __len__(self):
        return int(np.floor(len(self.paths_list) / self.batch_size))

    def __getitem__(self, index):
        # ** get batch indices
        start_index = index * self.batch_size
        end_index = min((index + 1) * self.batch_size, len(self.paths_list))
        indices = self.indices[start_index:end_index]

        # ** get batch inputs
        paths_batch_list = [self.paths_list[i] for i in indices]
        label_batch_list = [self.label_list[i] for i in indices]

        # ** generate batch data
        X, y = self.__data_generation(paths_batch_list, label_batch_list)

        return X, y

    def __data_generation(self, paths_batch_list, label_batch_list):
        X = np.empty((self.batch_size, self.image_height, self.image_width, 3))
        X = X.astype('float32')
        y = np.empty((self.batch_size, len(label_batch_list[0])))
        y = y.astype('float32')
        #print (len(self.label_list[0]))

        for i, (path, label) in enumerate(zip(paths_batch_list, label_batch_list)):
            X[i, :, :, :] = self.__image_augmentation(cv2.imread(path))
            y[i, :] = label

        return X, y

    def __image_augmentation(self, img):
        if img is None:
            raise '** Failed to read image.'
        # to rgb
        img_copy = img.copy()
        img_copy = img_copy[:, :, ::-1]

        # do aug
        if False and self.aug_freq > np.random.uniform(0, 1, 1): #for now, do not run the augmentor
            img_aug = self.aug_ops.augment_image(img_copy)
        else:
            img_aug = img_copy

        img_norm = self.__normalize(img_aug)

        #return cv2.resize(img_norm, (self.image_height, self.image_width))
        #return cv2.resize(img_norm, (self.image_width, self.image_height))
        return img_norm

    # def __augmentation_operations(self):
    #     self.aug_ops = iaa.Sequential(
    #         [
    #             self.__sometimes(iaa.Fliplr(1), 0.5),
    #             self.__sometimes(iaa.Affine(scale=iap.Uniform(1.0, 1.2).draw_samples(1)), 0.3),
    #             self.__sometimes(iaa.AdditiveGaussianNoise(scale=0.05*255), 0.2),
    #             self.__sometimes(iaa.OneOf(
    #                 [
    #                     iaa.CropAndPad(percent=(iap.Uniform(0.0, 0.20).draw_samples(1)[0],
    #                                             iap.Uniform(0.0, 0.20).draw_samples(1)[0]),
    #                                    pad_mode=["constant"],
    #                                    pad_cval=(0, 128)),
    #                     iaa.Crop(percent=(iap.Uniform(0.0, 0.15).draw_samples(1)[0],
    #                                       iap.Uniform(0.0, 0.15).draw_samples(1)[0]))
    #                 ]
    #             )),
    #             self.__sometimes(iaa.OneOf([
    #                 iaa.LogContrast(gain=iap.Uniform(0.9, 1.2).draw_samples(1)),
    #                 iaa.GammaContrast(gamma=iap.Uniform(1.5, 2.5).draw_samples(1))]))
    #         ],
    #         random_order=True
    #     )
    #     return None

    def __normalize(self, img):
        return img / 255.0

    def __sometimes(self, aug, prob=0.5):
        return iaa.Sometimes(prob, aug)
