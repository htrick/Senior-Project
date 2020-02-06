import os
import sys
import logging
import argparse
import configparser
import numpy as np
from keras.models import load_model
from src.generator import DataGenerator
from src.learning_rate_schedule import learning_rate_scheduler
from src.MobileNet_V3 import build_mobilenet_v3, Hswish
from keras.optimizers import Adam
from keras.callbacks import (ModelCheckpoint,
                             LearningRateScheduler,
                             ReduceLROnPlateau,
                             EarlyStopping)

logging.basicConfig(level=logging.INFO)

def _main(args):

    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

    # ** get configuration
    config_file = args.conf
    config_client = configparser.ConfigParser()
    config_client.read(config_file)

    # ** set gpu
    os.environ['CUDA_VISIBLE_DEVICES'] = config_client.get('gpu', 'gpu')

    # ** MobileNet V3 configuration
    input_width = config_client.getint('model', 'input_width')
    input_height = config_client.getint('model', 'input_height')
    model_size = config_client.get('model', 'model_size')
    pooling_type = config_client.get('model', 'pooling_type')
    num_outputs = config_client.getint('model', 'num_outputs')

    # ** training configuration
    epochs = config_client.getint('train', 'epochs')
    batch_size = config_client.getint('train', 'batch_size')
    save_path = config_client.get('train', 'save_path')

    # ** Dataset
    train_directory = config_client.get('data', 'train')
    valid_directory = config_client.get('data', 'valid')

    # ** initialize data generators
    train_generator = DataGenerator(dir_path=train_directory, batch_size=batch_size, aug_freq=0, image_width=640, image_height=360)
    valid_generator = DataGenerator(dir_path=valid_directory, batch_size=batch_size, aug_freq=0, image_width=640, image_height=360)

    # ** initalize model
    try:
        #model = load_model(os.path.join(ROOT_DIR, config_client.get('train', 'pretrained_path')))
        model = load_model(os.path.join(ROOT_DIR, config_client.get('train', 'pretrained_path')),
                           custom_objects={'Hswish':Hswish})
    except Exception as e:
        print("Error: {0}".format(e))
        logging.info('Failed to load pre-trained model.')
        model = build_mobilenet_v3(input_width, input_height, num_outputs, model_size, pooling_type)

    #model.compile(optimizer=Adam(lr=3e-3), loss='mean_absolute_error', metrics=['accuracy'])
    model.compile(optimizer=Adam(), loss='mean_absolute_error', metrics=['accuracy'])

    # ** setup keras callback
    filename = 'ep{epoch:03d}-loss{loss:.3f}.h5'
    weights_directory = os.path.join(ROOT_DIR, 'weights')

    if not os.path.exists(weights_directory):
      os.mkdir(weights_directory)

    save_path = os.path.join(weights_directory, filename)
    checkpoint = ModelCheckpoint(save_path, monitor='loss', save_best_only=True, period=50)
    scheduler = LearningRateScheduler(learning_rate_scheduler)
    # reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1)
    # early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=25, verbose=1)

    #print model information
    print(model.summary())
    #sys.exit()

    # ** start training
    model.fit_generator(generator       = train_generator,
                        epochs          = epochs,
                        #callbacks       = [checkpoint, scheduler],
                        callbacks       = [checkpoint],
                        )

    model.save(os.path.join(ROOT_DIR, save_path))

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='')
    argparser.add_argument('-c', '--conf', help='path to configuration file')

    args = argparser.parse_args()
    _main(args)

