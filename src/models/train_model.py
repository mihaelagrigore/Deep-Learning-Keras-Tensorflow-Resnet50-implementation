import os
import sys
import yaml
import time
import logging
from pathlib import Path
from typing import Tuple
import matplotlib.pyplot as plt
from ast import literal_eval as make_tuple
from src.utils.basic_functions import get_config

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from src.models.resnet50 import ResNet50

# folder to load config file
root = Path(__file__).parents[2]
CONFIG_FILE = os.path.join(root, 'config.yaml')


# use Keras api to create train and validation tf.data.Dataset
# instances from the directory of training images
def get_tf_datasets(data_dir: str, validation_split: float, label_mode:str, image_size: Tuple[int, int], batch_size: int):
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=validation_split,
        subset="training",
        label_mode=label_mode,  # default mode is 'int' label
        seed=123,
        image_size=image_size,
        batch_size=batch_size
    )

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=validation_split,
        subset="validation",
        label_mode=label_mode,
        seed=123,
        image_size=image_size,
        batch_size=batch_size
    )

    logging.debug(f'val split:{validation_split}, image_size:{image_size}, batch_size:{batch_size}')

    return train_ds, val_ds


# performs confiurations and inits necessary before model training
# instantiates, configs and trains a model
# return trained model instance and results
def train_model(img_fld: str, validation_split: float, label_mode: str, batch_size: int, image_size: Tuple[int, int], channels: int, num_classes: int, epochs: int):
    # split dataset into training and validation set
    train_ds, val_ds = get_tf_datasets(img_fld, validation_split, label_mode, image_size, batch_size)

    # normalize images (rescale pixel values from [0,255] to [0,1])
    normalization_layer = layers.experimental.preprocessing.Rescaling(1. / 255)
    norm_train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    norm_val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

    # instantiate model
    input_size = (image_size[0], image_size[1], channels)
    model = ResNet50.build(input_size, num_classes)

    # config model
    model.compile(
        optimizer='adam',  # optimizer
        loss='categorical_crossentropy',  # loss function to optimize
        metrics=['accuracy']  # metrics to monitor
    )

    # set prefetching for performance optimization
    AUTOTUNE = tf.data.AUTOTUNE
    norm_train_ds = norm_train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    norm_val_ds = norm_val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    # configure early stopping
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",  # monitor validation loss (that is, the loss computed for the validation holdout)
            min_delta=1e-2,  # "no longer improving" being defined as "an improvement lower than 1e-2"
            patience=10,  # "no longer improving" being further defined as "for at least 10 consecutive epochs"
            verbose=1
        )
    ]

    # check if GPU is available
    device_name = tf.test.gpu_device_name()
    logging.info(f'Training on: {device_name}')

    # train model
    start = time.time()
    with tf.device(device_name):
        history = model.fit(
            norm_train_ds,
            validation_data=norm_val_ds,
            epochs=epochs,
            callbacks=callbacks,
        )
    stop = time.time()
    logging.info(f'Training on {device_name} took: {(stop - start) / 60} minutes')

    return model, history, train_ds.class_names


# plot model training results:
#  accuracy on training and validations sets per epoch
#  loss on training and validations sets per epoch
#  save image plot to model output folder
def plot_training_history(history, save_to: str):
    # summarize history for accuracy
    plt.subplot(2, 1, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')

    # summarize history for loss
    plt.subplot(2, 1, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')

    plt.savefig(os.path.join(save_to, 'train_output.png'))


# main function that handles everything related to model training
def do_train(fld: str = None, epochs: int = None, batch_size: int = None, image_size: Tuple[int, int] = None,
             channels: int = None, validation_split: float = None, log_level: int = None):

    # read config file
    yaml_data = get_config(CONFIG_FILE)
    label_mode = yaml_data['training']['label_mode']
    if not fld:
        fld = yaml_data['dataset']['default_train_data']
    img_fld = os.path.join(root, yaml_data['dataset']['proc_data_path'], fld)
    if not validation_split:
        validation_split = yaml_data['training']['validation_split']
    if not epochs:
        epochs = yaml_data['training']['epochs']
    if not batch_size:
        batch_size = yaml_data['training']['batch_size']
    if not image_size:
        image_size = yaml_data['model']['input_size']
        image_size = make_tuple(image_size)
    if not channels:
        channels = yaml_data['model']['channels']
    model_out_folder = yaml_data['model']['save_path']
    if not log_level:
        log_level = yaml_data['logging']['level']

    # set up logging
    save_to = os.path.join(root, model_out_folder, time.strftime("%Y%m%d%H%M"))
    try:
        os.mkdir(save_to)
    except OSError:
        print("Creation of the directory %s failed" % save_to)
    logging.basicConfig(level=log_level,
                        #filename=os.path.join(save_to, 'training.log'),
                        stream=sys.stdout)

    logging.debug(f'training folder: {fld}, epochs: {epochs}, batch_size: {batch_size}, image_size: {image_size}, '
                  f'channels: {channels}, validation_split: {validation_split}, log_level: {log_level}')

    # number of classes in training dataset
    num_classes = len([f.path for f in os.scandir(img_fld) if f.is_dir()])

    # train model
    model, history, class_names = train_model(img_fld,  # name of folder containing training dataset
                                 validation_split, label_mode, batch_size,  # tf dataset config parameters
                                 image_size, channels, num_classes,  # model instance config params
                                 epochs)   # training params

    # save plot of model training results
    plot_training_history(history, save_to)

    # save trained model
    model.save(save_to)

    # save model input config
    cfg = {'training': {'image_size': image_size, 'channels': channels, 'class_names': class_names}}
    with open(os.path.join(save_to, 'cfg.yaml'), 'w') as outfile:
        yaml.dump(cfg, outfile, default_flow_style=False)

    return model, history

