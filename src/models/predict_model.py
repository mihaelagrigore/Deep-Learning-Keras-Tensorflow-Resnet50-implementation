import os
import cv2
import yaml
import numpy as np
import tensorflow as tf
from src.utils.basic_functions import get_config


# helper class for properly reading tuples from yaml config files
class PrettySafeLoader(yaml.SafeLoader):
    def construct_python_tuple(self, node):
        return tuple(self.construct_sequence(node))


# read config params used when the model was built
def get_model_config(model_path: str):
    # read model yaml config file
    # add custom loader for reading tuples correctly
    PrettySafeLoader.add_constructor(
        u'tag:yaml.org,2002:python/tuple',
        PrettySafeLoader.construct_python_tuple)
    with open(os.path.join(model_path, 'cfg.yaml')) as file:
        model_yaml = yaml.load(file, Loader=PrettySafeLoader)

    image_size = model_yaml['training']['image_size']
    channels = model_yaml['training']['channels']
    class_names = model_yaml['training']['class_names']

    return image_size, channels, class_names


# return pretrained model path (relative to root folder)
# if no model path was provided by user, choose
# a pretrained model from the default trained models folder
def get_model_path(model_path: str = None) -> str:
    yaml_data = get_config(CONFIG_FILE)
    checkpoint_dir = yaml_data['model']['save_path']

    if not model_path:
        models_list = os.listdir(checkpoint_dir)
        model_path = models_list[0]

    return os.path.join(checkpoint_dir, model_path)

# return test image path (relative to root folder)
# if no image path was provided by user, choose
# an image from the default test folder
def get_image_path(image_path: str=None):
    if not image_path:
        yaml_data = get_config(CONFIG_FILE)
        image_path = yaml_data['dataset']['default_test_path']
        img_list = os.listdir(image_path)
        img = img_list[0]
        image_path = os.path.join(image_path, img)
    return image_path


# preprocess image before feeding to model for predicion
def preprocess_image(image_path, image_size):
    img = cv2.imread(image_path)
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    x = cv2.resize(rgb_img, image_size, interpolation=cv2.INTER_AREA)
    x = np.expand_dims(x, axis=0)
    x = x / 255.0

    return x


# load trained model
def load_model(model_path: str):
    model_path = get_model_path(model_path)

    print(f'Load pretrained model from: {model_path}')

    model = tf.keras.models.load_model(model_path)

    return model


# make prediction on new image
def predict_model(model, image_path: str, model_path: str):
    # get path to pretrained model relative to root folder
    model_path = get_model_path(model_path)

    # read values for some parameters used when the model was built
    # so we know how to preprocess images we feed for prediction
    image_size, channels, class_names = get_model_config(model_path)

    # preprocess the raw image before we can feed it to the ResNet50 model for predictions
    x = preprocess_image(image_path, image_size)

    # predict the probabilities for belonging to target classes
    preds = model.predict(x)

    return preds, class_names
