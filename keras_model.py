import keras
from keras.models import Model, Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import to_categorical
from keras import backend as K
from keras import applications
from keras.callbacks import CSVLogger
import re
import cv2
import numpy as np
import os
import yaml

import constants
from utils import Utils

class KerasModel():

  def run_all_experiments(overwrite=False):
    configs = yaml.load(open('experiments/configs.yml'))
    existing_result_files = os.listdir('experiments')
    for experiment_id in configs.keys():
      result_file_name = experiment_id + '.csv'
      print((result_file_name in existing_result_files) and not overwrite)
      if (result_file_name in existing_result_files) and not overwrite:
        continue

      config = configs[experiment_id]
      KerasModel.run(epochs=config['epochs'],
                     base_model=config['base_model'],
                     base_model_layer=config['base_model_layer'],
                     learning_rate=config['learning_rate'],
                     result_file="experiments/{}".format(result_file_name))

  def run(epochs=100,
          batch_size=-1,
          base_model='VGG16',
          base_model_layer=0,
          learning_rate=0.0001,
          result_file='log.csv'):
    train_images, train_labels = KerasModel.load_images_and_labels(constants.FULL_SQUAT_TRAIN_FOLDER)
    dev_images, dev_labels = KerasModel.load_images_and_labels(constants.FULL_SQUAT_DEV_FOLDER)
    input_shape = (constants.IMAGE_HEIGHT, constants.IMAGE_WIDTH, 3)

    train_labels = to_categorical(train_labels)
    dev_labels = to_categorical(dev_labels)

    if base_model == 'VGG16':
      base_model = applications.VGG16(weights = "imagenet", include_top=False, input_shape = (constants.IMAGE_HEIGHT, constants.IMAGE_WIDTH, 3))
    elif base_model == 'ResNet50':
      base_model = applications.ResNet50(weights = "imagenet", include_top=False, input_shape = (constants.IMAGE_HEIGHT, constants.IMAGE_WIDTH, 3))
    elif base_model == 'Xception':
      base_model = applications.Xception(weights = "imagenet", include_top=False, input_shape = (constants.IMAGE_HEIGHT, constants.IMAGE_WIDTH, 3))
    else:
      raise 'Unknown base model'

    if base_model_layer == -1:
      x = base_model.layers[-1].output
    else:
      for layer in base_model.layers:
        layer.trainable = False
      x = base_model.layers[base_model_layer].output

    x = Flatten()(x)
    x = Dense(64, activation="relu")(x)
    predictions = Dense(2, activation='sigmoid')(x)
    model_final = Model(input=base_model.input, output=predictions)
    print(model_final.layers)

    model_final.compile(loss = "binary_crossentropy",
                        optimizer = keras.optimizers.Adam(lr=learning_rate),
                        metrics=["accuracy"])

    if batch_size == -1:
      batch_size = len(train_images)

    csv_logger = CSVLogger(result_file, separator=';')
    history_callback = model_final.fit(train_images, train_labels,
                                       batch_size=batch_size,
                                       epochs=epochs,
                                       verbose=1,
                                       validation_data=(dev_images, dev_labels),
                                       callbacks=[csv_logger])

  def extract_labels(file_names):
    labels = []
    for file_name in file_names:
      match = re.match(".*\.mp4_\d+_(\d)_\d+\.jpg", file_name)
      labels.append(int(match[1]))
    return labels

  def load_images_and_labels(folder):
    image_names = Utils.get_image_names(folder)

    # Load the images into an array
    images = []
    for image_name in image_names:
      images.append(cv2.imread("{}/{}".format(folder, image_name)))

    # Extract the labels from the image names
    labels = KerasModel.extract_labels(image_names)

    return np.array(images), np.array(labels)
