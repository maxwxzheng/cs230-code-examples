import keras
from keras.models import Model, Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import to_categorical
from keras import backend as K
from keras import applications
from keras.callbacks import CSVLogger
from keras.callbacks import EarlyStopping
from keras import regularizers
import re
import cv2
import numpy as np
import os
import yaml
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt

import constants
from utils import Utils

class KerasModel():

  def run_all_experiments(overwrite=False, test_mode=False):
    configs = yaml.load(open('experiments/configs.yml'))
    existing_result_files = os.listdir('experiments')
    for experiment_id in configs.keys():
      result_file_name_prefix = experiment_id
      if (result_file_name_prefix in existing_result_files) and not overwrite:
        continue

      config = configs[experiment_id]
      KerasModel.run(max_epochs=config['max_epochs'],
                     batch_size=100,
                     base_model=config['base_model'],
                     base_model_layer=config['base_model_layer'],
                     learning_rate=config['learning_rate'],
                     result_file_name_prefix="experiments/{}".format(result_file_name_prefix),
                     test_mode=test_mode)

  def run(max_epochs=50,
          batch_size=-1,
          base_model=None,
          base_model_layer=-1,
          learning_rate=0.0001,
          result_file_name_prefix='log',
          test_mode=False):
    train_images, train_labels = KerasModel.load_images_and_labels(constants.FULL_SQUAT_TRAIN_FOLDER)
    dev_images, dev_labels = KerasModel.load_images_and_labels(constants.FULL_SQUAT_DEV_FOLDER)
    input_shape = (constants.IMAGE_HEIGHT, constants.IMAGE_WIDTH, 3)

    train_labels = to_categorical(train_labels)
    dev_labels = to_categorical(dev_labels)

    if base_model in ['VGG16', 'ResNet50', 'Xception']:
      model_final = KerasModel.existing_model(base_model, base_model_layer)
    elif base_model.startswith('custom_model_'):
      model_final = getattr(KerasModel, base_model)()
    else:
      raise "Unrecognized model {}".format(base_model)

    print(model_final.layers)

    model_final.compile(loss = "binary_crossentropy",
                        optimizer = keras.optimizers.Adam(lr=learning_rate),
                        metrics=["accuracy"])

    # Save the result to csv file.
    csv_file_name = result_file_name_prefix + '.csv'
    csv_logger = CSVLogger(csv_file_name, separator=';')

    if test_mode:
      train_images = train_images[:50]
      train_labels = train_labels[:50]
      max_epochs = 50
      batch_size = -1
      min_delta = 1
    else:
      min_delta = 0.0008

    # Early stopping
    early_stopping = EarlyStopping(monitor='loss', min_delta=min_delta, patience=2)

    if batch_size == -1:
      batch_size = len(train_images)

    model_log = model_final.fit(train_images, train_labels,
                                batch_size=batch_size,
                                epochs=max_epochs,
                                verbose=1,
                                validation_data=(dev_images, dev_labels),
                                callbacks=[csv_logger, early_stopping])

    KerasModel.save_plot_history(model_log, result_file_name_prefix)
    KerasModel.save_model(model_final, result_file_name_prefix)
    KerasModel.save_test_result(model_final, result_file_name_prefix)

  def existing_model(base_model, base_model_layer):
    if base_model == 'VGG16':
      base_model = applications.VGG16(weights = "imagenet", include_top=False, input_shape = (constants.IMAGE_HEIGHT, constants.IMAGE_WIDTH, 3))
    elif base_model == 'ResNet50':
      base_model = applications.ResNet50(weights = "imagenet", include_top=False, input_shape = (constants.IMAGE_HEIGHT, constants.IMAGE_WIDTH, 3))
    elif base_model == 'Xception':
      base_model = applications.Xception(weights = "imagenet", include_top=False, input_shape = (constants.IMAGE_HEIGHT, constants.IMAGE_WIDTH, 3))
    else:
      raise "Unrecognized existing model {}".format(base_model)

    if base_model_layer == -1:
      x = base_model.layers[-1].output
    else:
      for layer in base_model.layers:
        layer.trainable = False
      x = base_model.layers[base_model_layer].output

    x = Flatten()(x)
    x = Dense(64, activation="relu")(x)
    x = Dense(2, activation='sigmoid')(x)
    return Model(input=base_model.input, output=x)

  def custom_model_1():
    model = Sequential()
    input_shape = (constants.IMAGE_HEIGHT, constants.IMAGE_WIDTH, 3)
    model.add(Conv2D(64, (3, 3), activation='relu', input_shape=input_shape, kernel_regularizer=regularizers.l2(0.001)))
    model.add(Conv2D(128, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.001)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation="relu", kernel_regularizer=regularizers.l2(0.001)))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='sigmoid'))
    return model

  def custom_model_2():
    model = Sequential()
    input_shape = (constants.IMAGE_HEIGHT, constants.IMAGE_WIDTH, 3)
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape, kernel_regularizer=regularizers.l2(0.001)))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.001)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(64, activation="relu", kernel_regularizer=regularizers.l2(0.001)))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='sigmoid'))
    return model

  def custom_model_3():
    model = Sequential()
    input_shape = (constants.IMAGE_HEIGHT, constants.IMAGE_WIDTH, 3)
    model.add(Conv2D(64, (3, 3), activation='relu', input_shape=input_shape, kernel_regularizer=regularizers.l2(0.001)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(32, activation="relu", kernel_regularizer=regularizers.l2(0.001)))
    model.add(Dense(2, activation='sigmoid'))
    return model

  def save_plot_history(model_log, result_file_name_prefix):
    fig = plt.figure()
    
    plt.subplot(2,1,1)
    plt.plot(model_log.history['acc'])
    plt.plot(model_log.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'dev'], loc='lower right')

    plt.subplot(2,1,2)
    plt.plot(model_log.history['loss'])
    plt.plot(model_log.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'dev'], loc='upper right')
    plt.tight_layout()

    plt_file_name = result_file_name_prefix + '.png'
    plt.savefig(plt_file_name)

  def save_model(model, result_file_name_prefix):
    model_file_name = result_file_name_prefix + '.json'
    model_digit_json = model.to_json()
    with open(model_file_name, 'w') as json_file:
      json_file.write(model_digit_json)
    
    model_weight_file_name = result_file_name_prefix + '.h5'
    model.save_weights(model_weight_file_name)

  def save_test_result(model, result_file_name_prefix):
    test_images, test_labels = KerasModel.load_images_and_labels(constants.FULL_SQUAT_TEST_FOLDER)
    test_labels = to_categorical(test_labels)

    score = model.evaluate(test_images, test_labels, verbose=0)
    csv_file_name = result_file_name_prefix + '.csv'
    with open(csv_file_name,'a') as fd:
      fd.write(';'.join(['test', str(score[1]), str(score[0])]))

  def extract_labels(file_names):
    labels = []
    for file_name in file_names:
      match = re.match(".*\.mp4_\d+_(\d)_\d+.*\.jpg", file_name)
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
