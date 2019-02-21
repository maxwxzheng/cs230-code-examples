import keras
from keras.models import Model, Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import to_categorical
from keras import backend as K
from keras import applications
import re
import cv2
import numpy as np

import constants
from utils import Utils

class KerasModel():

  def run():
    train_images, train_labels = KerasModel.load_images_and_labels(constants.FULL_SQUAT_TRAIN_FOLDER)
    dev_images, dev_labels = KerasModel.load_images_and_labels(constants.FULL_SQUAT_DEV_FOLDER)
    input_shape = (constants.IMAGE_HEIGHT, constants.IMAGE_WIDTH, 3)

    train_labels = to_categorical(train_labels)
    dev_labels = to_categorical(dev_labels)

    vgg16 = applications.VGG16(weights = "imagenet", include_top=False, input_shape = (constants.IMAGE_HEIGHT, constants.IMAGE_WIDTH, 3))

    for layer in vgg16.layers:
      layer.trainable = False
    x = vgg16.layers[3].output

    x = Flatten()(x)
    x = Dense(64, activation="relu")(x)
    predictions = Dense(2, activation='sigmoid')(x)
    model_final = Model(input=vgg16.input, output=predictions)
    print(model_final.layers)

    model_final.compile(loss = "binary_crossentropy",
                        optimizer = keras.optimizers.Adam(lr=0.0000001),
                        metrics=["accuracy"])

    model_final.fit(train_images, train_labels,
                    batch_size=len(train_images),
                    epochs=10,
                    verbose=1,
                    validation_data=(dev_images, dev_labels))
    score = model_final.evaluate(dev_images, dev_labels, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

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
