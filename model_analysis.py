import json
import keras
from keras.models import model_from_json
import numpy as np

import constants
from keras_model import KerasModel
from utils import Utils

"""
from model_analysis import ModelAnalysis
ModelAnalysis.analyze('experiments/experiment_27.json', 'experiments/experiment_27.h5')
"""
class ModelAnalysis():

  def analyze(json_path, weight_path):
    dev_images, dev_labels = KerasModel.load_images_and_labels(constants.FULL_SQUAT_DEV_FOLDER)
    image_names = Utils.get_image_names(constants.FULL_SQUAT_DEV_FOLDER)
    
    model = ModelAnalysis.load_model(json_path, weight_path)
    predictions = model.predict_on_batch(dev_images)
    prediction_labels = []
    for prediction in predictions:
      prediction_labels.append(np.argmax(prediction))
    
    for i in range(len(dev_labels)):
      if dev_labels[i] != prediction_labels[i]:
        print("{} label: {} predict: {}".format(image_names[i], dev_labels[i], prediction_labels[i]))

  def load_model(json_path, weight_path):
    with open(json_path) as f:
      model_json = f.read()

    model = model_from_json(model_json)
    model.load_weights(weight_path)
    return model
