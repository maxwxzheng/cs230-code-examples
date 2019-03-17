import cv2

from utils import Utils
import constants

"""
DataAugment is used to do the following data augmentations:
- Flip left and right
- Zoom

E.g.
from data_augment import DataAugment
DataAugment.run_for_full_squats()
"""
class DataAugment():

  def run_for_full_squats():
    DataAugment.run(constants.FULL_SQUAT_ALL_FOLDER)

  def run(folder):
    all_image_names = Utils.get_image_names(folder)
    for image_name in all_image_names:
      image = cv2.imread("{}/{}".format(folder, image_name))
      
      flipped = DataAugment.flip(image, folder, image_name)
      DataAugment.zoom(image, folder, image_name, False)
      DataAugment.zoom(flipped, folder, image_name, True)

  # Flip the image left and right
  def flip(image, folder, image_name):
    flipped = cv2.flip(image, 1)
    flipped_image_name = image_name[0:-4] + '_flip.jpg'
    cv2.imwrite("{}/{}".format(folder, flipped_image_name), flipped)
    return flipped

  # First resize the image and then pad the image to the original size.
  def zoom(image, folder, image_name, flipped):
    old_size = image.shape[:2]
    new_size = tuple([int(x * constants.ZOOM_RATIO) for x in old_size])
    resized_image = cv2.resize(image, (new_size[1], new_size[0]))
    delta_w = old_size[1] - new_size[1]
    delta_h = old_size[0] - new_size[0]
    top = int(delta_h / 2)
    bottom = top
    left = int(delta_w / 2)
    right = left
    color = [0, 0, 0]
    resized_image = cv2.copyMakeBorder(resized_image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    resized_image_name = image_name[0:-4]
    if flipped:
      resized_image_name += '_flip_resize.jpg'
    else:
      resized_image_name += '_resize.jpg'
    cv2.imwrite("{}/{}".format(folder, resized_image_name), resized_image)
