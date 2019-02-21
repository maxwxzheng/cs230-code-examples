import random
import shutil

import constants
from utils import Utils

"""
DataShuffler is used to shuffle squat images from 'all' folder into 'train' and 'dev' folders.

E.g.
from data_shuffler import DataShuffler
DataShuffler.shuffle_full_squat_data()
"""
class DataShuffler:

  # Shuffle full squat data from 'all' folder into 'train' and 'dev' folders.
  def shuffle_full_squat_data(train_ratio=0.8, random_seed=230):
    DataShuffler.shuffle_data(
      constants.FULL_SQUAT_ALL_FOLDER,
      constants.FULL_SQUAT_DEV_FOLDER,
      constants.FULL_SQUAT_TRAIN_FOLDER)

  # Shuffle sequence squat data from 'all' folder into 'train' and 'dev' folders.
  def shuffle_sequence_squat_data(train_ratio=0.8, random_seed=230):
    DataShuffler.shuffle_data(
      constants.SEQUENCE_SQUAT_ALL_FOLDER,
      constants.SEQUENCE_SQUAT_DEV_FOLDER,
      constants.SEQUENCE_SQUAT_TRAIN_FOLDER)

  def shuffle_data(all_dir, dev_dir, train_dir):
    Utils.remake_folder(dev_dir)
    Utils.remake_folder(train_dir)

    filenames = Utils.get_image_names(all_dir)

    # Split the images into 80% train and 20% dev
    # Make sure to always shuffle with a fixed seed so that the split is reproducible
    random.seed(230)
    filenames.sort()
    random.shuffle(filenames)

    split = int(0.8 * len(filenames))
    train_filenames = filenames[:split]
    dev_filenames = filenames[split:]

    # Copy files from all_dir to train_dir or dev_dir
    DataShuffler.copy_files(train_filenames, all_dir, train_dir)
    DataShuffler.copy_files(dev_filenames, all_dir, dev_dir)

  def copy_files(file_names, src_dir, dst_dir):
    for file_name in file_names:
      src_file = "{}/{}".format(src_dir, file_name)
      dst_file = "{}/{}".format(dst_dir, file_name)
      shutil.copyfile(src_file, dst_file)
