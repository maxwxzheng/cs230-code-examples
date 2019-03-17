import random
import shutil

import constants
from utils import Utils
import yaml

"""
DataShuffler is used to shuffle squat images from 'all' folder into 'train' and 'dev' folders.

E.g.
from data_shuffler import DataShuffler
DataShuffler.shuffle_full_squat_data()
"""
class DataShuffler:

  # Shuffle full squat data from 'all' folder into 'train' and 'dev' folders.
  def shuffle_full_squat_data(split_ratio=0.05, random_seed=230):
    DataShuffler.shuffle_data(
      split_ratio,
      random_seed,
      constants.FULL_SQUAT_ALL_FOLDER,
      constants.FULL_SQUAT_DEV_FOLDER,
      constants.FULL_SQUAT_TEST_FOLDER,
      constants.FULL_SQUAT_TRAIN_FOLDER)

  # Shuffle sequence squat data from 'all' folder into 'train' and 'dev' folders.
  def shuffle_sequence_squat_data(split_ratio=0.05, random_seed=230):
    DataShuffler.shuffle_data(
      split_ratio,
      random_seed,
      constants.SEQUENCE_SQUAT_ALL_FOLDER,
      constants.SEQUENCE_SQUAT_DEV_FOLDER,
      constants.SEQUENCE_SQUAT_TEST_FOLDER,
      constants.SEQUENCE_SQUAT_TRAIN_FOLDER)

  def shuffle_data(split_ratio, random_seed, all_dir, dev_dir, test_dir, train_dir):
    Utils.remake_folder(dev_dir)
    Utils.remake_folder(train_dir)
    Utils.remake_folder(test_dir)

    # Make sure to always shuffle with a fixed seed so that the split is reproducible
    random.seed(random_seed)

    # Shuffle the squats instead of individual images. So that the squat in
    # dev set and test set are not in training set.
    # Also Get equal number of positive samples and negative samples into each set.
    raw_data_params = yaml.load(open('raw_data_params.yml'))
    positive_prefixes = []
    negative_prefixes = []
    for video_name in raw_data_params.keys():
      for i in range(len(raw_data_params[video_name])):
        video_params = raw_data_params[video_name][i]
        image_name_prefix = "{}_{}_{}_".format(video_name, i, video_params['label'])
        if video_params['label'] == 1:
          positive_prefixes.append(image_name_prefix)
        else:
          negative_prefixes.append(image_name_prefix)
    
    random.shuffle(positive_prefixes)
    random.shuffle(negative_prefixes)

    num_positive_dev = int(split_ratio * len(positive_prefixes))
    num_positive_test = num_positive_dev
    num_negative_dev = int(split_ratio * len(negative_prefixes))
    num_negative_test = num_negative_dev

    positive_dev_prefixes = positive_prefixes[:num_positive_dev]
    positive_test_prefixes = positive_prefixes[num_positive_dev:num_positive_dev+num_positive_test]
    positive_train_prefixes = positive_prefixes[num_positive_dev+num_positive_test:]
    negative_dev_prefixes = negative_prefixes[:num_negative_dev]
    negative_test_prefixes = negative_prefixes[num_negative_dev:num_negative_dev+num_negative_test]
    negative_train_prefixes = negative_prefixes[num_negative_dev+num_negative_test:]

    filenames = Utils.get_image_names(all_dir)
    DataShuffler.copy_files_for_prefix(filenames, positive_train_prefixes + negative_train_prefixes, all_dir, train_dir)
    DataShuffler.copy_files_for_prefix(filenames, positive_dev_prefixes + negative_dev_prefixes, all_dir, dev_dir)
    DataShuffler.copy_files_for_prefix(filenames, positive_test_prefixes + negative_test_prefixes, all_dir, test_dir)

  def copy_files_for_prefix(all_file_names, file_name_prefixes, src_dir, dst_dir):
    image_names = []
    for prefix in file_name_prefixes:
      image_names += DataShuffler.all_image_names(all_file_names, prefix)
    DataShuffler.copy_files(image_names, src_dir, dst_dir)

  def copy_files(file_names, src_dir, dst_dir):
    for file_name in file_names:
      src_file = "{}/{}".format(src_dir, file_name)
      dst_file = "{}/{}".format(dst_dir, file_name)
      shutil.copyfile(src_file, dst_file)

  def all_image_names(all_file_names, image_name_prefix):
    image_names = []
    for file_name in all_file_names:
      if file_name.startswith(image_name_prefix):
        image_names.append(file_name)
    return image_names
