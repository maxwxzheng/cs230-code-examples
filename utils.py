import os
import shutil

class Utils:

  # Return an array of image names in dir.
  def get_image_names(dir):
    # List all files in all_dir
    filenames = os.listdir(dir)
    
    # Only keep .jpg. Get rid of .DS_Store and other random files.
    filenames = [f for f in filenames if f.endswith('.jpg')]

    return filenames

  # Deletes the given dir and remake a new one.
  def remake_folder(dir):
    shutil.rmtree(dir)
    os.mkdir(dir)
