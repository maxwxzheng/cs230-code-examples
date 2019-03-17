## First Time Setup

Make sure your python3 version is python3.6. This repo does not work with python3.7.

#### Setup Main Module
```
virtualenv -p python3 .env
source .env/bin/activate
pip install -r requirements.txt
```

#### Setup tf-pose
```
cd tf_pose/pafprocess
brew install swig
swig -python -c++ pafprocess.i && python3 setup.py build_ext --inplace
```

#### Setup data
1. Create a folder under deep-squat called 'data'
2. Create folders under data called 'raw_data', 'full_squat_all', 'full_squat_train', 'full_squat_dev', 'full_squat_test'
3. Put all the videos into 'raw_data' folder
4. In console, if you are not in virtual env, run:
```
source .env/bin/activate
```
5. Start python3:
```
python3
```
6. Run the following command in python3:
```
from data_processor import DataProcessor
processor = DataProcessor()
processor.extract_all_videos(False, True, False)

from data_augment import DataAugment
DataAugment.run_for_full_squats()

from data_shuffler import DataShuffler
DataShuffler.shuffle_full_squat_data()
```

'full_squat_all' stores all the images for full squats. 'full_squat_train', 'full_squat_dev', 'full_squat_test' stores training data, dev data and test data for full squats.
Note that a 'full_squat' is when the person squat to the bottom. It's mainly used by CNN to analyze a static image.

#### Train CNN

1. In console, if you are not in virtual env, run:
```
source .env/bin/activate
```
2. Run python3:
```
python3
```
3. Change the model architecture and hyperparameters in keras_model.py and run:
```
from keras_model import KerasModel
KerasModel.run()
```

Note that every time you change keras_model.py, you need to exit python3 console and restart it.

## Everytime before working on the project
```
source .env/bin/activate
```

## Everytime when done with the project
```
deactivate
```
