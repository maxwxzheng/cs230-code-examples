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
2. Create a folder under data called 'raw_data'
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
processor.extract_all_videos(False, True)

from data_shuffler import DataShuffler
DataShuffler.shuffle_full_squat_data()
```

You should have 6 new folders created under data folder.
'full_squat_all' and 'sequence_squat_all' stores all the images for full squat and sequence squat.
Note that a 'full_squat' is when the person squat to the bottom. It's mainly used by CNN to analyze a static image.
A 'sequence_squat' includes all the frames of a squat from the beginning to the end. It's mainly used by Sequence model to analyze videos (TODO)

'full_squat_train' and 'full_squat_dev' stores training data and dev data for full squats.
'sequence_squat_train' and 'sequence_squat_dev' stores training data and dev data for sequence squats.

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
