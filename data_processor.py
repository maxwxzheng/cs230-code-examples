import cv2
import yaml
import constants
from tf_pose.estimator import TfPoseEstimator

class DataProcessor():

  def __init__(self):
    self.raw_data_params = yaml.load(open('raw_data_params.yml'))

    graph_path = 'tf_pose/graph/mobilenet_thin/graph_opt.pb'
    self.pose_estimator = TfPoseEstimator(graph_path, target_size=(constants.IMAGE_WIDTH, constants.IMAGE_HEIGHT))

  def extract_all_videos(self, extract_sequence_frames=False, extract_full_frames=False):
    for file_name in self.raw_data_params.keys():
      self.extract_all_frames_from_video(file_name, extract_sequence_frames, extract_full_frames)

  def extract_all_frames_from_video(self, file_name, extract_sequence_frames=False, extract_full_frames=False):
    cap = cv2.VideoCapture("data/raw_data/{}".format(file_name))

    video_params = self.raw_data_params[file_name]
    # Each dictionary in the array represents a full squat
    for i in range(len(video_params)):
      squat_params = video_params[i]
      start = squat_params['start']
      end = squat_params['end']
      label = squat_params['label']
      file_name_prefix = "{}_{}_{}".format(file_name, i, label)
      self.extract_frames_from_start_to_end(cap, file_name_prefix, start, end, extract_sequence_frames, extract_full_frames)

  """
  Extracts the individual frames from the video in cap. Resizes and adds pose
  to the frames. Optionally saves the frames to either sequence_squat folder or
  full_squat folder.

  Args:
    cap: (cv2.VideoCapture) the object that contains the video.
    file_name_prefix: (String) the name prefix to export the file with.
    start: (String) start time to extract frames with.
    end: (String) end time to extract frames with.
    extract_sequence_frames: (Boolean) if true, saves every frame into the
        sequence_squat_all folder.
    extract_full_frames: (Boolean) if true, saves the middle frame into the
        full_squat_all_folder.
  """
  def extract_frames_from_start_to_end(self, cap, file_name_prefix, start, end, extract_sequence_frames=False, extract_full_frames=False):
    # Input video fps
    fps = cap.get(cv2.CAP_PROP_FPS)

    start_sec = DataProcessor.time_to_seconds(start)
    end_sec = DataProcessor.time_to_seconds(end)

    start_frame = int(fps * start_sec)
    end_frame = int(fps * end_sec)

    # Run transformation on each frame. Append transformed frame to writer.
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    for i in range(end_frame - start_frame + 1):
      success, frame = cap.read()
      if not success:
        continue
        # raise Exception("Failed to read frame {} in file {}".format(i + start_frame, file_name_prefix))

      # We only need to extract the middle frame for full-squat 
      extract_full_frames_for_i = (extract_full_frames and i == int((end_frame - start_frame) / 2) + 1)

      if extract_sequence_frames or extract_full_frames_for_i:
        frame = cv2.resize(frame, (0,0), fx=constants.IMAGE_WIDTH / frame.shape[0], fy=constants.IMAGE_HEIGHT / frame.shape[1])
        rotated_frame = DataProcessor.rotate_frame_clockwise_90_degrees(frame)
        pose_frame = self.add_pose(rotated_frame)

        # Write all frames to sequence_squat_all
        if extract_sequence_frames:
          file_path = "{}/{}_{}.jpg".format(constants.SEQUENCE_SQUAT_ALL_FOLDER, file_name_prefix, i)
          cv2.imwrite(file_path, pose_frame)

        # Write the middle frame to full_squat_all
        if extract_full_frames_for_i:
          file_path = "{}/{}_{}.jpg".format(constants.FULL_SQUAT_ALL_FOLDER, file_name_prefix, i)
          cv2.imwrite(file_path, pose_frame)

  # Use the pose_estimator to add the pose on top of the given frame.
  def add_pose(self, frame):
    humans = self.pose_estimator.inference(frame, resize_to_default=True, upsample_size=4.0)
    frame = TfPoseEstimator.draw_humans(frame, humans, imgcopy=False)
    return frame

  # Rotate the frame clockwise by 90 degrees.
  def rotate_frame_clockwise_90_degrees(frame):
    rotated = cv2.transpose(frame)
    rotated = cv2.flip(rotated, 1)
    return rotated

  """
  Converts SS:ss formatted time to seconds.
  Args:
    time: (string) in the format 'SS:ss'. 'ss' represents fractional
          second with range [0, 29].
  E.g. time is '1:15'. Returns 1.5.
  """
  def time_to_seconds(time):
    if time[-3] != ':':
      raise Exception("time mal-formatted. -3 element is not ':'. time is {}".format(time))

    fractional_second = int(time[-2:])
    second = int(time[:-3])
    return second + fractional_second / 30

  # def to_video():
  # # https://gist.github.com/takuma7/44f9ecb028ff00e2132e
  # fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
  # writer = cv2.VideoWriter('output.avi', fourcc, fps, (1920, 1080))
  #   writer.write(pose_frame)
  # writer.release()
