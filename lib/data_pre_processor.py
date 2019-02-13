"""
Resize image.
Use tf-pose-estimation to analyze the image.
"""
import cv2
from tf_pose.estimator import TfPoseEstimator

class DataPreProcessor():

  def __init__(self):
    graph_path = 'tf_pose/graph/mobilenet_thin/graph_opt.pb'

    # Somehow only this target_size works.
    self.pose_estimator = TfPoseEstimator(graph_path, target_size=(432, 368))

  def get_video_clip(self, cap, start, end):
    # Input video fps
    fps = cap.get(cv2.CAP_PROP_FPS)

    start_sec = DataPreProcessor.time_to_seconds(start)
    end_sec = DataPreProcessor.time_to_seconds(end)

    start_frame = int(fps * start_sec)
    end_frame = int(fps * end_sec)

    # # https://gist.github.com/takuma7/44f9ecb028ff00e2132e
    # fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    # writer = cv2.VideoWriter('output.avi', fourcc, fps, (1920, 1080))

    # Run transformation on each frame. Append transformed frame to writer.
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    for i in range(end_frame - start_frame + 1):
      success, frame = cap.read()
      if not success:
        raise Exception("Failed to read frame {}".format(i + start_frame))

      rotated_frame = DataPreProcessor.rotate_frame_clockwise_90_degrees(frame)
      pose_frame = self.add_pose(rotated_frame)
      cv2.imwrite("tmp_data/img_{}.jpg".format(i), pose_frame)
    #   writer.write(pose_frame)
    # writer.release()

  def add_pose(self, frame):
    humans = self.pose_estimator.inference(frame, resize_to_default=True, upsample_size=4.0)
    frame = TfPoseEstimator.draw_humans(frame, humans, imgcopy=False)
    return frame

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
