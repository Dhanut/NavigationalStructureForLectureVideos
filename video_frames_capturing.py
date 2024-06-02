import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import pathlib
import cv2
from PIL import Image
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
from IPython.display import display
from google.protobuf import text_format
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
utils_ops.tf = tf.compat.v1
tf.gfile = tf.io.gfile
import cv2


class Frame:
  def __init__(self, timestamp, frame):
    self.timestamp = timestamp
    self.frame = frame

frames_list =[]

def reset_global_list():
    global frames_list
    frames_list = []

def capturingVideoFrames(filename):
        reset_global_list()
        file_path = "static/uploads/" + filename
        cap = cv2.VideoCapture(file_path)
        count = 0
        fps = cap.get(cv2.CAP_PROP_FPS)
        rounded_frames_per_second=round(fps)
        #print("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(rounded_frames_per_second))

        while cap.isOpened():
          ret, frame = cap.read()
          if ret:
              frame = cv2.resize(frame, (1300, 700)) 
              #saving the video frame with timestamp(milliseconds)
              cv2.imwrite('object_detection/test_images/timestamp'+str(cap.get(cv2.CAP_PROP_POS_MSEC))+'.jpg', frame)
              frames_list.append(Frame(str(cap.get(cv2.CAP_PROP_POS_MSEC)),frame))
              #print("for frame : " + str(count) + "   timestamp is: ", str(cap.get(cv2.CAP_PROP_POS_MSEC)))
              count += (rounded_frames_per_second)*1 # i.e. at 30 fps, this advances one second
              cap.set(1, count)
          else:
              cap.release()
              break
               
        return frames_list





















