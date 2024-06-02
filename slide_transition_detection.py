import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import pathlib
import cv2
import math
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
from skimage.metrics import structural_similarity

# patch tf1 into `utils.ops`
utils_ops.tf = tf.compat.v1

# Patch the location of gfile
tf.gfile = tf.io.gfile

def mse(img1, img2):
       h, w = img1.shape
       diff = cv2.subtract(img1, img2)
       err = np.sum(diff**2)
       mse = err/(float(h*w))
       return mse, diff

def SSIM(img1,img2):

    (score, diff) = structural_similarity(img1, img2, full=True)
    diff = (diff * 255).astype("uint8")
    return score

def calculate_error(frame1, frame2):

    frame1_converted=cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    frame2_converted=cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    #match_error12, diff12 = mse(frame1_converted, frame2_converted)
    #return match_error12
    score = SSIM(frame1_converted, frame2_converted)
    return score











