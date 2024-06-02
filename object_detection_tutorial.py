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

import re
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
from IPython.display import display
from google.protobuf import text_format

# Import the object detection module.



from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util


utils_ops.tf = tf.compat.v1

tf.gfile = tf.io.gfile


underlined_bounding_boxes_list=[]
highlighted_bounding_boxes_list=[]
explanations_bounding_boxes_list=[]
detection_model = None


def load_model(model_name):
  base_url = 'http://download.tensorflow.org/models/object_detection/'
  model_file = model_name + '.tar.gz'
  model_dir = tf.keras.utils.get_file(
    fname=model_name, 
    origin=base_url + model_file,
    untar=True)

  model_dir = pathlib.Path(model_dir)/"saved_model"

  model = tf.saved_model.load(str(model_dir))

  return model

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = 'object_detection\images\labelmap.pbtxt'
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)


# If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.
import pathlib
PATH_TO_TEST_IMAGES_DIR = pathlib.Path('object_detection/test_images')
TEST_IMAGE_PATHS = sorted(list(PATH_TO_TEST_IMAGES_DIR.glob("*.jpg")))
print(TEST_IMAGE_PATHS)
def reset_detection_model():
    global detection_model
    detection_model = None

def getDetectionModel():
    # # Detection
    # Load an object detection model:
    #model_name = 'ssd_mobilenet_v1_coco_2017_11_17'
    #detection_model = load_model(model_name)
    reset_detection_model()
    detection_model = tf.saved_model.load('object_detection/inference_graph/saved_model')
    # Check the model's input signature, it expects a batch of 3-color images of type uint8:
    #print(detection_model.signatures['serving_default'].inputs)
    # And returns several outputs:
    detection_model.signatures['serving_default'].output_dtypes
    detection_model.signatures['serving_default'].output_shapes
    # Add a wrapper function to call the model, and cleanup the outputs:
    return detection_model




def run_inference_for_single_image(model, img):
  image = np.asarray(img)
  # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
  input_tensor = tf.convert_to_tensor(image)
  # The model expects a batch of images, so add an axis with `tf.newaxis`.
  input_tensor = input_tensor[tf.newaxis,...]

  # Run inference
  model_fn = model.signatures['serving_default']
  output_dict = model_fn(input_tensor)
  # All outputs are batches tensors.
  # Convert to numpy arrays, and take index [0] to remove the batch dimension.
  # We're only interested in the first num_detections.
  num_detections = int(output_dict.pop('num_detections'))
  output_dict = {key:value[0, :num_detections].numpy()
                 for key,value in output_dict.items()}
  output_dict['num_detections'] = num_detections

  # detection_classes should be ints.
  output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)

  # Handle models with masks:
  detection_masks_reframed = None
  if 'detection_masks' in output_dict:
    print("Handle models with masks:")
    # Reframe the the bbox mask to the image size.
    detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
              output_dict['detection_masks'], output_dict['detection_boxes'],
               image.shape[0], image.shape[1])
    detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5,
                                       tf.uint8)
    output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()
  return output_dict



def show_inference(model, image_path):
  # the array based representation of the image will be used later in order to prepare the
  # result image with boxes and labels on it.
  image_np = np.array(Image.open(image_path))
  # Actual detection.
  output_dict = run_inference_for_single_image(model, image_np)
  # Visualization of the results of a detection.
  get_boundingBoxes(output_dict)
  final_img =vis_util.visualize_boxes_and_labels_on_image_array(
      image_np,
      output_dict['detection_boxes'],
      output_dict['detection_classes'],
      output_dict['detection_scores'],
      category_index,
      instance_masks=output_dict.get('detection_masks_reframed', None),
      use_normalized_coordinates=True,
      line_thickness=4,min_score_thresh=0.22)

  return(final_img)



def get_boundingBoxes(output_dict):
    bounding_boxes=output_dict['detection_boxes']
    detection_classes=output_dict['detection_classes']
    detection_scores=output_dict['detection_scores']
    
    underlined_indices = []
    highlighted_indices = []
    explanations_indices = []
    for idx, value in enumerate(detection_classes):
        if value == 1:
            underlined_indices.append(idx)
        if value == 2:
            highlighted_indices.append(idx)
        if value ==3:
            explanations_indices.append(idx)
      
    #print(underlined_indices)
    #print(highlighted_indices)

    if(len(underlined_indices) >= 1):
        res_list = None
        res_list = [detection_scores[i] for i in underlined_indices]
        #print ("Resultant list : " + str(res_list))

        scores_with_threshold  = [index for (index, item) in enumerate(res_list) if item >= 0.22]

        #print(scores_with_threshold)
        global underlined_bounding_boxes_list
        underlined_bounding_boxes_list = []
        underlined_bounding_boxes_list=[bounding_boxes[i] for i in scores_with_threshold]
        print("underlined_bounding_boxes_list")
        print(len(underlined_bounding_boxes_list))
        print(underlined_bounding_boxes_list)

    if(len(highlighted_indices) >= 1):
        res_list2 = None
        res_list2 = [detection_scores[i] for i in highlighted_indices]
        #print ("Resultant list : " + str(res_list2))

        scores_with_threshold2  = [index for (index, item) in enumerate(res_list2) if item >= 0.22]

        #print(scores_with_threshold2)
        global highlighted_bounding_boxes_list
        highlighted_bounding_boxes_list=[]
        highlighted_bounding_boxes_list=[bounding_boxes[i] for i in scores_with_threshold2]
        print("highlighted_bounding_boxes_list")
        print(len(highlighted_bounding_boxes_list))
        print(highlighted_bounding_boxes_list)
    
    if(len(explanations_indices) >= 1):
        res_list = None
        res_list = [detection_scores[i] for i in explanations_indices]
        #print ("Resultant list : " + str(res_list))

        scores_with_threshold  = [index for (index, item) in enumerate(res_list) if item >= 0.22]

        #print(scores_with_threshold)
        global explanations_bounding_boxes_list
        explanations_bounding_boxes_list=[]
        explanations_bounding_boxes_list=[bounding_boxes[i] for i in scores_with_threshold]
        print("explanations_bounding_boxes_list")
        print(len(explanations_bounding_boxes_list))
        print(explanations_bounding_boxes_list)
def extract_number(file_path):
    # Extract the numeric part using regular expression
    match = re.search(r'\d+\.\d+', file_path)
    if match:
        return float(match.group())
    else:
        return None

finalFramesDict = {}
highlighted_texts_frames_list=[]
underlined_texts_frames_list=[]
extra_explanations_frames_list=[]

def reset_global_lists_and_dictionary():
    global finalFramesDict
    global highlighted_texts_frames_list
    global underlined_texts_frames_list
    global extra_explanations_frames_list
    finalFramesDict = {}
    highlighted_texts_frames_list = []
    underlined_texts_frames_list = []
    extra_explanations_frames_list = []

def identifyingMostAttentionalSlideFrames(listOfFrames):
    print("Starting Identifying Most Attentional Slide Frames")
    reset_global_lists_and_dictionary()
    print("finalFramesDict:", finalFramesDict)
    print("highlighted_texts_frames_list:", highlighted_texts_frames_list)
    print("underlined_texts_frames_list:", underlined_texts_frames_list)
    print("extra_explanations_frames_list:", extra_explanations_frames_list)
    images_folder_path= "object_detection/test_images/"
    images_path =[]
    object_detection_model=getDetectionModel()
    for filename in os.listdir(images_folder_path):
        # Check if the file has a .jpg extension
        if filename.endswith(".jpg"):
            # Print the path to the file
            path = os.path.join(images_folder_path, filename)
            images_path.append(path)
    sorted_file_paths = sorted(images_path, key=extract_number)
    for path in sorted_file_paths:
        print(path)
    for image_path in images_path:
          print(image_path)
          resulted_image=show_inference(object_detection_model, image_path)
          print("---------------------------------------------------------------------------")
          if(len(underlined_bounding_boxes_list)>=1):
                underlined_texts_frames_list.append(str(image_path))
          if(len(highlighted_bounding_boxes_list)>=1):
                highlighted_texts_frames_list.append(str(image_path))
          if(len(explanations_bounding_boxes_list)>=1):
                extra_explanations_frames_list.append(str(image_path))

          finalFramesDict["underlined_texts"] = underlined_texts_frames_list
          finalFramesDict["highlighted_texts"] = highlighted_texts_frames_list
          finalFramesDict["extra_explanations"] = extra_explanations_frames_list
          print(finalFramesDict)
    print("Ending Identifying Most Attentional Slide Frames")      
    return finalFramesDict
