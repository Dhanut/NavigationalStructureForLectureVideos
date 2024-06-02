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
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
from video_frames_capturing import capturingVideoFrames
from slide_transition_detection import calculate_error
from object_detection_tutorial import identifyingMostAttentionalSlideFrames
from ocr import content_list_from_frames

# patch tf1 into `utils.ops`
utils_ops.tf = tf.compat.v1

# Patch the location of gfile
tf.gfile = tf.io.gfile

#Binary Search -Group the captured video image frames to each slide in the presentation - slide transition detection and grouping
transition_frames_list = []
grouped_frames_list = []
grouped_timestamps_list = []
dictionary = {}
most_attentional_timeframes_list = []

def reset_global_lists_and_dictionary():
    global transition_frames_list
    global grouped_frames_list
    global grouped_timestamps_list
    global dictionary
    global most_attentional_timeframes_list

    transition_frames_list = []
    grouped_frames_list = []
    grouped_timestamps_list = []
    dictionary = {}
    most_attentional_timeframes_list = []

def getSlideTransitionFrames(response_list):
    if len(response_list) == 0:
        return transition_frames_list
    elif len(response_list) == 1:
        transition_frames_list.append("All frames")
        return transition_frames_list
    else:
        #first_frame = response_list[0]
        #last_frame = response_list[len(response_list)-1]
        #print("First Frame Timestamp ::" + first_frame.timestamp)
        #print("Last Frame Timestamp ::" + last_frame.timestamp)
        #low= 0
        #high= len(response_list) -1
        return sequentialSearch(response_list)


def binarySearch(response_list,low, high):
    if low != high:
        mid = low + (high - low)//2
        frame1=response_list[low].frame
        frame2=response_list[mid].frame
        error_value=calculate_error(frame1,frame2)
        print("Image matching Error between " + response_list[low].timestamp+ "and" +response_list[mid].timestamp +":",error_value)
        
        if error_value < 2:
            transition_frames_list.append(mid)
            binarySearch(response_list,mid+1,high)
            
        if error_value >= 2:
            transition_frames_list.append(mid)
            binarySearch(response_list,low,mid-1)
        
def  sequentialSearch(response_list):
    transition_frames_list.append(response_list[0])  
    for index, item in enumerate(response_list):
    
        frame1=item.frame
        #frame1=removingRedColor(frame1)//not used
        if  index+1 == len(response_list):
            #transition_frames_list.append(response_list[index+1])
            break
        else:
            frame2=response_list[index+1].frame
            #frame2=removingRedColor(frame2)//not used
        
        error_value=calculate_error(frame1,frame2)
        print("Image matching Similarity Between " + item.timestamp+ "and" +response_list[index+1].timestamp +":",error_value)
        #if error_value >= 2.5:
        if error_value <= 0.97:
            transition_frames_list.append(response_list[index+1])
    return transition_frames_list

def removingRedColor(frame):

    img_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    ## Red lower mask (0-10) and upper mask (160-180) of RED
    mask1 = cv2.inRange(img_hsv, (0,100,100), (10,255,255))
    mask2 = cv2.inRange(img_hsv, (160,100,100), (180,255,255))

    mask = cv2.bitwise_or(mask1, mask2 )
    croped = cv2.bitwise_and(frame, frame, mask=mask)

    cv2.imshow("mask", mask)
    cv2.imshow("croped", croped)
    cv2.waitKey(0)
    return croped


def groupingFramesBasedOnSlideTransitionFramesList(transition_frames,response_list):
    
    for index, item in enumerate(transition_frames):
        setOfFramesList =[]
        startingIndex = response_list.index(item)
        if index+1 == len(transition_frames):
            endingIndex = len(response_list)
        else:       
            nextItem=transition_frames[index+1]
            endingIndex = response_list.index(nextItem)
            
        if startingIndex==endingIndex:
            setOfFramesList.append([response_list[startingIndex]])
        #elif endingIndex == len(response_list) -1:
            #setOfFramesList.append(response_list[startingIndex : endingIndex+1])
        else:
            setOfFramesList.append(response_list[startingIndex : endingIndex])
        grouped_frames_list.append(setOfFramesList)
    return grouped_frames_list    
        
def convert_milliseconds_to_minutes_and_seconds(milliseconds_str):
    milliseconds = float(milliseconds_str)
    seconds = milliseconds / 1000
    minutes = seconds // 60
    seconds = seconds % 60
    return int(minutes), int(seconds)

def sort_list_ascending(input_list):
    sorted_list = sorted(input_list, key=lambda x: float(x))
    return sorted_list

def getUniqueFramesList(language,filename):
    reset_global_lists_and_dictionary()
    print("transition_frames_list:", transition_frames_list)
    print("grouped_frames_list:", grouped_frames_list)
    print("grouped_timestamps_list:", grouped_timestamps_list)
    print("dictionary:", dictionary)
    print("most_attentional_timeframes_list:", most_attentional_timeframes_list)
    response_list = capturingVideoFrames(filename)
    print("Total Number of Frames Captured ::::: ",len(response_list))

    for item in response_list:
            print(item.timestamp)
       
    print("transition frames list :::::")
    transition_frames=getSlideTransitionFrames(response_list)

    print("Total Number of Slide Transitions ::::: ",len(transition_frames))
    for item in transition_frames:
            print("Slide Transition occured in "+item.timestamp+" millisecond")
            

    grouped_frames=groupingFramesBasedOnSlideTransitionFramesList(transition_frames,response_list)
    print("Grouped Frames List :::",grouped_frames)
    print("::::::::::::::::::::::::::::::::::::::")
    for groupedList in grouped_frames:
        if(1<= len(groupedList)):
            for item in groupedList:
                test1 =[]
                for x in item:
                    test1.append(x.timestamp)
                    print("TimeStamp of the Frame :::",x.timestamp)
                grouped_timestamps_list.append(test1)    
                print("--------------------------------------")
    most_attentional_frames_list = {}
    most_attentional_frames_list.clear()
    most_attentional_frames_list = identifyingMostAttentionalSlideFrames(response_list)
    print("")
    print("------------After detecting most attentional contents in the slide frames---------------")
    print("highlighted_texts :::")
    print(most_attentional_frames_list.get('highlighted_texts'))
    print("")
    print("underlined_texts :::")
    print(most_attentional_frames_list.get('underlined_texts'))
    print("")
    print("extra_explanations :::")
    print(most_attentional_frames_list.get('extra_explanations'))


    time_stamp_list =[]

    for valuesList in most_attentional_frames_list.values():
        for item in valuesList:
            start_index = item.find("timestamp") + len("timestamp")
            end_index = item.find(".jpg")
            numeric_part = item[start_index:end_index]
            time_stamp_list.append(numeric_part)

    #sample_list=['object_detection\\test_images\\timestamp0.0.jpg', 'object_detection\\test_images\\timestamp60000.0.jpg','object_detection\\test_images\\timestamp240000.0.jpg']
    #time_stamp_list =[]
    #for x in sample_list:
       #y=x[38:]
       #z=y[:-4]
       #time_stamp_list.append(z)
    for t1 in time_stamp_list:
        for itemList in grouped_timestamps_list:
                if t1 in itemList:
                    most_attentional_timeframes_list.append(itemList[0])
    print("")
    print("First frame from the grouped frames list")
    print(most_attentional_timeframes_list)
    print("")
    print("")
    final_result=[]
    unique_timestamps_list = [*set(most_attentional_timeframes_list)]
    print("Unique timestamps list in ascending order")
    sorted_timestamps_list = sort_list_ascending(unique_timestamps_list)
    print(sorted_timestamps_list)
    if(1<= len(sorted_timestamps_list)):
        texts_list=content_list_from_frames(sorted_timestamps_list,language)
        final_result.append(texts_list)
        final_result.append(sorted_timestamps_list)
    return final_result
