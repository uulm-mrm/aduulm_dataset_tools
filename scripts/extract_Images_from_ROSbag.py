#!/usr/bin/python
# -*- coding: utf-8-*

###################################################################################################################################
### This script extracts the <prevImg> images of the provided ros-sequences, as required in the following paper:                ###
### "Robust Semantic Segmentation in Adverse Weather Conditions bymeans  of  Fast  Video-Sequence  Segmentation"                ###
### (https://arxiv.org/pdf/2007.00290.pdf)                                                                                      ###
### The following folders are created:                                                                                          ###
### 1) img_sequences:       contains the current and the <prevImg> previous images of the ros-sequences                         ### 
### 2) training_scrips:     folder, where the txt-file is stored                                                                ###
###                                                                                                                             ###
### required arguments:                                                                                                         ###
###     - argument 1: <path2dataset>/ADUUlm_Dataset                                                                             ###
###     - argument 2: image_split ('split_training_good_weather_only' or 'split_training_adverse_weather')                      ###
###     - argument 3: training_set ('train','val','test', ...)                                                                  ###
###                                                                                                                             ###
###                                                                                                                             ###
### usage: python extract_Images_from_ROSbag.py  <path2dataset>/ADUUlm_Dataset <image_split> <training_set>                     ###
### e.g. python extract_Images_from_ROSbag.py  /home/user/ADUUlm_Dataset split_training_good_weather_only train                 ###
###                                                                                                                             ###
### Version 1.0                                                                                                                 ###
### (c) Andreas Pfeuffer, 30.08.2020                                                                                            ###
###################################################################################################################################



import numpy as np
import cv2
from collections import namedtuple
import os
import rospy
import rosbag
import cv2
import cv_bridge
import sys

from map_uint16_to_uint8 import map_uint16_to_uint8

# amount of previous images to save
prevImg = 3

# -------------------------------------
# define input-pathes
# -------------------------------------

if (len(sys.argv) != 4):
    print ("[ERROR] some arguments are missing. The following arguments are necessary: ")
    print ("[ERROR] argument 1: <path2dataset>/ADUUlm_Dataset ")
    print ("[ERROR] argument 2: image_split ('split_training_good_weather_only' or 'split_training_adverse_weather')")
    print ("[ERROR] argument 3: training_set ('train','val','test', ...)")
    print ("[ERROR] python prepare_TrainingData_SemanticSegmentation.py <path2dataset>/ADUUlm_Dataset <image_split> <training_set> ")
    sys.exit()

# path to dataset
DATADIR_ADUULM_DATASET=sys.argv[1]

# image_split (choose between "split_training_good_weather_only" and "split_training_adverse_weather")
#IMAGE_SPLIT = "split_training_good_weather_only"
IMAGE_SPLIT=sys.argv[2]

# training set (chosse between "all", "train", "val", "test", ...)
#TRAINING_SET = "train"
TRAINING_SET=sys.argv[3]

# path to image list: each line of the txt-file contains the corresponding image-name
path2list = os.path.join(DATADIR_ADUULM_DATASET, "image_splits/semantic_segmentation/", IMAGE_SPLIT, TRAINING_SET+".txt")
# prefix of the camera-image
prefix_camera = '_camera_16bit.tiff'

# topic name of camera image
topicName_camera =  '/camera/camera_wideangle/image' 

DATADIR_ADUULM_DATASET_DATA = os.path.join(DATADIR_ADUULM_DATASET, "data")
DATADIR_ADUULM_DATASET_ROSSEQUENCES = os.path.join(DATADIR_ADUULM_DATASET, "ros_sequences")
DATADIR_ADUULM_DATASET_GT = os.path.join(DATADIR_ADUULM_DATASET, 'gt')

# -------------------------------------
# define output-pathes
# -------------------------------------

# path to the output txt-file, which can be used for training. Each line contains the path to the image and the corresponding ground-truth
# eg. <path2image> <path2gt>
output_trainingFiles = os.path.join(DATADIR_ADUULM_DATASET, "training_scrips", TRAINING_SET+"_sequences.txt")
# path to directory, where to save the processed images
saveDir_img = os.path.join(DATADIR_ADUULM_DATASET, 'img_sequences')

# -------------------------------------
# create directory, if not exist
# -------------------------------------

if not os.path.exists(saveDir_img):
    os.makedirs(saveDir_img)
if not os.path.exists(os.path.join(DATADIR_ADUULM_DATASET, "training_scrips")):
    os.makedirs(os.path.join(DATADIR_ADUULM_DATASET, "training_scrips"))

# create dontCare label

if not os.path.exists(DATADIR_ADUULM_DATASET_GT):
    print ("[ERROR] ground-truth folder does not exist. Please create ground-truth by means of script 'prepare_TrainingData_SemanticSegmentation.py'")
    sys.exit()
    

cv2.imwrite(os.path.join(DATADIR_ADUULM_DATASET_GT, "dontCare.png"), np.zeros([850, 1920]))

# -------------------------------------------
# open txt-files 
# -------------------------------------------

# open files
image_file = open(path2list,"r")
image_nr = 0

# open file to write
save_file = open(output_trainingFiles,"w") 


for line in image_file:
 
    buff = line.rstrip('\n')
    img_name = buff.split('/')[1]
    weather_cond = buff.split('/')[0]
    img_timeStamp = img_name.split('_')[-1] 
    path2rosbag = os.path.join(DATADIR_ADUULM_DATASET_ROSSEQUENCES, weather_cond, img_name+'.bag')

    if not os.path.exists(path2rosbag):
        print ("[ERROR] path '{}' does not exit".format(path2rosbag))
        print ("[ERROR] please check your path!")
        sys.exit()

    print (path2rosbag)

    # ----------------
    # load rosbag
    # -----------------

    bag = rosbag.Bag(path2rosbag)     
    images_ROSbag = bag.read_messages(topics=[topicName_camera])

    # check, if topicName exists in ROS-bag
    if bag.get_message_count(topicName_camera) == 0:
        print("[ERROR] cannot find images with topic_name {} in ROS-bag!".format(topicName_camera))
        continue

    # ----------------
    # save actual image sample
    # ----------------

    hit = 0;
    iter = 0;
    img_list = []

    for img in images_ROSbag:

        img_list.append(img)

        # convert image from ROS-Format to openCV-format
        image1 = cv_bridge.CvBridge().imgmsg_to_cv2(img[1], desired_encoding="passthrough")

        # convert image to mono8
        image1 = map_uint16_to_uint8(image1)[0:850, ...]

        # histogram equalization
        image1 = cv2.equalizeHist(image1)
        
        if (str(img[2]) == img_timeStamp): 
            hit = iter

            # save image

            if not os.path.exists(os.path.join(saveDir_img,weather_cond)):
                os.makedirs(os.path.join(saveDir_img,weather_cond))

            cv2.imwrite(os.path.join(saveDir_img,weather_cond, img_name+"_camera_0.png"), image1)   

        iter += 1

        
    # ----------------
    # save previous image samples
    # ----------------

    if (prevImg > len(img_list)):
        print ("[ERROR] ros-sequence does not contain so many images: {} are requested, but only {} are provided".format(prevImg, img_list))
        sys.exit()

    for iter in range(prevImg):

        img = img_list[hit-1-iter]
        
        # convert image from ROS-Format to openCV-format
        image1 = cv_bridge.CvBridge().imgmsg_to_cv2(img[1], desired_encoding="passthrough")

        # convert image to mono8
        image1 = map_uint16_to_uint8(image1)[0:850, ...]

        # histogram equalization
        image1 = cv2.equalizeHist(image1)

        # save image

        if not os.path.exists(os.path.join(saveDir_img,weather_cond)):
            os.makedirs(os.path.join(saveDir_img,weather_cond))

        cv2.imwrite(os.path.join(saveDir_img,weather_cond, img_name+"_camera_"+str(iter+1)+".png"), image1)
        print ("save previous image {}".format(iter+1))

    # ---------------------------------------
    # add pathes to txt-file
    # ---------------------------------------

    path2DontCare = os.path.join(DATADIR_ADUULM_DATASET_GT, "dontCare.png") 

    for iter in range(prevImg):
        
        path2IMG = os.path.join(saveDir_img, weather_cond, img_name+"_camera_"+str(prevImg-iter)+".png")
        save_file.write(path2IMG + " " + path2DontCare + "\n")

    path2IMG = os.path.join(saveDir_img,  weather_cond, img_name+"_camera_0.png")
    path2GT = os.path.join(DATADIR_ADUULM_DATASET_GT, weather_cond, img_name+"_gtFine_labelIds.png")
    save_file.write(path2IMG + " " + path2GT + "\n")


# ---------------------------------------
# close output-files
# --------------------------------------- 

save_file.close()
        
        

