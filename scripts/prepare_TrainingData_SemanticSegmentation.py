#!/usr/bin/python
# -*- coding: utf-8-*

###################################################################################################################################
### This script creates the ground-truth images for semantic labeling from a given json-file of the corresponding image.        ###
### Furthermore, the 16bit images are converted into 8-bit images and a txt-file is created, which can be used for training.    ###
### The following folders are created:                                                                                          ###
### 1) img:                 contains the 8-bit images                                                                           ### 
### 2) gt:                  contains the ground-truth-images                                                                    ###
### 3) gt_color:            colored illustration of the ground-truth                                                            ###
### 4) gt_colorOverlay:     ground-truth and corresponding camera image are overlayed                                           ###
### 5) training_scrips:     folder, where the txt-file is stored                                                                ###
###                                                                                                                             ###
### required arguments:                                                                                                         ###
###     - argument 1: <path2dataset>                                                                                            ###
###     - argument 2: image_split ('split_training_good_weather_only' or 'split_training_adverse_weather')                      ###
###     - argument 3: training_set ('train','val','test', ...)                                                                  ###
###                                                                                                                             ###
###                                                                                                                             ###
### usage: python prepare_TrainingData_SemanticSegmentation.py <path2dataset> <image_split> <training_set>                      ###
### e.g. python prepare_TrainingData_SemanticSegmentation.py /home/user/ADUUlm_Dataset split_training_good_weather_only train   ###
###                                                                                                                             ###
### Version 1.0                                                                                                                 ###
### (c) Andreas Pfeuffer, 30.08.2020                                                                                            ###
###################################################################################################################################


import numpy as np
import cv2
from collections import namedtuple
import os
import sys

from map_uint16_to_uint8 import map_uint16_to_uint8

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

DATADIR_ADUULM_DATASET_DATA = os.path.join(DATADIR_ADUULM_DATASET, "data")

TASK = "semantic_segmentation"

# -------------------------------------
# define output-pathes
# -------------------------------------

# path to the output txt-file, which can be used for training. Each line contains the path to the image and the corresponding ground-truth
# eg. <path2image> <path2gt>
output_trainingFiles = os.path.join(DATADIR_ADUULM_DATASET, TASK, "training_scrips", TRAINING_SET+".txt")
# path to directory, where to save the processed images
saveDir_img = os.path.join(DATADIR_ADUULM_DATASET, TASK, 'img')
# path to directory, where to save the groundtruth (id-images)
saveDir_gt = os.path.join(DATADIR_ADUULM_DATASET, TASK, 'gt')
# path to directory, where to save the colored groundtruth-images
saveDir_gtcolor = os.path.join(DATADIR_ADUULM_DATASET, TASK, 'gt_color')
# path to directory, where to save the colored groundtruth-images
saveDir_gtcolorOverlay = os.path.join(DATADIR_ADUULM_DATASET, TASK, 'gt_colorOverlay')

# -------------------------------------
# create directory, if not exist
# -------------------------------------

if not os.path.exists(saveDir_gtcolor):
    os.makedirs(saveDir_gtcolor)
if not os.path.exists(saveDir_gtcolorOverlay):
    os.makedirs(saveDir_gtcolorOverlay)
if not os.path.exists(saveDir_gt):
    os.makedirs(saveDir_gt)
if not os.path.exists(saveDir_img):
    os.makedirs(saveDir_img)
if not os.path.exists(os.path.join(DATADIR_ADUULM_DATASET, TASK, "training_scrips")):
    os.makedirs(os.path.join(DATADIR_ADUULM_DATASET, TASK, "training_scrips"))

# -------------------------------------------
# open txt-files 
# -------------------------------------------

# open files
image_file = open(path2list,"r")
image_nr = 0

# open file to write
save_file = open(output_trainingFiles,"w") 


# -------------------------------------------
# create ground-truth for each file of list
# -------------------------------------------

imgHeight = 0
imgWidth = 0
gt = []

for line in image_file:
 
    buff = line.rstrip('\n')
    img_name = buff.split('/')[1]
    weather_cond = buff.split('/')[0]

    # -------------------------------------
    # imread image
    # -------------------------------------

    path2IMG = os.path.join(DATADIR_ADUULM_DATASET_DATA, weather_cond, img_name, img_name + prefix_camera)
    if not os.path.exists(path2IMG):
        print ("[ERROR] path '{}' does not exit".format(path2IMG))
        print ("[ERROR] please check your path!")
        sys.exit()
        
    print ('process {}'.format(path2IMG))

    img = cv2.imread(path2IMG,3)

    # convert 16bit image to 8bit image  
 
    img = map_uint16_to_uint8(img)

    # --------------------------------------
    # imread json file
    # --------------------------------------


    path2JSON = os.path.join(DATADIR_ADUULM_DATASET_DATA, weather_cond, img_name, img_name + ".json")
    if not os.path.exists(path2JSON):
        print ("[ERROR] path '{}' does not exit".format(path2IMG))
        print ("[ERROR] please check your path!")
        sys.exit()

    with open(path2JSON, 'r') as f:
        rois = f.readlines()


    # --------------------------------------
    # get polygons of objects
    # --------------------------------------

    for ix, roi in enumerate(rois):
        s = roi.split()
        if s[0] == '\"imgHeight\":':
            imgHeight = int(s[1].rstrip(','))
        if s[0] == '\"imgWidth\":':
            imgWidth = int(s[1].rstrip(','))
            gt = np.zeros((np.shape(img)[0],np.shape(img)[1]),dtype=np.uint8)
            gt_color = np.zeros((np.shape(img)[0],np.shape(img)[1],3),dtype=np.uint8)
            
        if s[0] == '\"label\":': 
            if s[1] == '\"traffic':
                s[1] = s[1] + ' ' + s[2]
            if s[1] == '\"ego':
                s[1] = s[1] + ' ' + s[2]
            if s[1] == '\"rectification':
                s[1] = s[1] + ' ' + s[2]
            if s[1] == '\"out':
                s[1] = s[1] + ' ' + s[2]+ ' ' + s[3]
            name = s[1].rstrip('\",').strip('\"')

            if rois[ix+1].split()[0] == '\"instance\":':
                ix = ix + 1
                #print ('skip instance')
                

            ix = ix + 3
            object_polygon = []

            while s[0] != ']':
                s = rois[ix].split(',')
                x = int(s[0])
                s = rois[ix+1].split(',')
                y = int(s[0])
                s = rois[ix+2].split()
                ix = ix + 4
                object_polygon.append([x,y])

            # get label-id for each class

            if len(object_polygon) > 0:
                if name == "Car":
                    color = np.array([0,	0,	255])
                    img_id = 8
                elif name == "Truck":
                    color = np.array([0,	255,	255])
                    img_id = 9
                elif name == "Bus":
                    img_id = 10
                    color = np.array([153,	153,	255])
                elif name == "Motorcycle":
                    color = np.array([0,	0,	230	])
                    img_id = 11
                elif name == "Pedestrian":
                    color = np.array([220,	20,	60])
                    img_id = 6
                elif name == "Bicyclist":
                    color = np.array([155,	40,	0])
                    img_id = 7
                elif name == "Traffic_sign":
                    color = np.array([220,	220,	0])
                    img_id = 5
                elif name == "Traffic_light":
                    color = np.array([250,	170,	30])
                    img_id = 4
                elif name == "Road":
                    color = np.array([170,	85,	255])
                    img_id = 1
                elif name == "EgoLane":   # map to road
                    color = np.array([170,	85,	255])
                    img_id = 1
                    continue
                elif name == "ParallelLane":   # map to road
                    color = np.array([170,	85,	255])
                    img_id = 1
                    continue
                elif name == "OppositeLane":   # map to road
                    color = np.array([170,	85,	255])
                    img_id = 1
                    continue
                elif name == "ParkingLane":   # map to road
                    color = np.array([170,	85,	255])
                    img_id = 1
                    continue
                elif name == "CrossingArea":   # map to road
                    color = np.array([170,	85,	255])
                    img_id = 1
                    continue
                elif name == "OppositeTurnLane":   # map to road
                    color = np.array([170,	85,	255])
                    img_id = 1
                    continue
                elif name == "ParallelTurnLane":   # map to road
                    color = np.array([170,	85,	255])
                    img_id = 1
                    continue
                elif name == "Sidewalk":
                    color = np.array([244,	35,	232])
                    img_id = 2
                elif name == "Pole":
                    color = np.array([152,	251,	152])
                    img_id = 3
                elif name == "ego_vehicle":
                    color = np.array([0,0,0])
                    img_id = 0
                else: 
                    color = np.array([0,0,0])
                    img_id = 0
                

                if len(object_polygon) > 0:
                    cv2.fillPoly(gt,[np.array(object_polygon)],(img_id))
                    cv2.fillPoly(gt_color,[np.array(object_polygon)],[int(color[2]),int(color[1]),int(color[0])])
                    

    # ---------------------------------------
    # save image/ground-truth
    # ---------------------------------------

    # save colorized ground-truth

    path2GTcolor = saveDir_gtcolor + '/' + img_name + "_gtFine_color.png"
    cv2.imwrite(path2GTcolor, gt_color[0:850,:,:])

    # save colorized ground-truth

    gt_colorOverlay = cv2.addWeighted(img, 0.5, gt_color,0.5,0)

    path2GTcolorOverlay = saveDir_gtcolorOverlay + '/' + img_name + "_gtFine_color.png"
    cv2.imwrite(path2GTcolorOverlay, gt_colorOverlay[0:850,:,:])

    saveDir_gtcolorOverlay

    # save ground-truth as ID-image

    if not os.path.exists(os.path.join(saveDir_gt, weather_cond)):
        os.makedirs(os.path.join(saveDir_gt, weather_cond))

    path2GT = os.path.join(saveDir_gt, weather_cond, img_name + "_gtFine_labelIds.png")
    cv2.imwrite(path2GT, gt[0:850,:])

    # save image converted as 8-bit (Note, that parts of the engine bonnet are removed)

    if not os.path.exists(os.path.join(saveDir_img, weather_cond)):
        os.makedirs(os.path.join(saveDir_img, weather_cond))

    path2IMG = os.path.join(saveDir_img, weather_cond, img_name + "_camera.png")
    cv2.imwrite(path2IMG, img[0:850,:,:])

    # ---------------------------------------
    # add pathes to txt-file
    # --------------------------------------- 

    save_file.write(path2IMG + " " + path2GT + "\n")

    print ('Process img ', image_nr)
    image_nr = image_nr + 1


# ---------------------------------------
# close output-files
# --------------------------------------- 

save_file.close()


