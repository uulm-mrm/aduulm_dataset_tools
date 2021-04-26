#!/usr/bin/python
# -*- coding: utf-8-*

###################################################################################################################################
### This script creates dense-depth-images according to the following paper: "Robust Semantic Segmentation in Adverse           ###
### Weather Conditions by means of Sensor Data Fusion" (https://arxiv.org/pdf/1905.10117.pdf)                                   ###
### The following folders are created:                                                                                          ###
### 1) depth:               contains the generated dense-depth-images from the lidar data                                       ### 
### 2) training_scrips:     folder, where the txt-file is stored                                                                ###
###                                                                                                                             ###
### required arguments:                                                                                                         ###
###     - argument 1: <path2dataset>/ADUUlm_Dataset                                                                             ###
###     - argument 2: image_split ('split_training_good_weather_only' or 'split_training_adverse_weather')                      ###
###     - argument 3: training_set ('train','val','test', ...)                                                                  ###
###                                                                                                                             ###
###                                                                                                                             ###
### usage: python generate_DepthImage.py  <path2dataset>/ADUUlm_Dataset <image_split> <training_set>                            ###
### e.g. python generate_DepthImage.py  /home/user/ADUUlm_Dataset split_training_good_weather_only train                        ###
###                                                                                                                             ###
### Version 1.0                                                                                                                 ###
### (c) Andreas Pfeuffer, 30.08.2020                                                                                            ###
###################################################################################################################################


import numpy as np
import cv2
import os
import yaml
import sys

from map_uint16_to_uint8 import map_uint16_to_uint8
from pcl import *


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

DATADIR_ADUULM_DATASET_DATA = os.path.join(DATADIR_ADUULM_DATASET, "data")
DATADIR_ADUULM_DATASET_METADATA = os.path.join(DATADIR_ADUULM_DATASET, "Metadata")
DATADIR_ADUULM_DATASET_GT = os.path.join(DATADIR_ADUULM_DATASET, 'gt')
DATADIR_ADUULM_DATASET_IMG = os.path.join(DATADIR_ADUULM_DATASET, 'img')

# -------------------------------------
# define output-pathes
# -------------------------------------

# path to the output txt-file, which can be used for training. Each line contains the path to the image and the corresponding ground-truth
# eg. <path2image> <path2depth> <path2gt>
output_trainingFiles = os.path.join(DATADIR_ADUULM_DATASET, "training_scrips", TRAINING_SET+"_depth.txt")
# path to directory, where to save the processed images
saveDir_depth = os.path.join(DATADIR_ADUULM_DATASET, 'depth')

# -------------------------------------
# create directory, if not exist
# -------------------------------------

if not os.path.exists(saveDir_depth):
    os.makedirs(saveDir_depth)
if not os.path.exists(os.path.join(DATADIR_ADUULM_DATASET, "training_scrips")):
    os.makedirs(os.path.join(DATADIR_ADUULM_DATASET, "training_scrips"))

#-----------------------
# some parameters
#-----------------------

# determine colored representation of depth image (True/False)
CREATE_COLOROUTPUT = False
# whether to apply interpolation to generate depth-image (if false, the lidar points are only plot into the camera image)
APPLY_INTERPOLATION = True
# interpolation size of single lidar point
rectAngleSize_x = 20
rectAngleSize_y = 5 

# -------------------------------------------
# open txt-files 
# -------------------------------------------

# open files
image_file = open(path2list,"r")
image_nr = 0

# open file to write
save_file = open(output_trainingFiles,"w") 


# ------------------------------------------------
# camera paramters
# ------------------------------------------------

          
# load yaml-file

if not os.path.exists(os.path.join(DATADIR_ADUULM_DATASET_METADATA, "camera_wideangle.yaml")):
    print ("[ERROR] path '{}' does not exit".format(os.path.join(DATADIR_ADUULM_DATASET_METADATA, "camera_wideangle.yaml")))
    print ("[ERROR] please check your path!")
    sys.exit()

yaml_file = open(os.path.join(DATADIR_ADUULM_DATASET_METADATA, "camera_wideangle.yaml"))
camera_parameters = yaml.load(yaml_file, Loader=yaml.FullLoader)['camera_wideangle']

# load intrinsic camera parameters

cx, cy = camera_parameters['optical_center']   
skew = camera_parameters['skew'] 
fx, fy = camera_parameters['focal_length']   
k1,k2,k5 = camera_parameters['radial_dist']
k3,k4 = camera_parameters['tangent_dist']

cameraParams_K = np.array([[fx,  skew, cx], [ 0, fy, cy], [ 0,  0,  1]], np.float32)                          
cameraParams_distCoeffs = np.array([k1,k2,k3,k4,k5], np.float32)

# load extrinsic camera parameters

T = camera_parameters['alignment']

R = np.array([[float(T[0][0]), float(T[0][1]),  float(T[0][2])],
        [float(T[1][0]), float(T[1][1]),  float(T[1][2])],
        [float(T[2][0]), float(T[2][1]),  float(T[2][2])]], np.float32)


t = np.array([[float(T[0][3])],
        [float(T[1][3])],
        [float(T[2][3])]], np.float32)


# invert transformationmatrix

cameraParams_tvec = (-1) * np.matmul(R.transpose((1, 0)),t)
cameraParams_rvec = cv2.Rodrigues(R.transpose((1, 0)))[0]



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

    # load lidar-data

    path2LidarData_VeloFront = os.path.join(DATADIR_ADUULM_DATASET_DATA, weather_cond, img_name, img_name+"_VeloFront_syn.pcd")
    path2LidarData_VeloRear = os.path.join(DATADIR_ADUULM_DATASET_DATA, weather_cond, img_name, img_name+"_VeloRear_syn.pcd")
    path2LidarData_VeloLeft = os.path.join(DATADIR_ADUULM_DATASET_DATA, weather_cond, img_name, img_name+"_VeloLeft_syn.pcd")
    path2LidarData_VeloRight = os.path.join(DATADIR_ADUULM_DATASET_DATA, weather_cond, img_name, img_name+"_VeloRight_syn.pcd")

    if not os.path.exists(path2LidarData_VeloFront):
        print ("[ERROR] path '{}' does not exit".format(path2LidarData_VeloFront))
        print ("[ERROR] please check your path!")
        sys.exit()
    print (path2LidarData_VeloFront)

    # open Point Cloud

    pointCloud = imreadPCD(path2LidarData_VeloFront)
    pointCloud = addPCD(pointCloud, imreadPCD(path2LidarData_VeloRear))
    pointCloud = addPCD(pointCloud, imreadPCD(path2LidarData_VeloLeft))
    pointCloud = addPCD(pointCloud, imreadPCD(path2LidarData_VeloRight))

    # load image

    path2Img = os.path.join(DATADIR_ADUULM_DATASET_DATA, weather_cond, img_name, img_name+"_camera_16bit.tiff")
    if not os.path.exists(path2Img):
        print ("[ERROR] path '{}' does not exit".format(path2Img))
        print ("[ERROR] please check your path!")
        sys.exit()

    img = cv2.imread(path2Img,3)

    # convert 16bit image to 8bit image  
 
    img = map_uint16_to_uint8(img)[0:850,:,:]

    # --------------------------------
    # convert lidar point to image
    # --------------------------------

    image_points = []

    for iter in range(len(pointCloud)):  

        proImg = cv2.projectPoints(np.array([pointCloud[iter].x, pointCloud[iter].y, pointCloud[iter].z]), cameraParams_rvec, cameraParams_tvec, cameraParams_K, cameraParams_distCoeffs)[0][0][0];
    
        # check, if image points are in image range  
    
        if (int (proImg[0]) < 0) or (int(proImg[0]) > 1920) or (int(proImg[1]) < 0) or (int(proImg[1]) > 850):
            continue

        # determine color and plot to image

        point = PointXYZ()
        point.x = int (proImg[0])
        point.y = int (proImg[1])
        point.z = float(pointCloud[iter].x) - t[2] # add depth to 3. channel

        # remove all points behind the car:
        if (point.z > 0.0):
            image_points.append(point)
        


    # --------------------------------
    # generate depth image
    # --------------------------------

    # get image size of image
    size = np.shape(img)

    # initialize arrys
    lidar2Image = np.zeros((size[0],size[1],1), dtype=np.float32)
    supportedPoints = np.zeros((size[0],size[1],1), dtype=np.float32)
    interpolationResult = np.zeros((size[0],size[1],1), dtype=np.float32)
    interpolationResult_color = []
    if CREATE_COLOROUTPUT:
        interpolationResult_color = np.zeros((size[0],size[1],3), dtype=np.uint8)
        imgLidar = 255*np.ones((size[0],size[1],3), dtype=np.uint8)

    # fill arrays with lidar points
    for i in range(len(image_points)):
        cv2.circle(supportedPoints, (int(image_points[i].x), int(image_points[i].y)), 1, 1, -1)
        cv2.circle(lidar2Image, (int(image_points[i].x), int(image_points[i].y)), 1, float(image_points[i].z), -1)

       
    # determine integral image

    if APPLY_INTERPOLATION:
        supportedPoints_Integral = cv2.integral(supportedPoints)
        lidar2Image_Integral = cv2.integral(lidar2Image)

    # interpolate

    if APPLY_INTERPOLATION:
        for i_x in range(size[0]):
            x1 = max(i_x -rectAngleSize_x, 0)
            x2 = min(i_x +rectAngleSize_x, (size[0])-1)
            for i_y in range(size[1]):
                y1 = max(i_y -rectAngleSize_y, 0)
                y2 = min(i_y +rectAngleSize_y, (size[1])-1)

                # determine weighting
                supportedPoints_amount = supportedPoints_Integral[x1,y1] + supportedPoints_Integral[x2,y2] - supportedPoints_Integral[x1,y2] - supportedPoints_Integral[x2,y1]

                # average depth values
                if supportedPoints_amount > 0:
                    interpolationResult[i_x, i_y] = (lidar2Image_Integral[x1,y1] + lidar2Image_Integral[x2,y2] - lidar2Image_Integral[x1,y2] - lidar2Image_Integral[x2,y1]) / supportedPoints_amount
    else:
        interpolationResult = lidar2Image

    if CREATE_COLOROUTPUT:
        interpolationResult_color[:,:,0] = interpolationResult[:,:,0]
        interpolationResult_color[:,:,1] = interpolationResult[:,:,0]
        interpolationResult_color[:,:,2] = interpolationResult[:,:,0]

        interpolationResult_color = cv2.applyColorMap(interpolationResult_color*3, cv2.COLORMAP_JET)
        
        for ix in range(np.shape(interpolationResult_color)[0]):
            for iy in range(np.shape(interpolationResult_color)[1]):
                if interpolationResult[ix,iy,:] == 0:
                    interpolationResult_color[ix,iy,:] = [255,255,255]

        # show result

        #cv2.imshow('image',cv2.addWeighted(interpolationResult_color, 0.5, img, 0.5,0.0))
        #cv2.waitKey(0)

    # batch normalization

    interpolationResult = interpolationResult / np.max(interpolationResult)

    # save result

    path2IMG = os.path.join(DATADIR_ADUULM_DATASET_IMG,  weather_cond, img_name+"_camera.png")
    path2GT = os.path.join(DATADIR_ADUULM_DATASET_GT, weather_cond, img_name+"_gtFine_labelIds.png")
    path2Depth = os.path.join(saveDir_depth, weather_cond, img_name+"_depth.png")

    # save image converted as 8-bit (Note, that parts of the engine bonnet are removed)

    if not os.path.exists(os.path.join(saveDir_depth,weather_cond)):
            os.makedirs(os.path.join(saveDir_depth,weather_cond))

    cv2.imwrite(path2Depth, interpolationResult[0:850,:,:]*255)

    # ---------------------------------------
    # add pathes to txt-file
    # --------------------------------------- 

    save_file.write(path2IMG + " " + path2Depth + " " + path2GT + "\n")


# ---------------------------------------
# close output-files
# --------------------------------------- 

save_file.close()

    

    


