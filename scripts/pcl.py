#!/usr/bin/python
# -*- coding: utf-8-*

import numpy as np
import os


# Point definition for a point cloud analogous to the Point Cloud Library (PCL)
class PointXYZ:
    def __init__(self):
        # point information
        self.x = 0
        self.y = 0
        self.z = 0

class PointXYZRGB:
    def __init__(self):
        # point information
        self.x = 0
        self.y = 0
        self.z = 0
        # color information
        self.r = 0
        self.g = 0
        self.b = 0


def imreadPCD(file_name):
    line_number = 0

    pointcloud = []

    # open files

    if (os.path.isfile(file_name)):
        image_file = open(file_name,"r")
        for line in image_file:
            line_number = line_number + 1
            if line_number > 11:
                buff = line.split()
                
                point = PointXYZ()

                point.x = float(buff[0])
                point.y = float(buff[1])
                point.z = float(buff[2])

                pointcloud.append(point)

    return pointcloud

def addPCD(pointcloud_org, pointcloud2Add):

    for iter in range(len(pointcloud2Add)):
        pointcloud_org.append(pointcloud2Add[iter])
        

    return pointcloud_org


