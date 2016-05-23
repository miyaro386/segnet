import cv2
import numpy as np
import csv
import CSVReader
import os
import collections
import Forest

ROOT_FILE_PATH = "C:/segnet/DataSet"

def assess_label_coincidence(img_a,img_b):
    all_pixel_num = 0
    coin_pixel_num = 0

    width = img_a.shape[1]
    height = img_a.shape[0]

    all_pixel_num = width* height
    print all_pixel_num
    
    for y in range(height):
        for x in range(width):
            if img_a[y,x] == img_b[y,x]:
                coin_pixel_num += 1
                #print coin_pixel_num

    rate = 1.0*coin_pixel_num/all_pixel_num
    return rate


img_a = cv2.imread('12.png',0)
img_b = cv2.imread('output.jpg',0)
#print assess_label_coincidence(img_a,img_b)


forest = Forest.Forest(10,5)
forest.create_forest()



