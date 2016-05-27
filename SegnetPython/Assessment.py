# coding=utf_8
import cv2
import numpy as np
import csv
import CSVReader
import os
import collections
import Forest
import matplotlib.pyplot as plt

ROOT_FILE_PATH = "C:/segnet/DataSet"

def AssessmentByMIOU(img_a,img_b):
    rate=1.0
    LabelNum=38
    PredictionLabelNum=0
    Ic_Uc=0.0
    width = img_a.shape[1]
    height = img_a.shape[0]
    GrandTruth=np.zeros(LabelNum,dtype=np.int32)
    Prediction=np.zeros(LabelNum,dtype=np.int32)
    Intersection=np.zeros(LabelNum,dtype=np.int32)
    IntersectionRate=np.zeros(LabelNum,dtype=np.float32)
    #ラベルも同時に出力したい場合
    label = CSVReader.read_csv('seg37list.csv')
    LabelList = ['other']
    for row in label:
        for i in range(len(row)):
            LabelList.append(row[i])    
    #ここまで
    for y in range(height):
        for x in range(width):
            GrandTruth[img_a[y,x]] += 1
            Prediction[img_b[y,x]] += 1
            if img_a[y,x] == img_b[y,x]:
                Intersection[img_a[y,x]] += 1    
    print (GrandTruth)
    print (Prediction)
    print(Intersection)
    for i in range(LabelNum):
        if(GrandTruth[i]+Prediction[i]-Intersection[i]>0):
            IntersectionRate[i]=1.0*Intersection[i]/(GrandTruth[i]+Prediction[i]-Intersection[i])
            print(i,LabelList[i],IntersectionRate[i])
            Ic_Uc += 1.0*IntersectionRate[i]
        if(Prediction[i]>0):
            PredictionLabelNum += 1
    rate = Ic_Uc/PredictionLabelNum
    '''
    #ラベルごとIOU値の棒グラフ出力
    X=np.arange(0,LabelNum,1)
    plt.bar(X,IntersectionRate,width=0.7)
    plt.xticks(X, LabelList)    # 目盛りを数字からラベルに書き換える
    plt.xticks(rotation=90)     # 目盛りを傾ける
    plt.show()
    #ここまで
    '''
    return rate

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





