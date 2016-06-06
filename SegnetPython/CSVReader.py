#coding:utf-8

import os
import csv
import string

CSV_FILE_PATH = ""
ROOT_PATH = ""
ROOT_FILE_PATH = "C:/segnet/DataSet"

def read_csv(CSV_FILE_PATH):
    f= open(CSV_FILE_PATH,'rb')
    dataReader = csv.reader(f)
    return dataReader

#この関数に,csvで保存したファイルパスと何行目かを渡すとListで返す

def get_path_list(CSV_FILE_PATH,ROOT_FILE_PATH, number):
    csv_data = read_csv(CSV_FILE_PATH)
    path_list = []
    #print csv_data
    for line in csv_data:
        #print line
        path = line[number]
        path = path.replace('$ROOT_PATH$',ROOT_FILE_PATH)
        path.replace('\\','/') 
        path_list.append(path)
    return path_list

def get_path_list2(CSV_FILE_PATH, number):
    csv_data = read_csv(CSV_FILE_PATH)
    path_list = []
    #print csv_data
    for line in csv_data:
        #print line
        path = line[number]
        path = path.replace('$ROOT_PATH$',ROOT_FILE_PATH)
        path.replace('\\','/') 
        path_list.append(path)
    return path_list

def read_csv_as_int(CSV_FILE_PATH):
    f= open(CSV_FILE_PATH,'rb')
    dataReader = csv.reader(f)
    data = []
    list = []
    for line in dataReader:
        list = []
        for item in line:
            list.append(int(float(item)))
        data.append(list)

    return data

def read_csv_as_float(CSV_FILE_PATH):
    f= open(CSV_FILE_PATH,'rb')
    dataReader = csv.reader(f)
    data = []
    list = []
    for line in dataReader:
        list = []
        for item in line:
            #if (item.replace(".","").isdigit() == False) :
            #    list.append(0)
            #    continue
            list.append(float(item))
        data.append(list)
        #print list
    return data