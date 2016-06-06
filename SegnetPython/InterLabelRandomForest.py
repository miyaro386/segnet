# coding=utf_8
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_validation import train_test_split
from sklearn.externals import joblib
from sklearn import svm
import cv2
import numpy as np
import csv
import ImgToVec
import CSVReader
import Assessment
import time




class ILRF:
    start_num = 0
    end_num = 0
    layer = 0
    label = 0
    forest_path = ""
    forest = None

    m = "1000"
    s_weight = "0"
    

    def __init__(self,start_num, end_num, layer, label):
        self.start_num = start_num
        self.end_num = end_num +1
        self.layer = layer
        self.label = label
        self.forest_path = './supervised/layer'+str(layer)+'/label'+str(label)+'/data'+str(start_num)+'-'+str(end_num)
        self.forest = self.laod_forest()

    def laod_forest(self):
        if os.path.exists(self.forest_path+'/forest.bin'):
            return joblib.load(self.forest_path+'/forest.bin')
        else :
            print "ILRF does not exist"
            print "making new ILRF",self.layer
            return self.create_forest()

    def create_forest(self):
        start_time = time.time()
        forest = RandomForestRegressor(n_estimators = 10)

       
        sp_vec_path_list = CSVReader.get_path_list2("./output/csvpath/spdata_path_m"+self.m+".csv",0)
        sp_neighbors_list = CSVReader.get_path_list2("./output/csvpath/spdata_path_m"+self.m+".csv",2)
      
        if (len(sp_vec_path_list)<self.end_num):
            print "end_num is larger than data length"
            return -1


        trainingdata = []
        traininglabel = []
        
        print "ILRF making training data Layer",self.layer,'Label',self.label
        for i in range(self.start_num ,self.end_num):
            
            #print "inputting",i,sp_vec_path_list[i]
            data = CSVReader.read_csv_as_float(sp_vec_path_list[i])
            neighbors = CSVReader.read_csv_as_int(sp_neighbors_list[i])
            
            probs = self.load_probs(data, i )
            #print len(data),len(probs)
            label_col = len(data[0])-1
            for j in range(len(data)):
                vector = []
                #print j,neighbors[j]
                vector.extend( probs[j] )
                vector.extend( probs[neighbors[j][0]] )
                vector.extend( probs[neighbors[j][1]] )
                vector.extend( probs[neighbors[j][2]] )
                vector.extend( probs[neighbors[j][3]] )
                #print "training vector",len(vector)
                trainingdata += [ vector ]
                if data[j][label_col] == self.label:
                    traininglabel +=[1]
                else:
                    traininglabel +=[0]

        print "training data Complete"
 

        forest.fit(trainingdata , traininglabel )
        
        # Save 
        print self.layer,"save to", self.forest_path
        if not os.path.exists(self.forest_path):
            os.makedirs(self.forest_path)
        joblib.dump(forest, self.forest_path+'/forest.bin')
        
        print 'InterLabelRondomForest layer',self.layer,'Label', self.label,'Complete' ,time.time() - start_time
        return forest


    def load_probs(self, data, data_num):
        if self.layer == 0:
            return CSVReader.read_csv_as_float('./output/probs/base/'+str(data_num)+'.csv')
        else: 
            return CSVReader.read_csv_as_float('./output/probs/layer'+str(self.layer-1)+'/'+str(data_num)+'.csv')


    def get_prob(self, probs):
        output = self.forest.predict( [ probs ] )
        return output.tolist()
     
    def combine(self, forest):
        self.forest.estimators_ += forest.forest.estimators_
        self.forest.n_estimators = len(self.forest.estimators_)
        return self.forest


    def save_combined_forest(self,start_num, end_num):
        self.forest_path = './supervised/layer'+str(self.layer)+'/label'+str(self.label)+'/data'+str(start_num)+'-'+str(end_num)
        
        print "save to", self.forest_path
        if not os.path.exists(self.forest_path):
            os.makedirs(self.forest_path)
        joblib.dump(self.forest, self.forest_path+'/forest.bin')
        
        
         
    # forest内の各種データ確認用
    def show_detail(self):
        print "n_estimators",self.forest.n_estimators
        print "criterion",self.forest.criterion
        print "max_depth",self.forest.max_depth
        print "n_features",self.forest.n_features_
        print "n_outputs_",self.forest.n_outputs_
        print "feature_importances_",self.forest.feature_importances_