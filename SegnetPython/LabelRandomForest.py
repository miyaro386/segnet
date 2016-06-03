# coding=utf_8
import os

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






class LRF:

    start_num = 0
    end_num = 0
    label =0
    forest_path = ""
    
    #warm_start こいつがすべてのバグの元凶
    forest = RandomForestRegressor()

    m = "1000"
    s_weight = "0"

    def __init__(self,start_num, end_num, label):
        self.start_num = start_num
        self.end_num = end_num +1
        self.label = label
        self.forest_path = './supervised/label'+str(label)+'/data'+str(start_num)+'-'+str(end_num)
        self.forest = self.laod_forest()

    def laod_forest(self):
        if os.path.exists(self.forest_path+'/forest.bin'):
            return joblib.load(self.forest_path+'/forest.bin')
        else :
            print "LRF forest does not exist",self.label
            print "making new LRF forest",self.label
            #こいつでフォレストを返すと全てが狂う
            return self.create_forest()


    def create_forest(self):

        forest = RandomForestRegressor(n_estimators = 20)
        
        #sp_vec_path_list = CSVReader.get_path_list2("./output/csvpath/spdata_path_m"+self.m+"cs"+self.cs+"w"+self.s_weight+".csv",0)
        sp_vec_path_list = CSVReader.get_path_list2("./output/csvpath/spdata_path_m"+self.m+".csv",0)
        
        if (len(sp_vec_path_list)<self.end_num):
            print "end_num is larger than data length"
            return -1


        trainingdata = []
        traininglabel = []
        
        count = 0
        n_count = 0
        print "making training data"
        for i in range(self.start_num ,self.end_num):
            #print "inputting",i,sp_vec_path_list[i]
            data = CSVReader.read_csv_as_float(sp_vec_path_list[i])
            #self.data_deteil(sp_vec_path_list[i],i)
            label_col = len(data[0])-1
            for line in data:
                trainingdata += [ line[0:label_col] ]
                if line[label_col] == self.label:
                    traininglabel +=[1]
                else:
                    traininglabel +=[0]
        print "training data Complete"
        
        
        forest.fit(trainingdata , traininglabel )
        # Save 
        print "save to", self.forest_path
        if not os.path.exists(self.forest_path):
            os.makedirs(self.forest_path)
        joblib.dump(forest, self.forest_path+'/forest.bin')
    
        return forest

    
    def get_prob(self, sp_data):
        output = self.forest.predict( [ sp_data ] )
        return output.tolist()

    def predict(self, num):


        src_jpg_path = CSVReader.get_path_list2('SUNRGBDMeta_reduced.csv',4)
        src_dep_path = CSVReader.get_path_list2('SUNRGBDMeta_reduced.csv',3)
        src_label_path = CSVReader.get_path_list2('SUNRGBDMeta_reduced.csv',9)

     
        jpg_path = src_jpg_path[2*num]
        dep_path = src_dep_path[2*num]
        label_path = src_label_path[2*num]

        
        print "making test data"
        print "inputting",jpg_path
        print "inputting",label_path

        if os.path.exists(jpg_path+"_m"+self.m+"spdata.csv"):
            print "exist",jpg_path
        else:
            cmd = u'"slic\SLICOSuperpixel.exe"'
            os.system(cmd+" "+self.m+" "+jpg_path+" "+dep_path+" "+label_path+" "+ self.s_weight);

        vec_data_path = jpg_path+"_m"+self.m+"spdata.csv"
        sp_map_path = jpg_path+"_m"+self.m+"spmap.csv"

      
        
        data = CSVReader.read_csv_as_float(vec_data_path)
        sp_map =  CSVReader.read_csv_as_int(sp_map_path)
        test_data = []
        print "making test data"
        #print "inputting",jpg_path
        label_col = len(data[0])-1
        for line in data:
            test_data += [ line[0:label_col] ]

        print "test data Complete"

        output = self.forest.predict( test_data )
        #self.output_file(output, num)

        return output

    def output_file(self, output, num):
        if not os.path.exists('./output'):
            os.makedirs('./output')
        f = open('./output/outputlabel_data'+str(num)+'.csv', 'w')
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow(output)
        f.close()

    def show_detail(self):
        print "n_estimators",self.forest.n_estimators
        print "criterion",self.forest.criterion
        print "max_depth",self.forest.max_depth
        print "n_features",self.forest.n_features_
        print "n_outputs_",self.forest.n_outputs_
        print "feature_importances_",self.forest.feature_importances_
        # print "oob_score_",self.forest.oob_score_
        # print "oob_decision_function_",self.forest.oob_decision_function_


    def combine(self, forest):
        self.forest.estimators_ += forest.forest.estimators_
        self.forest.n_estimators = len(self.forest.estimators_)
        return self.forest


    def save_combined_forest(self,start_num, end_num):
        self.forest_path = './supervised/label'+str(self.label)+'/data'+str(start_num)+'-'+str(end_num)
        
        print "save to", self.forest_path
        if not os.path.exists(self.forest_path):
            os.makedirs(self.forest_path)
        joblib.dump(self.forest, self.forest_path+'/forest.bin')



    def data_statistics(self):
        if not os.path.exists('./output/statistics'):
            os.makedirs('./output/statistics')
        sp_vec_path_list = CSVReader.get_path_list2("./output/csvpath/spdata_path_m1000.csv",0)
        for i in range(5000):
            f = open('./output/statistics/data'+str(i)+'.csv', 'w')
            writer = csv.writer(f, lineterminator='\n')
            output = [0 for x in range(38)]
            data = CSVReader.read_csv_as_float(sp_vec_path_list[i])
            label_col = len(data[0])-1
            for line in data:
                output[int(line[label_col])] += 1
            print sp_vec_path_list[i]
            for j in range( len(output) ):
                print j,output[j]
                writer.writerow([j,output[j]])
            f.close()

    def data_deteil(self, sp_data_path ,i):
        if not os.path.exists('./output/statistics'):
            os.makedirs('./output/statistics')
        f = open('./output/statistics/data'+str(i)+'.csv', 'w')
        writer = csv.writer(f, lineterminator='\n')
        output = [0 for x in range(38)]
        data = CSVReader.read_csv_as_float(sp_data_path)
        label_col = len(data[0])-1
        for line in data:
            output[int(line[label_col])] += 1
        print sp_data_path
        for j in range( len(output) ):
            print j,output[j]
            writer.writerow([j,output[j]])
        f.close()

