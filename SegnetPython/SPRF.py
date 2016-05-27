# coding=utf_8
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
from sklearn.externals import joblib
from sklearn import svm
import cv2
import numpy as np
import csv
import ImgToVec
import CSVReader
import Converter
import Assessment



class SPRF:

    start_num = 0
    end_num = 0
    m = "0"
    cs = "0"
    weight ="0"
    s_weight =""
    forest_path = ""
    forest = RandomForestClassifier(n_estimators = 5,warm_start=True)

    

    def __init__(self,start_num, end_num, m,cs ,weight):
        self.start_num = start_num
        self.end_num = end_num +1
        self.m = str(m)
        self.cs = str(cs)
        self.weight = str(weight)
        self.s_weight = str(int( float(weight)  * 10))
        self.forest_path = './supervised/data'+str(start_num)+'-'+str(end_num)+'m'+self.m+'cs'+self.cs+'w'+self.s_weight
        self.forest = self.laod_forest()
      
 

    def save_forest(self):
        if not os.path.exists(self.forest_path):
            os.makedirs(self.forest_path)
        joblib.dump(self.forest, forest_path+'/forest.bin')

  

    def create_forest(self):

        print "save to", self.forest_path
        
        trainingdata = []
        traininglabel = []


        sp_vec_path_list = CSVReader.get_path_list2("./output/csvpath/spdata_path_m"+self.m+"cs"+self.cs+"w"+self.s_weight+".csv",0)
        if (len(sp_vec_path_list)<self.end_num):
            print "end_num is larger than data length"
            return -1

        print "making training data"
        for i in range(self.end_num - self.start_num):
            print "inputting",i,sp_vec_path_list[i]
            data = CSVReader.read_csv_as_float(sp_vec_path_list[i])
            label_col = len(data[0]) -1
            for line in data:
                #print line[0:256/int(self.cs)*4] 
                trainingdata += [ line[0:label_col] ]
                #print int(line[label_col])
                traininglabel+= [int(line[label_col])]
        print "training data Complete"
 
        #print traininglabel
        self.forest.fit(trainingdata , traininglabel )
        
        # Save 
        print "save to", self.forest_path
        if not os.path.exists(self.forest_path):
            os.makedirs(self.forest_path)
        joblib.dump(self.forest, self.forest_path+'/forest.bin')
    

        return self.forest



    def laod_forest(self):
        if os.path.exists(self.forest_path+'/forest.bin'):
            return joblib.load(self.forest_path+'/forest.bin')
        else :
            print "forest does not exist"
            print "making new forest"
            return self.create_forest()


 
        
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

        if os.path.exists(jpg_path+"_m"+self.m+"c"+self.cs+".csv"):
            print "exist",jpg_path
        else:
            cmd = u'"slic\SLICOSuperpixel.exe"'
            os.system(cmd+" "+ self.m + " "+self.cs+" "+jpg_path+" "+dep_path+" "+label_path+" "+ self.s_weight);

        vec_data_path = jpg_path+"_m"+self.m+"c"+self.cs+".csv"
        sp_map_path = jpg_path+"_m"+self.m+"spmap.csv"

      
        
        data = CSVReader.read_csv_as_float(vec_data_path)
        sp_map =  CSVReader.read_csv_as_int(sp_map_path)
        test_data = []
        label_col = len(data[0]) -1
        for line in data:
            test_data += [ line[0:label_col] ]

        print "test data Complete"

        output = []
        output = self.forest.predict( test_data )

        self.output_file(output, sp_map  , num)
        
        img_a = cv2.imread(label_path,0)
        img_b = cv2.imread('output_data'+str(num)+'.png',0)
        return Assessment.AssessmentByMIOU(img_a,img_b)

    def output_file(self, output, sp_map , num):
        if not os.path.exists('./output'):
            os.makedirs('./output')
        f = open('./output/outputlabel_data'+str(num)+'.csv', 'w')
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow(output)
        f.close()

     
        
        size =  [len(sp_map), len(sp_map[0])]  
        dstimg = np.zeros((size[0],size[1],3), np.uint8)
        count = 0
        for y in range(size[0]):
            for x in range(size[1]):
                
                dstimg.itemset((y,x,0),output[sp_map[y][x]])
                dstimg.itemset((y,x,1),output[sp_map[y][x]])
                dstimg.itemset((y,x,2),output[sp_map[y][x]])
                count+=1
        
        cv2.imwrite('output_data'+str(num)+'.png',dstimg)
       





    #??????forest?͕ʂɍ?????forest
    #
    # forest =  Forest( start_num = 0, end_num = 3, cut_size = 3)
    # forest2 =  Forest( 4, 6, 3)
    # forest.combine(forest2.forest)
    #
    #?̂悤?ɂ??Ďg??
    def combine(self, forest):
        self.forest.estimators_ += forest.estimators_
        self.forest.n_estimators = len(self.forest.estimators_)
        
        # Save 
        self.forest_path += 'Combined'
        print "save to", self.forest_path
        if not os.path.exists(self.forest_path):
            os.makedirs(self.forest_path)
        joblib.dump(self.forest, self.forest_path+'/forest.bin')

        return self.forest


    # forest???̊e???f?[?^?m?F?p
    def show_detail(self):
        print "n_estimators",self.forest.n_estimators
        print "criterion",self.forest.criterion
        print "max_depth",self.forest.max_depth
        print "class_", self.forest.classes_
        print "class_weight", self.forest.class_weight
        print "n_classes_", self.forest.n_classes_
        print "n_features",self.forest.n_features_
        print "n_outputs_",self.forest.n_outputs_
        print "feature_importances_",self.forest.feature_importances_
       # print "oob_score_",self.forest.oob_score_
       # print "oob_decision_function_",self.forest.oob_decision_function_

    