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


class LRF:

    start_num = 0
    end_num = 0
    label =0
    forest_path = ""
    lrf = RandomForestRegressor(warm_start=True)

    

    def __init__(self,start_num, end_num, label):
        self.start_num = start_num
        self.end_num = end_num +1
        self.label = label
        self.forest_path = './supervised/data'+str(start_num)+'-'+str(end_num)+'lab'+str(label)
        self.lrf = self.laod_forest()

    def laod_forest(self):
        if os.path.exists(self.forest_path+'/forest.bin'):
            return joblib.load(self.forest_path+'/forest.bin')
        else :
            print "forest does not exist"
            print "making new forest"
            return self.create_forest()


    def create_forest(self):

        
        trainingdata = []
        traininglabel = []

        data = CSVReader.read_csv_as_float("img.jpg_m300c4a.csv")
        spmap= CSVReader.read_csv_as_int("img.jpg_m300spmapa.csv")


        
        label_col = len(data[0])-1
        print "making training data"
        for i in range(self.end_num - self.start_num):

            for line in data:
                trainingdata += [ line[0:label_col] ]
                if line[label_col] == self.label:
                    traininglabel +=[1]
                else:
                    traininglabel +=[0]

        print "training data Complete"
 

        self.lrf.fit(trainingdata , traininglabel )
        
        # Save 
        print "save to", self.forest_path
        if not os.path.exists(self.forest_path):
            os.makedirs(self.forest_path)
        joblib.dump(self.lrf, self.forest_path+'/forest.bin')
    
        return self.lrf

    
    def predict(self, num):


        data = CSVReader.read_csv_as_float("img.jpg_m300c4a.csv")
        spmap= CSVReader.read_csv_as_int("img.jpg_m300spmapa.csv")

        test_data= []
    
        label_col = len(data[0])

        print "making test data"
        #print "inputting",jpg_path
        
        label_col = len(data[0])-1
        for line in data:
            test_data += [ line[0:label_col] ]

       
        print "test data Complete"

        output = self.lrf.predict( test_data )
        self.output_file(output, num)

        return output

    def output_file(self, output, num):
        if not os.path.exists('./output'):
            os.makedirs('./output')
        f = open('./output/outputlabel_data'+str(num)+'.csv', 'w')
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow(output)
        f.close()

# 使い方
#1, 'from Forest import Forest' を上に追加  
#2,  forest =  Forest( start_num = 0, end_num = 3, cut_size = 3) を書く。 インスタンス化というやつ。 forestが本体
#3, 松下先生のexcelファイルのパスのリストからstart_num～end_numの間のデータセットのフォレストを作る。cut_sizeは正方形の辺の長さ
#4, 過去に同じ引数をしているなら、保存してあるforestを参照する
#5, テストデータを渡すときは 'forest.forest.predict( testdata )' という形で。
#
# forest.Combine(forest2) で一応はforestがforest2を吸収合併する。
#
#使う前に！
# CSVReader.py のファイルパスを各自の環境に合わせて変更してから使って
#

class Forest:

    start_num = 0
    end_num = 0
    cut_size = 0
    forest_path = ""
    forest = RandomForestClassifier(warm_start=True)

    

    def __init__(self,start_num, end_num, cut_size):
        self.start_num = start_num
        self.end_num = end_num +1
        self.cut_size = cut_size
        self.forest_path = './supervised/data'+str(start_num)+'-'+str(end_num)+'size'+str(cut_size)
        self.forest = self.laod_forest()
      
 

    def save_forest(self):
        if not os.path.exists(self.forest_path):
            os.makedirs(self.forest_path)
        joblib.dump(self.forest, forest_path+'/forest.bin')

    #
    # 
    #
    #

    def create_forest(self):

        print "save to", self.forest_path
        
        trainingdata = []
        traininglabel = []

        src_jpg_path = CSVReader.get_path_list2('SUNRGBDMeta_reduced.csv',4)
        src_dep_path = CSVReader.get_path_list2('SUNRGBDMeta_reduced.csv',3)
        src_label_path = CSVReader.get_path_list2('SUNRGBDMeta_reduced.csv',9)

        jpg_path = []
        dep_path = []
        label_path = []

        for i in range(self.start_num, self.end_num):
            jpg_path.append(src_jpg_path[2*i-1])
            dep_path.append(src_dep_path[2*i-1])
            label_path.append(src_label_path[2*i-1])


    
        print "making training data"
        for i in range(self.end_num - self.start_num):
            print "inputting",i,jpg_path[i]
            jpg_img = cv2.imread(jpg_path[i])
            label_img = cv2.imread(label_path[i],0)
            depth_img = cv2.imread(dep_path[i],0)
            data = ImgToVec.GetRGBDVecList( jpg_img, depth_img, label_img, self.cut_size)
            trainingdata += data.vec_list
            traininglabel += data.label_list
        print "training data Complete"
 

        self.forest.fit(trainingdata , traininglabel )
        
        # Save 
        print "save to", self.forest_path
        if not os.path.exists(self.forest_path):
            os.makedirs(self.forest_path)
        joblib.dump(self.forest, self.forest_path+'/forest.bin')
    
        return self.forest






    def predictRGBD(self, num):

        src_jpg_path = CSVReader.get_path_list2('SUNRGBDMeta_reduced.csv',4)
        src_dep_path = CSVReader.get_path_list2('SUNRGBDMeta_reduced.csv',3)
        src_label_path = CSVReader.get_path_list2('SUNRGBDMeta_reduced.csv',9)

     
        jpg_path = src_jpg_path[2*num]
        dep_path = src_dep_path[2*num]
        label_path = src_label_path[2*num]



    
        print "making test data"
        print "inputting",jpg_path
        jpg_img = cv2.imread(jpg_path)
        label_img = cv2.imread(label_path,0)
        depth_img = cv2.imread(dep_path,0)
        size = jpg_img.shape
        data = ImgToVec.GetRGBDVecList( jpg_img, depth_img, label_img, self.cut_size)
        #data = ImgToVec.get_vec_list( jpg_img,label_img, self.cut_size)
        jpg_data = data.vec_list
        print "test data Complete"

        output = self.forest.predict( jpg_data )

        self.output_file(output , size , num)
        

        return output
        
    def predict(self, num):

        src_jpg_path = CSVReader.get_path_list2('SUNRGBDMeta_reduced.csv',4)
        src_dep_path = CSVReader.get_path_list2('SUNRGBDMeta_reduced.csv',3)
        src_label_path = CSVReader.get_path_list2('SUNRGBDMeta_reduced.csv',9)

     
        jpg_path = src_jpg_path[2*num]
        dep_path = src_dep_path[2*num]
        label_path = src_label_path[2*num]



    
        print "making test data"
        print "inputting",jpg_path
        jpg_img = cv2.imread(jpg_path)
        label_img = cv2.imread(label_path,0)
        size = jpg_img.shape
        data = ImgToVec.get_vec_list( jpg_img, label_img, self.cut_size)
        jpg_data = data.vec_list
        print "test data Complete"

        output = self.forest.predict( jpg_data )

        self.output_file(output , size , num)
        

        return output

    # 出力ファイル名は各自で設定
    def output_file(self, output, size, num):
        if not os.path.exists('./output'):
            os.makedirs('./output')
        f = open('./output/outputlabel_data'+str(num)+'.csv', 'w')
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow(output)
        f.close()

        print "size0",size[0]
        print "size1",size[1]
        dstimg = np.zeros((size[0],size[1],3), np.uint8)
        '''
        for i in range(count):
            dstimg.itemset((y,x,0),output[i])
            dstimg.itemset((y,x,1),output[i])
            dstimg.itemset((y,x,2),output[i])
                
            
        
        '''
        for y in range(size[0]):
            for x in range(size[1]):
                
                dstimg.itemset((y,x,0),output[count])
                dstimg.itemset((y,x,1),output[count])
                dstimg.itemset((y,x,2),output[count])
        
        cv2.imwrite('output_data'+str(num)+'.png',dstimg)
       



    #引数のforestは別に作ったforest
    #
    # forest =  Forest( start_num = 0, end_num = 3, cut_size = 3)
    # forest2 =  Forest( 4, 6, 3)
    # forest.combine(forest2.forest)
    #
    #のようにして使う
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


    # forest内の各種データ確認用
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

    