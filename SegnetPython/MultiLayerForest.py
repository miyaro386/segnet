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
from InterLabelRandomForest import ILRF
from LabelRandomForest import LRF




class MLF:
    start_num = 0
    end_num = 0
    layer_num = 0
    forest_path = ""
    #warm_start こいつがすべてのバグの元凶
    forest = RandomForestClassifier()
    m = "1000"
    s_weight = "0"
    

    def __init__(self,start_num, end_num, layer_num):
        self.start_num = start_num
        self.end_num = end_num +1
        self.layer_num = layer_num
        self.forest_path = './supervised/MLF_layer'+str(layer_num)+'_data'+str(start_num)+'-'+str(end_num)
        self.forest = self.laod_forest()

    #ロードはうまく行く
    def laod_forest(self):
        if os.path.exists(self.forest_path+'/forest.bin'):
            return joblib.load(self.forest_path+'/forest.bin')
        else :
            print "MLF does not exist"
            print "making new MLF"
            return self.create_forest()

    def create_forest(self):
        
        forest = RandomForestClassifier(n_estimators = 20)


        sp_vec_path_list = CSVReader.get_path_list2("./output/csvpath/spdata_path_m"+self.m+".csv",0)
        sp_neighbors_list = CSVReader.get_path_list2("./output/csvpath/spdata_path_m"+self.m+".csv",2)
        if (len(sp_vec_path_list)<self.end_num):
            print "end_num is larger than data length"
            return -1


        trainingdata = []
        traininglabel = []
        
        print "making training data MLF"
        for i in range(self.start_num ,self.end_num):
            print "inputting",i,sp_vec_path_list[i]
            #ある画像に対するスーパーピクセルのデータ集合
            data = CSVReader.read_csv_as_float(sp_vec_path_list[i])
            #ある画像に対するスーパーピクセルの近傍データ
            neighbors = CSVReader.read_csv_as_int(sp_neighbors_list[i])
            probs = self.load_probs(sp_vec_path_list, sp_neighbors_list, self.layer_num, i)
            label_col = len(data[0])-1
            for j in range(len(data)):
                vector = []
                vector.extend( probs[j] )
                vector.extend( probs[neighbors[j][0]] )
                vector.extend( probs[neighbors[j][1]] )
                vector.extend( probs[neighbors[j][2]] )
                vector.extend( probs[neighbors[j][3]] )
                #print "training vector",len(vector)
                trainingdata += [ vector ]
                traininglabel+= [ data[j][label_col] ]


        print "training data Complete"
 

        forest.fit(trainingdata , traininglabel )
        
        # Save 
        print "save to", self.forest_path
        if not os.path.exists(self.forest_path):
            os.makedirs(self.forest_path)
        joblib.dump(forest, self.forest_path+'/forest.bin')
    
        return forest
    
    #layer_numの回数だけprobs.csvのデータの変換を繰り返す
    def load_probs(self, sp_vec_path_list, sp_neighbors_list, layer_num, img_num):
      
        #(1)spDataをLRFを用いてprobs.csvに変換
        forests = []
        for i in range(38):
            forests += [ LRF(self.start_num, self.end_num-1, i) ]

        for i in range(self.start_num, self.end_num):
            if not os.path.exists('./output/probs/base'):
                os.makedirs('./output/probs/base')
            if os.path.exists('./output/probs/base/'+str(i)+'.csv'): continue

            data = CSVReader.read_csv_as_float(sp_vec_path_list[i])
            output = []
            label_col = len(data[0])-1
            for line in data:
                probs = []
                for f in forests:
                    probs.extend( f.get_prob (line[0:label_col]) )
                output += [probs]

            
            f = open('./output/probs/base/'+str(i)+'.csv', 'w')
            writer = csv.writer(f, lineterminator='\n')
            for line in output:
                writer.writerow(line)
            f.close()
            

        #(2)probs.csvをILRFを用いてlayer_numの回数分変換
            
        for layer in range(layer_num):
            forests = []
            for label in range(38):
                print 'ILRF',label
                forests += [ ILRF(self.start_num, self.end_num-1, layer, label) ]

            for n in range(self.start_num, self.end_num):
                data_num = n 
                if os.path.exists('./output/probs/layer'+str(layer)+'/'+str(data_num)+'.csv'): continue
                if layer == 0:
                    src_probs = CSVReader.read_csv_as_float('./output/probs/base/'+str(data_num)+'.csv')
                else: 
                    src_probs = CSVReader.read_csv_as_float('./output/probs/layer'+str(layer-1)+'/'+str(data_num)+'.csv')
                dist_probs = []
                neighbors = CSVReader.read_csv_as_int(sp_neighbors_list[n])
                for i in range(len(src_probs)):
                    vector = []
                    vector.extend( src_probs[i] )
                    vector.extend( src_probs[neighbors[i][0]] )
                    vector.extend( src_probs[neighbors[i][1]] )
                    vector.extend( src_probs[neighbors[i][2]] )
                    vector.extend( src_probs[neighbors[i][3]] )
                    probs = []
                    for f in forests:
                        probs.extend( f.get_prob (vector) )
                    dist_probs += [probs]
                    
                if not os.path.exists('./output/probs/layer'+str(layer)):
                    os.makedirs('./output/probs/layer'+str(layer))
                f = open('./output/probs/layer'+str(layer)+'/'+str(data_num)+'.csv', 'w')
                writer = csv.writer(f, lineterminator='\n')
                for line in dist_probs:
                    writer.writerow(line)
                f.close()


        if layer_num == 0:
            return CSVReader.read_csv_as_float('./output/probs/base/'+str(img_num)+'.csv')
        else: 
            return CSVReader.read_csv_as_float('./output/probs/layer'+str(layer_num-1)+'/'+str(img_num)+'.csv')

   

    def predict(self, num):


        src_jpg_path = CSVReader.get_path_list2('SUNRGBDMeta_reduced.csv',4)
        src_dep_path = CSVReader.get_path_list2('SUNRGBDMeta_reduced.csv',3)
        src_label_path = CSVReader.get_path_list2('SUNRGBDMeta_reduced.csv',9)

     
        jpg_path = src_jpg_path[2*num]
        dep_path = src_dep_path[2*num]
        label_path = src_label_path[2*num]

        
        if os.path.exists(jpg_path+"_m"+self.m+"spdata.csv"):
            print "exist",jpg_path
        else:
            cmd = u'"slic\SLICOSuperpixel.exe"'
            os.system(cmd+" "+ self.m + " "+jpg_path+" "+dep_path+" "+label_path+" "+ self.s_weight);

        vec_data_path = jpg_path+"_m"+self.m+"spdata.csv"
        sp_map_path = jpg_path+"_m"+self.m+"spmap.csv"
        sp_neighbors_path = jpg_path+"_m"+self.m+"neighbors.csv"

        data = CSVReader.read_csv_as_float(vec_data_path)
        sp_map =  CSVReader.read_csv_as_int(sp_map_path)
        neighbors = CSVReader.read_csv_as_int(sp_neighbors_path)

        test_data = []

        probs = self.load_probs_for_predict(data, neighbors, self.layer_num)
        label_col = len(data[0])-1
        for j in range(len(data)):
            vector = []
            vector.extend( probs[j] )
            vector.extend( probs[neighbors[j][0]] )
            vector.extend( probs[neighbors[j][1]] )
            vector.extend( probs[neighbors[j][2]] )
            vector.extend( probs[neighbors[j][3]] )
            test_data += [ vector ]
            #print len(vector)

        print "test data Complete"
     
        #print len(test_data), len(test_data[0])
        self.show_detail()
        output = self.forest.predict( test_data )
        self.output_file(output, sp_map, num)

        img_a = cv2.imread(label_path,0)
        img_b = cv2.imread('output_MLF'+str(self.start_num)+'-'+str(self.end_num)+'_layer'+str(self.layer_num)+'data'+str(num)+'.png',0)
        return [label_path ,Assessment.AssessmentByMIOU(img_a,img_b)]


    def load_probs_for_predict(self, data, neighbors, layer_num):
      
        #(1)spDataをLRFを用いてprobs.csvに変換
            
        if not os.path.exists('./output/probs.csv'): 
            forests = []
            for i in range(38):
                forests += [ LRF(self.start_num, self.end_num-1, i) ]

            output = []
            label_col = len(data[0])-1
            for line in data:
                probs = []
                for f in forests:
                    probs.extend( f.get_prob (line[0:label_col]) )
                output += [probs]

            if not os.path.exists('./output'):
                os.makedirs('./output')
            f = open('./output/probs.csv', 'w')
            writer = csv.writer(f, lineterminator='\n')
            for line in output:
                writer.writerow(line)
            f.close()
            

        #(2)probs.csvをILRFを用いてlayer_numの回数分変換
        
        for layer in range(layer_num):
            if os.path.exists('./output/probs_layer'+str(layer)+'.csv'):continue
            forests = []
            for label in range(38):
                forests += [ ILRF(self.start_num, self.end_num-1, layer, label) ]

            if layer == 0:
                src_probs = CSVReader.read_csv_as_float('./output/probs.csv')
            else: 
                src_probs = CSVReader.read_csv_as_float('./output/probs_layer'+str(layer-1)+'.csv')
            dist_probs = []
            for i in range(len(src_probs)):
                vector = []
                vector.extend( src_probs[i] )
                vector.extend( src_probs[neighbors[i][0]] )
                vector.extend( src_probs[neighbors[i][1]] )
                vector.extend( src_probs[neighbors[i][2]] )
                vector.extend( src_probs[neighbors[i][3]] )
                probs = []
                for f in forests:
                    probs.extend( f.get_prob (vector) )
                dist_probs += [probs]
                    
            if not os.path.exists('./output'):
                os.makedirs('./output')
            f = open('./output/probs_layer'+str(layer)+'.csv', 'w')
            writer = csv.writer(f, lineterminator='\n')
            for line in dist_probs:
                writer.writerow(line)
            f.close()


        if layer_num == 0:
            return CSVReader.read_csv_as_float('./output/probs.csv')
        else: 
            return CSVReader.read_csv_as_float('./output/probs_layer'+str(layer_num-1)+'.csv')

    def output_file(self, output, sp_map , num):
        #result =[]
        #if not os.path.exists('./output'):
        #    os.makedirs('./output')
        #f = open('./output/outputmiou_mlf'+str(self.start_num)+'-'+str(self.end_num)+'l'+str(self.layer_num)+'data'+str(num)+'.csv', 'w')
        #writer = csv.writer(f, lineterminator='\n')
        #for line in result:
        #    writer.writerow(result)

        #f.close()

     
        
        size =  [len(sp_map), len(sp_map[0])]  
        dstimg = np.zeros((size[0],size[1],3), np.uint8)
        count = 0
        for y in range(size[0]):
            for x in range(size[1]):
                
                dstimg.itemset((y,x,0),output[sp_map[y][x]])
                dstimg.itemset((y,x,1),output[sp_map[y][x]])
                dstimg.itemset((y,x,2),output[sp_map[y][x]])
                count+=1
        
        cv2.imwrite('output_MLF'+str(self.start_num)+'-'+str(self.end_num)+'_layer'+str(self.layer_num)+'data'+str(num)+'.png',dstimg)


    def combine(self, forest):
        self.forest.estimators_ += forest.forest.estimators_
        self.forest.n_estimators = len(self.forest.estimators_)
        return self.forest


    def save_combined_forest(self,start_num, end_num):
        self.forest_path = './supervised/MLF_layer'+str(self.layer_num)+'_data'+str(start_num)+'-'+str(end_num)
        
        print "save to", self.forest_path
        if not os.path.exists(self.forest_path):
            os.makedirs(self.forest_path)
        joblib.dump(self.forest, self.forest_path+'/forest.bin')


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
