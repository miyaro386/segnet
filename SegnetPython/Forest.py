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

class MLRF:
    start_num = 0
    end_num = 0
    forest_path = ""
    #warm_start こいつがすべてのバグの元凶
    forest = RandomForestClassifier()
    lrforests = []
    m = "3000"
    s_weight = "0"
    

    def __init__(self,start_num, end_num):
        self.start_num = start_num
        self.end_num = end_num +1
        self.forest_path = './supervised/MLRF_data'+str(start_num)+'-'+str(end_num)
        self.forest = self.laod_forest()

    #ロードはうまく行く
    def laod_forest(self):
        if os.path.exists(self.forest_path+'/forest.bin'):
            for i in range(38):
                self.lrforests += [ LRF(self.start_num, self.end_num-1, i) ]
            return joblib.load(self.forest_path+'/forest.bin')
        else :
            print "MLRF does not exist"
            print "making new MLRF"
            #こいつでフォレストを返すと全てが狂う
            return self.create_forest()

    def create_forest(self):
        
        forest = RandomForestClassifier()

        for i in range(38):
            self.lrforests += [ LRF(self.start_num, self.end_num-1, i) ]
       
        sp_vec_path_list = CSVReader.get_path_list2("./output/csvpath/spdata_path_m"+self.m+".csv",0)
        sp_neighbors_list = CSVReader.get_path_list2("./output/csvpath/spdata_path_m"+self.m+".csv",2)
        if (len(sp_vec_path_list)<self.end_num):
            print "end_num is larger than data length"
            return -1


        trainingdata = []
        traininglabel = []
        
        print "making training data"
        for i in range(self.end_num - self.start_num):
            print "inputting",i,sp_vec_path_list[i]
            data = CSVReader.read_csv_as_float(sp_vec_path_list[i])
            label_col = len(data[0])-1
            for line in data:
                prob = []
                for lrf in self.lrforests:
                    prob.extend( lrf.get_prob (line[0:label_col]) )
                #print "prob",prob
                trainingdata += [ prob ]
                traininglabel+= [ line[label_col] ]

        print "training data Complete"
 

        forest.fit(trainingdata , traininglabel )
        
        # Save 
        print "save to", self.forest_path
        if not os.path.exists(self.forest_path):
            os.makedirs(self.forest_path)
        joblib.dump(forest, self.forest_path+'/forest.bin')
    
        return forest

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

        data = CSVReader.read_csv_as_float(vec_data_path)
        sp_map =  CSVReader.read_csv_as_int(sp_map_path)
        test_data = []


        label_col = len(data[0])-1
        print "making test data"
        print "inputting",vec_data_path
        data = CSVReader.read_csv_as_float(vec_data_path)
        for line in data:
            prob = []
            for forest in self.lrforests:
                prob.extend( forest.get_prob (line[0:label_col]) )
            test_data += [ prob ]
        print "test data Complete"

        output = self.forest.predict( test_data )
        self.output_file(output, sp_map, num)

        img_a = cv2.imread(label_path,0)
        img_b = cv2.imread('output_MLRFdata'+str(num)+'.png',0)
        return Assessment.AssessmentByMIOU(img_a,img_b)




    def output_file(self, output, sp_map , num):
        if not os.path.exists('./output'):
            os.makedirs('./output')
        f = open('./output/outputMLRFlabel_data'+str(num)+'.csv', 'w')
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
        
        cv2.imwrite('output_MLRFdata'+str(num)+'.png',dstimg)


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

class CLRF:
    start_num = 0
    end_num = 0
    forest_path = ""
    #warm_start こいつがすべてのバグの元凶
    forest = RandomForestClassifier()
    lrforests = []
    m = "1000"
    s_weight = "0"
    

    def __init__(self,start_num, end_num):
        self.start_num = start_num
        self.end_num = end_num +1
        self.forest_path = './supervised/CLRF_data'+str(start_num)+'-'+str(end_num)
        self.forest = self.laod_forest()

    #ロードはうまく行く
    def laod_forest(self):
        if os.path.exists(self.forest_path+'/forest.bin'):
            for i in range(38):
                self.lrforests += [ LRF(self.start_num, self.end_num-1, i) ]
            return joblib.load(self.forest_path+'/forest.bin')
        else :
            print "CLRF does not exist"
            print "making new CLRF"
            #こいつでフォレストを返すと全てが狂う
            return self.create_forest()

    def create_forest(self):
        
        forest = RandomForestClassifier()

        for i in range(38):
            self.lrforests += [ LRF(self.start_num, self.end_num-1, i) ]
       
        sp_vec_path_list = CSVReader.get_path_list2("./output/csvpath/spdata_path_m"+self.m+".csv",0)
        sp_neighbors_list = CSVReader.get_path_list2("./output/csvpath/spdata_path_m"+self.m+".csv",2)
        if (len(sp_vec_path_list)<self.end_num):
            print "end_num is larger than data length"
            return -1


        trainingdata = []
        traininglabel = []
        
        print "making training data"
        for i in range(self.end_num - self.start_num):
            
            print "inputting",i,sp_vec_path_list[i]
            data = CSVReader.read_csv_as_float(sp_vec_path_list[i])
            neighbors = CSVReader.read_csv_as_int(sp_neighbors_list[i])
            probs = self.load_probs(data)
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
            os.remove('./output/probs.csv')

        print "training data Complete"
 

        forest.fit(trainingdata , traininglabel )
        
        # Save 
        print "save to", self.forest_path
        if not os.path.exists(self.forest_path):
            os.makedirs(self.forest_path)
        joblib.dump(forest, self.forest_path+'/forest.bin')
    
        return forest

    def load_probs(self, data):
        if os.path.exists('./output/probs.csv'):
            return CSVReader.read_csv_as_float('./output/probs.csv')
        else :
            output = []
            label_col = len(data[0])-1
            for line in data:
                probs = []
                for forest in self.lrforests:
                    probs.extend( forest.get_prob (line[0:label_col]) )
                output += [probs]

            if not os.path.exists('./output'):
                os.makedirs('./output')
            f = open('./output/probs.csv', 'w')
            writer = csv.writer(f, lineterminator='\n')
            for line in output:
                writer.writerow(line)
            f.close()
            return CSVReader.read_csv_as_float('./output/probs.csv')

    def load_inter_probs(self, data, neighbors):
        if os.path.exists('./output/probs.csv'):
            return CSVReader.read_csv_as_float('./output/probs.csv')
        else :
            output = []
            probs = self.load_probs(data)
            label_col = len(data[0])-1
            for j in range(len(data)):
                new_prob = []
                vector = []
                vector.extend( probs[j] )
                vector.extend( probs[neighbors[j][0]] )
                vector.extend( probs[neighbors[j][1]] )
                vector.extend( probs[neighbors[j][2]] )
                vector.extend( probs[neighbors[j][3]] )
                for forest in self.lrforests:
                    new_prob.extend( forest.get_prob (vector) )
                output += [new_prob]

            os.remove('./output/probs.csv')
            if not os.path.exists('./output'):
                os.makedirs('./output')
            f = open('./output/inter_probs.csv', 'w')
            writer = csv.writer(f, lineterminator='\n')
            for line in output:
                writer.writerow(line)
            f.close()
            return CSVReader.read_csv_as_float('./output/layer_probs.csv')



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

        probs = self.load_probs(data)
        label_col = len(data[0])-1
        for j in range(len(data)):
            vector = []
            vector.extend( probs[j] )
            vector.extend( probs[neighbors[j][0]] )
            vector.extend( probs[neighbors[j][1]] )
            vector.extend( probs[neighbors[j][2]] )
            vector.extend( probs[neighbors[j][3]] )
            test_data += [ vector ]
        os.remove('./output/probs.csv')

        print "test data Complete"

        output = self.forest.predict( test_data )
        self.output_file(output, sp_map, num)

        img_a = cv2.imread(label_path,0)
        img_b = cv2.imread('output_CLRFdata'+str(num)+'.png',0)
        return [label_path ,Assessment.AssessmentByMIOU(img_a,img_b)]




    def output_file(self, output, sp_map , num):
        if not os.path.exists('./output'):
            os.makedirs('./output')
        f = open('./output/outputCLRFlabel_data'+str(num)+'.csv', 'w')
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
        
        cv2.imwrite('output_CLRFdata'+str(num)+'.png',dstimg)


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

class ILRF:
    start_num = 0
    end_num = 0
    label = 0
    layer = 0
    forest_path = ""
    #warm_start こいつがすべてのバグの元凶
    forest = RandomForestClassifier()
    lrforests = []
    m = "1000"
    s_weight = "0"
    

    def __init__(self,start_num, end_num, label, layer):
        self.start_num = start_num
        self.end_num = end_num +1
        self.label = label
        self.layer = layer
        self.forest_path = './supervised/layer'+str(layer)+'/label'+str(label)+'/data'+str(start_num)+'-'+str(end_num)
        self.forest = self.laod_forest()

    #ロードはうまく行く
    def laod_forest(self):
        if os.path.exists(self.forest_path+'/forest.bin'):
            for i in range(38):
                self.lrforests += [ LRF(self.start_num, self.end_num-1, i) ]
            return joblib.load(self.forest_path+'/forest.bin')
        else :
            print "ILRF does not exist"
            print "making new ILRF"
            #こいつでフォレストを返すと全てが狂う
            return self.create_forest()

    def create_forest(self):
        
        forest = RandomForestRegressor()

        for i in range(38):
            self.lrforests += [ LRF(self.start_num, self.end_num-1, i) ]
       
        sp_vec_path_list = CSVReader.get_path_list2("./output/csvpath/spdata_path_m"+self.m+".csv",0)
        if (len(sp_vec_path_list)<self.end_num):
            print "end_num is larger than data length"
            return -1


        trainingdata = []
        traininglabel = []
        
        print "making training data"
        for i in range(self.end_num - self.start_num):
            
            print "inputting",i,sp_vec_path_list[i]
            data = CSVReader.read_csv_as_float(sp_vec_path_list[i])
            neighbors = CSVReader.read_csv_as_int(sp_neighbors_list[i])
            probs = self.load_probs(data)
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
                if line[label_col] == self.label:
                    traininglabel +=[1]
                else:
                    traininglabel +=[0]

        print "training data Complete"
 

        self.forest.fit(trainingdata , traininglabel )
        
        # Save 
        print "save to", self.forest_path
        if not os.path.exists(self.forest_path):
            os.makedirs(self.forest_path)
        joblib.dump(self.forest, self.forest_path+'/forest.bin')
    
        return self.forest


    def load_probs(self, data):
        if os.path.exists('./output/probs.csv'):
            return CSVReader.read_csv_as_float('./output/probs.csv')
        else :
            output = []
            label_col = len(data[0])-1
            for line in data:
                probs = []
                #print line[0:label_col]
                for forest in self.lrforests:
                    probs.extend( forest.get_prob (line[0:label_col]) )
                output += [probs]
                #print "probs",probs
            if not os.path.exists('./output'):
                os.makedirs('./output')
            f = open('./output/probs.csv', 'w')
            writer = csv.writer(f, lineterminator='\n')
            for line in output:
                writer.writerow(line)
            f.close()
            return CSVReader.read_csv_as_float('./output/probs.csv')

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

        data = CSVReader.read_csv_as_float(vec_data_path)
        sp_map =  CSVReader.read_csv_as_int(sp_map_path)
        test_data = []


        label_col = len(data[0])-1
        print "making test data"
        print "inputting",vec_data_path
        data = CSVReader.read_csv_as_float(vec_data_path)
        for line in data:
            prob = []
            for forest in self.lrforests:
                prob.extend( forest.get_prob (line[0:label_col]) )
            test_data += [ prob ]
        print "test data Complete"

        output = self.forest.predict( test_data )
        self.output_file(output, sp_map, num)

        return output


    def output_file(self, output, sp_map , num):
        if not os.path.exists('./output'):
            os.makedirs('./output')
        f = open('./output/outputMLRFlabel_data'+str(num)+'.csv', 'w')
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
        
        cv2.imwrite('output_MLRFdata'+str(num)+'.png',dstimg)

    
    
    def get_prob(self, probs):
        prob = []
        for forest in self.lrforests:
            prob.extend( forest.get_prob ( probs ) )
        output = self.forest.predict( [ prob ] )
        return output.tolist()
     
            
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

        forest = RandomForestRegressor()
        
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
        for i in range(self.end_num - self.start_num):
            print "inputting",i,sp_vec_path_list[i]
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
        label_col = len(data[0])
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
        self.forest.estimators_ += forest.estimators_
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



    def laod_forest(self):
        if os.path.exists(self.forest_path+'/forest.bin'):
            return joblib.load(self.forest_path+'/forest.bin')
        else :
            print "forest does not exist"
            print "making new forest"
            return self.create_forest()


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
        

        return output[0]
        
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
        count = size[0] * size[1] - size[0] * self.cut_size * 2 - (size[1] - 4 ) * self.cut_size * 2
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
                count+=1
        
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

    