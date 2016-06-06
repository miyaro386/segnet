
# coding=utf_8

import cv2
import numpy as np
from numba import autojit
import csv
import ImgToVec
import CSVReader
import os
from sklearn.ensemble import RandomForestClassifier
from InterLabelRandomForest import ILRF
from Forest import MLRF
from Forest import CLRF
import Converter
from SPRF import SPRF
from MultiLayerForest import MLF
from LabelRandomForest import LRF





#ImgToVec.sp_data_path("3000","4","0")

result = []

#forest = MLF(0,2500,0)
#for i in range(5):
#    result += [forest.predict(2000+i)]
#if not os.path.exists('./output'):
#    os.makedirs('./output')
#f = open('./output/outputMIOU_MLF0-2500Layer'+str(0)+'data2000-2004.csv', 'w')
#writer = csv.writer(f, lineterminator='\n')
#for line in result:
#    writer.writerow(line)
#f.close()

#for label in range(38):
#    forest = LRF(0,500,label)
#    for j in [500,1000,1500,2000]:
#        forest.combine( LRF(j,j+500,label) )
#    forest.save_combined_forest(0,2500)

#for layer_num in range(1):
#    forest = MLF(0,500,layer_num)
#    for j in [500,1000,1500,2000]:
#        forest.combine( MLF(j,j+500,layer_num) )
#    forest.save_combined_forest(0,2500)

#for i in range(38):
#    forest = ILRF(0,0,0,i)
#    forest.forest = forest.create_forest()
#    tmp_forest = ILRF(1,1,0,i)
#    tmp_forest.forest = tmp_forest.create_forest()
#    forest.combine(tmp_forest )
#    forest.save_combined_forest(0,1)

#for layer_num in range(10):
#    result = []
#    forest = MLF(0,1000,layer_num)
#    for i in [1000,2000,3000,4000]:
#        forest.combine( MLF(i,i+1000,layer_num) )
#    forest.save_combined_forest(0,5000)

#    for i in range(100):
#        result += [forest.predict(i)]

#    if not os.path.exists('./output'):
#        os.makedirs('./output')
#    f = open('./output/outputMIOU_MLF0-5000Layer'+str(layer_num)+'data0-99.csv', 'w')
#    writer = csv.writer(f, lineterminator='\n')
#    for line in result:
#        writer.writerow(result)

start_num = 2000
end_num = start_num + 500
for layer_num in range(50):
    result = []
    forest = MLF(start_num,end_num,layer_num)

    #for i in range(10):
    #    result += [forest.predict(2000+i)]
    #if not os.path.exists('./output'):
    #    os.makedirs('./output')
    #f = open('./output/outputMIOU_MLF'+str(start_num)+'-'+str(end_num)+'Layer'+str(layer_num)+'data'+str(start_num)+'+10.csv', 'w')
    #writer = csv.writer(f, lineterminator='\n')
    #for line in result:
    #    writer.writerow(line)



#for i in [ "3000","5000","10000"]:
#    print i
#    ImgToVec.convert_sp_to_rgbdrate(i,cs,weight)


#ImgToVec.convert_sp_to_rgbdrate(m)
#ImgToVec.sp_data_path(m)

#data_num = 0
#start_num = 0
#end_num = 2500
#for layer_num in range(50):
#    forest = MLF(start_num,end_num,layer_num)
#    result += [forest.predict(0)] 
#    if not os.path.exists('./output'):
#        os.makedirs('./output')
#    f = open('./output/outputMIOU_MLF'+str(start_num)+'-'+str(end_num)+'l'+str(layer_num)+'data'+str(data_num)+'.csv', 'w')
#    writer = csv.writer(f, lineterminator='\n')
#    for line in result:
#        writer.writerow(result)

#    f.close()


#start_num = 0
#end_num = 1
#layer_num = 0
#data_num = 10


#for i in range(38):
#    ilrf = ILRF(0,500,0,i)
#    for j in [500,1000,1500]:
#        ilrf.combine(ILRF(j,j+500,0,i))

#    ilrf.save_combined_forest(0,2000)
 





#start_num = 0
#end_num = 2000
#layer_num = 1
#data_num = 10
#forest = MLF(start_num,end_num,layer_num)
#result += [forest.predict(data_num)] 





#labelmap = "img.jpg_m300spmap.csv"
#data= CSVReader.read_csv_as_int(labelmap)
#size = [ len (data) , len(data[0] ) ]
#if not os.path.exists('./output'):
#    os.makedirs('./output')
#print "size0",size[0]
#print "size1",size[1]
#dstimg = np.zeros((size[0],size[1],3), np.uint8)
   
#for y in range(size[0]):
#    for x in range(size[1]):
                
#        dstimg.itemset((y,x,0),int(data[y][x])*2)
#        dstimg.itemset((y,x,1),int(data[y][x])*2)
#        dstimg.itemset((y,x,2),int(data[y][x])*2)
        
#cv2.imwrite('output_data.png',dstimg)






