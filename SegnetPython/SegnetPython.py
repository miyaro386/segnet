# coding=utf_8

import cv2
import numpy as np
from numba import autojit
import csv
import ImgToVec
import CSVReader
import os
from sklearn.ensemble import RandomForestClassifier
from Forest import LRF
from Forest import MLRF
import Converter
from SPRF import SPRF



#ImgToVec.sp_data_path("3000","4","0")

m = "3000"
cs = "4"
weight ="0.0"

#ImgToVec.convert_sp_to_rgbdrate(m,cs,weight)

#for i in [ "3000","5000","10000"]:
#    print i
#    ImgToVec.convert_sp_to_rgbdrate(i,cs,weight)


#ImgToVec.convert_sp_to_rgbdrate(m)
#ImgToVec.sp_data_path(m)



#for i in range(38):
#    forest1 = LRF(0,500,i)
#    forest2 = LRF(500,1000,i)
#    forest3 = LRF(1000,1500,i)
#    forest1.combine(forest2.lrf)
#    forest1.combine(forest3.lrf)
#    forest1.save_combined_forest(0,1500)

#forest = MLRF(1000,1000)

#forest.predict(100)
forest1 = LRF(1000,1003,17)
forest2 = LRF(1000,1003,6)
#forest1.show_detail()
#forest2.show_detail()
#forest.data_statistics();
#forest.create_forest()
#forest.show_detail()
#forest.predict(100)


'''
sprf = SPRF(0,2000,3000,4,0)
result = []
for i in range(10):
    print i
    result += [ sprf.predict(i) ]

if not os.path.exists('./output'):
    os.makedirs('./output')
f = open('./output/outputMIOU.csv', 'w')
writer = csv.writer(f, lineterminator='\n')
writer.writerow(result)
f.close()


'''


'''

data = CSVReader.read_csv_as_float("0000133.jpg_m3000c4.csv")
sp_map =  CSVReader.read_csv_as_int("0000133.jpg_m3000spmap.csv")
test_data = []
label = []
for line in data:
    test_data += [ line[0:256/int(4)*4] ]
    label += [int(line[256/int(4)*4])]

print label
        
size =  [len(sp_map), len(sp_map[0])]  
dstimg = np.zeros((size[0],size[1],3), np.uint8)
count = 0
for y in range(size[0]):
    for x in range(size[1]):
        #print label[sp_map[y][x]]
        dstimg.itemset((y,x,0),label[sp_map[y][x]])
        dstimg.itemset((y,x,1),label[sp_map[y][x]])
        dstimg.itemset((y,x,2),label[sp_map[y][x]])
        count+=1



cv2.imwrite('test.png',dstimg)
        

Converter.convert_label_to_color ("test.png")


s_weight = str(int( float(weight)  * 10))
model = RandomForestClassifier()
ROOT_FILE_PATH = "C:\segnet\DataSet"


cmd = u'"slic\SLICOSuperpixel.exe"'
#os.system(cmd+" 300 img.jpg dep.png label.png");

src_jpg_path = CSVReader.get_path_list2('SUNRGBDMeta_reduced.csv',4)
src_dep_path = CSVReader.get_path_list2('SUNRGBDMeta_reduced.csv',3)
src_label_path = CSVReader.get_path_list2('SUNRGBDMeta_reduced.csv',9)

jpg_path = []
dep_path = []
label_path = []

for i in range(5000):

    jpg_path.append(src_jpg_path[2*i-1])
    dep_path.append(src_dep_path[2*i-1])
    label_path.append(src_label_path[2*i-1])


    
print "making training data"
for i in range(5000):
    if os.path.exists(jpg_path[i]+"_m"+m+"cs"+cs+"w"+s_weight+".csv"):
        print "pass",jpg_path[i]
    else:
        print "inputting",i,jpg_path[i]
        os.system(cmd+" "+ m + " "+cs+" "+jpg_path[i]+" "+dep_path[i]+" "+label_path[i]+" "+ weight);
print "training data Complete"

'''

'''
labelmap = "img.jpg_m300spmap.csv"
data= CSVReader.read_csv_as_int(labelmap)

size = [ len (data) , len(data[0] ) ]

if not os.path.exists('./output'):
    os.makedirs('./output')

print "size0",size[0]
print "size1",size[1]
dstimg = np.zeros((size[0],size[1],3), np.uint8)
   
for y in range(size[0]):
    for x in range(size[1]):
                
        dstimg.itemset((y,x,0),int(data[y][x])*2)
        dstimg.itemset((y,x,1),int(data[y][x])*2)
        dstimg.itemset((y,x,2),int(data[y][x])*2)
        
cv2.imwrite('output_data.png',dstimg)

'''


'''
start_num = 0
end_num = 3
cut_size = 3

forest =  Forest( start_num = 0, end_num = 1, cut_size = 5)
forest.show_detail();
forest.predict(10)
'''

