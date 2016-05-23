
import cv2
import numpy as np
from numba import autojit
import csv
import ImgToVec
import CSVReader
import Converter
import os
from sklearn.ensemble import RandomForestClassifier
from Forest import LRF
 
lrf = LRF(0,0,1)
lrf.predict(0)


'''
#Forest使用例

model = RandomForestClassifier()
ROOT_FILE_PATH = "C:\segnet\DataSet"

data = CSVReader.read_csv_as_float("img.jpg_m300c4a.csv")
spmap= CSVReader.read_csv_as_int("img.jpg_m300spmapa.csv")

trainingvec = []
label = []
for line in data:
    trainingvec += [ line[0:256] ]
    label +=[ line[256] ]

model.fit(trainingvec, label)
output = model.predict(trainingvec)



label = []
for line in data:
    label += [int(line[256])]


size = [ len (spmap) , len(spmap[0] ) ]

if not os.path.exists('./output'):
    os.makedirs('./output')

print "size0",size[0]
print "size1",size[1]
dstimg = np.zeros((size[0],size[1],3), np.uint8)
   
for y in range(size[0]):
    for x in range(size[1]):
                
        dstimg.itemset((y,x,0),output[spmap[y][x]] ) 
        dstimg.itemset((y,x,1),output[spmap[y][x]] )
        dstimg.itemset((y,x,2),output[spmap[y][x]] )

        #dstimg.itemset((y,x,0),spmap[y][x]) 
        #dstimg.itemset((y,x,1),spmap[y][x])
        #dstimg.itemset((y,x,2),spmap[y][x])
        
cv2.imwrite('output_data.png',dstimg)

cmd = u'"color\LabelToColor.exe"'
os.system(cmd+" output_data.png");

#Converter.convert_label_to_color('output_data.png')

#cmd = u'"slic\SLICOSuperpixel.exe"'
#os.system(cmd+" 300 img.jpg dep.png label.png");
'''

'''
start_num = 0
end_num = 3
cut_size = 3

forest =  Forest( start_num = 0, end_num = 1, cut_size = 5)
forest.show_detail();
forest.predict(10)
'''

