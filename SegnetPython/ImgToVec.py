import cv2
import numpy as np
import collections
import CSVReader

import csv
import os

def GetRGBDVecList(jpg_img, depth_img, label_img, cut_size):

    if ( cut_size % 2 ) == 0 :
        print "Error : cut size needs to be odd number"
        return 0;
    
    side = (cut_size - 1 ) / 2
    height = jpg_img.shape[0]
    width = jpg_img.shape[1]
    vec_list = []
    label_list = []
    count = 0
    for y in range(height):
        for x in range(width) :
            
            if y-side < 0 or height <= y+side or x-side < 0 or width <= x+side:
                
                mat = np.zeros((cut_size,cut_size),dtype = np.int )
                vec_list.append( np.hstack (np.hstack( mat.tolist() ).tolist() +
                                            np.hstack( mat.tolist() ).tolist() +
                                            np.hstack( mat.tolist() ).tolist() +
                                            np.hstack( mat.tolist() ).tolist()
                                            ).tolist()
                                )
                
                label_list.append(label_img[y,x])
            else :
                mat = jpg_img[y-side:y+side+1,x-side:x+side+1]
                matDepth = depth_img[y-side:y+side+1,x-side:x+side+1]
                
                print "mat"
                tmp = 0
                for arr in mat:
                    print tmp,np.array( mat )                
                    tmp += 1
                #print "+matDepth",matDepth 
                #print "np.hstack ( np.hstack( mat.tolist() ) ).tolist() ",np.hstack ( np.hstack( mat.tolist() ) ).tolist() 
                #print "np.hstack( matDepth.tolist() )",np.hstack( matDepth.tolist() )
                
                vec_list.append( ( np.r_[np.hstack ( np.hstack( mat.tolist() ) ),  np.hstack( matDepth.tolist() )]  ).tolist()  )
                
                #print x,y,vec_list[vec_list]
                #vec_list += ( np.hstack( matDepth.tolist() ) )
        
                     
                label_list.append(label_img[y,x])
        #print(vec_list)
    result = collections.namedtuple('result', 'vec_list, label_list')
    return result(vec_list=vec_list, label_list=label_list)
    
def get_vec_list(jpg_img, label_img, cut_size):

    if ( cut_size % 2 ) == 0 :
        print "Error : cut size needs to be odd number"
        return 0;
    
    side = (cut_size - 1 ) / 2
    height = jpg_img.shape[0]
    width = jpg_img.shape[1]
    vec_list = []
    label_list = []
    count = 0
    for y in range(height):
        for x in range(width) :
            if y-side < 0 or height < y+side or x-side < 0 or width < x+side:
                '''
                mat = np.zeros((cut_size,cut_size),dtype = np.int )
                vec_list.append( np.hstack (np.hstack( mat.tolist() ).tolist() +
                                            np.hstack( mat.tolist() ).tolist() +
                                            np.hstack( mat.tolist() ).tolist() 
                                            ).tolist()
                                )
                '''

            else :
                mat = jpg_img[y-side:y+side,x-side:x+side]
                vec_list.append( np.hstack ( np.hstack( mat.tolist() ) ).tolist()  )
            
                print x,y,vec_list[count]
                count += 1
                
                label_list.append(label_img[y,x])

    result = collections.namedtuple('result', 'vec_list, label_list')
    return result(vec_list=vec_list, label_list=label_list)


def get_vec_list_fast(test_img, label_img, cut_size):

    if ( cut_size % 2 ) == 0 :
        print "Error : cut size needs to be odd number"
        return 0;
    
    side = (cut_size - 1 ) / 2
    height = test_img.shape[0]
    width = test_img.shape[1]
    vec_list = []
    label_list = []
    for y in range(height):
        for x in range(width) :
            if y-side < 0 or height < y+side or x-side < 0 or width < x+side:
                '''
                mat = np.zeros((cut_size,cut_size),dtype = np.int )
                vec_list.append( np.hstack (np.hstack( mat.tolist() ).tolist() +
                                            np.hstack( mat.tolist() ).tolist() +
                                            np.hstack( mat.tolist() ).tolist() 
                                            ).tolist()
                                )
                '''

            else :
                mat = test_img[y-side+1:y+side,x-side+1:x+side]
                vec_list.append( np.hstack ( np.hstack( mat.tolist() ) ).tolist()  )
                label_list.append(label_img[y,x])

    result = collections.namedtuple('result', 'vec_list, label_list')
    return result(vec_list=vec_list, label_list=label_list)


def convert_sp_to_rgbdrate(m):



    cmd = u'"slic\SLICOSuperpixel.exe"'
    #os.system(cmd+" 300 img.jpg dep.png label.png");

    src_jpg_path = CSVReader.get_path_list2('SUNRGBDMeta_reduced.csv',4)
    src_dep_path = CSVReader.get_path_list2('SUNRGBDMeta_reduced.csv',3)
    src_label_path = CSVReader.get_path_list2('SUNRGBDMeta_reduced.csv',9)

    jpg_path = []
    dep_path = []
    label_path = []
    line = []

    for i in range(5001):

        jpg_path.append(src_jpg_path[2*i+1])
        dep_path.append(src_dep_path[2*i+1])
        label_path.append(src_label_path[2*i+1])


    if not os.path.exists('./output/csvpath'):
        os.makedirs('./output/csvpath')
    f = open("./output/csvpath/spdata_path_m"+m+".csv", 'w')
    writer = csv.writer(f, lineterminator='\n')
    print "making training sp data"
    temp = 4000
    for i in range(temp,temp+1001):
        if os.path.exists(jpg_path[i]+"_m"+m+"spdata.csv"):
            print "pass",i,jpg_path[i]
        else:
            print "inputting",i,jpg_path[i]
            os.system(cmd+" "+ m +" "+jpg_path[i]+" "+dep_path[i]+" "+label_path[i]+" 0");
        ##print "inputting",i,jpg_path[i]
        ##os.system(cmd+" "+ m +" "+jpg_path[i]+" "+dep_path[i]+" "+label_path[i]+" 0");
        
        
    print "spdata data Complete"
    f.close()

def sp_data_path(m):
    

    src_jpg_path = CSVReader.get_path_list2('SUNRGBDMeta_reduced.csv',4)
    src_dep_path = CSVReader.get_path_list2('SUNRGBDMeta_reduced.csv',3)
    src_label_path = CSVReader.get_path_list2('SUNRGBDMeta_reduced.csv',9)

    jpg_path = []
    dep_path = []
    label_path = []
    line = []

    if not os.path.exists('./output/csvpath'):
        os.makedirs('./output/csvpath')
    f = open("./output/csvpath/spdata_path_m"+m+".csv", 'w')
    writer = csv.writer(f, lineterminator='\n')

    for i in range(5001):

        jpg_path.append(src_jpg_path[2*i+1])
        dep_path.append(src_dep_path[2*i+1])
        label_path.append(src_label_path[2*i+1])
        line = [jpg_path[i]+"_m"+m+"spdata.csv",jpg_path[i]+"_m"+m+"spmap.csv",jpg_path[i]+"_m"+m+"neighbors.csv"]
        writer.writerow(line)

def chk_spdata():
    return 0

    
