import cv2
import numpy as np
import collections


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
