import cv2
import numpy as np
import csv
import CSVReader
import os
import collections



def to_color(luminance):
    label = CSVReader.read_csv('seg37list.csv')
    red = 0
    green = 0
    blue = 0
    if luminance<=13:
        red = 255 * luminance/13
        blue = 255 - 255 * luminance/13
        green = 0
    elif 13 < luminance <=26:
        red = 0
        blue = 255 * luminance/13
        green = 255 - 255 * luminance/13
    if 26< luminance <= 36:
        red = 255 - 255 * luminance/13
        blue = 0
        green = 255 * luminance/13

    #result = collections.namedtuple('result', 'red, blue, green')
    #return result(red=red, blue=blue,green=green)

    result = [ green, blue, red ]
    return result


def convert_label_to_color(FILE_PATH):
    test_img = cv2.imread(FILE_PATH,0)
    height = test_img.shape[0]
    width = test_img.shape[1]
    dstimg = np.zeros((height,width,3), np.uint8)
    print "start"
    for y in range(height):
        for x in range(width):
            
            color = to_color(test_img[y,x])
            dstimg[y][x][0] = color[0]
            dstimg[y][x][1] = color[1]
            dstimg[y][x][2] = color[2]

    print "finish"
    cv2.imwrite(FILE_PATH+'true12color.jpg',dstimg)



