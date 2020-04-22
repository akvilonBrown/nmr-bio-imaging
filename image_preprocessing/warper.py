# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 16:42:10 2020

@author: Yaroslav
found at https://subscription.packtpub.com/book/application_development/9781785283932/1/ch01lvl1sec16/image-warping
"""
import os
import cv2
import numpy as np
import math

input_directory = "E:/Projects/NMR/grandset/warp/label"
output_directory = "E:/Projects/NMR/grandset/warp_done/label"

#cv2.imshow('Input', img)

#####################
# Both horizontal and vertical
def warp(img):
    strength = 8.0
    frequency = 150.0
    rows = img.shape[0]
    cols = img.shape[1]
    #rows, cols = img.shape  for grayscale only
    img_output = np.zeros(img.shape, dtype=img.dtype)

    for i in range(rows):
        for j in range(cols):
            offset_x = int(strength * math.sin(2 * 3.14 * i / frequency))
            offset_y = int(strength * math.cos(2 * 3.14 * j / frequency))
            if i+offset_y < rows and j+offset_x < cols:
                img_output[i,j] = img[(i+offset_y)%rows,(j+offset_x)%cols]
            else:
                img_output[i,j] = 0	
                
    return img_output            

#cv2.imshow('Multidirectional wave', img_output)
#cv2.waitKey()

for i, filename in enumerate(os.listdir(input_directory)):
    if filename.endswith(".bmp"):
         src = os.path.join(input_directory, filename)
         dst = os.path.join(output_directory, filename)
         print(src)
         img = cv2.imread(src)  #img = cv2.imread(src, 0) 0 for grayscale
         img = warp(img)
         cv2.imwrite(dst, img)
         print(dst)        
         #gray_anotate(path)
         #decode_gray(path)
         
        #continue
    else:
        continue