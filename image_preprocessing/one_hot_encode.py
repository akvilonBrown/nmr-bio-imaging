# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 17:14:25 2020

@author: Yaroslav
"""

import cv2
import numpy as np


directory = "E:/Docs/Downloads/datasets/test_annotations"
num_classes = 5

img = cv2.imread("E:/Docs/Downloads/datasets/train/label/50.bmp",0)
print(img.shape)
img_out = np.zeros(img.shape + (num_classes,))
print(img_out.shape)

#class0 = np.where(img==77, 1, 0)
#class1 = np.where(img==129, 1, 0)
#class2 = np.where(img==177, 1, 0)
#class3 = np.where(img==255, 1, 0)
#class_bkg = np.where(img==0, 1, 0)

class0 = np.where(img==1, 1, 0)
class1 = np.where(img==2, 1, 0)
class2 = np.where(img==3, 1, 0)
class3 = np.where(img==4, 1, 0)
class_bkg = np.where(img==0, 1, 0)

#img_out[:, :, 0] = class0
#img_out[:, :, 1] = class1
#img_out[:, :, 2] = class2
#img_out[:, :, 3] = class3
#img_out[:, :, 4] = class_bkg

new_arr = np.stack((class0, class1, class2, class3, class_bkg), 2)

print(new_arr[10])
#print(img_out[100, 70])
#print(np.unique(img_out))
#print(new_arr[100, 70])

'''
#i=0
for filename in os.listdir(directory):
    if filename.endswith(".bmp"):
         src = os.path.join(directory, filename)
         img = cv2.imread(src,0)
         print(img.shape)
                  #os.rename(src, dst)
         
         #gray_anotate(path)
         #decode_gray(path)
         
        #continue
    else:
        continue
        
'''