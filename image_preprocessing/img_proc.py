# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 13:38:55 2020

@author: Yaroslav
"""
import cv2
import numpy as np
import os

def grayscale(file_location):
    img = cv2.imread(path, 0)
    cv2.imwrite(path, img)

def gray_anotate(file_location):
    img = cv2.imread(path, 0)
    img = np.where(img==77, 1, img)
    img = np.where(img==129, 2, img)
    img = np.where(img==177, 3, img)
    img = np.where(img==255, 4, img)
    print(np.unique(img))
    cv2.imwrite(path, img)
    
def decode_gray(file_location):
    img = cv2.imread(path, 0)
    img = np.where(img==1, 77, img)
    img = np.where(img==2, 129, img)
    img = np.where(img==3, 177, img)
    img = np.where(img==4, 255, img)
    print(np.unique(img))
    cv2.imwrite(path, img)    

directory = "E:/Projects/NMR/drive-download-20200325T173948Z-001/train_annotations_decoded/"

for filename in os.listdir(directory):
    if filename.endswith(".bmp"):
         path = os.path.join(directory, filename)
         print(path)
         #gray_anotate(path)
         decode_gray(path)
         
        #continue
    else:
        continue


