# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 14:24:45 2020

@author: Yaroslav
"""

import cv2
import numpy as np


directory = "E:/Projects/NMR/drive-download-20200325T173948Z-001/train/"

img = cv2.imread(directory + 'flair0021.bmp' )
print(img[100, 70])
