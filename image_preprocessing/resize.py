import os
import cv2

src_directory = "E:/Docs/Downloads/datasets/small/train/train_painted/"
target_directory = "E:/Docs/Downloads/datasets/small/train/probe/"
gray = True
target_shape = (192,64)

print('starting reading files')
for filename in os.listdir(src_directory):
    if filename.endswith(".bmp"):
         src = os.path.join(src_directory, filename)
         dst = os.path.join(target_directory, filename)
         #print(filename)
         if(gray):
             img = cv2.imread(src, 0)
         else:
             img = cv2.imread(src)
         
         resized = cv2.resize(img, target_shape, interpolation = cv2.INTER_AREA)   
         cv2.imwrite(dst, resized)
         print('resized: ' + dst)

    else:
        continue
