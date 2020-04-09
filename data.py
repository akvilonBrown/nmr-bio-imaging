#Copyright (c) 2019 zhixuhao
#https://github.com/zhixuhao/unet

from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator
import numpy as np 
import os
import glob
import skimage.io as io
import skimage.transform as trans
import cv2

class0 = 77
class1 = 129
class2 = 177
class3 = 255
background = 0

GRAY_DICT = np.array([class0, class1, class2, class3, background ])


'''
'TODO: make this func more univarsal
'''
def adjustData5cl(img,mask,num_class, target_size, batch_size):    
    img = img / 255
    #mask = mask[:,:,:,0] if(len(mask.shape) == 4) else mask[:,:,0]
    #print(f"image shape {img.shape} mask shape {mask.shape}")
    #new_mask = np.zeros(mask.shape + (num_class,))
    
    class0 = np.where(mask==1, 1, 0)
    class1 = np.where(mask==2, 1, 0)
    class2 = np.where(mask==3, 1, 0)
    class3 = np.where(mask==4, 1, 0)
    class_bkg = np.where(mask==0, 1, 0)
    new_mask = np.stack((class0, class1, class2, class3, class_bkg), 3)
    new_mask = np.reshape(new_mask, (batch_size, target_size[0], target_size[1], num_class))
    mask = new_mask
    #print(mask.shape)
    #print(mask[0,100, 100])
    return (img,mask)

def trainGenerator(batch_size,train_path,image_folder,mask_folder,aug_dict,image_color_mode = "grayscale",
                    mask_color_mode = "grayscale",image_save_prefix  = "image",mask_save_prefix  = "mask",
                    flag_multi_class = False,num_class = 5,save_to_dir = None,target_size = (160,480),seed = 1):
    '''
    can generate image and mask at the same time
    use the same seed for image_datagen and mask_datagen to ensure the transformation for image and mask is the same
    if you want to visualize the results of generator, set save_to_dir = "your path"
    '''
    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)
    image_generator = image_datagen.flow_from_directory(
        train_path,
        classes = [image_folder],
        class_mode = None,
        color_mode = image_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = image_save_prefix,
        seed = seed)
    mask_generator = mask_datagen.flow_from_directory(
        train_path,
        classes = [mask_folder],
        class_mode = None,
        color_mode = mask_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = mask_save_prefix,
        seed = seed)
    train_generator = zip(image_generator, mask_generator)
    for (img,mask) in train_generator:        
        img,mask = adjustData5cl(img,mask,num_class, target_size, batch_size)
        yield (img,mask)



def testGenerator(test_path,num_image = 30, as_gray = True):
    for i in range(num_image):
        img = io.imread(os.path.join(test_path,"%d.bmp"%i),as_gray = as_gray)
        img = img / 255
        img = np.reshape(img,img.shape+(1,))
        img = np.reshape(img,(1,)+img.shape)
        yield img


def saveResult(save_dir,results, num_class = 5):
    rshape = results.shape
    batch = np.reshape(results, (rshape[0] * rshape[1] * rshape[2], num_class))
    new_batch = np.zeros(batch.shape[0])
    
    for i in range (batch.shape[0]):
        new_batch[i] = GRAY_DICT[np.argmax(batch[i, :])]
    
    new_batch = np.reshape(new_batch, (rshape[0], rshape[1], rshape[2]), 'C');	
    
    for i in range(new_batch.shape[0]):
      img = new_batch[i]
      fl = os.path.join(save_dir, str(i)+".bmp")      
      cv2.imwrite(fl, img)
    print(f"Results saved to folder {save_dir}")	