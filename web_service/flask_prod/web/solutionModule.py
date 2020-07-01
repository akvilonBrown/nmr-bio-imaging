import sys
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
import numpy as np
import os
import glob
#import skimage.io as io
import skimage.transform as trans
import cv2
from keras.models import load_model

BATCH_SIZE = 2
NUM_CLASSES=5

background = 0
class1 = 105 #77
class2 = 176  #129
class3 = 222 #177
class4 = 255

# sometimes ImageGeneratro converts colored mask into grayscale pixels differently - for example 105 or 106, or 222/223
class1ext = 106
class3ext = 223

INPUT_SIZE = (160,480)
# Grayscale dictionary to convert mask image into one-hot encoded 5-class label (and back if required).
# Colored mask becomes graysclale when opened in respective mode
GRAY_DICT = np.array([background, class1, class2, class3, class4, class1ext, class3ext])

def testGenerator(test_path,num_image = 30, resize = False, target_size = INPUT_SIZE):
    files = os.listdir(test_path)
    files.sort()
    for i,  filename in enumerate (files):
        img = cv2.imread(os.path.join(test_path, filename), 0)    
        if(resize):
            img = cv2.resize(img, (target_size[1], target_size[0]), interpolation = cv2.INTER_NEAREST)  # cv image size format (width, heigth) so I need to swap  
        img = img / 255
        img = np.reshape(img,img.shape+(1,))
        img = np.reshape(img,(1,)+img.shape)
        yield img


def saveResultColored(save_dir, results, num_class = 5):
    background = np.array([0,0,0])
    class1 = np.array([0,102,153])
    class2 = np.array([0,170,255])
    class3 = np.array([0,255,244])
    class4 = np.array([255,255,255])
    COLOR_DICT = np.array([background, class1, class2, class3, class4])

    rshape = results.shape

    batch = np.reshape(results, (rshape[0] * rshape[1] * rshape[2], num_class))
    new_batch = np.zeros((batch.shape[0], 3))

    for i in range (batch.shape[0]):
        new_batch[i] = COLOR_DICT[np.argmax(batch[i, :])]

    new_batch = np.reshape(new_batch, (rshape[0], rshape[1], rshape[2], 3), 'C');

    for i in range(new_batch.shape[0]):
      img = new_batch[i]
      fl = os.path.join(save_dir, str(i)+".bmp")
      cv2.imwrite(fl, img)
    print(f"Results saved to folder {save_dir}")


def runModel(source_folder, destination_folder):
    test_size = len(os.listdir(source_folder))    
    #test_size = 10
    testGene = testGenerator(source_folder, num_image = test_size, resize=True, target_size = INPUT_SIZE)
    model = load_model('./model_full_saved.hd')
    results = model.predict_generator(testGene, test_size, verbose=1)
    saveResultColored(destination_folder, results)
