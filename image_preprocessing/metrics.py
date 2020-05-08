import os
import cv2
import numpy as np

#pred_folder = "E:/Docs/Downloads/datasets/junk/predicted/predicted1_1"
pred_folder = "E:/Projects/NMR/grandset/big_data/test_results/FINAL/test_results_extra_training"
true_folder = "E:/Projects/NMR/grandset/big_data/test/label"
#true_folder = "E:/Docs/Downloads/datasets/junk/predicted/test_label"

target_shape = (160,480)

def to_precategorical5(mask):
    #this map is how cv2 converts colored images into grayscale pixels
    background = 0   #corresponds to color 0,0,0        black  (blue, green, red)
    class1 = 77      #corresponds to color 0,102,153   brown
    class2 = 129     #corresponds to color 0,170,255   orange
    class3 = 177     #corresponds to color 0,255,244   yellow
    class4 = 255     #corresponds to color 255,255,255  white
    
    # Grayscale dictionary to convert mask image into one-hot encoded 5-class label (and back if required).
    GRAY_DICT_CV = np.array([background, class1, class2, class3, class4])
    #num_classes = len(GRAY_DICT_CV)
    
    #mask = np.where(mask==GRAY_DICT_CV[4], 4, mask)   # backround pixels with value 0 are already treated as class0
    mask = np.where(mask==GRAY_DICT_CV[1], 1, mask) 
    mask = np.where(mask==GRAY_DICT_CV[2], 2, mask)
    mask = np.where(mask==GRAY_DICT_CV[3], 3, mask)
    mask = np.where(mask==GRAY_DICT_CV[4], 4, mask)
    
    return mask

def to_categorical_i(img, num_class=5):
    img_shape = img.shape
    encoded = np.zeros(img_shape + (num_class,))
    
    for i in range(num_class):
        encoded[:, :, i] = np.where(img==i, 1, 0)
    
    return encoded


def load_images(truth_folder, predicted_folder, input_size = target_shape, num_classes=5):
    
    batch_size= len(os.listdir(truth_folder))
    if batch_size != len(os.listdir(predicted_folder)):
      print("The number of images in truth folder differs from the number of predicted images")
      return
    files_truth = os.listdir(truth_folder)
    files_truth.sort()
    files_predicted = os.listdir(predicted_folder)
    files_predicted.sort()
    truth_hot_encoded=np.zeros((batch_size, input_size[0], input_size[1], num_classes))
    predicted_hot_encoded=np.zeros((batch_size, input_size[0], input_size[1], num_classes)) 
    print("Loading truth images")
    counter = 0;
    for i,  filename in enumerate (files_truth):
        #print(filename)
        mask = cv2.imread(os.path.join(truth_folder, filename), 0)
        if(mask.shape != input_size):
          counter+=1
          mask = cv2.resize(mask, (input_size[1], input_size[0]), interpolation = cv2.INTER_NEAREST)  # cv image size format (width, heigth) so I need to swap
        mask = to_precategorical5(mask)
        mask = to_categorical_i(mask)
        truth_hot_encoded[i] = mask  
    print(f"Resized {counter} truth images")
    counter=0     
    print("Loading predicted images")     
    for i,  filename in enumerate (files_predicted):
        mask = cv2.imread(os.path.join(predicted_folder, filename), 0)
        if(mask.shape != input_size):
          counter+=1
          #print(f"resizing predicted masks, {filename}, shape: {mask.shape}")
          mask = cv2.resize(mask, (input_size[1], input_size[0]), interpolation = cv2.INTER_NEAREST)  # cv image size format (width, heigth) so I need to swap
        mask = to_precategorical5(mask)
        mask = to_categorical_i(mask)   
        predicted_hot_encoded[i] = mask
    print(f"Resized {counter} predicted images")
    print(f"truth shape: {truth_hot_encoded.shape}, predicted shape: {predicted_hot_encoded.shape}")
    if (truth_hot_encoded.shape != predicted_hot_encoded.shape):
      print(f"Error during loading, shapes don't match, truth shape")
    return   truth_hot_encoded,   predicted_hot_encoded            

def iou_score_single (y_true, y_pred):
    num_cl = y_true.shape[-1]
    score = np.zeros(num_cl)
    for i in range(num_cl):
      intersection = np.logical_and(y_true[:,:,i], y_pred[:,:,i])
      union = np.logical_or(y_true[:,:,i], y_pred[:,:,i])
      #if(np.sum(union))==0:
        #print(f"Class{i} np.sum(union))==0")
      score[i] = np.sum(intersection) / np.sum(union)    
    return score

def dice_score_single (y_true, y_pred):
    num_cl = y_true.shape[-1]
    score = np.zeros(num_cl)
    for i in range(num_cl):
      intersection = np.logical_and(y_true[:,:,i], y_pred[:,:,i])
      score[i] = 2 * np.sum(intersection) / (np.sum(y_true[:,:,i]) + np.sum(y_pred[:,:,i]))      
    return score                
def iou_score_total (y_true, y_pred):
    num_cl = y_true.shape[-1]
    score = np.zeros(num_cl)
    for i in range(num_cl):
      intersection = np.logical_and(y_true[:,:,:,i], y_pred[:,:,:,i])
      union = np.logical_or(y_true[:,:,:,i], y_pred[:,:,:,i])
      score[i] = np.sum(intersection) / np.sum(union)
      print(f"IoU score for class{i} = {round(score[i], 4)}")
    
    print(f"Total IoU score: {round(score.mean(), 4)}")  
    return score.mean()

def dice_score_total (y_true, y_pred):
    num_cl = y_true.shape[-1]
    score = np.zeros(num_cl)
    for i in range(num_cl):
      intersection = np.logical_and(y_true[:,:,:,i], y_pred[:,:,:,i])          
      score[i] = 2 * np.sum(intersection) / (np.sum(y_true[:,:,:,i]) + np.sum(y_pred[:,:,:,i]))
      print(f"Dice score for class{i} = {round(score[i], 4)}")
    
    print(f"Total Dice score: {round(score.mean(), 4)}")
    return score.mean()


true_img, predicted_img = load_images(true_folder, pred_folder)
iou = iou_score_total(true_img, predicted_img)
dice = dice_score_total(true_img, predicted_img)

print("-------------")
print("-------------")

'''
for i in range(len(true_img)):
  sc = iou_score_single (true_img[i], predicted_img[i])
  dice = dice_score_single(true_img[i], predicted_img[i])
  print(f"Image {str(i).zfill(5)}, total IoU: {round(sc[~np.isnan(sc)].mean(), 4)}, total dice: {round(dice[~np.isnan(dice)].mean(), 4)}")
  for i in range(5):
     print(f"    Class{i} IoU score {round(sc[i], 4)}           dice score {round(dice[i], 4)}")
'''
   