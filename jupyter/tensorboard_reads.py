# Load the TensorBoard notebook extension
%load_ext tensorboard
#################################################################################################################
import tensorflow as tf
import datetime, os
import os
from google.colab import drive

ROOT = "/content/drive"
PROJ = "My Drive/DriveUploader/data"

drive.mount(ROOT)
#################################################################################################################
%tensorboard --logdir "/content/drive/My Drive/DriveUploader/big_data/debug/tensorlogs_auto"
#################################################################################################################
!find ./tensorlogs_junk | grep tfevents
#################################################################################################################
%tensorboard --inspect --logdir "/content/drive/My Drive/DriveUploader/big_data/debug/tensorlogs_auto"