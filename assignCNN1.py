import math 
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import scipy 
import h5py

def load_dataset(path_to_train):
    train_dataset = h5py.File(path_to_train)
    train_x = np.array(train_dataset['train_set_x'][:])
    train_y = np.array(train_dataset['train_set_y'][:])



    # y reshaped
    train_y = train_y.reshape((1, train_x.shape[0]))

    return train_x, train_y

train_path = "images/train_happy.h5"

train_x, train_y = load_dataset(train_path)

cv2.imshow("new_image", train_x[0])


cv2.waitKey(5000)
cv2.destroyAllWindows()

