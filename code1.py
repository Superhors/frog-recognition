
# coding: utf-8

# In[1]:

import os
import random
import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
import tensorflow as tf
from tensorflow.python.framework import ops
import math

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten, Input
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.applications.vgg16 import preprocess_input
from keras.optimizers import RMSprop, Adam
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit


# In[ ]:

# In[2]:

train_images = ["./input/" + i for i in os.listdir("./input") if i != '.DS_Store']
random.shuffle(train_images)


# In[3]:

data = np.ndarray((124, 64,64,3), dtype=np.uint8)
for j,i in enumerate(train_images):
    img = cv2.imread(i)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_CUBIC)
    data[j] = img


# In[4]:

labels = []
for i in train_images:
    labels.append(i.split("/")[2].split(".")[0].split("_")[0])


# # 将标签one-hot及数据归一化处理

# In[27]:


label = LabelEncoder().fit_transform(labels).reshape(-1,1)


# In[6]:

y = OneHotEncoder().fit_transform(label).toarray()


# In[7]:

x = data /255


# In[8]:

X_train, X_test, Y_train, Y_test = train_test_split(x,y, test_size=0.3, random_state=0)

