
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


# In[3]:

def vgg16_model(input_shape):
    vgg16 = VGG16(include_top = False, weights = 'imagenet', input_shape = input_shape)

    for layer in vgg16.layers:
        layer.trainable = False

    last = vgg16.output
    
    #Please modify this part to fill your own fully connected layers.
    x = Flatten()(last)
    #x = Dense(256, activation = 'relu')(x)
    #x = Dropout(0.5)(x)
    x = Dense(256, activation = 'relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(4, activation = 'softmax')(x)
    
    model = Model(inputs=vgg16.input, outputs=x)
    
    return model


# In[4]:

model_vgg16 = vgg16_model()
model_vgg16.compile(loss='categorical_crossentropy', optimizer = Adam(0.0001), metrics = ['accuracy'])
history = model_vgg16.fit(train_X, train_y, validation_data=(test_X, test_y), epochs = 20, batch_size = 64)
# Final evaluation of the model
scores = model_vgg16.evaluate(test_X, test_y, verbose = True)
print("VGG-16 Pretrained Model Error: %.2f%%" % (100 - scores[1] * 100))


# In[ ]:



