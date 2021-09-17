#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# Imports
import numpy as np 
import pandas as pd 
import warnings 
from glob import glob 
from skimage.io import imread
import os
import shutil
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.model_selection import train_test_split
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.nasnet import NASNetMobile
from keras.applications.xception import Xception
from keras.applications import VGG16
from keras.utils.vis_utils import plot_model
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D, Average, Input, Concatenate, GlobalMaxPooling2D,Conv2D
from keras.models import Model
from keras.callbacks import CSVLogger, ModelCheckpoint
from keras.optimizers import Adam
from livelossplot import PlotLossesKeras


# In[3]:


warnings.filterwarnings('ignore')
IMAGE_SIZE = (224, 224)
INPUT_SHAPE = (224, 224, 3)
TRAINING_RATIO = 0.0001


# In[4]:


train_datagen = ImageDataGenerator()
test_datagen = ImageDataGenerator()

training_set = train_datagen.flow_from_directory(
            'data/training',
            target_size = IMAGE_SIZE,
            class_mode = 'categorical',
            batch_size = 32)

test_set = test_datagen.flow_from_directory(
            'data/validation',
            target_size = IMAGE_SIZE,
            class_mode = 'categorical',
            batch_size = 32)
classes = {v: k for k, v in training_set.class_indices.items()}
classes


# In[5]:


#Model
input_shape = INPUT_SHAPE
inputs = Input(input_shape)
input_tensor = Input(shape= (224,224,3))
xception = Xception(include_top=False, input_shape = input_shape)(inputs)
nas_net = NASNetMobile(input_tensor = input_tensor, include_top = False, weights = 'imagenet')(inputs)


outputs = Concatenate(axis=-1)([GlobalAveragePooling2D()(xception), GlobalAveragePooling2D()(nas_net)])
outputs = Dropout(0.5)(outputs)
outputs = Dense(1, activation='sigmoid')(outputs)

model = Model(inputs, outputs)
model.compile(optimizer=Adam(lr=0.0001, decay=0.00001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.summary()


# In[6]:


#  Training
history = model.fit_generator(training_set,
                              steps_per_epoch=30,
                              epochs=10,
                              validation_data=test_set)

model.save("LastModel.h5")
model.save_weights("LastModelWeights.h5") 


# In[21]:


fig, ax = plt.subplots(1,2, figsize=(10,5))
ax = ax.ravel()
for i, met in enumerate(['accuracy','loss']):
    ax[i].plot(history.history[met])
    ax[i].plot(history.history['val_' + met])
    ax[i].set_title('Model {}'.format(met))
    ax[i].set_xlabel('spochs')
    ax[i].set_ylabel(met)
    ax[i].legend(['train', 'val'])
    
plt.show()


# In[ ]:




