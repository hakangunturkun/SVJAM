import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
#keras = tf.keras
import pandas as pd
from PIL import Image
import sys
import glob

import scipy
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix, roc_curve, auc, roc_auc_score

from keras.models import Model, Sequential
#from tensorflow.keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization, Input
from keras.layers import Conv2D, MaxPooling2D
from keras.datasets import cifar10
from keras import regularizers, optimizers
from keras.optimizers import SGD
from keras.callbacks import LearningRateScheduler
from sklearn.metrics import roc_auc_score
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.utils import class_weight
from tensorflow.keras.optimizers import Adam
from keras.layers.merge import Concatenate
from keras.layers.merge import concatenate

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

def create_model(kernel_size ,
                pool_size ,
                first_filters  ,
                second_filters ,
                third_filters  ,
                first_dense,
                second_dense,
                dropout_conv ,
                dropout_dense ):

    model = Sequential()
    # First conv filters
    model.add(Conv2D(first_filters, kernel_size, activation = 'relu', padding="same",
                     input_shape = (512,512,3)))
    model.add(Conv2D(first_filters, kernel_size, padding="same", activation = 'relu'))
    model.add(Conv2D(first_filters, kernel_size, padding="same", activation = 'relu'))
    model.add(MaxPooling2D(pool_size = pool_size)) 
    model.add(Dropout(dropout_conv))

    # Second conv filter
    model.add(Conv2D(second_filters, kernel_size, padding="same", activation ='relu'))
    model.add(Conv2D(second_filters, kernel_size, padding="same", activation ='relu'))
    model.add(Conv2D(second_filters, kernel_size, padding="same", activation ='relu'))
    model.add(Conv2D(second_filters, kernel_size, padding="same", activation ='relu'))
    model.add(MaxPooling2D(pool_size = pool_size))
    model.add(Dropout(dropout_conv))

    # Third conv filter
    model.add(Conv2D(third_filters, kernel_size, padding="same", activation ='relu'))
    model.add(Conv2D(third_filters, kernel_size, padding="same", activation ='relu'))
    model.add(Conv2D(third_filters, kernel_size, padding="same", activation ='relu'))
    model.add(Conv2D(third_filters, kernel_size, padding="same", activation ='relu'))
    model.add(MaxPooling2D(pool_size = pool_size))
    model.add(Dropout(dropout_conv))

    model.add(Flatten())
    
    # Out layer
    model.add(Dense(3, activation = "softmax"))

    model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy', metrics=["categorical_accuracy"])
    return model

def prediction(args1,args2,args3):
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)

    model = create_model(kernel_size = (9,9),
                    pool_size= (3,3),
                    first_filters = 32,
                    second_filters =64 ,
                    third_filters =128,
                    first_dense=256,
                    second_dense=128,
                    dropout_conv = 0.4,
                    dropout_dense = 0.3)
    model.build((512,512,3))
    #model.summary()

    checkpoint_path = "./tools/cnn_weights.ckpt"

    # Loads the weights
    model.load_weights(checkpoint_path)

    def files(args1):
        for file in os.listdir(args1):
            if not os.path.isfile(os.path.join(args1, file)):
                yield file
    file_ls=[]
    for file in files(args1):
        file_ls.append(file)
    for file in files(args1):
        name_of_folder = str(file)

        path_manual = args1+name_of_folder+'/'
        try:
            testdf_manual=pd.read_csv(args2+name_of_folder+'.csv',sep='\t',dtype=str)
        except FileNotFoundError:
            print('FILE_NOT_FOUND',file)
            continue
        if os.path.exists(args3+name_of_folder+'_pred.csv'):
            print("exists:", file)
            continue
        else:
            print(file)
        for filename in os.listdir(path_manual):
            img = load_img(path_manual+filename, target_size=(512,512))
            img_array = img_to_array(img)
            img_array = tf.expand_dims(img_array, 0)  # Create batch axis
            predictions = model.predict(img_array, steps=1)
            score = predictions[0]
            if round(score[0]) == 1:
                testdf_manual.loc[testdf_manual['fullname'] == filename, 'predictions'] = 0
            elif round(score[1]) == 1:
                testdf_manual.loc[testdf_manual['fullname'] == filename, 'predictions'] = 1
            elif round(score[2]) == 1:
                testdf_manual.loc[testdf_manual['fullname'] == filename, 'predictions'] = 2
        testdf_manual.to_csv(args3+name_of_folder+'_pred.csv')
