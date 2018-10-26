import os, sys

import numpy as np 
import pandas as pd 

from keras.models import Model, Sequential, load_model
from keras.layers import Conv2D, Dropout, BatchNormalization, Flatten, Dense, Activation
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping

def DoodleConv1(xdim, cdim):
    # Simple CNN with batch normalization and dropout
    # Last update: 2018.10.26 -I. Colbert
    model = Sequential()

    model.add(BatchNormalization())
    model.add(Conv2D(32, (3,3)))
    model.add(Activation('relu'))
    model.add(Dropout(0.3))

    model.add(BatchNormalization())
    model.add(Conv2D(64, (3,3)))
    model.add(Activation('relu'))
    model.add(Dropout(0.3))

    model.add(BatchNormalization())
    model.add(Conv2D(64, (3,3)))
    model.add(Activation('relu'))
    model.add(Dropout(0.3))

    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))

    model.add(Dense(cdim))
    model.add(Activation('softmax'))

    return model