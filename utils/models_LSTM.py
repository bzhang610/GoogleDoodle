import os
import numpy as np
from keras.utils.np_utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder

import pandas as pd
from keras.metrics import top_k_categorical_accuracy
from keras.models import Sequential
from keras.layers import BatchNormalization, Conv1D, LSTM, Dense, Dropout

from keras import backend
if len(backend.tensorflow_backend._get_available_gpus()) > 0:
    from keras.layers import CuDNNLSTM as LSTM 

dp = 0.1 # dropout 
def DoodleLSTM(xdim,cdim):
    model = Sequential()
    model.add(BatchNormalization(input_shape = xdim))
    model.add(Conv1D(48, (5,)))
    model.add(Dropout(dp))
    model.add(Conv1D(64, (5,)))
    model.add(Dropout(dp))
    model.add(Conv1D(96, (3,)))
    model.add(Dropout(dp))
    model.add(LSTM(128, return_sequences = True))
    model.add(Dropout(dp))
    model.add(LSTM(512, return_sequences = False))
    model.add(Dropout(dp))
    model.add(Dense(128))
    model.add(Dropout(dp))
    model.add(Dense(cdim, activation = 'softmax'))
    return model



