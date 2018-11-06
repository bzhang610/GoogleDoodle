import os, sys
import ast
import importlib
import keras

import pandas as pd
import numpy as np

from .load import DataLoader, classes
from keras.preprocessing.sequence import pad_sequences

class SeqGenerator():
    def __init__(self, access_files, batch_size=16,**kwargs):
        self.loader     = DataLoader(access_files, batch_size=batch_size)
        self.dim = kwargs.pop('dim', 100)
        self.batchSize  = batch_size*8
        self.n_channels = kwargs.pop('n_channels', 3)
        self.n_classes  = len(classes)

    def generateSeq(self):
        while True:
            X = np.empty((self.batchSize, self.dim, self.n_channels))
            y = np.empty((self.batchSize,), dtype=int)
            df = next(self.loader)
            for i, idx in enumerate(range(df.shape[0])):
                (seq, lbl) = self.pts2seq(df, idx)
                X[i,] = seq
                y[i,] = lbl
            yield X, self.onehot(y)

    def onehot(self, y):
        b = np.zeros((y.shape[0], self.n_classes))
        b[np.arange(y.shape[0]), y] = 1
        return b

    def pts2seq(self, df, row=None):
        if row is None:
            row = np.random.randint(0,df.shape[0])
        points = df.iloc[row]['drawing']
        # strokes = = get_strokes(points,self.dim) #currently not working
        strokes = stack_it(points,self.dim)
        return strokes, classes.index(df.iloc[row]['word'])


def get_strokes(xy_data,maxlen=200):
    """preprocess the points to a standard Nx3 sequence vector, N = maxlen"""
    '''currently not working'''
    xy = ast.literal_eval(xy_data) # string->list
    strokes = np.zeros([maxlen,3])
    idx_begin = 0
    for i in xy:
        x,y = i
        idx_end = idx_begin + len(x)
        strokes[idx_begin:idx_end,0] = x # x values for stroke
        strokes[idx_begin:idx_end,1] = y # y values for stroke
        strokes[idx_begin:idx_end,2] = 1 # index 1 for same stroke
        strokes[idx_begin,2] = 2 # index 2 for new stroke (0 for no stroke)
        idx_begin = idx_end #update next index
    return strokes
    

def stack_it(raw_strokes,STROKE_COUNT=100):
    
    """preprocess the string and make a standard Nx3 stroke vector"""
    '''https://www.kaggle.com/kmader/quickdraw-lstm-thoughts/notebook'''
    
    stroke_vec = ast.literal_eval(raw_strokes) # string->list
    # unwrap the list
    in_strokes = [(xi,yi,i)  
     for i,(x,y) in enumerate(stroke_vec) 
     for xi,yi in zip(x,y)]
    c_strokes = np.stack(in_strokes)
    # replace stroke id with 1 for continue, 2 for new
    c_strokes[:,2] = [1]+np.diff(c_strokes[:,2]).tolist()
    c_strokes[:,2] += 1 # since 0 is no stroke
    # pad the strokes with zeros
    return pad_sequences(c_strokes.swapaxes(0, 1), 
                         maxlen=STROKE_COUNT, 
                         padding='post').swapaxes(0, 1)