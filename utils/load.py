import os, sys
import ast

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from IPython.display import display, HTML

classes = ["airplane","angel","basket","bear","belt","candle","crown","cat","dog","fish"]

class data_loader:
    def __init__(self,access_files):
        '''
        Initialize a base data loader to load samples from a given access file
        inputs:
        -------
        access_file: location of source folder
        self: pointer to current class instance
        '''
        #self.access_file = access_file
        frames = []
        for index,file in enumerate(access_files):
            data = pd.read_csv(file,usecols=["drawing","word"])
            frames.append(data)
                
        self.data = pd.concat(frames)
        self.data = self.data.sample(frac=1).reset_index(drop=True)
        display(HTML(self.data.head().to_html()))
        
    def __getitem__(self,index):
        '''
        Indexer of class to return specific locations during calls
        inputs:
        ------
        self: pointer to current class instance
        '''
        return self.data.iloc[index];
    
    def head(self):
        '''
        To visualize sample values in the data
        inputs:
        -------
        self: pointer to current class instance
        '''
        display(HTML(self.data.head().to_html()))

    @property
    def shape(self):
        return self.data.shape

    @property
    def labels(self):
        return pd.get_dummies(self.data.word)
