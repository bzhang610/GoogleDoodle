import os, sys
import ast

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from IPython.display import display, HTML

classes = ["airplane","angel","basket","bear","belt","candle","crown","cat"]

class DataLoader:
    '''
    Data Loader for google AI challenge used for generating batches of data from training source with minimal delay
    '''
    def __init__(self,access_files,batch_size=16):
        '''
        Initialize a base data loader to load samples from a given access file
        inputs:
        -------
        access_file: location of source folder
        self: pointer to current class instance
        batch_size:size of batch to be taken from each file in the access_files set
        '''
        assert isinstance(access_files,map),"Undesired input to the function, map iterable not provided to iterate through the files of the data"
        assert isinstance(batch_size,int),"Undesired input to the function, batch size must be an integer"
        self.iterators = []#Create an empty list of generators to access data from multiple classes
        
        for index,file in enumerate(access_files):
            assert isinstance(file,str),"Undesired input, file name must be a string"
            data_generator = pd.read_csv(file,usecols=["drawing","word"],chunksize=batch_size)
            self.iterators.append(data_generator)
        
    def __next__(self):
        '''
        Proceed to the next batch of data
        
        inputs:
        ------
        self: pointer to class instance
        
        outputs:
        -------
        dataframe: (dtype:pandas dataframe)Dataframe of samples form different classes
        '''
        data = []
        for it in self.iterators:
            data.append(next(it));
        dataframe = pd.concat(data)
        dataframe = dataframe.sample(frac=1).reset_index(drop=True)
        return dataframe
    
    def __iter__(self):
        '''
        Iterator to allow for iteration along dataset
        
        inputs:
        -------
        self:pointer to the instance of the class
        
        outputs:
        -------
        self:pointer to the instance of the class
        '''
        return self
            

class data_loader_v0:
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
