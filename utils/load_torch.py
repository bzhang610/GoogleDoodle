import os, sys
import ast
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw 
import torch

class DataLoader():
    '''
    Data hander class to read data from csv files from the quickdraw dataset
    '''
    def __init__(self,classes,root_location,read_size=4,batch_size=16):
        '''
        Initializes class to read the data from the CSV files
        Args:
        ----
            self:(datatype: DataLoader)pointer to current instance of the class
            classes:(data type: list)list of classes to load data for training under a subset of the data
            root_location:(data type: string) root location of the folder under consideration
        Kwargs:
        ------
            read_size:(data type: int)number of instances to read from each file.This value controls the speed of reading the data from the 
                .csv files. Default value is set at 4.
            batch_size:(default type: int) number of instances in a batch of data used for training
        '''
        assert isinstance(classes,list),"Instance of classes must be a list"
        assert all([isinstance(class_name,str) for class_name in classes]),"All class names must be strings"
        assert isinstance(root_location,str),"Root location of CSV files must be string"
        assert isinstance(read_size,int),"Readsize must be an integer"
        
        self.name_encoder = {name:index for index,name in enumerate(classes)}
        self.batch_size = batch_size
        self.num_classes = len(classes)

        access_files = map(lambda x:root_location + x + '.csv',classes)
        self.iterators = []
        
        for file in access_files:
            data_generator = pd.read_csv(file,usecols=["drawing","word","recognized"],chunksize=read_size)
            self.iterators.append(data_generator)
    
    def _generate_block(self):
        '''
        Generate block of data using iterators using the init definition of the class
        Args:
        -----
            self: pointer to the current instance of the class
        Returns:
        -------
            dataframe:(datatype : pandas dataframe)shuffled dataframe of different classes 
        '''
        data = []
        for it in self.iterators:
            data.append(next(it))
        dataframe = pd.concat(data)
        dataframe = dataframe.sample(frac=1).reset_index(drop=True)
        dataframe['drawing'] = dataframe['drawing'].apply(lambda x:eval(x))
        dataframe['word'] = dataframe['word'].apply(lambda x: self.name_encoder[x]) 
        return dataframe
    
    @staticmethod
    def chunker(seq,batch_size):
        '''
        Function to split dataframe to blocks
        Args:
        ----
            seq:dataframe to split into chunks
            batch_size: block of data to break into for processing purposes
        Returns:
        -------
            generator of blocks of data for processing
        '''
        return (seq[pos:pos + batch_size] for pos in range(0, len(seq), batch_size))
    
    def __iter__(self):
        '''
        Generator for class implementation of data loader to generate batches of data from .csv files
        
        '''
        raise NotImplementedError

class SequenceGenerator(DataLoader):
    
    @staticmethod
    def convert(strokes,MIN_LENGTH=5):
        '''
        Converts a sequence of strokes into a numpy array of sequential x,y locations with -1,-1 as demarcator
        Args:
        ----
            strokes:(datatype:int)list of strokes from dataframe
        Kwargs:
        -----
            MIN_LENGTH:min length sequence to filter out noise
        '''
        strokes = list(filter(lambda x:len(x[0])>MIN_LENGTH,strokes))
        
        x = []
        y = []
        z = []
        for stroke in strokes:
            x = x + stroke[0]
            y = y + stroke[1]
            z = z + [1]+ [0]*(len(stroke[0])-1)
        normalized = np.array([x,y,z]).T
        return normalized

    def __iter__(self):
        MAX_LENGTH = 100
        def padding(sample):
            '''
            Function to pad the input sequence
            Args:
            ----
                sample:drawing strokes from pandas dataframe
            '''
            stroke = np.zeros((MAX_LENGTH,3))
            stroke[:sample.shape[0],:] = sample
            return stroke

        while(True):
            try:
                data = self._generate_block()
                data['drawing'] = data['drawing'].apply(self.convert)
                data['lengths'] = data['drawing'].apply(lambda x:x.shape[0])
                data = data[data['lengths']!=0]
                data = data[data['lengths']<MAX_LENGTH]
                max_length = np.max(data['lengths'].values)
                data.sort_values(by=['lengths'],inplace=True,ascending=False)
                data['drawing'] = data['drawing'].apply(padding)
            except:
                break
            for chunk in self.chunker(data,self.batch_size):
                Sequence = torch.from_numpy(np.stack(chunk['drawing'].values.tolist()))
                Lengths = torch.from_numpy(chunk['lengths'].values)
                Category = torch.from_numpy(chunk['word'].values)
                yield Sequence,Category

             
            
class ImageLoader(DataLoader):
    
    @staticmethod
    def draw_image(strokes,imheight=32,imwidth=32):
        '''
        Draws images given strokes from dataset
        Args:
        ------
            strokes:(datatype:string) defines a single stroke of the image
        Kwargs:
        -------
            imheight:(default value:32) defines default height of returned image
            imwidth:(default value:32) defines default width of returned image
        Returns:
        --------
            image:(data type: np.ndarray) image in 32x32 shape with information regarding the image
        '''
        image = Image.new("P",(256,256), color=255)
        image_draw = ImageDraw.Draw(image)
        for stroke in strokes:
            for i in range(len(stroke[0])-1):
                image_draw.line([stroke[0][i], 
                             stroke[1][i],
                             stroke[0][i+1], 
                             stroke[1][i+1]],
                            fill=0, width=5)
        image = image.resize((imheight, imwidth))
        return np.array(image)/255
    
    def __iter__(self):
        while(True):
            try:
                data = self._generate_block()
                data['drawing'] = data['drawing'].apply(self.draw_image)
            except:
                break
            for chunk in self.chunker(data,self.batch_size):
                images = torch.from_numpy(np.stack(chunk['drawing'].values.tolist()))
                categories = torch.from_numpy(chunk['word'].values)
                recognized = chunk['recognized'].values
                yield images.view(images.shape[0],1,images.shape[1],images.shape[2]),categories,recognized
