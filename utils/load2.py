import os, sys
import ast

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

from IPython.display import display, HTML

from keras.preprocessing.sequence import pad_sequences

class DataLoader():
    '''
    Data Loader for google AI challenge used for generating batches of data from training source with minimal delay
    '''
    def __init__(self,classes,data_location,batch_size=16):
        '''
        Initialize a base data loader to load samples from a given access file
        inputs:
        -------
        access_file: location of source folder
        self: pointer to current class instance
        batch_size:size of batch to be taken from each file in the access_files set
        '''
        assert isinstance(classes,list),"Undesired input. Expects list of classes"
        assert isinstance(data_location,str),"Data location of .csv files of training data"
        assert isinstance(batch_size,int),"Undesired input to the function, batch size must be an integer"

        access_files = map(lambda x : data_location + x + '.csv', classes)
        self.iterators = []#Create an empty list of generators to access data from multiple classes
        
        for index,file in enumerate(access_files):
            assert isinstance(file,str),"Undesired input, file name must be a string"
            data_generator = pd.read_csv(file,usecols=["drawing","word"],chunksize=batch_size)
            self.iterators.append(data_generator)
        
        self.num_classes = len(classes)
        self.class_encoder = {class_name:index for index,class_name in enumerate(classes)}

    def _onehotencoder(self,labels):
        '''
        One hot encoding of input labels using number of classes for reference
        Args:
        ----
            self:pointer to current instance of class
            labels:(default type:numpy.ndarray) array of labels from 'word' field from data frame
        returns:
        -------
            one_hot:(default type:numpy.ndarray) array of one hot vectors of the labels provided
        '''
        one_hot = np.zeros((labels.shape[0],self.num_classes))
        one_hot[np.arange(labels.shape[0]),labels] = 1
        return one_hot
        
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
            data.append(next(it))
        dataframe = pd.concat(data)
        dataframe = dataframe.sample(frac=1).reset_index(drop=True)
        return dataframe
    
    def __iter__(self):
        '''
        Iterator to allow for iteration along dataset
        
        inputs:
        self:pointer to the instance of the class
        
        outputs:
        -------
        self:pointer to the instance of the class
        '''
        return self
            





class SeqGenerator(DataLoader):

    @staticmethod
    def convert(strokes,MAXLEN=75):
        '''
        Converts input sequence( list of lists) containing cordinate locations of points on canvas to usable format for RNN/LSTM.
        The individual strokes are concatenated with each other with a separate dimension specifically  to mark start of a new stroke

        Args:
        ----
            strokes:(default type:List) List of lists containing x,y coordinates
        Kwargs:
        ------
            MAXLEN:(default value:100) Maximum length of input sequence to sequence model
        outputs:
        -------
            padded_seq:padded  numpy matrix of the shape (Number of points,3) The first channel corresponds to the x locations, the 
                        the second channel corresponds to the y locations and the third coorespond the starts of new stroke.
        '''
        assert isinstance(strokes,str),"Confirm if input to function is a string for eval function"

        seq = eval(strokes)
        in_strokes = [(xi,yi,i)  for i,(x,y) in enumerate(seq) for xi,yi in zip(x,y)]
        c_strokes = np.stack(in_strokes)
        c_strokes[:,2] = [1]+np.diff(c_strokes[:,2]).tolist()
        c_strokes = c_strokes.astype(np.float64)
        c_strokes[:,0] = c_strokes[:,0]/np.max(c_strokes[:,0])
        c_strokes[:,1] = c_strokes[:,1]/np.max(c_strokes[:,1])
        #Preprocessing to scale values to a range more convinient for operations
        return pad_sequences(c_strokes.swapaxes(0, 1), maxlen=MAXLEN,dtype='float64', padding='post').swapaxes(0, 1)

    def __next__(self):
        '''
        Generator to produce next batch of data
        inputs:
        -------
        self: pointer to class instance

        outputs:
        -------
        X: Numpy array of batch
        Y: label associated to input set
        '''
        data = []
        for it in self.iterators:
            data.append(next(it))
        dataframe = pd.concat(data)
        dataframe = dataframe.sample(frac=1).reset_index(drop=True)

        dataframe['drawing'] = dataframe['drawing'].apply(self.convert)

        dataframe['word'] = dataframe['word'].apply(lambda x:self.class_encoder[x])

        return np.stack(dataframe['drawing'].values.tolist()),self._onehotencoder(dataframe['word'].values)


class ImgGenerator(DataLoader):
    imsize = (450,450)

    def fig2data (self,fig):
        """
        @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
        @param fig a matplotlib figure
        @return a numpy 3D array of RGBA values

        http://www.icare.univ-lille1.fr/tutorials/convert_a_matplotlib_figure
        """
        # draw the renderer
        fig.canvas.draw ( )
    
        # Get the RGBA buffer from the figure
        w,h = fig.canvas.get_width_height()
        buf = np.fromstring ( fig.canvas.tostring_argb(), dtype=np.uint8 )
        buf.shape = ( w, h,4 )
    
        # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
        buf = np.roll ( buf, 3, axis = 2 )
        return buf
    
    def fig2img (self,fig, size=None ):
        """
        @brief Convert a Matplotlib figure to a PIL Image in RGBA format and return it
        @param fig a matplotlib figure
        @return a Python Imaging Library ( PIL ) image

        http://www.icare.univ-lille1.fr/tutorials/convert_a_matplotlib_figure
        """
        # put the figure pixmap into a numpy array
        buf     = self.fig2data ( fig )
        w, h, d = buf.shape
        image   = Image.frombytes( "RGBA", ( w ,h ), buf.tostring( ) )

        if size is not None:
            image.thumbnail(size, Image.ANTIALIAS)
            background = Image.new('RGBA', size, (255, 255, 255, 0))
            background.paste(
                image, (int((size[0] - image.size[0]) / 2), int((size[1] - image.size[1]) / 2))
            )
            return background
        else:
            return image

    def vec2img(self,strokes):
        '''
        Converts given strokes to 2D image. Uses faculties of matpltlib for plotting purposes
        Args:
        ----
            strokes:sequence of points marking path of drawing
        returns:
        -------
            img:
        '''
        points = eval(strokes)
        plt.close()
        figure = plt.figure()
        for xy in points:
            [x,y] = xy
            plt.plot(x, y, marker='.')
        plt.axis('off')
        return self.fig2img(figure, size=ImgGenerator.imsize)

    def __next__(self):
        '''
        Generator to produce next batch of data
        inputs:
        -------
        self: pointer to class instance

        outputs:
        -------
        X: Numpy array of batch
        Y: label associated to input set
        '''
        data = []
        for it in self.iterators:
            data.append(next(it))
        dataframe = pd.concat(data)
        dataframe = dataframe.sample(frac=1).reset_index(drop=True)

        dataframe['drawing'] = dataframe['drawing'].apply(self.vec2img)
        dataframe['word'] = dataframe['word'].apply(lambda x:self.class_encoder[x])
        return np.stack(dataframe['drawing'].values.tolist()),self._onehotencoder(dataframe['word'].values)