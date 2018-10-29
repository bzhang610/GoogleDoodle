import os, sys
import ast
import importlib
import keras

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from glob import glob
from PIL import Image

from .load import data_loader

def sketchplot(data,row=None):
    assert isinstance(data, (data_loader, ImageGenerator)), "data needs to be of type data_loader"
    if row is None:
        row = np.random.randint(0,data.shape[0])
    points = data[row]['drawing']
    points = ast.literal_eval(points)
    for xy in points:
        [x,y] = xy
        plt.plot(x, y, marker='.')
    plt.title(data[row]['word'])
    plt.axis('off')

class ImageGenerator(data_loader, keras.utils.Sequence):
    imsize = (450, 450)
    def __init__(self, access_files, **kwargs):
        data_loader.__init__(self, access_files)
        self.dim = kwargs.pop('dim', self.imsize)
        self.batch_size = kwargs.pop('batch_size', 128)
        self.n_channels = kwargs.pop('n_channels', 4)
        self.n_classes  = self.labels.shape[-1]
        self.indexes    = np.arange(self.data.shape[0])
        self.shuffle    = kwargs.pop('shuffle', True)
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(self.shape[0] / self.batch_size))
    
    def __getitem__(self, index):
        '''
        Returns images using the timestamped vectors in the data
        '''
        idxs = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        X, y = self.__generate_data(idxs)

        return X, y

    def __generate_data(self, idxs):
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size, self.n_classes), dtype=int)

        for i, idx in enumerate(idxs):
            (img, wrd) = self.vec2img(idx)
            X[i,] = np.asarray(img)
            y[i,] = wrd.values

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        if self.shuffle:
            self.data = self.data.sample(frac=1).reset_index(drop=True)

    def vec2img(self,row=None):
        if row is None:
            row = np.random.randint(0,self.data.shape[0])
        points = self.data.iloc[row]['drawing']
        points = ast.literal_eval(points)
        plt.close()
        figure = plt.figure()
        for xy in points:
            [x,y] = xy
            plt.plot(x, y, marker='.')
        plt.axis('off')
        return fig2img(figure, size=self.imsize), self.labels.iloc[row]

def fig2data ( fig ):
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

def fig2img ( fig, size=None ):
    """
    @brief Convert a Matplotlib figure to a PIL Image in RGBA format and return it
    @param fig a matplotlib figure
    @return a Python Imaging Library ( PIL ) image

    http://www.icare.univ-lille1.fr/tutorials/convert_a_matplotlib_figure
    """
    # put the figure pixmap into a numpy array
    buf     = fig2data ( fig )
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


'''
Use this as a template
'''

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, labels, batch_size=32, dim=(32,32,32), n_channels=1,
                 n_classes=10, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            X[i,] = np.load('data/' + ID + '.npy')

            # Store class
            y[i] = self.labels[ID]

        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)