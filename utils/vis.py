import os, sys
import ast
import importlib
import keras

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from glob import glob
from PIL import Image

from .load import DataLoader, classes

def sketchplot(data,row=None):
    assert isinstance(data, (DataLoader, ImageGenerator)), "data needs to be of type data_loader"
    if row is None:
        row = np.random.randint(0,data.shape[0])
    points = data[row]['drawing']
    points = ast.literal_eval(points)
    for xy in points:
        [x,y] = xy
        plt.plot(x, y, marker='.')
    plt.title(data[row]['word'])
    plt.axis('off')

class ImageGenerator():
    imsize = (450, 450)
    def __init__(self, access_files, batch_size=16, **kwargs):
        self.loader     = DataLoader(access_files, batch_size=batch_size)
        self.dim        = kwargs.pop('dim', self.imsize)
        self.batchSize  = batch_size*8
        self.n_channels = kwargs.pop('n_channels', 4)
        self.n_classes  = len(classes)
        #self.indexes    = np.arange(self.data.shape[0])
        #self.shuffle    = kwargs.pop('shuffle', True)

    def generateImages(self):
        while True:
            X = np.empty((self.batchSize, *self.dim, self.n_channels))
            y = np.empty((self.batchSize,), dtype=int)

            df = next(self.loader)
            for i, idx in enumerate(range(df.shape[0])):
                (img, lbl) = self.vec2img(df, idx)
                X[i,] = np.asarray(img)
                y[i,] = lbl

            yield X, self.onehot(y)

    def onehot(self, y):
        b = np.zeros((y.shape[0], self.n_classes))
        b[np.arange(y.shape[0]), y] = 1
        return b

    def vec2img(self, df, row=None):
        if row is None:
            row = np.random.randint(0,df.shape[0])
        points = df.iloc[row]['drawing']
        points = ast.literal_eval(points)
        plt.close()
        figure = plt.figure()
        for xy in points:
            [x,y] = xy
            plt.plot(x, y, marker='.')
        plt.axis('off')
        return fig2img(figure, size=self.imsize), classes.index(df.iloc[row]['word'])

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