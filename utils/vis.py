import os, sys
import ast
import importlib

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from .load import data_loader

def sketchplot(data,row=None):
    assert isinstance(data, data_loader), "data needs to be of type data_loader"
    if row is None:
        row = np.random.randint(0,data.shape[0])
    points = data[row]['drawing']
    points = ast.literal_eval(points)
    for xy in points:
        [x,y] = xy
        plt.plot(x, y, marker='.')
    plt.title(data[row]['word'])
    plt.axis('off')