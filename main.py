import sys
import os
import skimage
import skimage.io as io
import scipy.ndimage as ndi
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn import cluster
from sklearn import metrics
from scipy.optimize import minimize_scalar
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import squareform

from scipy.spatial.distance import pdist

import optuna
import time

# Creates file with random paths to source objects.
def randomize_file(n=5000):
    src = "./data/"
    data = os.listdir(src)
    print(os.getcwd()+src[1:])
    print(data[:100])

def inSRC():
    pass

def outTXT():
    pass

def outHTML():
    pass

def main():
    input_filename = sys.argv[1]
    print((input_filename))
    if input_filename == "random":
        randomize_file()


if __name__ == "__main__":
    main()