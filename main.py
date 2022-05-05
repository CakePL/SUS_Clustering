import sys
import os
import fnmatch
import random
from typing import List

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


def automated_clustering(data) -> List[str]:
    return ["None"]


def randomize_file(n=5000):
    src = "./data/"
    data = fnmatch.filter(os.listdir(src), '*.png')
    data = [src + data[i] for i in range(len(data))]
    random.shuffle(data)
    fn = "rand_input.txt"
    with open(fn, "w") as f:
        n = min(n, len(data))
        for i in range(n):
            f.write(data[i] + "\n")
    return fn


def inSRC(input_filename):
    return list(pd.read_csv(input_filename, header=None)[0])


def name(path):
    return os.path.basename(path)


def make_clustering(data, manual=False):
    if manual:
        csv_filename = "data/@0CLUSTERING.csv"
        src = "./data/"
        # data = ['7_517-141.png', '7_517-266.png', '7_517-395.png', '7_519-40.png', '7_519-514.png', '7_522-189.png']

        df = pd.read_csv(csv_filename)
        df.set_index("Filename", inplace=True)
        res = [df.loc[name(fl), "Cluster"] for fl in data]
    else:
        res = automated_clustering(data)
    return res


def outTXT(data, filename="res.txt"):
    with open(filename, "w") as res_file:
        act = data[0][1]
        for d in data:
            if d[1] != act:
                res_file.write("\n")
                act = d[1]
            res_file.write(name(d[0]) + " ")


def outHTML(data, filename="res.html"):
    HTML = '''
    <!DOCTYPE html>
    <html>
    <head>
        <title> Clustering results </title>
    </head>
    <body>
    %s
    </body>
    </html>
    '''
    IMG = '<img src="%s" />'

    clustering = []
    clust = []
    act = data[0][1]
    for d in data:
        if d[1] != act:
            act = d[1]
            clustering.append(" ".join(clust))
            clust = []
        clust.append(IMG % d[0])
    final_html = HTML % "\n<hr />\n".join(clustering)
    with open(filename, "w") as res_file:
        res_file.write(final_html)


def main():
    input_filename = sys.argv[1]
    print((input_filename))
    if input_filename == "random":
        input_filename = randomize_file(10000)

    data = inSRC(input_filename)
    clustered = make_clustering(data, manual=True)
    res = [(data[i], clustered[i]) for i in range(len(data))]
    res = sorted(res, key=lambda x: x[1])
    outTXT(res)
    outHTML(res)


if __name__ == "__main__":
    main()
