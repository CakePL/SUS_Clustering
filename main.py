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

import plotly.io as pio
pio.renderers.default = "browser"

P = 2
EPS = 2.21


def show_plot(data_x, y):
    pca = PCA(n_components=2)
    data2D = pca.fit_transform(data_x)
    print(f"nr of classes: {len(set(y))}")
    fig = px.scatter(x=data2D[:, 0], y=data2D[:, 1], color=[str(v) for v in y], width=900, height=600)
    fig.show()


def show_results(data, res, imgs):
    correct = make_clustering(data, manual=True)
    print("correct clustering")
    show_plot(imgs, correct)
    print("\n")
    acc = metrics.adjusted_rand_score(correct, res)
    print(f"found clustering")
    print(f"eps: {EPS}")
    print(f"p: {P}")
    print(f"ACCURACY: {metrics.rand_score(correct, res)}")
    print(f"BALANCE ACCURACY: {acc}")
    show_plot(imgs, res)


def automated_clustering(data) -> List[int]:
    ic = io.imread_collection(data, conserve_memory=True)
    df = pd.DataFrame(dtype=object)
    df[0] = ic

    df[0] = pd.DataFrame(df.apply(lambda row: skimage.color.rgb2gray(row[0]), axis=1))
    max_sum = max(df.apply(lambda row: ndi.sum_labels(row[0]), axis=1))

    def img_center(img):
        cy, cx = ndi.center_of_mass(img)
        cy = round(cy)
        cx = round(cx)
        sy, sx = img.shape
        top = max(sy - 1 - cy - cy, 0)
        bot = max(cy - (sy - 1 - cy), 0)
        left = max(sx - 1 - cx - cx, 0)
        right = max(cx - (sx - 1 - cx), 0)
        return cv2.copyMakeBorder(img, top, bot, left, right, cv2.BORDER_CONSTANT, None, value=1.)

    df[0] = df.apply(lambda row: img_center(row[0]), axis=1)

    max_shape_y = max(df.apply(lambda row: row[0].shape[0], axis=1))
    max_shape_x = max(df.apply(lambda row: row[0].shape[1], axis=1))

    def img_equalize_size(img):
        sy, sx = img.shape
        top = (max_shape_y - sy) // 2
        bot = (max_shape_y - sy + 1) // 2
        left = (max_shape_x - sx) // 2
        right = (max_shape_x - sx + 1) // 2
        return cv2.copyMakeBorder(img, top, bot, left, right, cv2.BORDER_CONSTANT, None, value=1.)

    df[0] = df.apply(lambda row: img_equalize_size(row[0]), axis=1)

    df[0] = df.apply(lambda row: np.reshape(row[0], -1), axis=1)

    imgs = list(df[0])

    _, labels = cluster.dbscan(imgs, eps=EPS, min_samples=1, p=P)
    print(imgs)
    return labels, imgs


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
    return list(pd.read_csv(input_filename, header=None, dtype=object)[0])


def name(path):
    return os.path.basename(path)


def make_clustering(data, manual=False, show_results=True):
    if manual:
        csv_filename = "data/@0CLUSTERING.csv"
        src = "./data/"
        # data = ['7_517-141.png', '7_517-266.png', '7_517-395.png', '7_519-40.png', '7_519-514.png', '7_522-189.png']

        df = pd.read_csv(csv_filename)
        df.set_index("Filename", inplace=True)
        res = [df.loc[name(fl), "Cluster"] for fl in data]
    else:
        res, imgs = automated_clustering(data)

    # show_results(data, res, imgs)
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
        input_filename = randomize_file(5000)

    data = inSRC(input_filename)
    print("Begin clustering")
    clustered = make_clustering(data, manual=False)
    print("Generating output...", end="")
    res = [(data[i], clustered[i]) for i in range(len(data))]
    res = sorted(res, key=lambda x: x[1])
    outTXT(res)
    outHTML(res)
    print("DONE!")


if __name__ == "__main__":
    main()
