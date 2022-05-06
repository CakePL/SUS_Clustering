import sys
import os
import fnmatch
import random

from skimage import io
import scipy.ndimage as ndi
import pandas as pd
import numpy as np
from sklearn import cluster
import cv2
import time
from skimage.color import rgb2gray

TXT_RESULTS_FILENAME = "results.txt"
HTML_RESULTS_FILENAME = "results.html"
DIAGNOSTIC_INFO = True

N = 5000
P = 2.01
EPS = 2.21


def info(*args, **kwargs):
    if DIAGNOSTIC_INFO:
        print(*args, file=sys.stderr, **kwargs)


def img_center(img):
    center_y, center_x = ndi.center_of_mass(1 - img)
    center_y = round(center_y)
    center_x = round(center_x)
    shape_y, shape_x = img.shape
    top_pad = max((shape_y - 1 - center_y) - center_y, 0)
    bot_pad = max(center_y - (shape_y - 1 - center_y), 0)
    left_pad = max((shape_x - 1 - center_x) - center_x, 0)
    right_pad = max(center_x - (shape_x - 1 - center_x), 0)
    return cv2.copyMakeBorder(img, top_pad, bot_pad, left_pad, right_pad, cv2.BORDER_CONSTANT, None, value=1.)


def img_equalize_size(img, max_shape_x, max_shape_y):
    shape_y, shape_x = img.shape
    top_pad = (max_shape_y - shape_y) // 2
    bot_pad = (max_shape_y - shape_y + 1) // 2
    left_pad = (max_shape_x - shape_x) // 2
    right_pad = (max_shape_x - shape_x + 1) // 2
    return cv2.copyMakeBorder(img, top_pad, bot_pad, left_pad, right_pad, cv2.BORDER_CONSTANT, None, value=1.)


def preprocess_imgs(imgs):
    imgs = imgs.apply(rgb2gray)
    imgs = imgs.apply(img_center)
    max_shape_y = max(imgs.apply(lambda x: x.shape[0]))
    max_shape_x = max(imgs.apply(lambda x: x.shape[1]))
    imgs = imgs.apply(lambda img: img_equalize_size(img, max_shape_x, max_shape_y))
    imgs = imgs.apply(lambda x: np.reshape(x, -1))
    return imgs


def make_clustering(data):
    info("Loading data...")
    imgs = pd.Series([io.imread(path) for path in data.to_list()], index=data.index, dtype=object)
    info("Data loaded")

    info("Preprocessing images...")
    imgs = preprocess_imgs(imgs)
    info("Images preprocessed")

    info("Clustering...")
    _, labels = cluster.dbscan(imgs.to_list(), eps=EPS, min_samples=1, p=P)
    info("Clustering finished")

    return pd.Series(labels, index=data.index, name="clustering")


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
    df = pd.read_csv(input_filename, header=None, dtype=object)
    df.rename(columns={0: "path"}, inplace=True)
    df.index = df.apply(lambda row: name(row["path"]), axis=1)
    return df


def name(path):
    return os.path.basename(path)


def outTXT(data):
    with open(TXT_RESULTS_FILENAME, "w") as res_file:
        act = data.iloc[0, 1]
        for ind, d in data.iterrows():
            if d.loc["clustering"] != act:
                res_file.write("\n")
                act = d.loc["clustering"]
            res_file.write(ind + " ")


def outHTML(data):
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
    act = data.iloc[0, 1]
    for _, d in data.iterrows():
        if d.loc["clustering"] != act:
            act = d.loc["clustering"]
            clustering.append(" ".join(clust))
            clust = []
        clust.append(IMG % d.loc["path"])
    final_html = HTML % "\n<hr />\n".join(clustering)
    with open(HTML_RESULTS_FILENAME, "w") as res_file:
        res_file.write(final_html)


def main():
    start = time.time()  # ONLY FOT TEST!
    input_filename = sys.argv[1]
    info("Input filename: ", input_filename)
    if input_filename == "random":
        input_filename = randomize_file(N)

    data = inSRC(input_filename)
    data["clustering"] = make_clustering(data["path"])
    info("Generating output...")
    data.sort_values(by=["clustering"], inplace=True)
    outTXT(data)
    outHTML(data)
    info("Output generating finished")
    info(f"Text results saved to file {TXT_RESULTS_FILENAME}")
    info(f"HTML results saved to file {HTML_RESULTS_FILENAME}")
    info("Done!")
    stop = time.time()  # ONLY FOT TEST!
    info(f"time: {stop - start}")  # ONLY FOT TEST!


if __name__ == "__main__":
    main()
