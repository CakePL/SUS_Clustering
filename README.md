# SUS Clustering application

_Application for clustering sets of letters (stored as images). Hiperparameter optimization using multithread framework Optuna._
<br /> © Mateusz Boruń

## File structure description
Main program for clustering: `cluster.py`
Project directory contains much more than just simple clustering program. Here is a description of additional stuff:

1. `optimization/` - directory documents the hiperparameter optimization process, it is just a part of the method description - you do __not__ need it to run main program. It's for you to see exactly, how hiperparameters were optimized
2. `optimization/data/` - directory contains dataset of 7618 images of characters used for hiperparameter optimization.
3. `optimization/data/@0CLUSTERING.csv` - file contains manual clustering of whole dataset used for examining different clusterings (it took about 12h to cluster the dataset by hand)
4. `optimization/Clustering.ipynb` - jupyter notebook containing __all__ commands invoked during hiperparameter optimization process and results
5. `optimization/clustering.db` - database handling multithread processing
6. `optimization/plots/` - directory contains visualization of optimization process
7. `optimization/logs/` - directory contains logs from optimization process
8. `install.sh` - script creates virtual environment and installs libraries
9. `requirements.txt` - this file contains all requirements needed to run the `cluster.py` app.

## Installation

To install application (on Unix OS):

1. Some prerequisites are required (unless install script won't work properly):
    1. `python3`
    2. `python3-pip`
    3. `python3-virualenv` - package should be installed not by using `pip`, but using OS package manager. <br />
       For example, Debian/Ubuntu users should install `sudo apt install python3-virualenv`.
2. Run script `install.sh`. Script will automatically create virtual environment and install all needed libraries.

## How to run?

To run application, firstly you need to activate virtual environment.
<br />
Inside application directory run:

```
source ./venv/bin/activate
```

Then to run application, simply type:

```
python3 cluster.py [filename]
```

As  `[filename]` you have to provide path to file with paths to unclastered images. Names of files with images should be unique. This images will be clustered then.
<br />
You can also launch test run, by executing:

```
python3 cluster.py random
```

In this case input file is randomly generated from test data set - of course it requires the dataset (which is not normally needed).

### Output

As an output 2 files are generated:
> 1. `result.txt` - file of clustered results - in each line there will be filenames, which includes in single cluster.
> 2. `result.html` - file of clustered images. Every cluster will be separated by a single line (_it is __not__ suggested
     to open results using Google Chrome due to opening problems_).

### Computing time estimation

Due to distance calculations in Minkowski space (non-Euclidean), whole process can take a while.
For the input of __5000__ images,
process should take about __7-8 minutes__ to evaluate.

## How does the algorithm work?
The algorithm has 2 hiperparameters:
>- __P__ - order of Minkowski distance
>- __EPS__ - parameter of DBSCAN algorithm

The algorithm consists of some simple steps:
>1. Load images of characters from files
>2. Convert images to grayscale
>3. Compute the center of mass of each image
>4. Add white padding to each image, so that center of mass of each image is in the center of image
>5. Add minimal symmetric white padding to each image so that they all have the same width and height
>6. Cluster them with `DBSCAN` algorithm with parameters `min_samples=1`, and `eps=EPS`
using Minkowski P-distance as a dissimilarity measure

The reason we "shift" each image based on center of mass is: two images with the same character placed in different places
should be considered similar (and should be clustered together).\
I chose `DBSCAN` algorithm because it's a very convenient
algorithm to handle unknown number of cluster as long as parameter `eps` and dissimilarity measure are well-chosen.
Determining these values is the key part of this solution.

## Hiperparameters optimization

The key to good performance is to find best values of hiperparameters __EPS__ and __P__.\
In order to do that I used a modern optimization framework ___Optuna___ (see: [Optuna website](optuna.org)) and speed it up by multithreaded computation.\
Example dataset was clustered manually and then, Optuna was used to find P and EPS that maximize
__adjusted__ rand index score (clustering generated by algorithm was compared to manual clustering).

As it's necessary to specify finite interval for each hiperparameter
and order of Minkowski distance P can be any number from interval [1, +inf],
we define __Q__ := 1 / P, so Q is in interval [0, 1] optimize Q instead of P.

### Optimization algorithm description

To be a bit more formal let's introduce the following notation, for any values Q, EPS we define
>- f(Q, EPS) - adjusted rand score index of clustering of __the entire dataset__ obtained with these values of hiperparameters
>- g(Q, EPS) - average adjusted rand score index of clustering of __randomly chosen 5000-element subsets__ obtained with these values of hiperparameters
(we calculate the average of __50__ values)

Optimization process consists of two steps:
> 1. Find Q that maximize value of max( lambda EPS.f(Q, EPS) ), let BEST_Q be that optimal Q
> 2. Find EPS that maximize value of g(BEST_Q, EPS), let BEST_EPS be that optimal EPS

In other words:
> 1. Take Q that allows us (with any EPS) to get the highest score of clustering of __the entire dataset__.
> 2. Take EPS that allows us to get the highest average score of clustering of __randomly chosen 5000-element subset__ with Q fixed on value from previous step.

You can see optimization plots of step 1. and step 2. in files
`optimization/plots/optimization_q.html` and `optimization/plots/optimization_eps.html` respectively.\
You can also see Optuna logs from these steps in files `optimization/logs/optimization_q.log` and `optimization/logs/optimization_eps.log`

Someone may be surprised that we evaluate value of Q on the entire dataset, but value of EPS on randomly chosen 5000-element subsets.\
The reason for this is:\
The choice of a dissimilarity measure should result from the nature of the data rather than size of the test set,
so there is no particular need to restrict size of dataset.\
But the choice of eps parameter of DBSCAN algorithm should depend on the size of test set (the smaller the size of test set, the larger the eps value should be).

The reason for this order of optimization (first Q, then EPS) is:\
With Q fixed it is very cheap to test many values of EPS because we can calculate the distance matrix only once.
As computing the distances is the most expensive part of DBSCAN algorithm, we save a lot of time

### Optimization results

Result of optimization process are:

| __P__   | __3.5779679634436112__ |
|---------|------------------------|
| __EPS__ | __1.0694905187389605__ |

which leads to following average adjusted rand index score on randomly chosen 5000-element subset of example dataset:

| __average adjusted rand index score__ | __0.9567071769677712__ |
|---------------------------------------|------------------------|

You can see a visualization of an example outcome for these values of P and EPS on 5000-element subset in file `optimization/plots/clustering_example.png`.\
You can see a visualization of correct clustering of this set in file `optimization/plots/clustering_correct.png`.\
You can also see visualization of manual clustering of the entire dataset in file `optimization/plots/clustering_all_data.png`

rand index scores in this example are:

| __rand index score__          | __0.9955691138227646__ |
|-------------------------------|------------------------|
| __adjusted rand index score__ | __0.9534206920986646__ |

