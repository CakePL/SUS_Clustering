# SUS Clustering application
_Application for clustering sets of letters (stored as images). Model training using multithread optuna optimization._
<br /> © Mateusz Boruń 
## Installation
To install application (on Unix OS):
1. Some prerequisites are required (unless install script won't work properly):
   1. `python3`
   2. `python3-pip`
   3. `python3-virualenv`
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
As  `[filename]` you have to provide path to file with paths to unclastered images. This images will be clustered then.
<br />
You can also launch test run, by executing:
```
python3 cluster.py random
```
In this case input file is randomly generated from test data set.

### Output
As an output 2 files are generated:
1. `result.txt` - file of clustered results - in each line there will be filenames, which includes in single cluster.
2. `result.html` - file of clustered images. Every cluster will be separated by a single line (_it is __not__ suggested to open results using Google Chrome due to opening problems_).

## File structure description
Project directory contains much more than just simple clustering program. Here is a description of additional stuff:
1. `requirements.txt` - this file contains all requirements needed to run the `cluster.py` app.
2. `data/` - directory contains training data set.
3. `data/@0CLUSTERING.csv` - file contains manual clustering of whole training data set. This file is used to validate models in training process.
4. `preprocessing/Clustering.ipynb` - main jupyter notebook used to train and validate clustering model. Brief description of this process below.
5. `preprocessing/clustering.db` - database handling optuna studies for multithread processing

## Algorithm description

### How does the algorithm work?
The algorithm consists of some simple but effective steps:
1. Load images from files
2. Compute the center of mass of each image
3. Add white padding to each image, so that center of mass of each image is in the center of image
4. Add minimal symmetric white padding to each image so that they all have the same width and height
5. Cluster them with DBSCAN algorithm with hiperparameters min_samples=1, and some hiperparameter EPS, using Minkowski distance with some hiperparameter P as a dissimilarity measure

### Hiperparameters optimization
The key to good performance is to find best values of hiperparameters EPS and P.<br />
In order to do that I use a modern optimization framework "Optuna" (see: optuna.org)<br />
Example dataset was clustered manually and then I optuna was used to find P and EPS that maximize adjusted rand index score (clustering generated by algorithm was compared to manual clustering)<br />
Optimization process consisted of two steps:
 1. optimize P using the entire example dataset (7618 characters) - the plot of this optimization (adjusted rand index score depending on P)
 2. for P found in step one, optimize EPS using 5000 random characters from example dataset

Result of optimization process are:
* __P__ = 3.54941880103127
* __EPS__ = 1.0668941233212608

### Computing time estimation
Due to distance calculations in Minkowski space (non-Euclidean), whole process can take a while. For the input of __5000__ images,
process should take about __6 minutes__ to evaluate.

