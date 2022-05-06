# SUS Clustering application
This is an application (blah blah blah)
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
