# ML Project 1 -- Higgs boson challenge


This is a project that aim to find a binary classification in order to separate the Higgs Boson detection from background events using CERN’s Higgs Boson dataset. 

## dependencies

make sure you have installed the numpy python library


## Execute our best model 

to get the same result we got on aicrowd execute : `python run.py`

Make sure that that `train.csv` and `test.csv` are in a data/ folder
( or directly modify in run.py the path of the csv file  )
After excecution a file called output_sub.csv will be generated containing the preddicted labels.


> The model takes more than 30 minutes to train so we provide you with the pre-trained weights
> that produce exactly our best submission. Note that if you retrain the model the accuracy may vary
> as we did not seed the initial random weight matrix.


## Execute other models

`example.ipynb` gives you the comparison of our preprocessing techniques used on linear and logistic 
regression and outputs the accuracy and f1 score for both. 

You can uncomment the data augmentation step , to test over-sampling.


## executes tests on the requested basic functions

`basic_tests.ipynb` is the prewritten jupyter notebook file that you can run to test all the basic
functions with only standartisation and no other methods.