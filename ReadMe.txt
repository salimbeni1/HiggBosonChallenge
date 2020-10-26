The Zip file contains the following items:
implementations.py
run.py
saved_weights.py 
data folder 
basic_tests.ipynb
example.ipynb
 

basic_tests.ipynb
The basic_tests.ipynb is the prewritten jupyter notebook file that you can run to test all the basic
functions with only standartisation and no other methods.
If you encounter a keyerror please restart kernel and rerun the cell which raised the 
error.


run.py
The run.py is the main file with all our preprocessing and training that outputs our best submission 
in csv of the test file and prints the accuracy and the f1 score of the train set it was trained on.
The model takes more than 30 minutes to train so we provide you with the pre-trained weights
that produce exactly our best submission. Note that if you retrain the model the accuracy may vary
as we did not seed the initial radom weight matrix. Once you run the file you should see the file
named output_sub.csv appear which is our best submission csv.


example.ipynb
The example.ipynb gives you the comparison of our preprocessing techniques used on GD and logistic 
regression and outputs the accuracy and f1 score for both. (Note this is not our best model)


implementations.py
implementations.py contains all the functions that we use. The first part contains all the basic
ones, the second part contains our modifications and the last part contains the functions provided
to us during the labs that we found useful to use for this project.


saved_weights.py 
saved_weights.py  is the python file that contains the weights of our best model. It is automatically
imported in run.py file.

data folder contains the train.csv and test.csv that we call during our tests and run.py