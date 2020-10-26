import numpy as np
from implementations import *
from saved_weight import w_after_20000_steps


print("Project 1")

print("start : train and test csv parsing ...")

DATA_TRAIN_PATH = 'data/train.csv' 
y, tX, ids = load_csv_data(DATA_TRAIN_PATH)
DATA_TEST_PATH = 'data/test.csv'
y_test, tX_test, ids_test = load_csv_data(DATA_TEST_PATH)

print("done")

trainY , trainF = y, tX
testY , testF = y_test, tX_test

#### PREPROCESSING #################################################

# standardize
trainF, testF = standardize(trainF, testF)
trainF = np.insert(trainF ,0 , np.ones(trainF.shape[0]),axis = 1) # add bias
testF = np.insert(testF ,0 , np.ones(testF.shape[0]),axis = 1) # add bias

print("start : replacing missing values with mode ...")

# replace -999 with mode
trainF, testF = replace_missing_values(trainF,testF, 'mode')

print("done")

print("start : feature expansion ...")

# feature selection expansion
trainF_sel = select_and_expand_f_logistic(trainF)
testF_sel = select_and_expand_f_logistic(testF)

print("done")

#### TRAINING ######################################################

print("start : computing weights ...")
#w_inital = np.random.rand(trainF_sel.shape[1])

# commented because this training can last more than 40min
# you can make less steps but the accuracy wont be optimal
# w, _ = logistic_regression(trainY, trainF_sel, w_inital , 20000 ,0.07/y.shape[0])

# use pretrained array
w = np.array(w_after_20000_steps)

print("done")

#### prediction ###################################################

print("start : computing predict lables ...")
y_pred = predict_labels(w  , testF_sel)

# save prediction
OUTPUT_PATH = 'output_sub1000.csv'
create_csv_submission(ids_test, y_pred, OUTPUT_PATH)

print("done : saving prediction on ",OUTPUT_PATH)