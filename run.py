import numpy as np
from proj1_helpers import *
from preprocessing_helpers import *
from Basics import *
from saved_weight import w_after_20000_steps


DATA_TRAIN_PATH = 'data/train.csv' 
y, tX, ids = load_csv_data(DATA_TRAIN_PATH)
DATA_TEST_PATH = 'data/test.csv'
y_test, tX_test, ids_test = load_csv_data(DATA_TEST_PATH)

print("train and test csv parsed")

trainY , trainF = y, tX
testY , testF = y_test, tX_test

#### PREPROCESSING #################################################

# standardize
trainF, testF = standardize(trainF, testF)
trainF = np.insert(trainF ,0 , np.ones(trainF.shape[0]),axis = 1) # add bias
testF = np.insert(testF ,0 , np.ones(testF.shape[0]),axis = 1) # add bias

# replace -999 with mode
trainF, testF = replace_missing_values(trainF,testF, 'mode')

# feature selection expansion
trainF_sel = select_and_expand_f_logistic(trainF)
testF_sel = select_and_expand_f_logistic(testF)

print("preprocessing done")

#### TRAINING ######################################################
#w_inital = np.random.rand(trainF_sel.shape[1])

# commented because this training can last more than 30min
# you can make less steps but the accuracy wont be optimal
# w, _ = logistic_regression(trainY, trainF_sel, w_inital , 20000 ,0.07/y.shape[0])

# use pretrained array
w = np.array(w_after_20000_steps)


#### prediction ###################################################

y_pred = predict_labels(w  , testF_sel)


print("prediction done")

# save prediction
OUTPUT_PATH = 'output_sub1000.csv'
create_csv_submission(ids_test, y_pred, OUTPUT_PATH)