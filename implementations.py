# -*- coding: utf-8 -*-
"""some helper functions for project 1."""
import csv
import numpy as np

"""Basic implementations"""

def compute_residuals(y, tx, w):
    '''Compute residuals between the predicted regression and y'''
    e = y - (tx @ w.T)
    return e


def compute_grad(y, tx, w):
    '''Computes grad of the data'''
    resi = compute_residuals(y, tx, w)
    grad = -(tx.T @ resi)/y.shape[0]
    return grad


def compute_mse_loss(y, tx, w):
    '''Computes MSE loss calculator'''
    loss = np.sum((y - (tx @ w))**2)/(2*y.shape[0])
    return loss


######################################################################################

def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    '''Implements GD method with step size gamma and max number of iterations given'''
    w = initial_w 
    for n in range(max_iters):
        w +=  -gamma * compute_grad(y, tx, w)
        
    loss = compute_mse_loss(y, tx, w)
    
    return w, loss

def least_squares_SGD(y, x, w,  max_iters, gamma, batch_size=1):
    losses = np.zeros(max_iters)
    for n in range(max_iters):
        for minibatch_y, minibatch_x in batch_iter(y, x, batch_size):
            grad = compute_grad(minibatch_y, minibatch_x, w)
            w += - gamma*grad
    loss = compute_mse_loss(y, x, w)
    return w, loss

######################################################################################

def least_squares(y, tx):
    '''Simple OLS function'''
    w =  np.linalg.solve( tx.T @ tx, tx.T @ y)
    loss = compute_mse_loss(y, tx, w)
    
    return w, loss

######################################################################################

def ridge_regression(y, tx, lambda_):
    '''Implements ridge regression (similar to OLS)'''
    w = np.linalg.solve( (tx.T @ tx) + 2*tx.shape[0]*lambda_*np.eye(tx.shape[1]), tx.T @ y)
    loss = np.sum((y - (tx @ w))**2)/(2*tx.shape[0]) + lambda_*np.sum(w**2) 
    
    return w, loss  

######################################################################################

def sigmoid(x):
    return 1/(1+np.exp(-x))


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    '''Implement logistic regression with GD method'''
    y = (y+1.)/2.
    w = initial_w
    
    for n_iter in range(max_iters):
        grad = tx.T @ ( sigmoid( tx@w ) - y )/y.shape[0] 
        w += -gamma * grad
        loss = np.sum( np.logaddexp( 0,tx@w ) - y * (tx@w) ) # not using the mean loss
        #print(loss)
    loss = np.sum( np.logaddexp( 0,tx@w ) - y * (tx@w) ) # not using the mean loss

    return w, loss


######################################################################################



def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    '''Implement regularized logistic regression with GD method'''
    y = (y+1.)/2.
    w = initial_w   
    for n in range(max_iters):
        grad = tx.T @ ( sigmoid( tx@w ) - y )/y.shape[0] + lambda_*np.linalg.norm(w)*w #grad of loss of logistic function
        w += -gamma*grad/y.shape[0]    # take a step
        loss = loss = np.sum( np.logaddexp( 0,tx@w ) - y * (tx@w) ) +  lambda_*np.sum(w**2)/2  # computes loss
    return w, loss


 ##########################################################################################   
###########################################################################################


"""Additional Functions we used"""


def shuffle(x, y):
    shuffle_indices = np.random.permutation(np.arange(y.shape[0]))
    y = y[shuffle_indices]  # rearranges the y_train based on the shuffled indices
    x = x[shuffle_indices]  # rearranges the x_train based on the shuffled indices
    return x, y



def under_over(x, y, alpha=1, upsample=True, middle=True, gaussian=False, std=0.1, downsample=False):
    '''This function allows us to either undersample the labels -1 or upsample the labels1. There are 2 ways in which
    upsampling can happen. First it takes the middle values of the adjacent point or takes the existing standartized
    data and add gaussian noise with a mean 0 and provided std. Argument alpha allows us to choose what proportion of
    existing data point to add. Eg: if the labels are 10/5 of -1 and 1 respectively, setting alpha=1 will add 5 more
    1s, if alpha=0.2 it will add one etc.
    For the case of undesampling alpha determines the proportion of points that are left over. E.g if alpha=0.2 only
    2 points will be left, if alpha=0.5 5 will be left and so on.'''
    
    index = np.argsort(y)  # returns the indices of the sorted labels

    y = y[index]  # sorts the labels based on indices
    x = x[index]  # sorts the data based on indices

    if upsample:
        x_one, one = shuffle(x[np.where(y == 1)[0][0]:, :],
                             y[np.where(y == 1)[0][0]:])  # shuffles the sorted labels of 1
        if middle:
            x_one = (x_one[1:] + x_one[:-1]) / 2  # takes the middle value between adjacent points

        elif gaussian:
            x_one += std * np.random.randn(x_one.shape[0],
                                           x_one.shape[1])  # adds assign noise with 0 mean and std=0.1

        x = np.vstack((x, x_one[:int(alpha * x_one.shape[0]), :]))  # add the data point from the bottom
        y = np.hstack((y, one[:int(alpha * x_one.shape[0])]))  # stacks the full labels and the new labels of 1s

    elif downsample:
        x_m_one, m_one = shuffle(x[:np.where(y == 1)[0][0]],
                                 y[:np.where(y == 1)[0][0]])  # shuffles the sorted labels of -1

        x_m_one = x_m_one[:int(alpha * x_m_one.shape[0]), :]  # takes the slice
        m_one = m_one[:int(alpha * x_m_one.shape[0])]
        x = np.vstack((x[np.where(y == 1)[0][0]:, :], x_m_one))  # stacks 1s and the slice of -1
        y = np.hstack((y[np.where(y == 1)[0][0]:], m_one))

    x, y = shuffle(x, y)

    return x, y


############################################################################################################



def standardize(trainF, testF):
    """
        Standardize the original data set ignoring the missing values.
        
        Args:
            trainF : matrix with samples (dimensions: (N, M) where N is the number of samples and M the number 
                of features)
            testF : test dataset to standardize with the mean and std of the trianing dataset
        
        Returns: 
            standardized_train : standardized train data
            standardized_test : standardized test data
    """
    
    mask_train , mask_test = (trainF == -999) , (testF == -999.)
    trainF[mask_train] = np.nan
    testF[mask_test] = np.nan
    
    mean = np.nanmean(trainF, axis=0)
    centered_trainF = trainF - mean
    std = np.nanstd(centered_trainF, axis=0)
    standardized_train = centered_trainF / (std + 1e-10)
    
    # standardize test_set with x mean and std
    standardized_test = (testF-mean)/(std + 1e-10)
    
    # replace missing values
    standardized_train[mask_train] = -999.
    standardized_test[mask_test] = -999.
    
    return standardized_train, standardized_test

def normal_standardize(trainF, testF):
    """
        Standardize the original data.
        
        Args:
            trainF : matrix with samples (dimensions: (N, M) where N is the number of samples and M the number 
                of features)
            testF : test dataset to standardize with the mean and std of the trianing dataset
        
        Returns: 
            standardized_train : standardized train data
            standardized_test : standardized test data
    """
    
    mean = np.mean(trainF, axis=0)
    centered_trainF = trainF - mean
    std = np.std(centered_trainF, axis=0)
    standardized_train = centered_trainF / (std + 1e-10)
    
    # standardize test_set with x mean and std
    standardized_test = (testF-mean)/(std + 1e-10)
    
    return standardized_train, standardized_test

############################################################################################################



def replace_missing_values(x, x_test, val, cst = 0):
    """
        Replace missing values with constant values. There are four ways tested:
        it is possible to replace with a constant value for the whole matrix (0 by default)
        or with the mean, median or mode of the feature.
        
        Args:
            x : matrix with samples (dimensions: (N, M) where N is the number of samples and 
                M the number of features). It contains missing values.
            x_test : test dataset where missing values need to be replaced with the values obtained
                     with the trianing dataset
            val : it defines the constant and how the missing values are going to be replaced. 
                It can be 'constant', 'mean', 'median' or 'mode'. 
            cst : if val == 'constant', the constant can be defined (0 by default)
            
        Returns: 
            x : matrix with missing values replaced
            x_test : test matrix with missing values replaced
    """
    
    mask = x == -999.
    mask_test = x_test == -999.
    x[mask] = np.nan
    x_test[mask_test] = np.nan
    
    if val == 'constant':
        
        x[mask] = cst
        x_test[mask_test] = cst
        return x, x_test
        
    if val == 'mean':
        mean = np.nanmean(x, axis=0)
        
        for i in np.arange(x.shape[1]):
            x[np.isnan(x[:, i]), i] = mean[i]
            x_test[np.isnan(x_test[:, i]), i] = mean[i]
        return x, x_test
    
    if val == 'median':
        median = np.nanmedian(x, axis =0)
        
        for i in np.arange(x.shape[1]):
            x[np.isnan(x[:, i]), i] = median[i]
            x_test[np.isnan(x_test[:, i]), i] = median[i]
        return x, x_test
    
    if val == 'mode':
        
        for i in np.arange(x.shape[1]):
            bins = np.histogram(x[:,i][~np.isnan(x[:,i])], bins = x.shape[0]) #divide vect in bins
            idx = np.argmax(bins[0]) #index of the bins with the largest number of values
            mode = bins[1] #vector with the starting points of the bins
            mode = mode[idx] #take the starting value of the bins with the largest number of values in it
            x[np.isnan(x[:,i]),i] = mode #replace nan with the mode
            x_test[np.isnan(x_test[:,i]),i] = mode
            
        return x, x_test
    
    raise Exception("not defined method to replace missing values")



############################################################################################################


def outliers_removal(tx):
    
    """
        Remove outliers. Since the distribution is not normal, we remove outliers depending
        on the interquartile range. If the values are above Q3 + interquartile range x 1.5 or below
        Q1 - interquartile range x 1.5 they are replaced by the limit values (above and below)
        
        Args:
            tx : matrix with samples (dimensions: (N, M) where N is the number of samples and 
                M the number of features)   
        
        Returns: 
            tx : matrix without outliers
    """
    q1, q3 = np.percentile(tx, 25, axis=0), np.percentile(tx, 75, axis=0) #compute 25 and 75 quartile
    iqr = q3 - q1 #interquartile range
    thr = iqr * 1.5
  
    for i in np.arange(tx.shape[1]):
        mask_low = (tx[:,i] < q1[i] - thr[i]) #true value are outliers
        mask_high =  (tx[:,i] > q3[i] + thr[i])
        tx[mask_low,i] = q1[i] - thr[i] #replace outliers with the limits of the range we accept
        tx[mask_high,i] = q3[i] + thr[i]
        
    return tx


############################################################################################################



def get_accuracy(w, testy, testx):
    
    y_pred = predict_labels( w , testx )
    unique, counts = np.unique((y_pred == testy) , return_counts=True)
    res = dict(zip(unique, counts)) 

    return res[True]/(res[True]+res[False])


############################################################################################################




def select_and_expand_f_linear( arrF):
    """
        select main features and create new ones from them
        
        Args:
            arrF : (N,31) array of features , where first feature is the bias (arrF[:,0] == 1)
                    the order of the features must be the same of the csv file.
        
        Returns: 
            new_arr : (N,19) array , combination of selected and new features
    """
    
    new_arr = np.zeros(( arrF.shape[0], 19 ))
    
    temp = 0
    
    # selection
    selectInd = [ 0,  1,2,6, 12, 14, 17, 20, 23, 26]
    for ind in range(len(selectInd)):
        new_arr[:,ind] = arrF[:,selectInd[ind]]
     
    temp += len(selectInd)
    
    # combinations 2
    comb2Ind = [ (14 , 22) , (8,2) , (8,3)] 
    for ind in range(len(comb2Ind)):
        new_arr[:,temp + ind] = arrF[:,comb2Ind[ind][0]] *arrF[:,comb2Ind[ind][1]]
     
    temp += len(comb2Ind)
    
    # combination 3
    comb3Ind = [ (5,2,3) ]
    for ind in range(len(comb3Ind)):
        new_arr[:,temp + ind] = arrF[:,comb3Ind[ind][0]] *arrF[:,comb3Ind[ind][1]] *arrF[:,comb3Ind[ind][2]]
     
    temp += len(comb3Ind)
    
    # log(1**2)*5*2
    new_arr[:,temp] = np.log(arrF[:,1]**2)*arrF[:,5] *arrF[:,2]
    temp += 1
    
    # log(3**2)*3
    new_arr[:,temp] = np.log(arrF[:,3]**2)*arrF[:,3]
    temp += 1
    
    # logs(x**2)
    logInd = [3,1,14]
    for ind in range(len(logInd)):
        new_arr[:,temp + ind] =np.log(arrF[:,logInd[ind]]**2)
        
    temp += len(logInd)
      
    return new_arr


############################################################################################################

def select_and_expand_f_logistic( arrF):
    """
        select main features and create new ones from them
        
        Args:
            arrF : (N,31) array of features , where first feature is the bias (arrF[:,0] == 1)
                    the order of the features must be the same of the csv file.
        
        Returns: 
            new_arr : (N,526) array , combination of selected and new features
    """
    
    new_arr = np.zeros(( arrF.shape[0], 526 ))
    
    temp = 0
    
     # combinations of 2
    cnt = 0
    for ind1 in np.arange(31):
        for ind2 in np.arange(31)[ind1:]:
            new_arr[:,temp + cnt] = arrF[:,ind1]*arrF[:,ind2]
            cnt += 1
    temp += cnt-1
    
    
    # logs
    for ind in np.arange(31)[1:]:
            new_arr[:,temp + ind] = np.log(np.abs(arrF[:,ind]))       
    temp += 30
    
    
        
    return new_arr


############################################################################################################



def acc_f1(weights, data, ytest):
    y_pred = predict_labels(weights, data)
    ytest = (ytest + 1) / 2
    y_sum = y_pred + ytest  # adds true and predicted values
    tp = list(y_sum).count(2)  # true positives
    fp = list(y_sum).count(1)  # false positives
    tn = list(y_sum).count(-1)  # true negatives
    fn = list(y_sum).count(0)  # false negatives
    acc = 100 * (tp + tn) / (tp + fp + tn + fn)
    tpr = 100 * (tp) / (tp + fn)
    tnr = 100 * (tn) / (tn + fp)
    ppv = 100 * (tp) / (tp + fp)
    f1 = 2 * (tpr * ppv) / (tpr + ppv)
    return acc, f1


#####################################
###                               ###
#####################################


"""Project Helpers """


def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]




def load_csv_data(data_path, sub_sample=False):
    """Loads data and returns y (class labels), tX (features) and ids (event ids)"""
    y = np.genfromtxt(data_path, delimiter=",", skip_header=1, dtype=str, usecols=1)
    x = np.genfromtxt(data_path, delimiter=",", skip_header=1)
    ids = x[:, 0].astype(np.int)
    input_data = x[:, 2:]

    # convert class labels from strings to binary (-1,1)
    yb = np.ones(len(y))
    yb[np.where(y=='b')] = -1
    
    # sub-sample
    if sub_sample:
        yb = yb[::50]
        input_data = input_data[::50]
        ids = ids[::50]

    return yb, input_data, ids



def predict_labels(weights, data):
    """Generates class predictions given weights, and a test data matrix"""
    y_pred = np.dot(data, weights)
    y_pred[np.where(y_pred <= 0)] = -1
    y_pred[np.where(y_pred > 0)] = 1
    
    return y_pred



def create_csv_submission(ids, y_pred, name):
    """
    Creates an output file in csv format for submission to kaggle
    Arguments: ids (event ids associated with each prediction)
               y_pred (predicted class labels)
               name (string name of .csv output file to be created)
    """
    with open(name, 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({'Id':int(r1),'Prediction':int(r2)})



