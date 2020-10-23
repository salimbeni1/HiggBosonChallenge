import numpy as np
import sys
import time


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



def select_and_expand_f( arrF):
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


def find_modes(x, y):

    x_1 = []    # separate labels
    x_2 = []

    for y, x0 in zip(y, x):
        if y == 1:
            x_1.append(x0)
        else:
            x_2.append(x0)

    x_1 = np.array(x_1)
    x_2 = np.array(x_2)

    cols = []
    for col in range(x_2.shape[1]):
        if np.any(x_2[:, col] == -999):
            cols.append(col)

    x_1 = x_1[:, cols]  # only columns with missing values
    x_2 = x_2[:, cols]

    modes1 = []
    modes2 = []

    for i in range(len(cols)):  # go through the columns of x_1 and x_2
        # delete the missing values from the array
        del_miss_1 = np.delete(x_1[:, i], np.where(x_1[:, i] == -999)[0])
        del_miss_2 = np.delete(x_2[:, i], np.where(x_2[:, i] == -999)[0])

        # mean1 = np.mean(del_miss_1)   #optionally you can find means
        # mean2 = np.mean(del_miss_2)

        # select bin index with maximum count
        count1 = np.argmax(np.histogram(del_miss_1, bins=int(np.sqrt(del_miss_1.shape[0])))[0])
        count2 = np.argmax(np.histogram(del_miss_2, bins=int(np.sqrt(del_miss_2.shape[0])))[0])

        bins1 = np.histogram(del_miss_1, bins=int(np.sqrt(del_miss_1.shape[0])))[1]  # bin edges
        bins2 = np.histogram(del_miss_2, bins=int(np.sqrt(del_miss_2.shape[0])))[1]

        # mode of feature column without the missing values
        mode1 = (bins1[count1] + bins1[count1 + 1]) / 2
        mode2 = (bins2[count2] + bins2[count2 + 1]) / 2

        modes1.append(mode1)  # append the mode of each column into the list
        modes2.append(mode2)

        modes = [modes1, modes2]
    return modes, cols  # return modes list and list of indices of columns that are missing values


def find_and_replace(x, y, cols, bool=True, value=False):
    '''If bool is set to True in the argument valu, provide list (in the order: 1 and -1) of 2 lists (one for
    each label) of values to be replaced. The sublists   must be of length 11 (number of columns that have
    missing values, and must be in the same order as the columns given below). If set to False replace all the
    missing values with the constant term.
    Order in which the values should be provided [0, 4, 5, 6, 12, 23, 24, 25, 26, 27, 28] where numbers represent the
    column indices.'''

    miss_x = x[:, cols]    # select only columns that have missing values
    red_x = np.delete(x, cols, axis=1)     # delete columns that have missing values from the original x

    if bool:
        modes1 = value[0]
        modes2 = value[1]
        for i in range(y.shape[0]):
            if y[i] == 1:    # if label is 1
                for j in range(miss_x.shape[1]):    # go through row of X
                    if miss_x[i, j] == -999:     # if the data is missing
                        miss_x[i, j] = modes1[j]    # replace it the desired value
            else:
                for j in range(miss_x.shape[1]):
                    if miss_x[i, j] == -999:
                        miss_x[i, j] = modes2[j]
    else:
        for i in range(y.shape[0]):
            if y[i] == 1:    # if label is 1
                for j in range(miss_x.shape[1]):    # go through row of X
                    if miss_x[i, j] == -999:     # if the data is missing
                        miss_x[i, j] = value    # replace be the desired value
            else:
                for j in range(miss_x.shape[1]):
                    if miss_x[i, j] == -999:
                        miss_x[i, j] = value

    x = np.column_stack((red_x, miss_x))    # combine the 2 back together

    return x




def shuffle(x, y):
    shuffle_indices = np.random.permutation(np.arange(y.shape[0]))
    y = y[shuffle_indices]  # rearranges the Y_train based on the shuffled indices
    x = x[shuffle_indices]  # rearranges the X)train based on the shuffled indices
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
    

def outliers_removal(tx):
    
    """
        Remove outliers. Since the distribution is not normal, we remove outliers depending
        on the interquartile range. If the values are above or belove the interquartile range x 1.5
        they are replaced by the limit values (above and below)
        
        Args:
            tx : matrix with samples (dimensions: (N, M) where N is the number of samples and 
                M the number of features)   
        
        Returns: 
            tx : matrix without outliers
    """
    q25, q75 = np.percentile(tx, 25, axis=0), np.percentile(tx, 75, axis=0) #compute 25 and 75 quartile
    iqr = q75 - q25 #interquartile range
    thr = iqr * 1.5
  
    for i in np.arange(tx.shape[1]):
        mask_low = (tx[:,i] < q25[i] - thr[i]) #true value are outliers
        mask_high =  (tx[:,i] > q75[i] + thr[i])
        tx[mask_low,i] = q25[i] - thr[i] #replace outliers with the limits of the range we accept
        tx[mask_high,i] = q75[i] + thr[i]
        
    return tx




#//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
# what r we gonna do with this ?

def sigma(x, w):
    '''Define logistic function'''
    sigma = 1/(1+np.exp(np.dot(x, w)))
    return sigma


def predict_logistic_labels(xtest, w):
    """Changed version of predict labels function to account for the fact
    that logistic regression outputs answers between 0 and 1 and not -1 and 1"""
    y_pred = 1 / (1 + np.exp(np.dot(xtest, w.T)))  # prediction of the logistic function
    y_pred[np.where(y_pred <= 1 / 2)] = -1
    y_pred[np.where(y_pred > 1 / 2)] = 1
    return y_pred



def logistic_accuracy(ytest, xtest, w):
    y_pred = predict_logistic_labels(xtest, w)
    #positive == 1
    #negative == -1
    ytest = (ytest + 1)/2
    y_sum = y_pred + ytest  # adds true and predicted values
    tp = list(y_sum).count(2)   # true positives
    fp = list(y_sum).count(0)   # false positives
    tn = list(y_sum).count(-1)  # true negatives
    fn = list(y_sum).count(1)   # false negatives

    return tp, fp, tn, fn


def foo_logistic_regression(y, x, w, max_iters, gamma, ytest, xtest, lmbd):
    '''Implement regularized logistic regression with GD method'''

    P = list(ytest).count(1)   # number of 1s in test set
    N = list(ytest).count(-1)  # number of -1s in test set

    k = 200     # number of steps over which we observe loss to make sure it becomes smaller
    losses = np.zeros(max_iters)    # initialize matrix to record losses
    y = (y + 1)/2   # our labels are -1, 1 need to convert them to 0 and 1 for logistic regression

    i = 0
    for n in range(max_iters):
        if i <= k:
            grad = np.dot(x.T, (y - sigma(x, w)))/y.shape[0] + lmbd*np.linalg.norm(w)*w #grad of loss of logistic function
            w += -gamma*grad    # take a step

            loss = -(np.dot(y, np.log(sigma(x, w))) - np.dot((1 - y), np.log(1 - sigma(x, w)))) / y.shape[0] + \
                   lmbd * np.sum(w ** 2) / 2
            losses[n] = loss  # stores losses

            tp, fp, tn, fn = logistic_accuracy(ytest, xtest, w) # outputs confusion matrix
            sys.stdout.write('\rProgress={0}%, Accuracy={1}%'.format(100*n/max_iters, 100 * (tp + tn) / (N + P)))
            sys.stdout.flush()
            time.sleep(0.0001)

            #print(100 * (tp + tn) / (N + P))
        else:
            if losses[n-1] <= 1*losses[n-k]:  # if loss improves keep looping with the same gamma
                pass
            else:
                gamma = 0.9*gamma # if not make the step smaller
              
                i = 0

            grad = np.dot(x.T, (y - sigma(x, w)))/y.shape[0] + lmbd*np.linalg.norm(w)*w #grad of loss of logistic function
            w += -gamma*grad    # take a step

            loss = -(np.dot(y, np.log(sigma(x, w))) - np.dot((1 - y), np.log(1 - sigma(x, w))))/y.shape[0] + \
                   lmbd*np.sum(w**2)/2
            losses[n] = loss

            tp, fp, tn, fn = logistic_accuracy(ytest, xtest, w)
            sys.stdout.write('\rProgress={0}%, Accuracy={1}%'.format(100*n/max_iters, 100 * (tp + tn) / (N + P)))
            sys.stdout.flush()
            time.sleep(0.0001)
        i += 1
    #print(100*(tp+tn)/(N+P))
    return w, losses, tp, fp, tn, fn  # return the weight matrix and loss


def jet_number(x, y):
    ind = []
    yy = []
    xx = []
    for n in range(4):
        ind.append(np.where(x[:, 22] == n)[0])
        yy.append(y[ind[n]])
        xx.append(x[ind[n], :])
    return yy, xx
