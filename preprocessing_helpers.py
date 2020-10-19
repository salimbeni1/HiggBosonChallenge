import numpy as np
from tqdm import tqdm
import time
import sys


def standardize(x):
    
    centered_data = x - np.mean(x, axis=0)
    std_data = centered_data / np.std(centered_data, axis=0)
    
    return std_data , np.mean(x, axis=0) , np.std(centered_data, axis=0)





def select_and_expand_f( arrF):
    """
        select main features and create new ones from them
        
        Args:
            arrF : (N,31) array of features , where first feature is the bias (arrF[:,0] == 1)
                    the order of the features must be the same of the csv file.
                    ( only tested on stardardize data , with -999 and outliers )
        
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


def sigma(x, w):
    '''Define logistic function'''
    sigma = 1/(1+np.exp(np.dot(x, w)))
    return sigma


def predict_logistic_labels(xtest, w):
    '''Changed version of predict labels function to account for the fact
    that logistic regression outputs answers between 0 and 1 and not -1 and 1'''
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


def reg_logistic_regression(y, x, w, max_iters, gamma, ytest, xtest, lmbd):
    '''Implement regularized logistic regression with GD method'''

    P = list(ytest).count(1)   # number of 1s in test set
    N = list(ytest).count(-1)  # number of -1s in test set

    k = 200     # number of steps over which we observe loss to make sure it becomes smaller
    losses = np.zeros(max_iters)    # initialize matrix to record losses
    y = (y + 1)/2   # our labels are -1, 1 need to convert them to 0 and 1 for logistic regression

    i = 0
    for n in tqdm(range(max_iters)):
        if i <= k:
            grad = np.dot(x.T, (y - sigma(x, w)))/y.shape[0] + lmbd*np.linalg.norm(w)*w #grad of loss of logistic function
            w += -gamma*grad    # take a step

            loss = -(np.dot(y, np.log(sigma(x, w))) - np.dot((1 - y), np.log(1 - sigma(x, w)))) / y.shape[0] + \
                   lmbd * np.sum(w ** 2) / 2
            losses[n] = loss  # stores losses

            tp, fp, tn, fn = logistic_accuracy(ytest, xtest, w) # outputs confusion matrix

            sys.stdout.write('TN={0} TP={1} Accuracy={2}\r'.format("{:.2f}%".format(100*tn/N), '{: .2f}%'.format(100*tp/P), '{: .2f}%'.format(100*(tp+tn)/(N+P))))
            sys.stdout.flush()
            time.sleep(0.001)

        else:
            if losses[n-1] <= 0.9*losses[n-k]:  # if loss improves keep looping with the same gamma
                pass
            else:
                gamma = 0.9*gamma # if not make the step smaller
                print('\ngamma={0}'.format(gamma))
                i = 0

            grad = np.dot(x.T, (y - sigma(x, w)))/y.shape[0] + lmbd*np.linalg.norm(w)*w #grad of loss of logistic function
            w += -gamma*grad    # take a step

            loss = -(np.dot(y, np.log(sigma(x, w))) - np.dot((1 - y), np.log(1 - sigma(x, w))))/y.shape[0] + \
                   lmbd*np.sum(w**2)/2
            losses[n] = loss

            tp, fp, tn, fn = logistic_accuracy(ytest, xtest, w)

            sys.stdout.write('TN={0} TP={1} Accuracy={2}\r'.format("{:.2f}%".format(100*tn/N), '{: .2f}%'.format(100*tp/P), '{: .2f}%'.format(100*(tp+tn)/(N+P))))
            sys.stdout.flush()
            time.sleep(0.001)

      i += 1
    return w, losses, tp, fp, tn, fn  # return the weight matrix and loss


def shuffle(x, y):
    shuffle_indices = np.random.permutation(np.arange(y.shape[0]))
    y = y[shuffle_indices]  # rearranges the Y_train based on the shuffled indices
    x = x[shuffle_indices]  # rearranges the X)train based on the shuffled indices
    return x, y


def under_over(x, y, alpha=1, upsample=True, middle=True, gaussian=False, std=0.1, downsample=False):
    '''This function allows us to either undersample the labels -1 or upsample the labels1. There are 2 ways in which
    upsampling can happen. First it takes the middle values of the adjacent point or takes the existing standartized
    data and add gaussian noice with a mean 0 and provided std. Argument alpha allows us to choose what proportion of
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



