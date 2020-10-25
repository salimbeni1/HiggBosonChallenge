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

def select_and_expand_f_logistic( arrF):
    """
        select main features and create new ones from them
        
        Args:
            arrF : (N,31) array of features , where first feature is the bias (arrF[:,0] == 1)
                    the order of the features must be the same of the csv file.
        
        Returns: 
            new_arr : (N,19) array , combination of selected and new features
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
    
    
    print("TEMP ->" , temp)
    
        
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





w_win = np.array([-5.14495295e-01,  5.49907306e-01, -4.18800249e-01,  3.36162915e-01,
        6.33857610e-01,  2.90461436e-02,  4.27674442e-01, -2.27555210e-01,
        9.24862998e-01, -2.58054094e-02,  2.67440080e-01, -2.26534382e-01,
        5.73489796e-01,  5.91819016e-01,  5.82547745e-01, -6.17603541e-03,
        9.57667317e-02, -1.37373973e-02,  3.27061207e-02,  9.74395747e-03,
       -8.64899880e-02, -3.70020468e-02, -2.80673659e-01,  1.40910813e-01,
        4.95758037e-01,  5.08916487e-02, -3.46353028e-02, -1.42761124e-03,
       -2.53187166e-02, -6.98361367e-02, -3.78655632e-01, -4.64208708e-03,
       -4.79108548e-02,  1.86499188e-01, -3.61626927e-01, -1.87612008e-01,
        2.80017028e-01,  5.59680615e-02, -6.54241397e-01, -6.36329919e-02,
       -3.59508933e-01,  1.16106591e-01, -3.09408136e-02, -1.30886665e-01,
       -8.85753998e-02, -8.17673244e-02, -5.94076755e-02, -1.83827139e-01,
        7.54384905e-02, -1.98739215e-02,  6.23656394e-02, -3.12522921e-02,
       -4.54796114e-02,  2.34481740e-01, -1.07093236e-01, -1.75419089e-02,
       -2.19792724e-02, -2.62060720e-02, -7.82241925e-02, -1.34094170e-01,
        4.67850741e-01,  3.69085451e-02, -2.71714565e-01, -4.49795563e-01,
        1.73889525e-01, -7.23881201e-02,  6.00290748e-03, -3.85481423e-01,
        6.83693291e-02,  1.16686096e-01,  4.49218171e-03,  1.50829617e-01,
       -7.94383621e-02,  7.49340619e-02, -2.16809307e-02,  3.63532129e-04,
        1.32398724e-01, -2.26841250e-03,  3.99501492e-03,  1.39573250e-01,
        9.66251316e-03,  3.86238455e-02,  3.34314371e-02,  4.73746128e-02,
        1.34705715e-02, -1.33932417e-02,  3.70844390e-02,  9.70029632e-03,
       -4.66007040e-02, -2.25856126e-01,  1.30981516e-01, -9.12077656e-02,
        1.53262008e-01, -4.77864579e-02,  2.06628309e-02, -9.13035025e-01,
        6.77350604e-02,  1.84851228e-01, -3.50932099e-03,  3.94492494e-01,
        2.01492879e-01, -1.42742860e-01,  1.08955029e-01,  5.45928249e-02,
       -1.01847615e-01, -1.34063183e-01,  1.55000867e-03,  2.08500754e-01,
       -1.33737210e-02, -1.23269158e-01, -6.15958678e-02, -1.90786025e-01,
        5.47835485e-02, -7.03709917e-02, -1.79940277e-02,  1.03852855e-01,
        1.39502224e-01, -2.60038787e-02,  1.77146993e-01, -1.79542860e-01,
        4.26245631e-01,  1.69578389e-01,  3.27253674e-01,  1.53629875e-01,
        3.55631306e-01,  1.57164750e-01,  5.61053646e-02,  1.67639930e-01,
       -5.00165649e-02, -1.41534453e-02, -2.95963085e-02, -1.92536563e-01,
       -6.79650906e-02,  8.14182649e-02, -2.29441067e-01,  2.94561582e-03,
       -9.18842675e-02,  2.16423648e-02, -3.38393714e-02,  5.62713059e-02,
       -2.71190817e-02,  1.32948772e-02,  7.51896271e-02, -4.89682491e-02,
        2.38915345e-02, -1.70669408e-01,  6.07315293e-01,  1.63130177e-01,
        6.06458714e-03, -1.15833105e-01, -3.31697484e-01,  9.35725107e-02,
        3.65516863e-03,  2.97653200e-01,  1.32536630e-01, -1.23849886e-02,
       -3.75353112e-03,  5.83361174e-02, -3.22612645e-02, -4.74964040e-02,
        2.02615955e-01, -3.24175002e-02, -1.23198027e-02, -1.45087618e-02,
        4.95055537e-02,  4.90956889e-02,  9.07143707e-03,  2.75506364e-01,
       -2.97394622e-02, -9.22621927e-02,  2.69311201e-01,  5.26968251e-02,
        4.89742089e-01,  2.34798326e-01,  1.07602322e-01, -2.46019421e-01,
        7.02706005e-03, -9.33699678e-02,  5.63020715e-01, -9.52146717e-03,
       -5.18970944e-02,  5.76030745e-02, -5.83011539e-02,  1.89911052e-02,
        9.18527588e-02, -2.71236548e-01,  1.16808714e-02, -1.13948878e-01,
       -2.63038863e-01,  1.57040535e-01,  1.52278928e-01,  6.16947222e-02,
       -8.53115869e-02,  4.81757043e-02,  1.30847371e-01, -1.64862809e-02,
        8.35557454e-02,  1.33356951e-01,  1.00210568e-02, -1.16565780e-01,
        3.31298745e-02, -6.46555340e-02,  1.25355377e-01,  4.29532332e-02,
       -2.18472673e-03,  3.04185161e-02, -4.42075156e-02,  1.06213788e-02,
        1.57430722e-02, -1.94268197e-02,  1.08949587e-03, -5.93811192e-02,
        1.77772563e-01,  1.40965021e-01,  9.51324258e-02,  2.10087619e-02,
        8.97828793e-02, -7.32169237e-02,  7.93722294e-03, -1.68577312e-01,
        4.10745143e-02,  2.80099464e-02,  4.72048159e-03, -2.37652201e-01,
       -4.14647235e-01,  1.79510685e-01,  1.00836001e-01, -5.78981542e-03,
        8.87294375e-03,  1.91132378e-01, -5.79113157e-03,  3.95988976e-02,
       -5.62497334e-01,  4.99862684e-02,  9.57032047e-02, -2.95496221e-01,
        4.78699687e-01,  1.26215907e-02,  9.38458736e-02,  3.69351076e-02,
       -7.64092577e-03,  1.42583844e-03, -2.96716618e-01,  1.59259897e-02,
       -8.59457832e-02,  6.16014497e-03, -1.25438584e-01, -1.13340565e-02,
        7.11217477e-03, -1.95326417e-02, -2.04196958e-03, -3.83245649e-02,
       -2.97472372e-03, -9.19971591e-03, -6.37412438e-02,  5.69212894e-03,
       -8.97513987e-02,  8.29959162e-02,  1.20845466e-02,  1.53731239e-02,
       -3.08818727e-02, -2.09162706e-03,  7.96496612e-03, -1.77523341e-02,
        4.47843948e-02, -5.29853716e-03, -2.74363839e-01, -2.35549940e-01,
       -2.55593132e-01,  3.79095585e-02, -1.01429368e-01, -1.09901141e-01,
       -2.55754716e-01, -3.55422305e-01, -5.71459381e-02, -4.54953280e-02,
       -5.64037909e-02, -5.01107852e-01, -6.05320426e-01,  3.45820395e-02,
        8.62734581e-02,  1.40439609e-01, -4.02376088e-01, -3.09281433e-01,
       -5.22045522e-02, -2.33346030e-01, -1.48867963e-02,  1.20733408e-01,
       -1.39662008e-02,  1.08802498e-01,  1.71538347e-02, -9.64202645e-03,
        4.20776750e-02, -4.32481637e-02, -8.74273090e-03, -6.17422944e-02,
       -3.89153405e-02,  7.27679735e-03,  1.65846415e-01,  4.16859122e-03,
        2.21932400e-02, -5.63716179e-02,  7.38892055e-02, -8.94841666e-03,
       -3.97799893e-02,  3.13132951e-02,  2.78376664e-02,  4.31327198e-02,
       -1.42079905e-01,  1.74127092e-02,  9.91882684e-03, -1.53964786e-01,
       -3.23350140e-02, -1.32427457e-02,  1.83783562e-01,  7.11032286e-03,
        7.95029500e-02,  5.81520307e-02,  9.10232408e-02, -2.20537537e-03,
        2.23641429e-03,  1.65855108e-01,  1.08917632e-02,  1.88259736e-02,
       -1.11219564e-01, -2.05052742e-01,  4.47590299e-02,  2.22549342e-02,
       -2.33290957e-03,  3.10984519e-02,  5.65502792e-03,  3.55341002e-02,
       -1.49096070e-02, -1.30244430e-02, -9.42636435e-02,  2.17893153e-01,
        4.74996103e-02, -3.21074526e-02, -2.41377945e-02,  6.50142181e-02,
       -1.16139398e-02,  2.20145905e-02, -1.72953828e-02, -4.37340785e-02,
        2.87052742e-02,  4.67228225e-03,  8.90726962e-02,  6.41180221e-02,
       -6.13946096e-03, -5.69640462e-02,  9.21200584e-04,  1.62937404e-01,
        1.16209153e-01,  1.52155032e-01, -4.19805735e-02, -2.98706497e-02,
        1.10780298e-01,  2.65264985e-02,  1.94122670e-02, -2.91090845e-01,
       -4.41561723e-01, -1.13470885e-02, -2.51852849e-02,  6.11777113e-01,
       -8.93725041e-03,  3.49696443e-02,  1.12025535e-02,  2.04155203e-02,
        4.34652984e-02,  6.21278402e-02, -5.44495499e-02, -6.97355541e-03,
        4.91653633e-02,  9.24110636e-02,  1.35971612e-02, -2.06487645e-02,
       -5.48801183e-03,  2.74882980e-02,  1.36068372e-02,  2.58395232e-03,
        2.36675314e-02,  2.44356438e-02,  6.09718316e-04,  4.73213513e-02,
        1.19701406e-01, -1.35400590e-03,  1.86838941e-02,  5.47311250e-02,
        2.10353937e-02,  2.22943525e-02, -5.95187927e-02,  5.66400140e-02,
        1.37670479e-01, -3.37625264e-03, -2.90191577e-02,  3.38492474e-02,
        1.70304275e-01,  5.89525315e-02,  1.78287316e-01, -5.07013255e-02,
        1.77768797e-02,  4.78535631e-02,  3.18057945e-02,  4.45204846e-02,
        1.19006460e-01, -4.82860996e-01,  9.97276757e-03, -7.47830870e-03,
       -9.48544090e-03, -2.23039511e-02,  5.26646286e-02,  7.08520893e-02,
       -1.82001981e-02, -1.81044551e-03,  1.51288979e-02, -4.72476589e-02,
       -2.09411167e-02,  2.61134322e-01, -1.13162828e-03, -1.35106943e-02,
        3.72222235e-03,  8.14270247e-03, -1.99614416e-02, -3.11821909e-02,
       -1.34674578e-02, -1.55689859e-02, -2.47721002e-02, -5.78352909e-04,
        1.04275603e-03,  6.19781172e-02,  1.54977157e-02, -5.19624894e-03,
        1.29215495e-02, -8.28313207e-02,  1.01439309e-01, -1.47479104e-02,
        2.47427538e-02,  2.38195213e-02, -2.87772252e-02,  7.97579395e-02,
       -1.47212209e-01, -3.09295709e-03,  1.40768100e-02,  9.43273569e-02,
        5.85055796e-02,  8.03659780e-03,  3.86199390e-02,  4.60603977e-03,
       -2.35848288e-02, -1.73977260e-03, -3.66618927e-02,  1.93404122e-02,
       -4.78214956e-02, -7.56229668e-02, -4.94439862e-04,  7.35456992e-02,
        2.69417327e-02,  6.76124904e-03, -2.92675071e-02,  5.47445662e-01,
       -1.88871182e-03,  2.31816712e-01,  5.10080068e-03, -1.62333795e-02,
        2.65938935e-01, -5.73641118e-02,  5.83595247e-02, -9.44706040e-02,
       -5.85095345e-02,  1.82602191e-02, -9.31470075e-03,  1.19042400e-01,
       -8.39457476e-02, -9.71369043e-02, -1.76050822e-01,  3.30025060e-01,
       -3.97270279e-03, -2.83619953e-02,  1.35936498e-01, -3.69495052e-03,
       -1.12810768e-01, -9.45899587e-03, -7.11184768e-02,  3.01498938e-02,
        3.01197478e-02, -8.17778630e-02,  2.33263576e-02, -4.05575923e-02,
       -6.45231524e-03,  1.26566396e-01,  2.39524668e-01, -3.23962126e-03,
        3.57421791e-01, -7.85339556e-03,  8.90613835e-02,  4.36446013e-01,
       -3.95818841e-01, -7.37778228e-02, -1.33494904e-01, -3.35919073e-02,
        7.71221172e-02, -1.73904698e-02,  1.96347025e-02,  2.68690058e-02,
        1.39362375e-02, -5.97544992e-02,  2.38172394e-02,  4.82000704e-02,
        1.84290720e-02, -8.98521063e-03,  4.37646774e-02,  3.75534677e-03,
       -3.90752380e-03,  1.06858235e-02, -8.74887448e-03,  2.00904677e-02,
        1.96844318e-03, -3.44103220e-02,  7.79666468e-02, -4.47612567e-02,
        5.58862256e-02,  1.74494104e-02, -1.91143039e-02,  1.03781515e-01,
       -1.55385820e-02, -2.95542117e-02])




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


def jet_number(x, y, id):
    ind = []
    yy = []
    xx = []
    iid = []
    for n in range(4):
        ind.append(np.where(x[:, 22] == n)[0])
        yy.append(y[ind[n]])
        xx.append(x[ind[n], :])
        iid.append(id[ind[n]])
    return yy, xx, iid
