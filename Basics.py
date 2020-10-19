import numpy as np

 

def compute_residuals(y, tx, w):
    '''Compute residuals between the predicted regression and y'''
    e = y - np.dot(tx, w.T)
    return e

def compute_grad(y, tx, w):
    '''Computes grad of the data'''
    grad = -np.dot(tx.T, compute_residuals(y, tx, w))/y.shape[0]
    return grad

def compute_mse_loss(y, tx, w):
    '''Computes MSE loss calculator'''
    loss = np.sum((y - np.dot(tx, w))**2)/(2*y.shape[0])
    return loss


######################################################################################

def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    '''Implements GD method with step size gamma and max number of iterations given'''
    losses = np.zeros(max_iters) # initialize the matrix to record the losses
    w = initial_w 
    for n in range(max_iters):
        w +=  -gamma * compute_grad(y, tx, w)  #take a step
        losses[n] = compute_mse_loss(y, tx, w) #compute loss
        
    return w, losses[-1]

def least_squares_SGD(y, tx, initial_w,  max_iters, gamma, batch_size=1):
    '''This function iterates through the enitre data set with given
       minibatch size and repeats this max_iters number of times'''
    w = initial_w
    losses = np.zeros(max_iters) # initialize the losses vector
    num_batches = round(y.shape[0]/batch_size)
    for n in range(max_iters):
        for minibatch_y, minibatch_x in batch_iter(y, tx, batch_size, num_batches):
            grad = compute_grad(minibatch_y, minibatch_x, w)  #computed grad on the given minibatch
            w += - gamma*grad                                 #take a step 
        losses[n] = compute_mse_loss(y, tx, w)   # compute loss
        
    return w, losses[-1] 

######################################################################################

def least_squares(y, tx):
    '''Simple OLS function'''
    w =  np.linalg.solve( tx.t @ tx, tx.T @ y)
    loss = compute_mse_loss(y, tx, w)
    
    return w, loss

######################################################################################

def ridge_regression(y, tx, lambda_):
    '''Implements ridge regression (similar to OLS)'''
    w = np.linalg.solve( (tx.T @ tx) + 2*tx.shape[0]*lambda_*np.identity(tx.shape[1]), tx.T @ y)
    # modified loss for ridge regression 
    loss = np.sum((y - np.dot(tx, w))**2)/(2*y.shape[0]) + lambda_*np.sum(w**2) 
    
    return w, loss  

######################################################################################

def sigma(tx, w):
    '''Define logistic function'''
    return 1/( 1+np.exp(-(tx @ w)) )  


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    '''Implemensts logistic regression with GD method'''
    w = initial_w
    losses = np.zeros(max_iters)    # initialize matrix to record losses
    for n in range(max_iters):
        grad = tx.T @ (sigma(tx, w) - y) # grad of loss of logistic function
        w += -gamma*grad
        losses[n] = 1 
        # -(np.dot(y, np.log(sigma(tx, w))) - np.dot((1 - y), (np.log(1 - sigma(tx, w)))))

    return w, losses[-1]


######################################################################################


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    '''Implement regularized logistic regression with GD method'''

    w = initial_w   
    losses = np.zeros(max_iters) #initialize matrix to record losses
    for n in range(max_iters):
        grad = np.dot(tx.T, (y - sigma(tx, w))) + lambda_*np.linalg.norm(w)*w #grad of loss of logistic function
        w += -gamma*grad/y_shape[0]    # take a step
        loss = -(np.dot(y, np.log(sigma(tx, w))) - np.dot((1 - y), np.log(1 - sigma(tx, w)))) + lambda_*np.sum(w**2)/2  # computes loss
        losses[n] = loss  # stores losses
        
    return w, losses[-1]

