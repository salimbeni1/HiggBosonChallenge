import numpy as np


def compute_mse_loss(y, tx, w):
    '''Computes MSE loss calculator'''
    loss = np.sum((y - np.dot(tx, w))**2)/(2*y.shape[0]) #compute loss
    return loss

    
def compute_residuals(y, tx, w):
    '''Compute residuals between the predicted regression and y'''
    e = y - np.dot(tx, w.T)    #compute residual
    return e    # returns residuals


def compute_grad(y, tx, w):
    '''Computes grad of the data'''
    grad = -np.dot(tx.T, compute_residuals(y, tx, w))/y.shape[0] #compute grad
    return grad # returns grad


def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    '''Implements GD method with step size gamma and max number of iterations given'''
    losses = np.zeros(max_iters) # initialize the matrix to record the losses
    w = initial_w 
    for n in range(max_iters):
        w +=  - gamma*compute_grad(y, tx, w) #take a step
        loss = compute_mse_loss(y, tx, w) #compute loss
        losses[n] = loss #record loss
    return w, losses[-1]    # return the weight matrix and the last loss


def least_squares_SGD(y, tx, initial_w,  max_iters, gamma, batch_size=1):
    '''This function iterates through the enitre data set with given minibatch size and repeats this max_iters number of times'''
    w = initial_w
    losses = np.zeros(max_iters)    # initialize the losses vector
    num_batches = round(y.shape[0]/batch_size)  # number of batches to run through 
    for n in range(max_iters):  # iterate max_iter number of times
        for minibatch_y, minibatch_x in batch_iter(y, tx, batch_size, num_batches): #return minibatch_y and minibatch_x
            grad = compute_grad(minibatch_y, minibatch_x, w)    #computed grad or the given minibatch
            w += - gamma*grad   #take a step 
        loss = compute_mse_loss(y, tx, w)   # compute loss
        losses[n] = loss    # record loss
    return w, losses[-1]    # return the last weight and the last loss


def least_squares(y, tx):
    '''Simple OLS function'''
    w = np.linalg.multi_dot([np.linalg.inv(tx.T.dot(tx)), tx.T, y]) # the definition of the w estimator
    loss = compute_mse_loss(y, tx, w)   # compute loss
    return w, loss  # returns the estimator w and the loss 



def ridge_regression(y, tx, lambda_):
    '''Implements ridge regression (similar to OLS)'''
    w = np.dot(np.linalg.inv(tx.T.dot(tx) + 2*y.shape[0]*lambda_*np.identity(tx.shape[1])), np.dot(tx.T, y))    # modified weight matrix for ridge regression
    loss = np.sum((y - np.dot(tx, w))**2)/(2*y.shape[0]) + lambda_*np.sum(w**2) #modified loss for ridge regression 
    return w, loss  

def sigma(tx, w):
    '''Define logistic function'''
    sigma = 1/(1+np.exp(np.dot(tx, w))) #definition of sigma function
    return sigma  

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    '''Implemensts logistic regression with GD method'''
    w = initial_w
    losses = np.zeros(max_iters)    # initialize matrix to record losses
    for n in range(max_iters):
        grad = np.dot(tx.T, (y - sigma(tx, w)))/y.shape[0] #grad of loss of logistic function
        w += -gamma*grad    #   take a step
        loss = -(np.dot(y, np.log(sigma(tx, w))) - np.dot((1 - y), (np.log(1 - sigma(tx, w)))))    # loss of logistic function
        losses[n] = loss    # record losses

    return w, losses[-1] # return the weight matrix and the last loss  


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

