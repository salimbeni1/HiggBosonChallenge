import numpy as np


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Logistic regression with gradient descent algorithm."""
    
    w = initial_w
    w_t = []
    all_loss = []
    y_ = (y+1.)/2.
    for n_iter in range(max_iters):
        grad = compute_log_gradient(y_, tx, w)
        w = w - gamma * grad
        w_t.append(w)
        loss = compute_log_loss(y_, tx, w)
        all_loss.append(loss)
            
    return w, loss, w_t, all_loss

def compute_log_gradient(y, tx, w):
    """Compute the gradient for the logistic regression"""

    val = np.dot(tx, w)
    #y_ = np.expand_dims(y_, axis=1)
    grad = np.dot(tx.T, (sigma_fct(val)-y))
    
    return grad

def sigma_fct(val):
    """Sigma function"""
    fct = 1. / (np.exp(-val) + 1.)
    return fct

def compute_log_loss(y, tx, w):
    """Compute the cost function for the logistic regression"""
    
    val = np.dot(tx, w)
    loss = - np.dot(y, np.log(sigma_fct(val))) - np.dot((1-y), np.log(1-sigma_fct(val)))
    return loss
