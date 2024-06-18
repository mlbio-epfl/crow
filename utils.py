from scipy.optimize import linear_sum_assignment
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def accuracy(output, target):

    num_correct = np.sum(output == target)
    res = num_correct / len(target)

    return res

def cluster_acc(y_pred, y_true):
    """
    Calculate clustering accuracy. Require scikit-learn installed
    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        accuracy, in [0,1]
    """
    
    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)
    assert y_pred.size == y_true.size
    if y_pred.size == 0:
        return torch.zeros(1) - 1

    D = max(y_pred.max(), y_true.max()) + 1
    
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    row_ind, col_ind = linear_sum_assignment(w.max() - w)

    return w[row_ind, col_ind].sum() / y_pred.size

def cluster_acc_2(y_pred_truemask, y_true_truemask, seen_num):
    """
    This function is used to calculate the unseen accuracy.

    The main difference to the founction cluster_acc is 
    this function will directly consider the samples from 
    sunseen but being classified into seen to be worng cases.
    """

    y_pred_truemask=y_pred_truemask.astype(np.int64)
    y_true_truemask=y_true_truemask.astype(np.int64)

    assert y_pred_truemask.size == y_pred_truemask.size
    if y_pred_truemask.size == 0:
        return torch.zeros(1).item()
    
    D = max(y_pred_truemask.max(), y_true_truemask.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred_truemask.size):
        if y_pred_truemask[i] > seen_num:
            w[y_pred_truemask[i], y_true_truemask[i]] += 1
    row_ind, col_ind = linear_sum_assignment(w.max() - w)

    return w[row_ind, col_ind].sum() / y_true_truemask.size    

def entropy(x):

    EPS = 1e-8
    x_ =  torch.clamp(x, min = EPS)
    b =  x_ * torch.log(x_)

    if len(b.size()) == 2: # Sample-wise entropy
        return - b.sum(dim = 1).mean()
    elif len(b.size()) == 1: # Distribution-wise entropy
        return - b.sum()
    else:
        raise ValueError('Input tensor is %d-Dimensional' %(len(b.size())))