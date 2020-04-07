"""
Do not change the input and output format.
If our script cannot run your code or the format is improper, your code will not be graded.

The only functions you need to implement in this template is linear_regression_noreg, linear_regression_invertible，regularized_linear_regression,
tune_lambda, test_error and mapping_data.
"""

import numpy as np
import pandas as pd


###### Q1.1 ######
def mean_absolute_error(w, X, y):
    """
    Compute the mean absolute error on test set given X, y, and model parameter w.
    Inputs:
    - X: A numpy array of shape (num_samples, D) containing test feature.
    - y: A numpy array of shape (num_samples, ) containing test label
    - w: a numpy array of shape (D, )
    Returns:
    - err: the mean absolute error
    """

    # TODO 1: Fill in your code here #
    y_pred = np.dot(X, w)
    err = np.sum(np.abs(y_pred - y)) / len(y)

    return err


###### Q1.2 ######
def linear_regression_noreg(X, y):
    """
    Compute the weight parameter given X and y.
    Inputs:
    - X: A numpy array of shape (num_samples, D) containing feature.
    - y: A numpy array of shape (num_samples, ) containing label
    Returns:
    - w: a numpy array of shape (D, )
    """
    # TODO 2: Fill in your code here #
    X_trans = X.transpose()
    inverse = np.linalg.inv(np.dot(X_trans, X))
    w = inverse.dot(X_trans).dot(y)
    return w


###### Q1.3 ######
def linear_regression_invertible(X, y):
    """
    Compute the weight parameter given X and y.
    Inputs:
    - X: A numpy array of shape (num_samples, D) containing feature.
    - y: A numpy array of shape (num_samples, ) containing label
    Returns:
    - w: a numpy array of shape (D, )
    """
    # TODO 3: Fill in your code here #
    X_trans = X.transpose()
    mul = np.dot(X_trans, X)
    n = len(mul)
    entr = pow(10, -5)
    while np.min(np.linalg.eigvals(mul)) < entr:
        mul = mul + 0.1 * np.eye(n)
    inverse = np.linalg.inv(mul)
    w = inverse.dot(X_trans).dot(y)
    return w


###### Q1.4 ######
def regularized_linear_regression(X, y, lambd):
    """
    Compute the weight parameter given X, y and lambda.
    Inputs:
    - X: A numpy array of shape (num_samples, D) containing feature.
    - y: A numpy array of shape (num_samples, ) containing label
    - lambd: a float number containing regularization strength
    Returns:
    - w: a numpy array of shape (D, )
    """
    # TODO 4: Fill in your code here #
    mul = np.dot(X.transpose(), X)
    n = len(mul)
    inverse = np.linalg.inv(mul + lambd * np.eye(n))
    w = inverse.dot(X.transpose()).dot(y)
    return w


###### Q1.5 ######
def tune_lambda(Xtrain, ytrain, Xval, yval):
    """
    Find the best lambda value.
    Inputs:
    - Xtrain: A numpy array of shape (num_training_samples, D) containing training feature.
    - ytrain: A numpy array of shape (num_training_samples, ) containing training label
    - Xval: A numpy array of shape (num_val_samples, D) containing validation feature.
    - yval: A numpy array of shape (num_val_samples, ) containing validation label
    Returns:
    - bestlambda: the best lambda you find in lambds
    """
    #####################################################

    # TODO 5: Fill in your code here #

    min_mae = 1
    bestlambda = pow(10, -19)

    for x in range(-19, 20):
        lambd = pow(10, x)
        w = regularized_linear_regression(Xtrain, ytrain, lambd)
        mae = mean_absolute_error(w, Xval, yval)
        if mae < min_mae:
            min_mae = mae
            bestlambda = lambd
    return bestlambda


###### Q1.6 ######
def mapping_data(X, power):
    """
    Mapping the data.
    Inputs:
    - X: A numpy array of shape (num_training_samples, D) containing training feature.
    - power: A integer that indicate the power in polynomial regression
    Returns:
    - X: mapped_X, You can manually calculate the size of X based on the power and original size of X
    """
    #####################################################
    # TODO 6: Fill in your code here #
    mapped_X = np.copy(X)

    if power > 1:
        pow_mat = np.copy(X)
        for x in range(1, power):
            pow_mat = pow_mat * X
            for y in range(np.size(pow_mat, 1)):
                mapped_X = np.insert(mapped_X, -1, pow_mat[:, y], axis=1)

    return mapped_X


