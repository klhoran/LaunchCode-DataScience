import numpy as np


def computeCost(X, y, theta):
    """
       computes the cost of using theta as the parameter for linear 
       regression to fit the data points in X and y
    """

    J = 0
    m = y.size
    # ====================== YOUR CODE HERE ======================
    # Instructions: Compute the cost of a particular choice of theta
    #               You should set J to the cost.

    J = sum((X * theta - y)** 2) / (2 * m)
    # =========================================================================

    return J
