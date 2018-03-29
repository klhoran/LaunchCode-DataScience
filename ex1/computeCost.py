import numpy as np


def computeCost(X, y, theta):
    """
       computes the cost of using theta as the parameter for linear 
       regression to fit the data points in X and y
    """
    #m = y.size
    m = len(y)
    J = 0

    # ====================== YOUR CODE HERE ======================
    # Instructions: Compute the cost of a particular choice of theta
    #               You should set J to the cost.

    h = np.dot(X, theta)
    J = sum((h - y)** 2) / (2 * m)




    # =========================================================================

    return J
