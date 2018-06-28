#from ex2.costFunctionReg import costFunctionReg
import numpy as np
from sigmoid import sigmoid
# =============================================================


def lrCostFunction(theta, X, y, mylambda):
    """computes the cost of using
    theta as the parameter for regularized logistic regression and the
    gradient of the cost w.r.t. to the parameters.
    """
    m = y.size  # number of training examples
    J = 0
    h = sigmoid(np.dot (X, theta))


    #J = -(1.0 / m) * (np.sum (y.values.flatten () * np.log (h) + ((1 - y.values.flatten ()) * np.log (1 - h)))) + (
    #(mylambda / (2 * m) * (np.sum (theta[1:] ** 2))))

    # ====================== YOUR CODE HERE ======================
# Instructions: Compute the cost of a particular choice of theta.
#               You should set J to the cost.
#
# Hint: The computation of the cost function and gradients can be
#       efficiently vectorized. For example, consider the computation
#
#           sigmoid(X * theta)
#
#       Each row of the resulting matrix will contain the value of the
#       prediction for that example. You can make use of this to vectorize
#       the cost function and gradient computations. 
#

    J = -(1. / m) * ((y * np.transpose (np.log (sigmoid (np.dot (X, theta))))) + ((1 - y) * np.transpose (np.log (1 - sigmoid (np.dot (X, theta)))))).sum () + ((float (mylambda) / (2 * m)) * np.power (theta[1:theta.shape[0]], 2).sum ())

    grad = (1. / m) * np.dot (sigmoid (np.dot (X, theta)).T - y, X).T + (float (mylambda) / m) * theta

    # the case of j = 0 (recall that grad is a n+1 vector)
    nongrad = (1. / m) * np.dot (sigmoid (np.dot (X, theta)).T - y, X).T

    # and then assign only the first element of nongrad to grad
    grad[0] = nongrad[0]


    if return_grad:
        return J, grad.flatten ()
    else:
        return J
    #if np.isnan(J[0]):
       # return(np.inf)

    # =============================================================

    return (J[0])
