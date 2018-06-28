#from costFunction import costFunction
import numpy as np
from sigmoid import sigmoid

def costFunctionReg(theta, X, y, mylambda):
    """
    Compute cost and gradient for logistic regression with regularization

    computes the cost of using theta as the parameter for regularized logistic regression and the
    gradient of the cost w.r.t. to the parameters.
    """
    # Initialize some useful values
    m = y.size   # number of training examples
    J = 0
# ====================== YOUR CODE HERE ======================
# Instructions: Compute the cost of a particular choice of theta.
#               You should set J to the cost.
#               Compute the partial derivatives and set grad to the partial
#               derivatives of the cost w.r.t. each parameter in theta

# =============================================================
    #J=(costFunction(theta, X, y)) +((mylambda/(2*m) *(np.sum(theta[1:]**2))))

    h = sigmoid(np.dot (X, theta))

    #J = -(1 / m) * (np.sum (y * np.log (h) + ((1 - y) * np.log (1 - h))))

    J = -(1.0 / m) * (np.sum (y.values.flatten() * np.log(h) + ((1 - y.values.flatten()) * np.log(1 - h))))+((mylambda/(2*m) *(np.sum(theta[1:]**2))))


    #return (J.flatten())
    return J