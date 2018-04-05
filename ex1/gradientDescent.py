import numpy as np
from computeCost import computeCost


def gradientDescent(X, y, theta, alpha, iterations):
    """
     Performs gradient descent to learn theta
       theta = gradientDescent(x, y, theta, alpha, num_iters) updates theta by
       taking num_iters gradient steps with learning rate alpha
    """

    # Initialize some useful values
    J_history = []

    #J_history = np.zeros (iterations)
    # number of training examples
    m = len(y)


    for i in range(iterations):
    #for iter in np.arange (num_iters):

        #   ====================== YOUR CODE HERE ======================
        # Instructions: Perform a single gradient step on the parameter vector
        #               theta.
        #
        # Hint: While debugging, it can be useful to print out the values
        #       of the cost function (computeCost) and gradient here.
        #
        #h = X.dot(theta)
        #theta = theta - alpha * (1 / m) * (X.T.dot(h-y))

        h = X.dot(theta)
        errors = h - y
        delta = X.T.dot(errors)
        theta -= (alpha / m) * delta

        #h = X.dot(theta)
        #theta = theta - alpha * (1 / m) * (X.dot (h - y))
        #J_history[iterations] = computeCost (X, y, theta)
        #return (theta, J_history)



        # ============================================================

        # Save the cost J in every iteration
        J_history.append(computeCost(X, y, theta))


        return theta, J_history
