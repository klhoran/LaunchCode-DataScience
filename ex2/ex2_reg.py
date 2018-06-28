# Logistic Regression
from matplotlib import use

use('TkAgg')
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

import pandas as pd

from ml import mapFeature, plotData, plotDecisionBoundary
from show import show
from costFunctionReg import costFunctionReg
from gradientFunctionReg import gradientFunctionReg
from sigmoid import sigmoid


def optimize(mylambda):

    result = minimize(costFunctionReg, theta, method='L-BFGS-B',
               jac=gradientFunctionReg, args=(X.as_matrix(), y, mylambda),
               options={'gtol': 1e-4, 'disp': False, 'maxiter': 1000})

    return result


# Plot Boundary
def plotBoundary(theta, X, y):
    plotDecisionBoundary(theta, X.values, y.values)
    plt.title(r'$\mylambda$ = ') + (mylambda)

    # Labels and Legend
    plt.xlabel('Microchip Test 1')
    plt.ylabel('Microchip Test 2')
    plt.show()



# Initialization

# Load Data
#  The first two columns contains the X values and the third column
#  contains the label (y).

data = pd.read_csv('ex2data2.txt', header=None, names=[1,2,3])
X = data[[1, 2]]
y = data[[3]]

plotData(X.values, y.values)

# Labels and Legend
plt.xlabel('Microchip Test 1')
plt.ylabel('Microchip Test 2')
plt.show()
input("Program paused. Press Enter to continue...")


# =========== Part 1: Regularized Logistic Regression ============

# Add Polynomial Features

# Note that mapFeature also adds a column of ones for us, so the intercept
# term is handled
X = X.apply(mapFeature, axis=1)

# Initialize fitting parameters
theta = np.zeros(X.shape[1])

# Set regularization parameter lambda to 1
mylambda = 0.0

# Compute and display initial cost and gradient for regularized logistic
# regression
cost = costFunctionReg(theta, X, y, mylambda)

print('Cost at initial theta (zeros): %f' % cost)

# ============= Part 2: Regularization and Accuracies =============

# Optimize and plot boundary

mylambda = 1.0
result = optimize(mylambda)
theta = result.x
cost = result.fun

# Print to screen
print('lambda = ' + str(mylambda))
print('Cost at theta found by scipy: %f' % cost)
print('theta:', ["%0.4f" % i for i in theta])

input("Program paused. Press Enter to continue...")

plotBoundary(X, y, theta)

# Compute accuracy on our training set
p = np.round(sigmoid(X.dot(theta)))
acc = np.mean(np.where(p == y.T,1,0)) * 100
print('Train Accuracy: %f' % acc)

input("Program paused. Press Enter to continue...")

# ============= Part 3: Optional Exercises =============


for l in np.arange(0.0,10.1,1.0):
    result = optimize(mylambda)
    theta = result.x
    print('lambda = ' + str(mylambda))
    print('theta:', ["%0.4f" % i for i in theta])
    plotBoundary(theta, X, y)
input("Program paused. Press Enter to continue...")
