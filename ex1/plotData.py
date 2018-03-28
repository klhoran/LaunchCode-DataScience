import matplotlib.pyplot as plt
import numpy as np

def plotData(X, y):
    """
    plots the data points and gives the figure axes labels of
    population and profit.
    """

# ====================== YOUR CODE HERE ======================
# Instructions: Plot the training data into a figure using the
#               "figure" and "plot" commands. Set the axes labels using
#               the "xlabel" and "ylabel" commands. Assume the
#               population and revenue data have been passed in
#               as the x and y arguments of this function.
#
# Hint: You can use the 'rx' option with plot to have the markers
#       appear as red crosses. Furthermore, you can make the
#       markers larger by using plot(..., 'rx', 'MarkerSize', 10);
    plt.xlabel ("Population of City in 10,000s")
    plt.ylabel ("Profit in $10,000s")
    plt.plot (X, y, 'rx', markersize = 10)


    #plt.figure()  # open a new figure window

# ============================================================