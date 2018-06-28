import numpy as np


def sigmoid(z):

    g = np.zeros (z.shape)


    g = (1.0 / (1 + np.exp(-z)))

    return g