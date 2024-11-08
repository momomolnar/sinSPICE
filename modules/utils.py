#Random utility functions for supporting the synthesis

import numpy as np

def Gaussian(x, A, mu, sigma):

    return A / (sigma * np.sqrt(2 * np.pi)) * np.exp((x - mu) / (2 * sigma **2))
