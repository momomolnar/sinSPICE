#Random utility functions for supporting the synthesis

import numpy as np

def Gaussian(x, mu, sigma):

    return 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - mu)/ sigma)**2)
