import numpy as np
import scipy as sp

import ChiantiPy
import ChiantiPy.core as ch
import ChiantiPy.tools.filters as chfilters
import ChiantiPy.tools.io as chio

from utils import Gaussian

class spectralLine():
    """
    Define a class containing all the line attributes needed to synthesize coronal
    lines from CHIANTI
    """

    def __init__(self, ion, level, lineWavelength):
        self.Ion = ion
        self.level = level
        self.wvl = lineWavelength

    def G_tables(self, Trange, neRange):
        """
        Compute tables for interpolation of the G(T,n)

        :param Trange: [Tmin, Tmax, numTemperatures]
        :param neRange: [neMin, neMax, numNes]
        :return: table of GTne
        """



    def def_GTnfn(self):
        """ Interpolation function for the computation of the emergent intensity"""

    def computeGTne(self, T, ne):
        """
        Compute the G(T,ne) value for the line of interest
        :param T: temperature, K
        :param ne: electron density, cm-3
        :return: G(T, ne)

        """


class spectrum():
    """
    Main class of the exercise that synthesizes the lines
    """

    def __init__(self, ion, level, lineWavelength):
        self.Ion = ion
        self.level = level
        self.wvl = lineWavelength

        self.spectralLine = spectralLine(ion, level , lineWavelength)


    def lineProfile(self, spLine, numWvlPoints, temperature):
