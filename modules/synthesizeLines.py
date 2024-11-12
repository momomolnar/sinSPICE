import numpy as np
import scipy as sp
from scipy.interpolate import RectBivariateSpline

import ChiantiPy
import ChiantiPy.core as ch
import ChiantiPy.tools.filters as chfilters
import ChiantiPy.tools.io as chio
from sympy.physics.units.definitions.unit_definitions import gauss
from tqdm import tqdm
from utils import Gaussian

from joblib import Parallel, delayed

class spectralLine():
    """
    Define a class containing all the line attributes needed to synthesize coronal
    lines from CHIANTI
    """

    def __init__(self, ion, level, lineWavelength):
        self.Ion = ion
        self.level = level
        self.wvl = lineWavelength

    def def_GTnfn(self, TRange={}, neRange={}, loadGtable=False, gSaveFile=None):
        """ Interpolation function for the computation of the emergent intensity"""

        if loadGtable == False:
            TRange = {"Tmin":4, "Tmax":7.5, "numTemperatures":100}
            neRange = {"neMin":5, "neMax":12, "numNes":80}

            self.computeGtables(TRange, neRange)

        if loadGtable == True:
            self.temp, self.ne, self.GTne_table = np.load(gSaveFile)

        self.Gtne = RectBivariateSpline(self.temp, self.ne, self.GTne_table)


    def computeGTne_single(self, el):
        ion = ch.ion(self.Ion, self.temp, self.ne[el])
        ion.intensity()

        dist = np.abs(np.asarray(ion.Intensity['wvl']) - self.wvl)
        idx = np.argmin(dist)

        GTne = ion.Intensity['intensity'][:, idx]

        return GTne



    def computeGtables(self, Trange, neRange):
        """
        Compute tables for interpolation of the G(T,n)

        :param Trange: {Tmin, Tmax, numTemperatures} in log10 scale
        :param neRange: [neMin, neMax, numNes] in log 10 scale
        :return: table of GTne
        """
        self.temp = 10**(Trange["Tmin"]
                         + (Trange["Tmax"]
                            - Trange["Tmin"])
                         * np.arange(Trange["numTemperatures"]) / Trange["numTemperatures"])
        self.ne = 10**(neRange["neMin"]
                       + (neRange["neMax"]
                          - neRange["neMin"])
                       * np.arange(neRange["numNes"]) / neRange["numNes"])

        # self.GTne_table = np.zeros((Trange["numTemperatures"],
        #                             neRange["numNes"]))

        self.GTne_table = Parallel(n_jobs=40)(delayed(self.computeGTne_single)(el)
                                              for el in tqdm(range(neRange["numNes"])))
        self.GTne_table = np.array(self.GTne_table).T

        np.savez((f"G_Tmin_max_num_{Trange["Tmin"]}_{Trange["Tmax"]}_{Trange["numTemperatures"]}_"
                  + f"ne_{neRange["neMin"]}_{neRange["neMax"]}_{neRange["numNes"]}"),
                  self.temp, self.ne, self.GTne_table)

    def compute_G(self, T, ne):
        """
        Compute the G(T,ne) value for the line of interest
        :param T: temperature, K
        :param ne: electron density, cm-3
        :return: G(T, ne)

        """

        return self.Gtne(T, ne)

class spectrum():
    """
    Main class of the exercise that synthesizes the lines
    """

    def __init__(self, ionDict, loadGTables=False, GTablesFile=None):

        mass_dict = {"ne": 10, "fe":26, "n":7, "o": 8, "c":12,
                     "s":16}

        self.ionMass = mass_dict[ionDict["ion"]] * 1.66e-27
        self.Ion = str(ionDict["ion"]) + "_"+ str(ionDict["IonLevel"])
        self.IonLevel = ionDict["IonLevel"]
        self.numWavelengths = ionDict["numWaves"]
        self.lambda0 = ionDict["lambda0"]
        self.dlambda = ionDict["dlambda"]
        self.loadGTables = loadGTables
        self.GTablesFile = GTablesFile


        self.c = 3e10
        self.kb = 1.380649e-23
        self.lambdaGrid = (self.lambda0
                           + (self.numWavelengths // 2
                              - np.arange(self.numWavelengths)) * self.dlambda)

        self.spectralLine = spectralLine(self.Ion,
                                         self.IonLevel,
                                         self.lambda0)
        self.spectralLine.def_GTnfn(self.loadGTables,
                                    self.GTablesFile)

    def compute_spectrum1D(self, T, ne, vz):
        """ Function to compute 1D spectrum from a column"""

        intensity = np.zeros(self.numWavelengths)
        numZpoints = len(T)

        for el in range(numZpoints):

            dopplerShift = vz[el] /self.c * self.lambda0
            thermalLineWidth = 2 * np.sqrt(np.log(2)
                                           * (self.lambda0 / self.c)**2
                                           * (2 * self.kb * T[el] / self.ionMass))
            sigmaRadProfile = thermalLineWidth / 2.355

            if self.spectralLine.Gtne(T[el],ne[el]) > 0:
                lineProfile = Gaussian(self.lambdaGrid, (self.lambda0 + dopplerShift), sigmaRadProfile)
                Icontrib = self.spectralLine.Gtne(T[el],ne[el]) * lineProfile
            else:
                Icontrib = np.zeros((1, self.numWavelengths))

            intensity += Icontrib[0, :]
        return intensity


    def compute_2Dspectrum(self, Temp, ne, vz, TempClip=3e4):
            """
            Compute the 2D emergent spectrum from a MHD cube

            Model with features:
            :param T: temperature (nx, ny, nz)
            :param ne: electron density (nx, ny, nz)
            :param vz: Doppler velocity (nx, ny, nz)
            :return: I2 outgoing intensity (nx, ny, nLambda)
            """

            nx = Temp.shape[0]
            ny = Temp.shape[1]
            nz = Temp.shape[-1]

            indexOffset = 150 # account for the convection zone high temp region
            zClip = np.argmin(np.abs(Temp[:, :, indexOffset:] - TempClip), axis=-1)
            zClip += indexOffset

            Temp = Temp.reshape(nx*ny, nz)
            ne   = ne.reshape(nx*ny, nz)
            vz   = vz.reshape(nx*ny, nz)
            zClip = zClip.reshape(nx*ny)

            self.I2D = np.zeros((nx*ny, self.numWavelengths))

            self.I2D = Parallel(n_jobs=49)(delayed(self.compute_spectrum1D)(Temp[el, zClip[el]:],
                                                                            ne[el, zClip[el]:],
                                                                            vz[el, zClip[el]:])
                                           for el in tqdm(range(nx*ny)))
            self.I2D = np.array(self.I2D)

