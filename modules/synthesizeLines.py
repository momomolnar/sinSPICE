import numpy as np
import os
from scipy.interpolate import RectBivariateSpline

try:
    from .utils import Gaussian
except ImportError:  # Script drivers import from modules/ directly.
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
        import ChiantiPy.core as ch

        # G is per unit line-of-sight emission measure, integral(ne * nH * ds).
        ion = ch.ion(self.Ion, self.temp, self.ne[el], em=1.)
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
                                              for el in (range(neRange["numNes"])))
        self.GTne_table = np.array(self.GTne_table).T

        tmin = Trange["Tmin"]
        tmax = Trange["Tmax"]
        num_temperatures = Trange["numTemperatures"]
        ne_min = neRange["neMin"]
        ne_max = neRange["neMax"]
        num_nes = neRange["numNes"]
        output_file = (
            f"G_Tmin_max_num_{tmin}_{tmax}_{num_temperatures}_"
            + f"ne_{ne_min}_{ne_max}_{num_nes}"
        )

        np.savez(output_file, self.temp, self.ne, self.GTne_table)

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

        mass_dict = {"ne": 20.17, "fe":55,
                     "n":14, "o": 16, "c":12,
                     "s":32, "mg":24, "s":32}

        self.ionMass = mass_dict[ionDict["ion"]] * 1.66053906660e-24  # grams
        self.Ion = str(ionDict["ion"]) + "_"+ str(ionDict["IonLevel"])
        self.IonLevel = ionDict["IonLevel"]
        self.numWavelengths = ionDict["numWaves"]
        self.lambda0 = ionDict["lambda0"]
        self.dlambda = ionDict["dlambda"]
        self.loadGTables = loadGTables
        self.GTablesFile = GTablesFile


        self.c = 2.99792458e10  # cm/s
        self.kb = 1.380649e-16
        self.lambdaGrid = (self.lambda0
                           + (self.numWavelengths // 2
                              - np.arange(self.numWavelengths)) * self.dlambda)

        self.spectralLine = spectralLine(self.Ion,
                                         self.IonLevel,
                                         self.lambda0)
        self.spectralLine.def_GTnfn(self.loadGTables,
                                    self.GTablesFile)

    def compute_spectrum1D(self, T, ne, vz, *, rho, ds,
                           hydrogen_mass_fraction=0.74, observer="top"):
        """Integrate cell-centered emission in erg s^-1 cm^-2 sr^-1 Angstrom^-1.

        T is K, ne is cm^-3, rho is g cm^-3, and ds is the uniform cell
        length in cm. vz is native MURaM vx in cm/s, positive toward the
        top observer. nH counts hydrogen nuclei, including neutral hydrogen.
        """
        T, ne, vz, rho = [np.asarray(a, dtype=float) for a in (T, ne, vz, rho)]
        if T.ndim != 1 or any(a.shape != T.shape for a in (ne, vz, rho)):
            raise ValueError("Column inputs must be matching one-dimensional arrays.")
        if any(not np.all(np.isfinite(a)) for a in (T, ne, vz, rho)):
            raise ValueError("Column inputs must be finite.")
        if any(np.any(a <= 0) for a in (T, ne, rho)):
            raise ValueError("Temperature, electron density and mass density must be positive.")
        if not np.isfinite(ds) or ds <= 0:
            raise ValueError("ds must be a positive finite cell length in cm.")
        if not 0 < hydrogen_mass_fraction <= 1:
            raise ValueError("hydrogen_mass_fraction must be in (0, 1].")
        if observer not in ("top", "bottom"):
            raise ValueError("observer must be top or bottom.")

        n_h = hydrogen_mass_fraction * rho / (1.00794 * 1.66053906660e-24)
        emission_measure = ne * n_h * ds
        velocity_sign = -1 if observer == "top" else 1

        intensity = np.zeros(self.numWavelengths)
        numZpoints = len(T)

        for el in range(numZpoints):

            dopplerShift = velocity_sign * vz[el] / self.c * self.lambda0
            sigmaRadProfile = self.lambda0 / self.c * np.sqrt(self.kb * T[el] / self.ionMass)

            contribution = float(np.asarray(self.spectralLine.Gtne(T[el], ne[el])).item())
            if contribution > 0:
                lineProfile = Gaussian(self.lambdaGrid, (self.lambda0 + dopplerShift), sigmaRadProfile)
                intensity += contribution * emission_measure[el] * lineProfile

        return intensity


    def compute_spectrum_chunk(self, Temp, ne, vz, zClip, rho, ds,
                               hydrogen_mass_fraction, observer):
        return np.array([
            self.compute_spectrum1D(Temp[el, zClip[el]:],
                                    ne[el, zClip[el]:],
                                    vz[el, zClip[el]:], rho=rho[el, zClip[el]:], ds=ds,
                                    hydrogen_mass_fraction=hydrogen_mass_fraction,
                                    observer=observer)
            for el in range(Temp.shape[0])
        ])


    def compute_2Dspectrum(self, Temp, ne, vz, TempClip=2e4, n_jobs=1,
                           progress_label=None, *, rho, ds,
                           hydrogen_mass_fraction=0.74, observer="top"):
            """
            Compute the 2D emergent spectrum from a MHD cube

            Model with features:
            :param T: temperature (nx, ny, nz)
            :param ne: electron density (nx, ny, nz)
            :param vz: Doppler velocity (nx, ny, nz)
            :param rho: mass density (nx, ny, nz), g cm^-3
            :param ds: native vertical cell length dX[0], cm
            :return: I2 outgoing intensity (nx, ny, nLambda)
            """

            if Temp.ndim != 3 or any(a.shape != Temp.shape for a in (ne, vz, rho)):
                raise ValueError("Temperature, ne, vz and rho must be matching 3D cubes.")
            nx = Temp.shape[0]
            ny = Temp.shape[1]
            nz = Temp.shape[-1]

            nAtmos = nx * ny

            indexOffset = 150 # account for the convection zone high temp region
            zClip = np.argmin(np.abs(Temp[:, :, indexOffset:] - TempClip), axis=-1)
            zClip += indexOffset

            Temp = Temp.reshape(nx*ny, nz)
            ne   = ne.reshape(nx*ny, nz)
            vz   = vz.reshape(nx*ny, nz)
            rho  = rho.reshape(nx*ny, nz)
            zClip = zClip.reshape(nx*ny)

            if n_jobs == 0:
                raise ValueError("n_jobs must be nonzero")

            if n_jobs == 1:
                self.I2D = self.compute_spectrum_chunk(
                    Temp, ne, vz, zClip, rho, ds, hydrogen_mass_fraction, observer)
                return

            try:
                from .utils import joblib_progress
            except ImportError:
                from utils import joblib_progress

            if n_jobs == -1:
                chunkJobs = os.cpu_count() or 1
                parallelJobs = -1
            else:
                chunkJobs = min(abs(n_jobs), nAtmos)
                parallelJobs = chunkJobs

            nChunks = min(nAtmos, max(chunkJobs * 4, 1))
            chunkEdges = np.linspace(0, nAtmos, nChunks + 1, dtype=int)
            chunks = [
                (chunkEdges[el], chunkEdges[el + 1])
                for el in range(nChunks)
                if chunkEdges[el] < chunkEdges[el + 1]
            ]

            label = progress_label or "Synthesizing domain chunks"
            with joblib_progress(len(chunks), label=label):
                chunkSpectra = Parallel(n_jobs=parallelJobs)(
                    delayed(self.compute_spectrum_chunk)(
                        Temp[start:stop],
                        ne[start:stop],
                        vz[start:stop],
                        zClip[start:stop], rho[start:stop], ds,
                        hydrogen_mass_fraction, observer
                    )
                    for start, stop in chunks
                )
            self.I2D = np.concatenate(chunkSpectra, axis=0)
