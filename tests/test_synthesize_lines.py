"""Physical regressions for the original emission pipeline."""

import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
from scipy.integrate import trapezoid

from modules.dataLoader import MURaMData
from modules.synthesizeLines import spectralLine, spectrum


class ConstantContribution:
    def __call__(self, temperature, density):
        return np.array([[2e-24]])


def make_spectrum():
    # These tests isolate transport from the expensive atomic-table generation.
    with patch.object(spectralLine, "def_GTnfn"):
        result = spectrum(dict(ion="c", IonLevel=3, lambda0=977.,
                               numWaves=4001, dlambda=.0005))
    result.spectralLine.Gtne = ConstantContribution()
    return result


def moments(s, profile):
    w, p = s.lambdaGrid[::-1], profile[::-1]
    area = trapezoid(p, w)
    center = trapezoid(p*w, w)/area
    sigma = np.sqrt(trapezoid(p*(w-center)**2, w)/area)
    return area, center, sigma


class EmissionTests(unittest.TestCase):
    def test_absolute_intensity_width_and_cell_subdivision(self):
        s = make_spectrum()
        kwargs = dict(rho=[1e-14], ds=4e7)
        p = s.compute_spectrum1D([5e4], [1e10], [0.], **kwargs)
        area, center, sigma = moments(s, p)
        expected_nh = .74 * 1e-14 / (1.00794 * 1.66053906660e-24)
        self.assertAlmostEqual(area/(2e-24 * 1e10 * expected_nh * 4e7), 1., places=10)
        self.assertAlmostEqual(center, 977., places=10)
        expected_sigma = 977./2.99792458e10 * np.sqrt(
            1.380649e-16*5e4/(12*1.66053906660e-24))
        self.assertAlmostEqual(sigma/expected_sigma, 1., places=10)
        split = s.compute_spectrum1D([5e4]*8, [1e10]*8, [0.]*8,
                                     rho=[1e-14]*8, ds=5e6)
        np.testing.assert_allclose(split, p, rtol=1e-13, atol=1e-300)

    def test_density_composition_and_length_scaling(self):
        s = make_spectrum()
        base = dict(T=[5e4], ne=[1e10], vz=[0.], rho=[1e-14], ds=1e7,
                    hydrogen_mass_fraction=.74)
        p = s.compute_spectrum1D(**base)
        for changes, factor in ((dict(ne=[2e10]), 2), (dict(rho=[3e-14]), 3),
                                (dict(ds=4e7), 4), (dict(hydrogen_mass_fraction=.37), .5)):
            with self.subTest(changes=changes):
                np.testing.assert_allclose(s.compute_spectrum1D(**(base | changes)),
                                           factor*p, rtol=1e-13, atol=1e-300)

    def test_observer_sign(self):
        s = make_spectrum()
        for observer, expected in (("top", -30.), ("bottom", 30.)):
            p = s.compute_spectrum1D([5e4], [1e10], [30e5], rho=[1e-14],
                                     ds=1e7, observer=observer)
            center = moments(s, p)[1]
            self.assertAlmostEqual((center/977.-1)*299792.458, expected, places=8)

    def test_reader_and_parallel_clipped_rectangular_domain(self):
        with TemporaryDirectory() as folder:
            # Native axes (vertical x, horizontal y, horizontal z).
            shape = (154, 3, 2)
            np.savetxt(Path(folder)/"Header.000007", [*shape, 2e6, 3e6, 5e6, 42., .1, 1e8])
            temp = np.full(shape, 5e4)
            rho = np.broadcast_to((np.arange(154)+1)[:, None, None]*1e-16, shape)
            for y in range(3):
                for z in range(2):
                    temp[150+(y+z)%3, y, z] = 2e4
            for prefix, values in (("eosT", temp), ("eosne", np.full(shape, 1e10)),
                                   ("result_prim_0", rho), ("result_prim_1", np.full(shape, 30e5))):
                np.asarray(values.T, dtype=np.float32).tofile(Path(folder)/f"{prefix}.000007")
            model = MURaMData(folder, 7)
            model.loadModel()
            self.assertEqual(model.ds, 2e6)
            np.testing.assert_allclose(model.rho, rho.T)
            s = make_spectrum()
            expected = []
            for z in range(2):
                for y in range(3):
                    start = 150+(y+z)%3
                    nh_column = .5*model.rho[z, y, start:].astype(float)/(1.00794*1.66053906660e-24)
                    expected.append(2e-24*1e10*nh_column.sum()*2e6)
            for jobs in (1, 2):
                s.compute_2Dspectrum(model.Temp, model.ne, model.vz, rho=model.rho,
                                     ds=model.ds, n_jobs=jobs,
                                     hydrogen_mass_fraction=.5, observer="bottom")
                self.assertEqual(s.I2D.shape, (6, 4001))
                actual = [moments(s, p) for p in s.I2D]
                np.testing.assert_allclose(np.array(actual)[:, 0], expected, rtol=1e-12)
                np.testing.assert_allclose((np.array(actual)[:, 1]/977.-1)*299792.458,
                                           30., atol=1e-8)

    def test_missing_or_invalid_physical_inputs_fail(self):
        s = make_spectrum()
        with self.assertRaises(TypeError):
            s.compute_spectrum1D([5e4], [1e10], [0.])
        for kwargs in (dict(rho=[-1.], ds=1.), dict(rho=[1.], ds=0.),
                       dict(rho=[1.], ds=1., observer="side"),
                       dict(rho=[1.], ds=1., hydrogen_mass_fraction=np.nan)):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                s.compute_spectrum1D([5e4], [1e10], [0.], **kwargs)

    @unittest.skipUnless(os.environ.get("SINSPICE_TEST_CHIANTI") == "1", "Requires CHIANTI")
    def test_real_chianti_emission_measure_normalization(self):
        import ChiantiPy.core as ch

        s = make_spectrum()
        s.spectralLine.temp = np.array([5e4, 5e4])
        s.spectralLine.ne = np.array([1e10])
        coefficient = s.spectralLine.computeGTne_single(0)[0]
        s.spectralLine.Gtne = lambda t, ne: np.array([[coefficient]])
        area = moments(s, s.compute_spectrum1D([5e4], [1e10], [0.],
                                               rho=[1e-14], ds=2e6))[0]
        em = 1e10 * (.74*1e-14/(1.00794*1.66053906660e-24)) * 2e6
        ion = ch.ion("c_3", temperature=[5e4, 5e4], eDensity=1e10, em=em)
        ion.intensity()
        index = np.argmin(abs(ion.Intensity["wvl"]-977.))
        self.assertAlmostEqual(area/ion.Intensity["intensity"][0, index], 1., places=10)


if __name__ == "__main__":
    unittest.main()
