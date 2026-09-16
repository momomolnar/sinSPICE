"""Regression tests for the MURaM readers."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import warnings

import numpy as np

from modules.dataLoader import MURaMData
from modules.muram import MuramSnap


class MuramReaderTests(unittest.TestCase):
    def test_derives_density_and_electrons_from_pressure_and_temperature(self):
        with TemporaryDirectory() as folder:
            shape = (3, 2, 1)
            np.savetxt(Path(folder)/"Header.000005",
                       [*shape, 2e6, 3e6, 5e6, 42., .1, 1e8])
            temperature = np.linspace(4e5, 9e5, np.prod(shape),
                                      dtype=np.float32).reshape(shape)
            pressure = np.linspace(0.05, 0.3, np.prod(shape),
                                   dtype=np.float32).reshape(shape)
            velocity = np.full(shape, 3e6, dtype=np.float32)
            for prefix, values in (("eosT", temperature), ("eosP", pressure),
                                   ("result_prim_1", velocity)):
                np.asarray(values.T, dtype=np.float32).tofile(
                    Path(folder)/f"{prefix}.000005"
                )

            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                model = MuramSnap(folder, 5)

            n_h = pressure/(2.3*1.380649e-16*temperature)
            expected_rho = 1.4271*1.66053906660e-24*n_h
            expected_ne = 1.2*n_h
            np.testing.assert_allclose(np.asarray(model.rho), expected_rho, rtol=3e-7)
            np.testing.assert_allclose(np.asarray(model.ne), expected_ne, rtol=3e-7)
            np.testing.assert_allclose(model.ne.T[0], expected_ne.T[0], rtol=3e-7)
            self.assertEqual(model.rho_source, "eosP_eosT_fully_ionized")
            self.assertEqual(model.ne_source, "rho_fully_ionized")
            self.assertEqual(model.derived, ["rho", "ne"])
            self.assertEqual(len(caught), 2)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                loaded = MURaMData(folder, 5)
                loaded.loadModel()
            np.testing.assert_allclose(np.asarray(loaded.rho), expected_rho.T,
                                       rtol=3e-7)
            np.testing.assert_allclose(np.asarray(loaded.ne), expected_ne.T,
                                       rtol=3e-7)

    def test_derives_missing_electron_density_from_mass_density(self):
        with TemporaryDirectory() as folder:
            shape = (3, 2, 2)
            np.savetxt(Path(folder)/"Header.000007",
                       [*shape, 2e6, 3e6, 5e6, 42., .1, 1e8])
            rho = np.arange(1, np.prod(shape) + 1, dtype=np.float32).reshape(shape)*1e-14
            np.asarray(rho.T, dtype=np.float32).tofile(
                Path(folder)/"result_prim_0.000007"
            )
            for prefix, value in (("eosT", 6e5), ("result_prim_1", 3e6)):
                np.full(shape[::-1], value, dtype=np.float32).tofile(
                    Path(folder)/f"{prefix}.000007"
                )

            with self.assertWarnsRegex(RuntimeWarning, "deriving ne from rho"):
                model = MuramSnap(folder, 7)

            expected = 1.2*rho/(1.4271*1.66053906660e-24)
            np.testing.assert_allclose(np.asarray(model.ne), expected, rtol=2e-7)
            np.testing.assert_allclose(model.ne.T[::2], expected.T[::2], rtol=2e-7)
            np.testing.assert_allclose(model.ne.reshape(3, 4)[1],
                                       expected.reshape(3, 4)[1], rtol=2e-7)
            self.assertEqual(model.ne_source, "rho_fully_ionized")
            self.assertEqual(model.ne.nbytes, rho.nbytes)
            self.assertIn("ne", model.available)
            self.assertIn("ne", model.derived)
            self.assertNotIn("ne", model.unavailable)

            strict_model = MuramSnap(folder, 7, derive_ne=False)
            self.assertFalse(hasattr(strict_model, "ne"))
            self.assertIn("ne", strict_model.unavailable)

            with self.assertWarns(RuntimeWarning):
                loaded = MURaMData(folder, 7)
                loaded.loadModel()
            np.testing.assert_allclose(np.asarray(loaded.ne), expected.T, rtol=2e-7)
            self.assertEqual(loaded.ne.nbytes, expected.nbytes)

    def test_electron_density_file_takes_precedence(self):
        with TemporaryDirectory() as folder:
            shape = (2, 2, 1)
            np.savetxt(Path(folder)/"Header.000003",
                       [*shape, 2e6, 3e6, 5e6, 42., .1, 1e8])
            rho = np.full(shape, 1e-14, dtype=np.float32)
            ne = np.arange(1, np.prod(shape) + 1, dtype=np.float32).reshape(shape)*1e10
            for prefix, values in (("result_prim_0", rho), ("eosne", ne)):
                np.asarray(values.T, dtype=np.float32).tofile(
                    Path(folder)/f"{prefix}.000003"
                )

            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                model = MuramSnap(folder, 3)

            self.assertEqual(caught, [])
            np.testing.assert_array_equal(model.ne, ne)
            self.assertEqual(model.ne_source, "eosne")
            self.assertEqual(model.derived, [])


if __name__ == "__main__":
    unittest.main()
