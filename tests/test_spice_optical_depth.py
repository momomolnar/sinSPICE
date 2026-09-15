"""Physics/reader regression tests; optional tests use the installed CHIANTI data.

python -m unittest discover -s tests -v
SINSPICE_TEST_CHIANTI=1 python -m unittest discover -s tests -v
"""

import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import numpy as np
from scipy.integrate import trapezoid

from compute_spice_optical_depth import make_grid, parser, run
from modules.spiceOpticalDepth import (
    AMU, C, H, KB, MASS_U, M_H, LINE_BY_KEY, SPICE_LINES, PopulationTable,
    build_population_table, integrate_columns, load_snapshot, select_components,
    synthesize_snapshot, validate_block,
)


COMPONENT = dict(lower=1, upper=2, g_lower=1., g_upper=3.,
                 wavelength=977., a_value=1.769e9)


def write_snapshot(directory, shape=(4, 3, 2)):
    """Noncubic fixture in native (x vertical, y, z) indexing with unequal spacings."""
    np.savetxt(Path(directory)/"Header.000007", [*shape, 2e6, 3e6, 5e6, 42., .1, 1e8])
    rho = np.broadcast_to((np.arange(shape[0]) + 1)[:, None, None]*1e-14, shape)
    for prefix, values in (("eosT", np.full(shape, 5e4)),
                           ("eosne", np.full(shape, 1e10)),
                           ("result_prim_0", rho),
                           ("result_prim_1", np.full(shape, 3e6))):
        np.asarray(values.T, dtype=np.float32).tofile(Path(directory)/f"{prefix}.000007")
    return rho


class PhysicsTests(unittest.TestCase):
    def slab(self, temperature=5e4, velocity=0., ds=1e7, heights=4, upper=0., micro=0.):
        t = np.full((1, heights), temperature)
        n_h = np.full_like(t, 1e10)
        pop = np.broadcast_to([1e-4, upper], (*t.shape, 2))
        wavelengths = 977.*(1+np.linspace(-150, 150, 1201)*1e5/C)
        out = integrate_columns(t, n_h, np.full_like(t, velocity), pop, [COMPONENT],
                                [1, 2], wavelengths, ds, MASS_U["c"]*AMU, micro)
        return wavelengths, out

    def test_slab_line_center_and_path_length(self):
        _, (tau, _, static) = self.slab()
        b = np.sqrt(2*KB*5e4/(MASS_U["c"]*AMU))
        lam = 977e-8
        expected = lam**3 * 1.769e9 * 3 * (1e6 * 4e7) / (8*np.pi**1.5*b)
        np.testing.assert_allclose(tau[0, 600], expected, rtol=1e-13)
        np.testing.assert_allclose(static, expected, rtol=1e-13)
        # Cell splitting leaves the physical column invariant, including a one-cell slab.
        for heights, ds in ((1, 4e7), (8, 5e6)):
            np.testing.assert_allclose(self.slab(heights=heights, ds=ds)[1][0], tau, rtol=1e-13)

    def test_thermal_and_microturbulent_widths(self):
        tau = self.slab()[1][0]
        hotter = self.slab(temperature=2e5)[1][0]
        self.assertAlmostEqual(tau[0, 600]/hotter[0, 600], 2.)
        b_kms = np.sqrt(2*KB*5e4/(MASS_U["c"]*AMU))/1e5
        broadened = self.slab(micro=b_kms)[1][0]
        self.assertAlmostEqual(tau[0, 600]/broadened[0, 600], np.sqrt(2))

    def test_stimulated_emission_and_inversion(self):
        _, (tau, _, _) = self.slab(upper=1.5e-4)
        np.testing.assert_allclose(tau, .5*self.slab()[1][0], rtol=1e-13)
        np.testing.assert_allclose(self.slab(upper=3e-4)[1][0], 0, atol=1e-14)
        self.assertLess(self.slab(upper=4e-4)[1][0][0, 600], 0)

    def test_velocity_shift_and_static_diagnostic(self):
        w, (tau, _, static) = self.slab(velocity=40e5)
        self.assertAlmostEqual(w[tau.argmax()], 977*(1+40e5/C), places=10)
        self.assertLess(tau[0, 600], static[0]*.01)
        np.testing.assert_allclose(static, self.slab()[1][2])

    def test_profile_normalization_and_emission_units(self):
        w, (tau, intensity, _) = self.slab(upper=1e-6)
        # Integral of j_lambda equals h nu A N_upper/(4 pi), in CGS.
        expected_i = H*C/(977e-8)/(4*np.pi)*1.769e9*(1e10*1e-6*4e7)
        self.assertAlmostEqual(trapezoid(intensity[0], w)/expected_i, 1., places=7)
        expected_tau_area = (977e-8)**2/(8*np.pi)*1.769e9*(3e6-1e4)*4e7
        self.assertAlmostEqual(-trapezoid(tau[0], C/(w*1e-8))/expected_tau_area, 1., places=7)

    def test_multiplet_additivity(self):
        t = np.full((1, 2), 5e4)
        args = (t, np.ones_like(t)*1e10, np.zeros_like(t), np.ones((*t.shape, 2))*1e-5)
        a = integrate_columns(*args, [COMPONENT], [1, 2], [977.], 1e7, 12*AMU)
        b = integrate_columns(*args, [COMPONENT, COMPONENT], [1, 2], [977.], 1e7, 12*AMU)
        for single, double in zip(a, b):
            np.testing.assert_allclose(double, 2*single)


class ReaderAndSelectionTests(unittest.TestCase):
    def test_reader_axes_spacing_and_observer_sign(self):
        with TemporaryDirectory() as folder:
            rho = write_snapshot(folder)
            views, ds, info = load_snapshot(folder, 7, downsample=2, height_start=1, height_stop=3)
            self.assertEqual(views["temperature"].shape, (1, 2, 2))
            self.assertEqual(ds, 2e6)
            self.assertEqual(info["time_s"], 42.)
            np.testing.assert_allclose(views["rho"], rho.T[::2, ::2, 1:3])
            with self.assertRaises(ValueError):
                load_snapshot(folder, 7, height_stop=10)
            table = PopulationTable(np.array([4., 6.]), np.array([9., 11.]),
                                    np.broadcast_to([1e-4, 1e-6], (2, 2, 2)),
                                    dict(levels=[1, 2], components={"c_3_977": [COMPONENT]}))
            velocities = np.linspace(-100, 100, 201)
            top, _ = synthesize_snapshot(views, ds, LINE_BY_KEY["c_3_977"], table, velocities, chunk_columns=1)
            bottom, _ = synthesize_snapshot(views, ds, LINE_BY_KEY["c_3_977"], table, velocities, observer="bottom")
            self.assertEqual(top["tau2D"].shape, (1, 1, 2, 201))
            self.assertEqual(velocities[top["tau2D"][0, 0, 0].argmax()], -30)
            self.assertEqual(velocities[bottom["tau2D"][0, 0, 0].argmax()], 30)
            np.testing.assert_allclose(top["tau_static"], bottom["tau_static"])

    def test_strict_table_bounds_and_invalid_cells(self):
        table = PopulationTable(np.array([4., 6.]), np.array([9., 11.]), np.ones((2, 2, 1)), {})
        block = dict(temperature=np.array([1e5]), ne=np.array([1e10]), rho=np.array([1e-14]), velocity=np.array([0.]))
        validate_block(block, table)
        for key, value in (("temperature", 1e3), ("ne", 1e13), ("rho", -1.), ("velocity", np.nan)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_block(dict(block, **{key: np.array([value])}), table)
        with self.assertRaises(ValueError):
            table.evaluate(np.array([1e3]), np.array([1e10]))

    def test_observed_components_only_and_missing_line(self):
        ion = SimpleNamespace(Wgfa=dict(wvl=[-977.03, 977.02, 977.04, 1000.],
                                       avalue=[1e9]*4, lvl1=[1]*4, lvl2=[2]*4),
                              Elvlc=dict(lvl=[1, 2], mult=[1, 3]))
        cs = select_components(ion, LINE_BY_KEY["c_3_977"], .1)
        self.assertEqual([c["wavelength"] for c in cs], [977.02, 977.04])
        with self.assertRaises(ValueError):
            select_components(ion, LINE_BY_KEY["ne_8_770"], .1)

    def test_invalid_cli_grids_and_missing_files_fail_before_chianti(self):
        for grid in ([5, 4, 2], [4, 5, 2.5], [4, 5, float("nan")]):
            with self.assertRaises(ValueError):
                make_grid(grid)
        with TemporaryDirectory() as folder:
            args = parser().parse_args(["--model-dir", folder, "--iterations", "7"])
            with self.assertRaises(FileNotFoundError):
                run(args)


@unittest.skipUnless(os.environ.get("SINSPICE_TEST_CHIANTI") == "1", "Set SINSPICE_TEST_CHIANTI=1 for real atomic tests")
class ChiantiTests(unittest.TestCase):
    def test_table1_transitions_resolve(self):
        import ChiantiPy.tools.io as io
        for line in SPICE_LINES:
            with self.subTest(line=line.key):
                ion = SimpleNamespace(Wgfa=io.wgfaRead(line.ion), Elvlc=io.elvlcRead(line.ion))
                components = select_components(ion, line, .15)
                self.assertTrue(components)
                self.assertTrue(all(c["f_effective"] > 0 for c in components))

    def test_ciii_cache_populations_and_cli(self):
        with TemporaryDirectory() as folder:
            args = parser().parse_args([
                "--model-dir", folder, "--iterations", "7", "--lines", "c_3_977",
                "--log-t", "4", "6", "3", "--log-ne", "9", "11", "3",
                "--cache-dir", folder+"/tables", "--output-dir", folder+"/output",
                "--num-waves", "101", "--velocity-range", "100", "--plot"])
            write_snapshot(folder)
            run(args)
            self.assertTrue((Path(folder)/"output/spice_opacity_c_3_977_000007_tau_peak.png").is_file())
            cache_path = next((Path(folder)/"tables").glob("*.npz"))
            before = cache_path.stat().st_mtime_ns
            table = build_population_table([LINE_BY_KEY["c_3_977"]], make_grid(args.log_t),
                                           make_grid(args.log_ne), args.abundance, args.cache_dir)
            self.assertEqual(cache_path.stat().st_mtime_ns, before)
            import ChiantiPy.core as ch
            ion = ch.ion("c_3", temperature=1e5, eDensity=1e10, abundance=args.abundance)
            ion.populate()
            expected = ion.Abundance * np.asarray(ion.IoneqOne).item() * np.atleast_2d(ion.Population["population"])[0, np.array(table.metadata["levels"])-1]
            np.testing.assert_allclose(table.evaluate(np.array([1e5]), np.array([1e10]))[0], expected)
            with np.load(Path(folder)/"output/spice_opacity_c_3_977_000007.npz", allow_pickle=False) as output:
                self.assertEqual(output["tau2D"].shape, (1, 2, 3, 101))
                self.assertTrue(np.all(np.isfinite(output["I2D"])))
                self.assertGreater(output["tau_peak"].max(), 0)
                self.assertEqual(json.loads(str(output["metadata"]))["snapshot"]["time_s"], 42.)
            with self.assertRaises(FileExistsError):
                run(args)


if __name__ == "__main__":
    unittest.main()
