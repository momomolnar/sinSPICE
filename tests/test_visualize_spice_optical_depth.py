"""Check physical coordinates, edge cases, and plotting from saved maps alone."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from visualize_spice_optical_depth import main, make_tau_peak_figure, save_tau_peak_map


def metadata(shape=(4, 5, 7), stride=2):
    return dict(snapshot=dict(raw_shape=shape, cell_spacing_cm=[2e6, 3e6, 5e6],
                              downsample=stride, iteration=7, time_s=42.),
                line=dict(ion="c_3", wavelength=977.03),
                diagnostics=dict(reference_wavelength_A=977.02))


class MapTests(unittest.TestCase):
    def test_nonsquare_map_preserves_axes_stride_and_physical_extent(self):
        tau = np.arange(1., 13.).reshape(1, 4, 3)
        figure = make_tau_peak_figure(tau, metadata())
        try:
            axes = figure.axes[0]
            mesh = axes.collections[0]
            np.testing.assert_array_equal(mesh.get_array().reshape(4, 3), tau[0])
            edges = mesh.get_coordinates()
            np.testing.assert_allclose(edges[0, :, 0], [0., .045, .105, .15])
            np.testing.assert_allclose(edges[:, 0, 1], [0., .075, .175, .275, .35])
            np.testing.assert_allclose(axes.get_xlim(), [0., .15])
            np.testing.assert_allclose(axes.get_ylim(), [0., .35])
            self.assertEqual(axes.get_xlabel(), "x [Mm]")
            self.assertEqual(axes.get_ylabel(), "y [Mm]")
        finally:
            figure.clear()

    def test_zero_and_negative_values_are_black_in_default_log_colors(self):
        tau = np.array([[[0., -2., 3.], [np.nan, 10., 1.]]])
        info = metadata(shape=(4, 3, 2), stride=1)
        for scale, expected in (("log", [[False, False, False], [True, False, False]]),
                                ("linear", [[False, False, False], [True, False, False]])):
            figure = make_tau_peak_figure(tau, info, scale=scale)
            try:
                np.testing.assert_array_equal(np.ma.getmaskarray(figure.axes[0].collections[0].get_array()).reshape(2, 3), expected)
                figure.canvas.draw()
            finally:
                figure.clear()
        with TemporaryDirectory() as folder:
            path = Path(folder)/"empty.png"
            save_tau_peak_map(np.zeros((1, 1, 1)), metadata(shape=(2, 1, 1), stride=1), path)
            self.assertGreater(path.stat().st_size, 1000)

    def test_default_grayscale_limits_and_colorbar_height(self):
        tau = np.array([[[1e-4, 1e-3, 1.], [5., 10., 0.]]])
        figure = make_tau_peak_figure(tau, metadata(shape=(4, 3, 2), stride=1))
        try:
            figure.canvas.draw()
            axes, colorbar_axes = figure.axes
            mesh = axes.collections[0]
            self.assertEqual(mesh.cmap.name, "gray")
            self.assertEqual((mesh.norm.vmin, mesh.norm.vmax), (1e-3, 5.))
            np.testing.assert_allclose(mesh.cmap(mesh.norm(1e-4))[:3], [0., 0., 0.])
            np.testing.assert_allclose(mesh.cmap(mesh.norm(5.))[:3], [1., 1., 1.])
            np.testing.assert_allclose(mesh.cmap(mesh.norm(10.))[:3], [1., 1., 1.])
            map_box = axes.get_position()
            bar_box = colorbar_axes.get_position()
            self.assertAlmostEqual(bar_box.y0, map_box.y0, places=5)
            self.assertAlmostEqual(bar_box.height, map_box.height, places=5)
        finally:
            figure.clear()

    def test_saved_map_only_cli_and_collision(self):
        with TemporaryDirectory() as folder:
            source = Path(folder)/"spice_opacity_c_3_977_000007.npz"
            # There are deliberately no spectra or simulation inputs in this fixture.
            np.savez(source, tau_peak=np.ones((1, 4, 3)), metadata=json.dumps(metadata()))
            main([str(source), "--plot-limits", "0.01", "100"])
            output = source.with_name(source.stem + "_tau_peak.png")
            with output.open("rb") as stream:
                self.assertEqual(stream.read(8), b"\x89PNG\r\n\x1a\n")
            with self.assertRaises(FileExistsError):
                main([str(source)])
            with self.assertRaises(ValueError):
                main([str(source), "--plot-limits", "-1", "100"])


if __name__ == "__main__":
    unittest.main()
