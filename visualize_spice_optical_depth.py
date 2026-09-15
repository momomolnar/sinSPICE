#!/usr/bin/env python3
"""Plot horizontal maps of peak column optical depth from sinSPICE NPZ files.

The plotted quantity is tau_peak = max_wavelength(integral_LOS alpha_lambda ds).
Plot x is native MURaM y; plot y is native MURaM z. Native x is the LOS.
Only NumPy and Matplotlib are needed; no simulation cubes or CHIANTI are loaded.
"""

import argparse
import json
from pathlib import Path

import numpy as np


DEFAULT_LOG_LIMITS = (1e-5, 5.0)


def add_plot_arguments(parser):
    parser.add_argument("--plot-scale", choices=("log", "linear"), default="log",
                        help="Color normalization for tau_peak")
    parser.add_argument("--plot-limits", nargs=2, type=float, metavar=("MIN", "MAX"),
                        help="Fixed color limits in optical depth, useful for comparing snapshots")
    parser.add_argument("--plot-cmap", default="gray", help="Matplotlib colormap")
    parser.add_argument("--plot-dpi", type=int, default=180, help="PNG resolution")


def validate_plot_options(scale="log", limits=None, cmap="gray", dpi=180):
    import matplotlib

    if scale not in ("log", "linear") or dpi <= 0:
        raise ValueError("Plot scale must be log or linear, and DPI must be positive.")
    if limits is not None:
        if (len(limits) != 2 or not np.all(np.isfinite(limits)) or limits[0] >= limits[1]
                or (scale == "log" and limits[0] <= 0)):
            raise ValueError("Plot limits require finite MIN < MAX and MIN > 0 for logarithmic colors.")
    if cmap not in matplotlib.colormaps:
        raise ValueError(f"Unknown Matplotlib colormap: {cmap}")


def horizontal_coordinates(snapshot, map_shape):
    """Return x/y centers and display edges in Mm, preserving native ray positions.

    Edges lie halfway between sampled columns, clipped to the domain boundaries.
    This displays nearest-column values, including non-divisible downsampling.
    """
    shape = np.asarray(snapshot["raw_shape"])
    spacing = np.asarray(snapshot["cell_spacing_cm"], dtype=float)
    stride = snapshot["downsample"]
    if (shape.shape != (3,) or not np.all(np.isfinite(shape)) or np.any(shape < 1)
            or np.any(shape != shape.astype(int)) or spacing.shape != (3,)
            or not np.all(np.isfinite(spacing)) or np.any(spacing <= 0)
            or not np.isfinite(stride) or stride < 1 or stride != int(stride)):
        raise ValueError("Invalid raw_shape, cell_spacing_cm, or downsample in snapshot metadata.")
    centers, edges = [], []
    for axis in (1, 2):
        positions = (np.arange(0, int(shape[axis]), int(stride)) + .5) * spacing[axis] / 1e8
        centers.append(positions)
        edges.append(np.r_[0., (positions[:-1] + positions[1:]) / 2,
                           shape[axis] * spacing[axis] / 1e8])
    if tuple(map_shape) != (len(centers[1]), len(centers[0])):
        raise ValueError(f"tau_peak shape {map_shape} disagrees with snapshot geometry; "
                         f"expected {(len(centers[1]), len(centers[0]))} (native z, native y).")
    return centers[0], centers[1], edges[0], edges[1]


def make_tau_peak_figure(tau_peak, metadata, scale="log", limits=None, cmap="gray"):
    """Make a Matplotlib figure using a noninteractive canvas, suitable for HPC."""
    from matplotlib import colormaps
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.colors import LogNorm, Normalize
    from matplotlib.figure import Figure

    validate_plot_options(scale, limits, cmap)
    tau = np.asarray(tau_peak, dtype=float)
    if tau.ndim == 3 and tau.shape[0] == 1:
        tau = tau[0]
    if tau.ndim != 2 or 0 in tau.shape:
        raise ValueError("tau_peak must have shape (native_z, native_y) or (1, native_z, native_y).")
    snapshot = metadata["snapshot"]
    x, y, x_edges, y_edges = horizontal_coordinates(snapshot, tau.shape)
    finite = np.isfinite(tau)
    visible = finite
    if limits is not None:
        vmin, vmax = limits
    elif scale == "log":
        vmin, vmax = DEFAULT_LOG_LIMITS
    elif np.any(visible):
        vmin, vmax = float(tau[visible].min()), float(tau[visible].max())
        if vmin == vmax:
            if scale == "log":
                vmin, vmax = vmin / 10, vmax * 10
            else:
                padding = abs(vmin) * .1 if vmin else 1.
                vmin, vmax = vmin - padding, vmax + padding
    else:
        vmin, vmax = (1e-3, 1.) if scale == "log" else (0., 1.)
    norm = LogNorm(vmin=vmin, vmax=vmax) if scale == "log" else Normalize(vmin=vmin, vmax=vmax)
    colors = colormaps[cmap].copy()
    # LogNorm cannot represent zero or negative values.  Put those finite values
    # at the lower bound, while values in (0, vmin) use the under color.
    if scale == "log":
        plotted_values = np.where(finite & (tau <= 0), vmin, tau)
    else:
        plotted_values = tau
    plotted = np.ma.array(plotted_values, mask=~finite)
    colors.set_under("black")
    colors.set_over("white")
    colors.set_bad("lightgray")
    figure = Figure(figsize=(8, 6.7))
    FigureCanvasAgg(figure)
    axes = figure.subplots()
    figure.subplots_adjust(left=.12, right=.80, bottom=.12, top=.88)
    mesh = axes.pcolormesh(x_edges, y_edges, plotted, shading="flat", cmap=colors, norm=norm,
                          rasterized=True)
    axes.set(xlabel="x [Mm]", ylabel="y [Mm]", xlim=(x_edges[0], x_edges[-1]),
             ylim=(y_edges[0], y_edges[-1]), aspect="equal")
    line = metadata["line"]
    element, stage = line["ion"].split("_")
    roman = ("", "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X",
             "XI", "XII", "XIII", "XIV", "XV", "XVI", "XVII", "XVIII", "XIX", "XX")
    ion_label = f"{element.capitalize()} {roman[int(stage)] if 0 < int(stage) < len(roman) else stage}"
    wavelength = metadata.get("diagnostics", {}).get("reference_wavelength_A", line["wavelength"])
    axes.set_title(f"{ion_label} {wavelength:.3f} Å — maximum column optical depth\n"
                   f"Iteration {snapshot['iteration']:06d}, t = {snapshot['time_s']:g} s")
    notes = []
    if len(x) > 1 and len(y) > 1 and np.any(finite) and tau[finite].min() < 1 < tau[finite].max():
        axes.contour(x, y, np.ma.masked_invalid(tau), levels=[1.], colors="white", linewidths=.8)
        notes.append(r"White contour: $\tau_{\mathrm{peak}}=1$")
    if np.any(~visible):
        notes.append(f"Gray: {np.count_nonzero(~visible)} nonfinite pixels")
    if not np.any(visible):
        axes.text(.5, .5, "No finite optical depths",
                  transform=axes.transAxes, ha="center", va="center")
    if notes:
        figure.supxlabel("; ".join(notes), fontsize=9)
    # The equal-aspect map may be much shorter than the allocated subplot.  Draw
    # it once, then give the colorbar the exact data-axes height.
    figure.canvas.draw()
    box = axes.get_position()
    colorbar_axes = figure.add_axes([box.x1 + .025, box.y0, .035, box.height])
    colorbar = figure.colorbar(mesh, cax=colorbar_axes)
    if scale == "log" and (vmin, vmax) == DEFAULT_LOG_LIMITS:
        colorbar.set_ticks([1e-3, 1e-2, 1e-1, 1., 5.])
        colorbar.set_ticklabels([r"$10^{-3}$", r"$10^{-2}$", r"$10^{-1}$", "1", "5"])
    colorbar.set_label(r"Peak column optical depth $\tau_{\mathrm{peak}}$ (dimensionless)")
    return figure


def save_tau_peak_map(tau_peak, metadata, output_file, scale="log", limits=None,
                     cmap="gray", dpi=180, overwrite=False):
    validate_plot_options(scale, limits, cmap, dpi)
    output_file = Path(output_file)
    if output_file.exists() and not overwrite:
        raise FileExistsError(f"{output_file} exists; use --overwrite or another output directory.")
    figure = make_tau_peak_figure(tau_peak, metadata, scale, limits, cmap)
    try:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with output_file.open("wb" if overwrite else "xb") as stream:
            figure.savefig(stream, format="png", dpi=dpi)
    finally:
        figure.clear()
    return output_file


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_files", type=Path, nargs="+", help="One or more spice_opacity_*.npz results")
    parser.add_argument("--output-dir", type=Path, help="Default: save each PNG beside its input NPZ")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing PNG files")
    add_plot_arguments(parser)
    args = parser.parse_args(argv)
    validate_plot_options(args.plot_scale, args.plot_limits, args.plot_cmap, args.plot_dpi)
    for input_file in args.input_files:
        # NPZ loading is lazy: the large tau2D and I2D cubes are never read here.
        with np.load(input_file, allow_pickle=False) as saved:
            tau_peak = saved["tau_peak"]
            metadata = json.loads(str(saved["metadata"]))
        output = (args.output_dir or input_file.parent) / f"{input_file.stem}_tau_peak.png"
        save_tau_peak_map(tau_peak, metadata, output, args.plot_scale, args.plot_limits,
                          args.plot_cmap, args.plot_dpi, args.overwrite)
        print(f"Saved {output}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, KeyError, OSError, ImportError) as error:
        raise SystemExit(f"Error: {error}") from error
