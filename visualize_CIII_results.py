#!/usr/bin/env python3
"""Create 3-panel diagnostic frames from sinSPICE synthesis output."""

import argparse
import re
import sys
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize


PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR / "modules"))

from utils import ProgressBar


C_KMS = 299792.458


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Make 3-panel PNG frames from an example_CIII.py .npz file."
    )
    parser.add_argument("input_file", type=Path, help="Saved .npz file from example_CIII.py")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for PNG frames. Defaults to <input stem>_frames.",
    )
    parser.add_argument(
        "--lambda0",
        type=float,
        default=None,
        help="Rest wavelength in Angstrom. Defaults to filename value or wvlGrid center.",
    )
    parser.add_argument(
        "--line-label",
        default="C III 977 A",
        help="Label used in figure titles.",
    )
    parser.add_argument(
        "--prefix",
        default=None,
        help="Output filename prefix. Defaults to <input stem>_3panel.",
    )
    parser.add_argument("--t-start", type=int, default=0, help="First timestep index to plot.")
    parser.add_argument(
        "--t-stop",
        type=int,
        default=None,
        help="Stop timestep index, exclusive. Defaults to all timesteps.",
    )
    parser.add_argument(
        "--time-start",
        type=int,
        default=0,
        help="Physical/model timestep label for the first saved index.",
    )
    parser.add_argument(
        "--time-step",
        type=int,
        default=1,
        help="Physical/model timestep label spacing between saved indices.",
    )
    parser.add_argument(
        "--width-kind",
        choices=("fwhm", "sigma"),
        default="fwhm",
        help="Line-width map to write in Angstrom: wavelength FWHM or sigma.",
    )
    parser.add_argument(
        "--velocity-range",
        type=float,
        nargs=2,
        metavar=("VMIN", "VMAX"),
        default=None,
        help=(
            "Doppler-velocity color range in km/s, for example "
            "--velocity-range -30 30. Defaults to an automatic symmetric range."
        ),
    )
    parser.add_argument(
        "--intensity-scale",
        choices=("log", "linear"),
        default="log",
        help=(
            "Color normalization for line-integrated intensity. "
            "Defaults to logarithmic."
        ),
    )
    parser.add_argument(
        "--no-transpose",
        action="store_false",
        dest="transpose",
        help="Do not transpose maps before imshow.",
    )
    parser.add_argument("--dpi", type=int, default=180, help="PNG output resolution.")
    parser.add_argument(
        "--cmap-intensity",
        default="inferno",
        help="Matplotlib colormap for integrated intensity.",
    )
    parser.add_argument(
        "--cmap-velocity",
        default="RdBu_r",
        help="Matplotlib colormap for Doppler velocity.",
    )
    parser.add_argument(
        "--cmap-width",
        default="magma",
        help="Matplotlib colormap for line width.",
    )
    args = parser.parse_args()
    if args.velocity_range is not None:
        vmin, vmax = args.velocity_range
        if vmin >= vmax:
            parser.error("--velocity-range requires VMIN < VMAX")
    return args


def trapezoid(y, x, axis=-1):
    if hasattr(np, "trapezoid"):
        return np.trapezoid(y, x=x, axis=axis)
    return np.trapz(y, x=x, axis=axis)


def infer_lambda0(input_file, wvl_grid):
    match = re.search(r"_([0-9]+(?:\.[0-9]+)?)AA(?:_|$)", input_file.name)
    if match:
        return float(match.group(1))
    return float(wvl_grid[len(wvl_grid) // 2])


def normalize_i2d_shape(i2d, wvl_grid):
    n_wvl = len(wvl_grid)

    if i2d.ndim == 3 and i2d.shape[-1] == n_wvl:
        return i2d[np.newaxis, ...]

    if i2d.ndim == 4 and i2d.shape[-1] == n_wvl:
        return i2d

    if i2d.ndim == 4 and i2d.shape[2] == n_wvl:
        return np.moveaxis(i2d, -1, 0)

    raise ValueError(
        "Could not interpret I2D shape. Expected (nt, nx, ny, nwave), "
        "(nx, ny, nwave), or legacy (nx, ny, nwave, nt)."
    )


def compute_moments(spectrum_cube, wvl_grid, lambda0, width_kind):
    order = np.argsort(wvl_grid)
    wvl = wvl_grid[order]
    spectra = spectrum_cube[..., order]

    intensity = trapezoid(spectra, wvl, axis=-1)
    valid_intensity = np.where(intensity > 0.0, intensity, np.nan)

    cog = trapezoid(spectra * wvl, wvl, axis=-1) / valid_intensity
    doppler_velocity = C_KMS * (cog - lambda0) / lambda0

    variance = (
        trapezoid(spectra * (wvl - cog[..., np.newaxis]) ** 2, wvl, axis=-1)
        / valid_intensity
    )
    sigma_lambda = np.sqrt(np.maximum(variance, 0.0))
    line_width = sigma_lambda
    if width_kind == "fwhm":
        line_width *= 2.354820045

    return intensity, doppler_velocity, line_width


def percentile_limits(values, low, high, positive_only=False):
    finite = values[np.isfinite(values)]
    if positive_only:
        finite = finite[finite > 0.0]
    if finite.size == 0:
        if positive_only:
            raise ValueError(
                "Logarithmic intensity scaling requires at least one "
                "positive integrated-intensity value."
            )
        return 0.0, 1.0
    vmin, vmax = np.nanpercentile(finite, [low, high])
    if vmin == vmax:
        delta = abs(vmin) * 0.05 if vmin else 1.0
        vmin -= delta
        vmax += delta
    return float(vmin), float(vmax)


def find_color_limits(
    i2d,
    wvl_grid,
    lambda0,
    width_kind,
    timestep_indices,
    velocity_range=None,
    intensity_scale="log",
):
    intensity_values = []
    velocity_values = []
    width_values = []

    with ProgressBar(len(timestep_indices), label="Scanning color limits") as progress:
        for timestep_index in timestep_indices:
            intensity, velocity, width = compute_moments(
                i2d[timestep_index], wvl_grid, lambda0, width_kind
            )
            intensity_values.append(intensity.ravel())
            velocity_values.append(velocity.ravel())
            width_values.append(width.ravel())
            progress.update()

    intensity_values = np.concatenate(intensity_values)
    velocity_values = np.concatenate(velocity_values)
    width_values = np.concatenate(width_values)

    intensity_limits = percentile_limits(
        intensity_values,
        1.0,
        99.5,
        positive_only=intensity_scale == "log",
    )
    width_limits = percentile_limits(width_values, 1.0, 99.5)

    if velocity_range is not None:
        velocity_limits = tuple(velocity_range)
    else:
        finite_velocity = velocity_values[np.isfinite(velocity_values)]
        if finite_velocity.size == 0:
            velocity_limits = (-1.0, 1.0)
        else:
            vmax = float(np.nanpercentile(np.abs(finite_velocity), 99.0))
            velocity_limits = (-vmax, vmax)

    return intensity_limits, velocity_limits, width_limits


def map_for_plot(values, transpose):
    return values.T if transpose else values


def save_frame(
    output_file,
    timestep_index,
    timestep_label,
    line_label,
    intensity,
    velocity,
    width,
    limits,
    args,
):
    intensity_limits, velocity_limits, width_limits = limits
    width_label = "FWHM line width" if args.width_kind == "fwhm" else "Sigma line width"
    intensity_title = "Line integrated intensity"
    if args.intensity_scale == "log":
        intensity_title += " (log scale)"

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)

    panels = [
        (
            axes[0],
            map_for_plot(intensity, args.transpose),
            intensity_title,
            args.cmap_intensity,
            (
                LogNorm(vmin=intensity_limits[0], vmax=intensity_limits[1])
                if args.intensity_scale == "log"
                else Normalize(vmin=intensity_limits[0], vmax=intensity_limits[1])
            ),
            "Integrated intensity",
        ),
        (
            axes[1],
            map_for_plot(velocity, args.transpose),
            "Doppler velocity (COG)",
            args.cmap_velocity,
            Normalize(vmin=velocity_limits[0], vmax=velocity_limits[1]),
            "km/s",
        ),
        (
            axes[2],
            map_for_plot(width, args.transpose),
            width_label,
            args.cmap_width,
            Normalize(vmin=width_limits[0], vmax=width_limits[1]),
            "Å",
        ),
    ]

    for axis, image, title, cmap, norm, colorbar_label in panels:
        im = axis.imshow(image, origin="lower", cmap=cmap, norm=norm)
        axis.set_title(title)
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        fig.colorbar(im, ax=axis, label=colorbar_label)

    fig.suptitle(f"{line_label} | frame {timestep_index} | timestep {timestep_label}")
    fig.savefig(output_file, dpi=args.dpi)
    plt.close(fig)


def main():
    args = parse_arguments()
    input_file = args.input_file
    output_dir = args.output_dir or input_file.with_suffix("").with_name(
        input_file.stem + "_frames"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    with np.load(input_file) as data:
        i2d = normalize_i2d_shape(data["I2D"], data["wvlGrid"])
        wvl_grid = data["wvlGrid"]

    lambda0 = args.lambda0 if args.lambda0 is not None else infer_lambda0(input_file, wvl_grid)
    t_stop = i2d.shape[0] if args.t_stop is None else min(args.t_stop, i2d.shape[0])
    timestep_indices = list(range(args.t_start, t_stop))
    if not timestep_indices:
        raise ValueError("No timesteps selected.")

    prefix = args.prefix or input_file.stem + "_3panel"
    limits = find_color_limits(
        i2d,
        wvl_grid,
        lambda0,
        args.width_kind,
        timestep_indices,
        args.velocity_range,
        args.intensity_scale,
    )

    print(f"Input: {input_file}")
    print(f"I2D shape: {i2d.shape}")
    print(f"Using lambda0 = {lambda0}")
    print(f"Using {args.intensity_scale} integrated-intensity scaling")
    if args.velocity_range is not None:
        print(
            "Using Doppler velocity range = "
            f"{args.velocity_range[0]} to {args.velocity_range[1]} km/s"
        )
    print(f"Writing frames to: {output_dir}")

    with ProgressBar(len(timestep_indices), label="Writing frames") as progress:
        for timestep_index in timestep_indices:
            intensity, velocity, width = compute_moments(
                i2d[timestep_index], wvl_grid, lambda0, args.width_kind
            )
            timestep_label = args.time_start + timestep_index * args.time_step
            output_file = output_dir / f"{prefix}_{timestep_index:05d}.png"
            save_frame(
                output_file,
                timestep_index,
                timestep_label,
                args.line_label,
                intensity,
                velocity,
                width,
                limits,
                args,
            )
            progress.update()

    print("Done.")


if __name__ == "__main__":
    main()
