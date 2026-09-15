#!/usr/bin/env python3
"""Compute CHIANTI-based SPICE optical depths from MURaM full-domain cubes."""

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
import warnings

import numpy as np

from modules.spiceOpticalDepth import (
    DEFAULT_LINES, LINE_BY_KEY, PAPER_URL, SPICE_LINES,
    build_population_table, load_snapshot, synthesize_snapshot,
)
from visualize_spice_optical_depth import add_plot_arguments, save_tau_peak_map, validate_plot_options


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--model-dir", type=Path, help="Directory containing Header.*, eosT.*, eosne.*, result_prim_[01].*")
    p.add_argument("--output-dir", type=Path, default=Path("optical_depth_output"))
    p.add_argument("--cache-dir", type=Path, default=Path("opacity_tables"))
    p.add_argument("--iterations", nargs="+", type=int, help="Explicit MURaM iteration numbers")
    p.add_argument("--t-start", type=int, help="First iteration of an inclusive range")
    p.add_argument("--t-end", type=int, help="Last iteration of an inclusive range")
    p.add_argument("--time-step", type=int, default=2000, help="Iteration increment, not seconds")
    selection = p.add_mutually_exclusive_group()
    selection.add_argument("--lines", nargs="+", choices=LINE_BY_KEY, help="Default: eight transition-region/coronal diagnostics")
    selection.add_argument("--all-lines", action="store_true", help="All 18 Table 1 features, including approximate H I")
    p.add_argument("--list-lines", action="store_true", help="Print catalogue and exit without CHIANTI")
    p.add_argument("--tables-only", action="store_true", help="Build atomic caches without a MURaM snapshot")
    p.add_argument("--downsample", type=int, default=1, help="Horizontal stride; every vertical cell is retained")
    p.add_argument("--height-start", type=int, default=0, help="First included native vertical x index")
    p.add_argument("--height-stop", type=int, help="Exclusive vertical x stop; default: full height")
    p.add_argument("--observer", choices=("top", "bottom"), default="top")
    p.add_argument("--hydrogen-mass-fraction", type=float, default=0.74, help="X_H for n_H = X_H rho / m_H; match the MURaM EOS")
    p.add_argument("--abundance", default="sun_photospheric_2021_asplund", help="CHIANTI abundance filename stem")
    p.add_argument("--microturbulence", type=float, default=0., help="Nonthermal Doppler b parameter xi in km/s, not Gaussian sigma")
    p.add_argument("--velocity-range", type=float, default=200., help="Symmetric spectral half-range in km/s")
    p.add_argument("--num-waves", type=int, default=401, help="Odd number of spectral samples, including exact reference wavelength")
    p.add_argument("--log-t", nargs=3, type=float, default=(3., 8., 101), metavar=("MIN", "MAX", "COUNT"))
    p.add_argument("--log-ne", nargs=3, type=float, default=(6., 17., 45), metavar=("MIN", "MAX", "COUNT"))
    p.add_argument("--match-tolerance", type=float, default=0.15, help="Observed transition selection half-window in Angstrom")
    p.add_argument("--chunk-columns", type=int, default=128)
    p.add_argument("--jobs", type=int, default=1, help="Processes for table densities; -1 uses all CPUs")
    p.add_argument("--rebuild-tables", action="store_true")
    p.add_argument("--overwrite", action="store_true", help="Replace existing results")
    p.add_argument("--plot", action="store_true", help="Save a Matplotlib x/y map of peak column optical depth beside each NPZ")
    add_plot_arguments(p)
    return p


def make_grid(spec):
    low, high, count = spec
    if not np.all(np.isfinite(spec)) or low >= high or count < 2 or count != int(count):
        raise ValueError("Grid needs finite MIN < MAX and an integer COUNT >= 2.")
    if not -100 < low < high < 100:
        raise ValueError("Logarithmic grid endpoints must lie between -100 and 100.")
    return np.linspace(low, high, int(count))


def run(args):
    if args.list_lines:
        print("ID             Ion      Paper [A]  log T  Order  Default")
        for line in SPICE_LINES:
            print(f"{line.key:14} {line.ion:8} {line.wavelength:9.2f} {line.log_temperature:6.1f}"
                  f" {line.order:6}  {'yes' if line.key in DEFAULT_LINES else ''}")
        print(f"Source: {PAPER_URL}, Table 1; H I is an equilibrium approximation only.")
        return
    log_t, log_ne = make_grid(args.log_t), make_grid(args.log_ne)
    if (args.num_waves < 3 or args.num_waves % 2 != 1 or args.chunk_columns < 1 or
            args.downsample < 1 or args.height_start < 0 or args.jobs == 0 or
            not np.isfinite(args.velocity_range) or not 0 < args.velocity_range < 3000 or
            not np.isfinite(args.microturbulence) or args.microturbulence < 0 or
            not 0 < args.hydrogen_mass_fraction <= 1):
        raise ValueError("Invalid sampling, velocity, grid, job count, or hydrogen fraction; see --help.")
    keys = list(LINE_BY_KEY) if args.all_lines else list(dict.fromkeys(args.lines or DEFAULT_LINES))
    lines = [LINE_BY_KEY[key] for key in keys]
    if "h_1_1025" in keys:
        warnings.warn("H I Ly beta uses CHIANTI equilibrium populations. It is not an NLTE hydrogen calculation.")
    iterations = []
    if not args.tables_only:
        if args.plot:
            validate_plot_options(args.plot_scale, args.plot_limits, args.plot_cmap, args.plot_dpi)
        if args.model_dir is None:
            raise ValueError("--model-dir is required unless using --tables-only or --list-lines.")
        if args.iterations is not None:
            if args.t_start is not None or args.t_end is not None:
                raise ValueError("Choose --iterations or --t-start/--t-end.")
            iterations = list(dict.fromkeys(args.iterations))
        else:
            if args.t_start is None or args.t_end is None or args.time_step < 1 or args.t_end < args.t_start:
                raise ValueError("Specify --iterations or a valid --t-start/--t-end/--time-step range.")
            iterations = list(range(args.t_start, args.t_end + 1, args.time_step))
        if any(i < 0 for i in iterations):
            raise ValueError("Iterations must be nonnegative.")
        # Check files and output collisions before expensive atomic table generation.
        for iteration in iterations:
            for prefix in ("Header", "eosT", "eosne", "result_prim_0", "result_prim_1"):
                path = args.model_dir / f"{prefix}.{iteration:06d}"
                if not path.is_file():
                    raise FileNotFoundError(path)
            for key in keys:
                path = args.output_dir / f"spice_opacity_{key}_{iteration:06d}.npz"
                if path.exists() and not args.overwrite:
                    raise FileExistsError(f"{path} exists; use another output directory or --overwrite.")
                plot_path = path.with_name(f"{path.stem}_tau_peak.png")
                if args.plot and plot_path.exists() and not args.overwrite:
                    raise FileExistsError(f"{plot_path} exists; use another output directory or --overwrite.")
    if not os.environ.get("XUVTOP") or not Path(os.environ["XUVTOP"]).is_dir():
        raise ValueError("Set XUVTOP to an installed CHIANTI database before running.")
    tables = {}
    for name in dict.fromkeys(line.ion for line in lines):
        tables[name] = build_population_table(
            [line for line in lines if line.ion == name], log_t, log_ne, args.abundance,
            args.cache_dir, args.match_tolerance, args.jobs, args.rebuild_tables)
        for key, components in tables[name].metadata["components"].items():
            print(f"  {key}: " + ", ".join(f"{c['wavelength']:.5f} A ({c['lower']} -> {c['upper']})"
                                           for c in components), flush=True)
    if args.tables_only:
        return
    args.output_dir.mkdir(parents=True, exist_ok=True)
    velocity_grid = np.linspace(-args.velocity_range, args.velocity_range, args.num_waves)
    for iteration in iterations:
        views, ds, snapshot = load_snapshot(args.model_dir, iteration, args.downsample,
                                             args.height_start, args.height_stop)
        print(f"Snapshot {iteration:06d}: shape {views['temperature'].shape}, ds={ds:g} cm", flush=True)
        for line in lines:
            table = tables[line.ion]
            products, diagnostics = synthesize_snapshot(
                views, ds, line, table, velocity_grid, args.hydrogen_mass_fraction,
                args.microturbulence, args.observer, args.chunk_columns)
            metadata = dict(line=asdict(line), source=PAPER_URL, snapshot=snapshot,
                            atomic=table.metadata, diagnostics=diagnostics,
                            hydrogen_mass_fraction=args.hydrogen_mass_fraction,
                            microturbulence_kms=args.microturbulence, observer=args.observer,
                            units=dict(tau="dimensionless", I2D="erg s^-1 cm^-2 sr^-1 Angstrom^-1",
                                       wvlGrid="Angstrom", velocityGrid="km/s"),
                            intensity_model="optically thin spontaneous line emission",
                            opacity_model="equilibrium populations; Gaussian bound-bound opacity only",
                            command=sys.argv)
            if diagnostics["has_negative_opacity"]:
                warnings.warn(f"{line.key}: negative opacity; inspect populations for an inversion.")
            if diagnostics["max_abs_los_velocity_kms"] >= args.velocity_range:
                warnings.warn(f"{line.key}: some Doppler centers lie outside the spectral window; widen --velocity-range.")
            if velocity_grid[1]-velocity_grid[0] > diagnostics["minimum_doppler_b_kms"]/3:
                warnings.warn(f"{line.key}: narrow profiles may be undersampled; increase --num-waves.")
            path = args.output_dir / f"spice_opacity_{line.key}_{iteration:06d}.npz"
            # Exclusive creation prevents accidental replacement by a concurrent run.
            with path.open("wb" if args.overwrite else "xb") as output:
                np.savez_compressed(output, **products, metadata=json.dumps(metadata))
            print(f"Saved {path}; max sampled tau={products['tau_peak'].max():.5g}", flush=True)
            if args.plot:
                plot_path = path.with_name(f"{path.stem}_tau_peak.png")
                save_tau_peak_map(products["tau_peak"], metadata, plot_path, args.plot_scale,
                                  args.plot_limits, args.plot_cmap, args.plot_dpi, args.overwrite)
                print(f"Saved {plot_path}", flush=True)
            del products


if __name__ == "__main__":
    try:
        run(parser().parse_args())
    except (ValueError, FileNotFoundError, FileExistsError, ImportError) as error:
        raise SystemExit(f"Error: {error}") from error
