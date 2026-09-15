"""CHIANTI population tables and vertical SPICE line-opacity diagnostics (CGS).

See docs/spice_optical_depth.md for the equations, assumptions and file format.
ChiantiPy is imported only when building/loading atomic tables.
"""

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from .muram import MuramCube

C = 2.99792458e10                 # cm s^-1
H = 6.62607015e-27                # erg s
KB = 1.380649e-16                 # erg K^-1
AMU = 1.66053906660e-24           # g
M_H = 1.00794 * AMU              # mass per hydrogen nucleus, approximately
M_E = 9.1093837015e-28            # g
E_ESU = 4.80320471257e-10         # statcoulomb
MASS_U = {"h": 1.00794, "c": 12.0107, "o": 15.9994, "ne": 20.1797,
          "mg": 24.305, "si": 28.0855, "s": 32.065, "fe": 55.845}
PAPER_URL = "https://doi.org/10.1051/0004-6361/201935574"


@dataclass(frozen=True)
class SpiceLine:
    key: str
    ion: str
    wavelength: float             # paper wavelength, Angstrom
    log_temperature: float        # approximate value from paper, not a cutoff
    order: int = 1


# SPICE Consortium (2020), A&A 642 A14, Table 1. Not an exhaustive line list.
SPICE_LINES = tuple(SpiceLine(*row) for row in [
    ("h_1_1025", "h_1", 1025.72, 4.0),
    ("c_2_1036", "c_2", 1036.34, 4.3),
    ("c_3_977", "c_3", 977.03, 4.5),
    ("s_5_786", "s_5", 786.47, 5.2),
    ("o_4_787", "o_4", 787.72, 5.2),
    ("o_5_760", "o_5", 760.43, 5.4),
    ("o_6_1031", "o_6", 1031.93, 5.5),
    ("o_6_1037", "o_6", 1037.64, 5.5),
    ("ne_6_1005", "ne_6", 1005.79, 5.6),
    ("si_7_1049", "si_7", 1049.25, 5.6),
    ("ne_8_770", "ne_8", 770.42, 5.8),
    ("mg_8_772", "mg_8", 772.31, 5.9),
    ("mg_9_706", "mg_9", 706.02, 6.0),
    ("fe_10_1028", "fe_10", 1028.04, 6.0),
    ("mg_11_997", "mg_11", 997.44, 6.2),
    ("si_12_520", "si_12", 520.67, 6.3, 2),
    ("fe_18_974", "fe_18", 974.84, 6.9),
    ("fe_20_721", "fe_20", 721.55, 7.0),
])
LINE_BY_KEY = {line.key: line for line in SPICE_LINES}
DEFAULT_LINES = ("c_3_977", "s_5_786", "o_4_787", "o_5_760",
                 "o_6_1031", "o_6_1037", "ne_8_770", "mg_9_706")


def select_components(ion, line, tolerance):
    """Select all observed same-ion transitions inside a narrow matching window."""
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("Wavelength matching tolerance must be positive.")
    w = np.asarray(ion.Wgfa["wvl"], dtype=float)
    a = np.asarray(ion.Wgfa["avalue"], dtype=float)
    indices = np.flatnonzero((w > 0) & (a > 0) &
                             (np.abs(w - line.wavelength) <= tolerance))
    if not indices.size:
        raise ValueError(f"No observed CHIANTI transition for {line.key} within "
                         f"{tolerance} Angstrom of {line.wavelength}. "
                         "Inspect the database before increasing --match-tolerance.")
    level_ids = np.asarray(ion.Elvlc["lvl"], dtype=int)
    # ChiantiPy indexes population columns by the one-based CHIANTI level ID.
    if not np.array_equal(level_ids, np.arange(1, len(level_ids) + 1)):
        raise ValueError(f"Unsupported non-contiguous level numbering: {line.ion}")
    components = []
    for idx in indices:
        lower = int(ion.Wgfa["lvl1"][idx])
        upper = int(ion.Wgfa["lvl2"][idx])
        gl = float(ion.Elvlc["mult"][lower - 1])
        gu = float(ion.Elvlc["mult"][upper - 1])
        f = M_E * C / (8 * np.pi**2 * E_ESU**2) * (gu / gl) * (w[idx]*1e-8)**2 * a[idx]
        components.append(dict(lower=lower, upper=upper, g_lower=gl, g_upper=gu,
                               wavelength=float(w[idx]), a_value=float(a[idx]),
                               f_effective=float(f)))
    return components


def _population_column(ion_name, temperatures, density, abundance, levels):
    import ChiantiPy.core as ch

    ion = ch.ion(ion_name, temperature=temperatures, eDensity=float(density),
                 abundance=abundance)
    fraction = np.asarray(ion.IoneqOne, dtype=float).reshape(-1)
    if not np.all(np.isfinite(fraction)) or np.any((fraction < 0) | (fraction > 1 + 1e-8)):
        raise ValueError(f"Invalid ion fractions for {ion_name}.")
    result = np.zeros((len(temperatures), len(levels)))
    active = fraction > 0
    if not np.any(active):
        return result
    # Avoid running the level solver where the equilibrium fraction is exactly zero.
    ion = ch.ion(ion_name, temperature=temperatures[active], eDensity=float(density),
                 abundance=abundance)
    ion.populate()
    if "errorMessage" in ion.Population:
        raise ValueError(f"CHIANTI population solve failed: {ion.Population['errorMessage']}")
    pop = np.atleast_2d(np.asarray(ion.Population["population"], dtype=float))
    if (pop.shape[0] != active.sum() or not np.all(np.isfinite(pop)) or
            np.any(pop < 0) or not np.allclose(pop.sum(axis=1), 1, atol=1e-5)):
        raise ValueError(f"Invalid or unnormalized level populations for {ion_name}.")
    result[active] = float(ion.Abundance) * fraction[active, None] * pop[:, np.asarray(levels)-1]
    return result


@dataclass
class PopulationTable:
    log_t: np.ndarray
    log_ne: np.ndarray
    values: np.ndarray            # (T, ne, selected level); n_level / n_H
    metadata: dict

    def __post_init__(self):
        self.interpolate = RegularGridInterpolator(
            (self.log_t, self.log_ne), self.values, method="linear", bounds_error=True)

    def evaluate(self, temperature, electron_density):
        return self.interpolate(np.stack((np.log10(temperature),
                                          np.log10(electron_density)), axis=-1))


def build_population_table(lines, log_t, log_ne, abundance, cache_dir,
                           tolerance=0.15, jobs=1, rebuild=False):
    """Build/reuse one ion's selected-level tables; no silent extrapolation."""
    import ChiantiPy
    import ChiantiPy.core as ch
    import ChiantiPy.tools.data as chdata

    name = lines[0].ion
    if any(line.ion != name for line in lines):
        raise ValueError("A population table must contain exactly one ion.")
    for grid in (log_t, log_ne):
        if len(grid) < 2 or not np.all(np.isfinite(grid)) or np.any(np.diff(grid) <= 0):
            raise ValueError("Atomic grids must be finite and strictly increasing.")
    root = Path(os.environ["XUVTOP"]).expanduser().resolve()
    if not (root / "abundance" / (abundance + ".abund")).is_file():
        raise ValueError(f"Abundance file not found: {abundance}.abund")
    probe = ch.ion(name, temperature=10.**lines[0].log_temperature,
                   eDensity=1e10, abundance=abundance)
    if not hasattr(probe, "Wgfa"):
        raise ValueError(f"CHIANTI radiative data unavailable for {name}.")
    components = {line.key: select_components(probe, line, tolerance) for line in lines}
    levels = sorted({c[level] for cs in components.values() for c in cs
                     for level in ("lower", "upper")})
    # Fingerprint the files, ionization balance and abundance used by this solver.
    digest = hashlib.sha256()
    element, stage = name.split("_")
    for ion_name in (name, f"{element}_{int(stage)+1}"):
        for path in sorted((root / element / ion_name).glob("*")):
            if path.is_file():
                digest.update(path.name.encode())
                digest.update(path.read_bytes())
    digest.update(np.asarray(chdata.IoneqAll["ioneqAll"]).tobytes())
    digest.update(np.asarray(chdata.IoneqAll["ioneqTemperature"]).tobytes())
    digest.update(np.asarray(probe.AbundAll).tobytes())
    metadata = dict(schema=1, ion=name, levels=levels, components=components,
                    log_t=list(map(float, log_t)), log_ne=list(map(float, log_ne)),
                    abundance_name=abundance, abundance=float(probe.Abundance),
                    ioneq_name=probe.IoneqName, chiantipy_version=ChiantiPy.__version__,
                    database_root=str(root), database_version=(root / "VERSION").read_text().strip(),
                    atomic_sha256=digest.hexdigest(), match_tolerance=tolerance,
                    population_method="populate() with default corrections and proton density")
    signature = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()[:20]
    cache = Path(cache_dir) / f"{name}_{signature}.npz"
    if cache.exists() and not rebuild:
        with np.load(cache, allow_pickle=False) as saved:
            if json.loads(str(saved["metadata"])) != metadata:
                raise ValueError(f"Cache metadata mismatch: {cache}")
            values = saved["values"]
        print(f"  Reusing {cache}", flush=True)
    else:
        print(f"  Building {name}: {len(log_t)} temperatures x {len(log_ne)} densities", flush=True)
        if jobs == 1:
            columns = [_population_column(name, 10.**log_t, 10.**n, abundance, levels)
                       for n in log_ne]
        else:
            from joblib import Parallel, delayed
            columns = Parallel(n_jobs=jobs, verbose=5)(
                delayed(_population_column)(name, 10.**log_t, 10.**n, abundance, levels)
                for n in log_ne)
        values = np.stack(columns, axis=1)
        cache.parent.mkdir(parents=True, exist_ok=True)
        # A completed archive replaces the cache atomically, also with parallel runs.
        import tempfile
        with tempfile.NamedTemporaryFile(dir=cache.parent, suffix=".npz", delete=False) as f:
            temporary = Path(f.name)
        try:
            np.savez_compressed(temporary, values=values, metadata=json.dumps(metadata, sort_keys=True))
            temporary.replace(cache)
        finally:
            temporary.unlink(missing_ok=True)
    if values.shape != (len(log_t), len(log_ne), len(levels)) or not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError(f"Invalid population cache: {cache}")
    metadata["cache_file"] = str(cache.resolve())
    return PopulationTable(log_t, log_ne, values, metadata)


def load_snapshot(model_dir, iteration, downsample=1, height_start=0, height_stop=None):
    """Expose (horizontal z, horizontal y, vertical x) views, like MURaMData."""
    if downsample < 1 or height_start < 0:
        raise ValueError("downsample >= 1 and height_start >= 0 are required.")
    cubes = {key: MuramCube(str(model_dir), iteration, var) for key, var in
             (("temperature", "Temp"), ("ne", "ne"), ("rho", "rho"), ("velocity", "vx"))}
    raw = cubes["temperature"]
    stop = raw.shape[0] if height_stop is None else height_stop
    if not height_start < stop <= raw.shape[0]:
        raise ValueError(f"Height selection [{height_start}:{stop}] is outside {raw.shape[0]} cells.")
    if not np.all(np.isfinite(raw.dX)) or np.any(raw.dX <= 0):
        raise ValueError("Header cell spacings must be positive, finite cm values.")
    views = {key: cube.T[::downsample, ::downsample, height_start:stop]
             for key, cube in cubes.items()}
    info = dict(iteration=int(iteration), time_s=float(raw.time),
                raw_shape=list(raw.shape), output_axes=["muram_z", "muram_y", "wavelength"],
                cell_spacing_cm=list(map(float, raw.dX)), height_start=height_start,
                height_stop=stop, downsample=downsample, model_dir=str(Path(model_dir).resolve()))
    return views, float(raw.dX[0]), info


def validate_block(block, table):
    for key, array in block.items():
        if not np.all(np.isfinite(array)):
            raise ValueError(f"Non-finite MURaM {key} values.")
        if key in ("temperature", "ne", "rho") and np.any(array <= 0):
            raise ValueError(f"MURaM {key} must be strictly positive.")
    for key, grid in (("temperature", table.log_t), ("ne", table.log_ne)):
        limits = (float(block[key].min()), float(block[key].max()))
        if limits[0] < 10.**grid[0] or limits[1] > 10.**grid[-1]:
            raise ValueError(f"{key} range {limits} is outside the atomic grid "
                             f"[{10.**grid[0]:g}, {10.**grid[-1]:g}]. "
                             "Extend --log-t/--log-ne or explicitly select the desired heights.")
    if np.any(np.abs(block["velocity"]) > 0.01*C):
        raise ValueError("Velocity exceeds 0.01 c; check CGS units and the nonrelativistic assumption.")


def integrate_columns(temperature, n_h, red_velocity, level_per_h, components,
                      levels, wavelengths, ds, mass_g, microturbulence_kms=0.):
    """Return tau(lambda), optically thin I_lambda, and static reference tau.

    Atmospheric inputs have shape (rays, height); populations append a level axis.
    red_velocity is cm/s, positive away from the observer; wavelengths are Angstrom.
    Sum cell-centered coefficients times ds; no cell is dropped as in trapezoids.
    """
    if not np.isfinite(ds) or ds <= 0 or not np.isfinite(mass_g) or mass_g <= 0:
        raise ValueError("Cell length and particle mass must be positive and finite.")
    if not np.isfinite(microturbulence_kms) or microturbulence_kms < 0:
        raise ValueError("Microturbulent Doppler b parameter must be nonnegative.")
    wavelengths = np.asarray(wavelengths, dtype=float)
    if wavelengths.ndim != 1 or not np.all(np.isfinite(wavelengths)) or np.any(wavelengths <= 0):
        raise ValueError("Wavelengths must be a positive finite vector.")
    b = np.sqrt(2 * KB * temperature / mass_g + (microturbulence_kms * 1e5)**2)
    frequency = C / (wavelengths * 1e-8)
    tau = np.zeros((temperature.shape[0], wavelengths.size))
    intensity = np.zeros_like(tau)
    static = np.zeros(temperature.shape[0])
    reference_frequency = frequency[len(frequency)//2]
    index = {level: i for i, level in enumerate(levels)}
    for component in components:
        lam = component["wavelength"] * 1e-8
        nu0 = C / lam
        width = b / lam
        nl = n_h * level_per_h[..., index[component["lower"]]]
        nu = n_h * level_per_h[..., index[component["upper"]]]
        population_difference = component["g_upper"] / component["g_lower"] * nl - nu
        # Negative opacity is meaningful for an inversion; do not silently clip it.
        strength = lam**2 / (8*np.pi) * component["a_value"] * population_difference
        emission = H * nu0 / (4*np.pi) * component["a_value"] * nu
        norm = 1 / (np.sqrt(np.pi) * width)
        center = nu0 / (1 + red_velocity/C)
        static += np.sum(strength * norm * np.exp(-((reference_frequency-nu0)/width)**2), axis=1) * ds
        # Stream wavelength samples to avoid a rays x height x wavelengths allocation.
        for i, freq in enumerate(frequency):
            profile = norm * np.exp(-((freq-center)/width)**2)
            tau[:, i] += np.sum(strength * profile, axis=1) * ds
            intensity[:, i] += np.sum(emission * profile, axis=1) * ds
    intensity *= C / (wavelengths*1e-8)**2 * 1e-8  # I_nu -> I_lambda per Angstrom
    return tau, intensity, static


def synthesize_snapshot(views, ds, line, table, velocity_grid, hydrogen_mass_fraction=0.74,
                        microturbulence_kms=0., observer="top", chunk_columns=128):
    """Compute one line feature for one snapshot without copying full input cubes."""
    if not 0 < hydrogen_mass_fraction <= 1 or chunk_columns < 1:
        raise ValueError("Require 0 < hydrogen mass fraction <= 1 and chunk_columns >= 1.")
    if observer not in ("top", "bottom"):
        raise ValueError("Observer must be top or bottom.")
    components = table.metadata["components"][line.key]
    reference = min(components, key=lambda c: abs(c["wavelength"] - line.wavelength))["wavelength"]
    wavelengths = reference * (1 + np.asarray(velocity_grid)*1e5/C)
    shape = views["temperature"].shape
    count = int(np.prod(shape[:2]))
    tau = np.empty((count, len(wavelengths)))
    intensity = np.empty_like(tau)
    static = np.empty(count)
    max_speed = 0.
    min_b = np.inf
    for start in range(0, count, chunk_columns):
        end = min(start + chunk_columns, count)
        indices = np.unravel_index(np.arange(start, end), shape[:2])
        block = {key: np.asarray(value[indices], dtype=float) for key, value in views.items()}
        validate_block(block, table)
        populations = table.evaluate(block["temperature"], block["ne"])
        n_h = hydrogen_mass_fraction * block["rho"] / M_H
        # Increasing native x points toward an observer at the upper boundary.
        red_velocity = block["velocity"] * (-1 if observer == "top" else 1)
        result = integrate_columns(block["temperature"], n_h, red_velocity, populations,
                                   components, table.metadata["levels"], wavelengths, ds,
                                   MASS_U[line.ion.split("_")[0]]*AMU, microturbulence_kms)
        tau[start:end], intensity[start:end], static[start:end] = result
        max_speed = max(max_speed, float(np.max(np.abs(red_velocity)))/1e5)
        min_b = min(min_b, float(np.sqrt(2*KB*block["temperature"].min() /
                    (MASS_U[line.ion.split("_")[0]]*AMU) + (microturbulence_kms*1e5)**2))/1e5)
        if start == 0 or end == count or (start // chunk_columns) % 100 == 0:
            print(f"  {line.key}: {end}/{count} columns", flush=True)
    output_shape = (1, *shape[:2], len(wavelengths))
    maps_shape = (1, *shape[:2])
    return dict(tau2D=tau.reshape(output_shape), I2D=intensity.reshape(output_shape),
                tau_rest=tau[:, len(wavelengths)//2].reshape(maps_shape),
                tau_peak=tau.max(axis=1).reshape(maps_shape),
                tau_static=static.reshape(maps_shape), wvlGrid=wavelengths,
                velocityGrid=np.asarray(velocity_grid),
                detectorWvlGrid=line.order*wavelengths), dict(
                    reference_wavelength_A=reference, max_abs_los_velocity_kms=max_speed,
                    minimum_doppler_b_kms=min_b,
                    has_negative_opacity=bool(np.any(tau < 0)))
