"""Sensor models: gamma spectrometer, dose-rate meter, contamination probe.

The gamma spectrometer produces a synthetic spectrum with:
  * Gaussian photopeaks at each emitted gamma energy, broadened by detector
    resolution (NaI(Tl) ~7% FWHM at 662 keV, HPGe ~0.2% FWHM at 1332 keV,
    both scaling as 1/sqrt(E))
  * A simplified Compton continuum below each photopeak (Klein-Nishina-shaped
    edge approximated by a smoothed step)
  * Poisson counting statistics applied to the whole spectrum

It is intentionally not a full MCNP transport — but the photopeaks land at the
right energies, the resolution is right, and the noise is real Poisson, so a
trained eye sees something believable.
"""
from __future__ import annotations
import math
from dataclasses import dataclass
import numpy as np

from .radionuclides import Nuclide, NUCLIDES

# Spectrum binning
N_CHANNELS = 1024
E_MAX_KEV = 3000.0
CHANNEL_EDGES = np.linspace(0.0, E_MAX_KEV, N_CHANNELS + 1)
CHANNEL_CENTERS = 0.5 * (CHANNEL_EDGES[:-1] + CHANNEL_EDGES[1:])
CHANNEL_WIDTH = E_MAX_KEV / N_CHANNELS


@dataclass
class DetectorConfig:
    name: str
    # FWHM resolution at 662 keV (relative, e.g. 0.07 for 7%); scales 1/sqrt(E)
    fwhm_at_662: float
    # Intrinsic + geometric efficiency: counts per emitted photon at 1 m
    efficiency_at_662: float
    # Dead-time fraction at high count rates (simplified)
    dead_time_fraction: float = 0.0


# NaI(Tl) 2"x2" scintillator on mobile agents
NAI_DETECTOR = DetectorConfig(name="NaI(Tl) 2x2", fwhm_at_662=0.07, efficiency_at_662=0.02)

# HPGe coaxial on the fixed drum scanner
HPGE_DETECTOR = DetectorConfig(name="HPGe coax", fwhm_at_662=0.002, efficiency_at_662=0.005)


def _fwhm_keV(detector: DetectorConfig, energy_keV: float) -> float:
    """FWHM in keV at given energy. Scales as sqrt(E) (statistical broadening
    of charge carriers / scintillation photons)."""
    if energy_keV <= 0:
        return 1.0
    # FWHM_at_E = FWHM_at_662 * sqrt(E / 662) -> FWHM/E falls as 1/sqrt(E)
    fwhm_662_keV = detector.fwhm_at_662 * 662.0
    return fwhm_662_keV * math.sqrt(energy_keV / 662.0)


def _photopeak_counts(
    energy_keV: float,
    expected_counts: float,
    detector: DetectorConfig,
) -> np.ndarray:
    """Distribute expected_counts into a Gaussian centered on energy_keV."""
    if expected_counts <= 0 or energy_keV <= 0 or energy_keV >= E_MAX_KEV:
        return np.zeros(N_CHANNELS)
    fwhm = _fwhm_keV(detector, energy_keV)
    sigma = fwhm / 2.3548
    # Bin-integrated Gaussian -> use erf-based bin probabilities
    from scipy.special import erf
    z_lo = (CHANNEL_EDGES[:-1] - energy_keV) / (sigma * math.sqrt(2))
    z_hi = (CHANNEL_EDGES[1:] - energy_keV) / (sigma * math.sqrt(2))
    prob = 0.5 * (erf(z_hi) - erf(z_lo))
    return expected_counts * prob


def _compton_continuum(
    energy_keV: float,
    expected_counts: float,
    detector: DetectorConfig,
) -> np.ndarray:
    """Simplified Compton continuum: flat-ish from 0 up to the Compton edge,
    smoothed by detector resolution. Carries about 4x the photopeak counts
    for NaI (peak-to-Compton ratio ~5-10 for NaI vs >50 for HPGe at 662 keV).
    """
    if expected_counts <= 0 or energy_keV <= 0:
        return np.zeros(N_CHANNELS)

    # Compton edge: maximum energy transferred to electron in single scatter
    # E_c = E * 2E / (m_e*c^2 + 2E), m_e*c^2 = 511 keV
    edge = energy_keV * 2 * energy_keV / (511.0 + 2 * energy_keV)

    # Peak-to-Compton: NaI ~5, HPGe ~50 (very roughly at 662 keV)
    p_to_c = 5.0 if detector.fwhm_at_662 > 0.01 else 50.0
    continuum_total = expected_counts / p_to_c

    spectrum = np.zeros(N_CHANNELS)
    mask = (CHANNEL_CENTERS > 5.0) & (CHANNEL_CENTERS < edge)
    n_bins = max(int(mask.sum()), 1)
    spectrum[mask] = continuum_total / n_bins

    # Smooth the edge using detector resolution at the edge energy
    fwhm = _fwhm_keV(detector, max(edge, 50.0))
    sigma_bins = max((fwhm / 2.3548) / CHANNEL_WIDTH, 1.0)
    from scipy.ndimage import gaussian_filter1d
    return gaussian_filter1d(spectrum, sigma_bins, mode="nearest")


def simulate_spectrum(
    activities_bq: dict[str, float],
    integration_time_s: float,
    detector: DetectorConfig,
    distance_m: float = 0.1,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Simulate a measured gamma spectrum (counts per channel).

    activities_bq: nuclide name -> activity (Bq) in the source
    integration_time_s: live time of the measurement
    distance_m: source-to-detector distance (inverse-square scaling)
    """
    if rng is None:
        rng = np.random.default_rng()

    spectrum = np.zeros(N_CHANNELS)
    # Geometric factor: efficiency declared at 1 m -> scale by (1/d)^2
    geom = 1.0 / max(distance_m, 0.05) ** 2

    for nuclide_name, activity in activities_bq.items():
        if activity <= 0:
            continue
        nuc = NUCLIDES.get(nuclide_name)
        if nuc is None:
            continue
        # Total photons emitted in integration time per gamma line
        for line in nuc.gammas:
            emitted = activity * line.intensity * integration_time_s
            # Energy-dependent efficiency: approx Eff(E) = Eff_662 * (662/E)^0.5
            eff = detector.efficiency_at_662 * math.sqrt(662.0 / max(line.energy_keV, 50.0))
            expected = emitted * eff * geom
            spectrum += _photopeak_counts(line.energy_keV, expected, detector)
            spectrum += _compton_continuum(line.energy_keV, expected, detector)

        # Bremsstrahlung for pure-beta emitters: smooth low-energy continuum
        if nuc.brems_fraction > 0:
            total_decays = activity * integration_time_s
            brems_counts = (
                total_decays * nuc.brems_fraction
                * detector.efficiency_at_662 * geom
            )
            endpoint_keV = 600.0  # approximate beta endpoint
            mask = CHANNEL_CENTERS < endpoint_keV
            if mask.any():
                # 1/E-shaped bremsstrahlung continuum
                weights = 1.0 / np.maximum(CHANNEL_CENTERS[mask], 20.0)
                weights = weights / weights.sum()
                spectrum[mask] += brems_counts * weights

    # Detector dead time
    if detector.dead_time_fraction > 0:
        spectrum *= (1.0 - detector.dead_time_fraction)

    # Apply Poisson noise -- this is the *real* counting statistics part
    noisy = rng.poisson(np.maximum(spectrum, 0.0)).astype(np.float64)
    return noisy


# ---------- Dose-rate meter ----------

# Specific gamma-ray constants (µSv/h per MBq at 1 m). Order-of-magnitude
# values for shielded ion-chamber readings; sufficient for relative dose.
GAMMA_CONSTANTS_uSv_h_per_MBq_at_1m: dict[str, float] = {
    "Co-60": 0.351,
    "Co-58": 0.155,
    "Cs-137": 0.092,
    "Cs-134": 0.249,
    "Mn-54": 0.118,
    "I-131": 0.066,
    "Am-241": 0.0034,
    "U-235": 0.014,
    "U-238": 0.00084,
    "Pu-239": 6e-5,
    "Fe-55": 0.0,
    "Ni-63": 0.0,
    "Sr-90": 0.0,  # pure beta — dose dominated by brems, small at distance
}


def simulate_dose_rate(
    activities_bq: dict[str, float],
    distance_m: float = 0.3,
    rng: np.random.Generator | None = None,
) -> float:
    """Return measured dose rate in µSv/h. ~3% Gaussian instrument noise +
    a 0.05 µSv/h cosmic/background floor."""
    if rng is None:
        rng = np.random.default_rng()
    true_rate = 0.05  # background floor
    geom = 1.0 / max(distance_m, 0.05) ** 2
    for nuclide, activity in activities_bq.items():
        gamma_const = GAMMA_CONSTANTS_uSv_h_per_MBq_at_1m.get(nuclide, 0.0)
        true_rate += gamma_const * (activity / 1e6) * geom
    # Instrument noise: log-normal-ish via Gaussian on log
    measured = true_rate * rng.normal(1.0, 0.03)
    return max(measured, 0.0)


# ---------- Surface contamination probe (sorting station only) ----------

# ---------- Computer-vision classification (camera + ML on the robot) ----

# A real CV head on a sorting bot can pick up container type, packaging,
# shielding, decon-marker labels, and physical form (drum, bag, vitrified
# block) with non-trivial reliability. We model that as a noisy classifier
# that maps the true class to a confidence vector with a known confusion
# pattern. The output is independent of the gamma spectrum, so the hybrid
# classifier can use it as a corroborating signal: when CV disagrees with
# the gamma rules result, confidence drops and the item gets scrutinized.

_CV_CONFUSION: dict[str, dict[str, float]] = {
    # true -> {predicted: probability}. Per-class accuracy ~ 90% for VLLW/LLW
    # and ~93% for ILW (heavy shielded casks are easy to spot visually).
    "VLLW": {"VLLW": 0.90, "LLW": 0.08, "ILW": 0.02},
    "LLW":  {"VLLW": 0.06, "LLW": 0.88, "ILW": 0.06},
    "ILW":  {"VLLW": 0.01, "LLW": 0.06, "ILW": 0.93},
}


def simulate_cv_classification(
    true_class: str,
    tricky: bool,
    rng: np.random.Generator | None = None,
) -> dict[str, float]:
    """Return a class-probability dict for the CV head.

    `tricky` items have an actinide spike *inside* an LLW-looking package
    (per the WasteStream model). The container looks like LLW, so the CV
    head should be FOOLED — it tends to call them LLW even though they are
    truly ILW. This is intentional: CV alone cannot catch the actinide
    trick; the gamma side has to. CV's job here is to spot the more obvious
    cases where the gamma side might be ambiguous due to noise."""
    if rng is None:
        rng = np.random.default_rng()
    if tricky:
        # Fool the CV head — it sees an LLW container
        probs = {"VLLW": 0.08, "LLW": 0.80, "ILW": 0.12}
    else:
        probs = dict(_CV_CONFUSION.get(true_class, {"VLLW": 1 / 3, "LLW": 1 / 3, "ILW": 1 / 3}))
    # Add some Dirichlet noise so the same item type isn't always identical
    keys = list(probs.keys())
    base = np.asarray([probs[k] for k in keys]) * 20.0 + 0.5
    sample = rng.dirichlet(base)
    return {k: float(v) for k, v in zip(keys, sample)}


def simulate_surface_contamination(
    activities_bq: dict[str, float],
    surface_area_cm2: float = 100.0,
    rng: np.random.Generator | None = None,
) -> dict[str, float]:
    """Return alpha + beta surface count rates (cps).

    Assumes activities given here are the *surface* component.
    Alpha responders: actinides. Beta responders: most fission/activation."""
    if rng is None:
        rng = np.random.default_rng()
    alpha_emitters = {"U-235", "U-238", "Pu-239", "Am-241"}
    alpha_cps = 0.0
    beta_cps = 0.0
    for nuclide, activity in activities_bq.items():
        # Probe efficiency: ~30% alpha, ~30% beta, 2-pi geometry
        if nuclide in alpha_emitters:
            alpha_cps += 0.3 * activity
        else:
            beta_cps += 0.3 * activity
    # Poisson noise
    alpha = float(rng.poisson(max(alpha_cps, 0.0)))
    beta = float(rng.poisson(max(beta_cps, 0.0)))
    return {"alpha_cps": alpha, "beta_cps": beta}
