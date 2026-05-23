"""Radionuclide reference data.

Values are public-domain nuclear data (ENSDF / NNDC).
Activity ranges per waste class follow IAEA SRS-44 typical ranges
for LWR operational waste streams (order-of-magnitude figures).
"""
from __future__ import annotations
from dataclasses import dataclass, field


@dataclass(frozen=True)
class GammaLine:
    energy_keV: float
    intensity: float  # photons per decay (0-1)


@dataclass(frozen=True)
class Nuclide:
    name: str
    half_life_s: float
    gammas: tuple = field(default_factory=tuple)
    # Bremsstrahlung intensity for pure-beta emitters (relative units)
    brems_fraction: float = 0.0
    category: str = "activation"  # activation | fission | actinide


# Half-life conversions
_DAY = 86_400.0
_YEAR = 365.25 * _DAY

NUCLIDES: dict[str, Nuclide] = {
    # --- Activation products (corrosion / neutron activation in primary loop) ---
    "Co-60": Nuclide(
        "Co-60", 5.2711 * _YEAR,
        gammas=(GammaLine(1173.2, 0.9985), GammaLine(1332.5, 0.9998)),
        category="activation",
    ),
    "Co-58": Nuclide(
        "Co-58", 70.86 * _DAY,
        gammas=(GammaLine(810.8, 0.9945), GammaLine(511.0, 0.30)),
        category="activation",
    ),
    "Mn-54": Nuclide(
        "Mn-54", 312.2 * _DAY,
        gammas=(GammaLine(834.8, 0.9997),),
        category="activation",
    ),
    "Fe-55": Nuclide(
        # Pure EC, emits Mn K-X-rays at ~5.9 keV; below typical NaI window
        "Fe-55", 2.737 * _YEAR,
        gammas=(GammaLine(5.9, 0.28),),
        category="activation",
    ),
    "Ni-63": Nuclide(
        # Pure beta, no gamma; bremsstrahlung continuum only
        "Ni-63", 101.2 * _YEAR,
        gammas=(),
        brems_fraction=0.02,
        category="activation",
    ),
    # --- Fission products ---
    "Cs-137": Nuclide(
        # 661.7 keV from metastable Ba-137m daughter
        "Cs-137", 30.08 * _YEAR,
        gammas=(GammaLine(661.7, 0.851),),
        category="fission",
    ),
    "Cs-134": Nuclide(
        "Cs-134", 2.0648 * _YEAR,
        gammas=(
            GammaLine(604.7, 0.9762),
            GammaLine(795.9, 0.8553),
            GammaLine(569.3, 0.1538),
        ),
        category="fission",
    ),
    "Sr-90": Nuclide(
        # Pure beta -> Y-90 (also pure beta, 2.28 MeV endpoint -> strong brems)
        "Sr-90", 28.79 * _YEAR,
        gammas=(),
        brems_fraction=0.05,
        category="fission",
    ),
    "I-131": Nuclide(
        "I-131", 8.0252 * _DAY,
        gammas=(GammaLine(364.5, 0.815), GammaLine(637.0, 0.0716)),
        category="fission",
    ),
    # --- Actinides (trace; mostly in ILW from primary system or fuel handling) ---
    "U-235": Nuclide(
        "U-235", 7.04e8 * _YEAR,
        gammas=(GammaLine(185.7, 0.572), GammaLine(143.8, 0.109)),
        category="actinide",
    ),
    "U-238": Nuclide(
        # 1001 keV from Pa-234m daughter (in secular equilibrium)
        "U-238", 4.468e9 * _YEAR,
        gammas=(GammaLine(1001.0, 0.0084), GammaLine(766.4, 0.0029)),
        category="actinide",
    ),
    "Pu-239": Nuclide(
        "Pu-239", 24_110 * _YEAR,
        gammas=(GammaLine(129.3, 6.31e-5), GammaLine(413.7, 1.47e-5)),
        category="actinide",
    ),
    "Am-241": Nuclide(
        "Am-241", 432.6 * _YEAR,
        gammas=(GammaLine(59.54, 0.359),),
        category="actinide",
    ),
}


# Typical specific-activity ranges (Bq/g) per IAEA waste class.
# Mix dictionaries give relative composition weights, not concentrations.
# Total activity is drawn separately; composition fractions sum to ~1.
WASTE_PROFILES: dict[str, dict] = {
    "VLLW": {
        "total_activity_bq_per_g_range": (0.1, 100.0),
        "composition_weights": {
            "Co-60": 0.15, "Cs-137": 0.35, "Cs-134": 0.05,
            "Mn-54": 0.05, "Sr-90": 0.15, "Fe-55": 0.10, "Ni-63": 0.15,
        },
    },
    "LLW": {
        "total_activity_bq_per_g_range": (1e2, 1e5),
        "composition_weights": {
            "Co-60": 0.25, "Co-58": 0.10, "Cs-137": 0.25, "Cs-134": 0.05,
            "Mn-54": 0.05, "Sr-90": 0.10, "Fe-55": 0.10, "Ni-63": 0.10,
        },
    },
    "ILW": {
        "total_activity_bq_per_g_range": (1e5, 1e10),
        "composition_weights": {
            "Co-60": 0.30, "Co-58": 0.10, "Cs-137": 0.20, "Cs-134": 0.08,
            "Mn-54": 0.07, "Sr-90": 0.05, "Fe-55": 0.05, "Ni-63": 0.05,
            "Am-241": 0.04, "Pu-239": 0.03, "U-235": 0.02, "U-238": 0.01,
        },
    },
    # HLW: high-heat-generating liquid from PUREX dissolution. In the real
    # process the gamma-rules engine never even sees these — operations know
    # this stream is HLW. The sim treats HLW items as "pre-classified by
    # process knowledge": drones don't NaI-scan them, they route directly
    # through Solidification (vitrification) to HLW storage.
    "HLW": {
        "total_activity_bq_per_g_range": (1e10, 1e13),
        "composition_weights": {
            "Cs-137": 0.30, "Cs-134": 0.12, "Sr-90": 0.20,
            "Co-60": 0.05, "I-131": 0.03,
            "Am-241": 0.10, "Pu-239": 0.08, "U-235": 0.05, "U-238": 0.07,
        },
    },
}


def waste_classes() -> list[str]:
    return list(WASTE_PROFILES.keys())
