"""Generates synthetic waste items with ground-truth radionuclide compositions
drawn from WASTE_PROFILES.

Each item has:
  * a true IAEA class (VLLW/LLW/ILW)
  * a total specific activity (Bq/g)
  * a mass (g)
  * an activities_bq dict (nuclide -> Bq), with the per-nuclide mix sampled
    around the profile's composition_weights with Dirichlet noise
  * a generation_point and a creation time
"""
from __future__ import annotations
from dataclasses import dataclass, field
import itertools
import math
import numpy as np

from .radionuclides import WASTE_PROFILES


@dataclass
class WasteItem:
    item_id: str
    true_class: str
    mass_g: float
    total_specific_activity_bq_per_g: float
    activities_bq: dict[str, float]
    generation_point: str
    created_at_s: float
    # Optional "tricky" flag: composition was deliberately ambiguous
    tricky: bool = False
    notes: str = ""
    # When set (e.g. items from the Dissolution cell), operations *knows*
    # the waste class up-front and skips NaI gamma classification. Drones
    # carrying such an item follow the pre-determined route directly:
    # HLW -> Solidification -> HLW storage. Contact-handling items (the
    # bulk of the demo) leave this as None and go through Quick scan.
    process_known_class: str | None = None


class WasteStream:
    """Mixes the IAEA classes at realistic ratios for LWR operations:
    ~70% LLW, ~25% VLLW, ~5% ILW (rough plant-operations ballpark)."""

    def __init__(
        self,
        rng: np.random.Generator,
        class_mix: dict[str, float] | None = None,
        tricky_fraction: float = 0.10,
    ):
        self.rng = rng
        self.class_mix = class_mix or {"VLLW": 0.25, "LLW": 0.70, "ILW": 0.05}
        self.tricky_fraction = tricky_fraction
        self._counter = itertools.count(1)

    def _sample_class(self) -> str:
        classes = list(self.class_mix.keys())
        weights = np.array(list(self.class_mix.values()))
        weights = weights / weights.sum()
        return str(self.rng.choice(classes, p=weights))

    def _sample_composition(self, profile: dict) -> dict[str, float]:
        """Sample per-nuclide fractions by perturbing the profile weights
        with a Dirichlet draw, so two items in the same class do not look
        identical."""
        nuclides = list(profile["composition_weights"].keys())
        base = np.array([profile["composition_weights"][n] for n in nuclides])
        # Concentration parameter controls how much the draw can wander from
        # the mean. 30 = moderate wandering.
        alpha = base * 30.0 + 1e-3
        fractions = self.rng.dirichlet(alpha)
        return dict(zip(nuclides, fractions))

    def generate(
        self,
        generation_point: str,
        created_at_s: float,
        force_tricky: bool = False,
        forced_class: str | None = None,
        process_known: bool = False,
    ) -> WasteItem:
        """Generate one waste item.

        forced_class:    if set, true_class is forced to this value (skips the
                         statistical class_mix draw). Used by upstream process
                         stages that produce a known waste stream (e.g.
                         Dissolution -> HLW).
        process_known:   if True, mark the item as pre-classified by process
                         knowledge — drones will not run NaI classification
                         and instead follow the typed route. Tricky-actinide
                         masquerade is disabled for these items.
        """
        if forced_class is not None:
            true_class = forced_class
        else:
            true_class = self._sample_class()
        profile = WASTE_PROFILES[true_class]

        lo, hi = profile["total_activity_bq_per_g_range"]
        # Log-uniform across the activity range
        log_sa = self.rng.uniform(math.log10(lo), math.log10(hi))
        specific_activity = 10 ** log_sa

        # Mass: 1-30 kg per item (drums, bags, components)
        mass_g = float(self.rng.uniform(1e3, 3e4))
        total_activity_bq = specific_activity * mass_g

        fractions = self._sample_composition(profile)

        # Process-known items skip the actinide-masquerade trick: in real
        # plants, the operator already knows what came out of which cell.
        tricky = (not process_known) and (
            force_tricky or (self.rng.random() < self.tricky_fraction)
        )
        notes = ""
        if tricky:
            # Tricky case: low total activity (looks LLW or VLLW) but with
            # an actinide spike that should push it to ILW once identified.
            if true_class in ("LLW", "VLLW"):
                # Drop activity to upper end of LLW range — but stay strictly
                # below LLW_max so the rules engine alone genuinely classifies
                # the item as LLW. The trick can only be caught via the
                # actinide-spike signature.
                specific_activity = self.rng.uniform(1e3, 5e4)
                total_activity_bq = specific_activity * mass_g
                # But add Am-241 contamination that demands ILW handling
                fractions = {n: f * 0.6 for n, f in fractions.items()}
                fractions["Am-241"] = 0.30
                fractions["Pu-239"] = 0.10
                # Re-normalize
                s = sum(fractions.values())
                fractions = {n: f / s for n, f in fractions.items()}
                true_class = "ILW"  # truth shifts because of actinides
                notes = "actinide-spike masquerading as LLW"

        activities = {n: total_activity_bq * f for n, f in fractions.items()}

        item_id = f"W-{next(self._counter):05d}"
        return WasteItem(
            item_id=item_id,
            true_class=true_class,
            mass_g=mass_g,
            total_specific_activity_bq_per_g=specific_activity,
            activities_bq=activities,
            generation_point=generation_point,
            created_at_s=created_at_s,
            tricky=tricky,
            notes=notes,
            process_known_class=(true_class if process_known else None),
        )
