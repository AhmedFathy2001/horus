"""Run-time metrics. Three headline numbers, plus a confusion matrix for the
dashboard:

  * classification accuracy (correct / total)
  * cumulative worker dose (mSv)
  * throughput (items moved to final storage per simulated hour)

Worker-dose proxy: every simulated second, for every waste item not yet in its
correct final storage zone, accumulate dose based on its dose rate and the
inverse-square distance from the nearest worker. Items in the (shielded) ILW
storage do not contribute. ILW items mis-classified as LLW (i.e. not shielded
in transit) accumulate exposure at the full unshielded rate.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from collections import defaultdict
import numpy as np

from .facility import worker_zones, distance_m
from .radionuclides import waste_classes


@dataclass
class RunMetrics:
    mode: str
    classifications: list[dict] = field(default_factory=list)
    items_completed: int = 0
    items_released: int = 0
    items_shipped: int = 0
    shipments: list = field(default_factory=list)
    cumulative_dose_uSv: float = 0.0
    sim_duration_s: float = 0.0
    # Optional per-tick history for plotting
    accuracy_history: list[tuple[float, float]] = field(default_factory=list)
    dose_history: list[tuple[float, float]] = field(default_factory=list)

    def record_classification(
        self,
        item_id: str,
        true_class: str,
        predicted_class: str,
        confidence: float,
        scrutiny: bool,
        agent_id: str,
        timestamp_s: float,
    ) -> None:
        self.classifications.append({
            "item_id": item_id,
            "true_class": true_class,
            "predicted_class": predicted_class,
            "confidence": confidence,
            "scrutiny": scrutiny,
            "agent_id": agent_id,
            "t_s": timestamp_s,
        })

    def record_dose_tick(
        self,
        dt_s: float,
        active_items: list[dict],
        timestamp_s: float,
    ) -> None:
        """active_items: each is {pos, dose_rate_uSv_h, shielded}.
        Dose contribution this tick: sum over items of
            (dose_rate / 3600) * (1/distance^2 factor) * dt_s
        ignoring shielded items.
        """
        if not active_items:
            return
        workers = worker_zones()
        if not workers:
            return
        increment = 0.0
        for item in active_items:
            if item.get("shielded"):
                continue
            ix, iy = item["pos"]
            min_d = min(distance_m((ix, iy), (w.x, w.y)) for w in workers)
            min_d = max(min_d, 0.5)  # avoid singularity
            geom = 1.0 / (min_d ** 2)
            increment += item["dose_rate_uSv_h"] * (dt_s / 3600.0) * geom
        self.cumulative_dose_uSv += increment
        self.dose_history.append((timestamp_s, self.cumulative_dose_uSv))

    def accuracy(self) -> float:
        if not self.classifications:
            return 0.0
        correct = sum(c["true_class"] == c["predicted_class"] for c in self.classifications)
        return correct / len(self.classifications)

    def throughput_per_hour(self) -> float:
        if self.sim_duration_s <= 0:
            return 0.0
        return self.items_completed / (self.sim_duration_s / 3600.0)

    def confusion_matrix(self) -> tuple[list[str], np.ndarray]:
        classes = waste_classes()
        idx = {c: i for i, c in enumerate(classes)}
        m = np.zeros((len(classes), len(classes)), dtype=int)
        for c in self.classifications:
            t = idx.get(c["true_class"])
            p = idx.get(c["predicted_class"])
            if t is None or p is None:
                continue
            m[t, p] += 1
        return classes, m

    def summary(self) -> dict:
        return {
            "mode": self.mode,
            "n_classified": len(self.classifications),
            "n_completed": self.items_completed,
            "accuracy": self.accuracy(),
            "cumulative_dose_uSv": self.cumulative_dose_uSv,
            "throughput_per_hour": self.throughput_per_hour(),
            "sim_duration_s": self.sim_duration_s,
        }

    def update_accuracy_history(self, timestamp_s: float) -> None:
        self.accuracy_history.append((timestamp_s, self.accuracy()))
