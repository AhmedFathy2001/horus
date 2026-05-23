"""Per-arm redundancy state + failure model.

Every robotic arm in the plant comes as a primary + standby pair:
  * HPGe-A (primary) and HPGe-B (standby) in the QA lab
  * DECON-A and DECON-B at the Robot wash
  * Loader-A (currently active) and Loader-B (cold spare) at each
    storage door

When the primary unit fails (random fault, drawn from an exponential
MTBF), the standby takes over instantly. Drum throughput is unaffected;
the only visible change is which arm is doing the work. A repair
process puts the failed unit back into the rotation after
`repair_time_s` sim-seconds. This is the same federated-safety story
as the coordinator hot-standby — the system as a whole has zero
downtime so long as both units in a pair don't fail simultaneously.

The numbers here are chosen so the failover story is visible inside a
single 8 h demo shift but doesn't dominate the screen — roughly one
arm failure every 25 min on average, 4 min mean repair.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any


# Arm-failure tuning. Exponential MTBF across the fleet of pairs;
# divided across `n_pairs` so each individual pair sees a failure ≈
# every `ARM_MTBF_S` sim-seconds. ARM_REPAIR_S is kept short enough
# that the chance of BOTH units in a pair being down simultaneously is
# very low (≈ ARM_REPAIR_S / per-pair-MTBF squared per pair).
ARM_MTBF_S   = 1800.0   # ≈ 30 min between failures across the whole fleet
ARM_REPAIR_S =   45.0   # 45 s mean time-to-repair → "both down" is extremely rare


@dataclass
class ArmPair:
    """One redundant arm pair (HPGe, DECON, loader at one storage door)."""
    name: str                              # display name, e.g. "HPGe"
    units: tuple[str, str]                 # ("HPGe-A", "HPGe-B") etc.
    primary_idx: int = 0                   # 0 → A leads, 1 → B leads
    failed_units: set[int] = field(default_factory=set)
    repair_eta_s: dict[int, float] = field(default_factory=dict)
    failover_count: int = 0
    last_failover_t_s: float | None = None
    # True downtime — set when the failure process enters the BOTH-DOWN
    # state for this pair, cleared when either unit returns. The
    # downtime accumulator tracks the cumulative sim-seconds during
    # which neither unit was operating, so the panel can prove
    # "0 s downtime" honestly when redundancy worked, and call out the
    # rare exceptions.
    both_down_since_s: float | None = None
    total_downtime_s: float = 0.0

    def active_idx(self) -> int | None:
        """Which unit is currently doing the work. None means both units
        are down (system briefly degraded — shouldn't normally happen)."""
        if self.primary_idx not in self.failed_units:
            return self.primary_idx
        other = 1 - self.primary_idx
        if other not in self.failed_units:
            return other
        return None

    def status_for(self, idx: int) -> str:
        if idx in self.failed_units:
            return "failed"
        if idx == self.active_idx():
            return "active"
        return "standby"


def make_default_pairs() -> dict[str, ArmPair]:
    """Initial redundancy state for every arm pair in the plant."""
    return {
        "HPGe":           ArmPair("HPGe",           ("HPGe-A",  "HPGe-B")),
        "DECON":          ArmPair("DECON",          ("DECON-A", "DECON-B")),
        "VLLW_loader":    ArmPair("VLLW loader",    ("VLLW-LA", "VLLW-LB")),
        "LLW_loader":     ArmPair("LLW loader",     ("LLW-LA",  "LLW-LB")),
        "ILW_loader":     ArmPair("ILW loader",     ("ILW-LA",  "ILW-LB")),
        "HLW_loader":     ArmPair("HLW loader",     ("HLW-LA",  "HLW-LB")),
    }


def run_arm_failure_process(env, pairs: dict[str, ArmPair], rng,
                            event_sink: list[Any] | None = None):
    """Simpy process: forever, pick a random arm pair and fail its active
    unit. Schedule a repair `ARM_REPAIR_S` later. Push each event into
    `event_sink` (a plain list) so the live view can read + render them."""
    pair_names = list(pairs.keys())
    while True:
        # Time to next failure (exponential, scaled to overall MTBF
        # divided by number of pairs so per-pair rate stays reasonable
        # regardless of how many pairs we have).
        inter = float(rng.exponential(ARM_MTBF_S / max(len(pair_names), 1)))
        yield env.timeout(inter)

        # Pick a pair that has an active unit to fail. Skip pairs that
        # are entirely degraded (both units already down).
        candidates = [k for k in pair_names
                      if pairs[k].active_idx() is not None]
        if not candidates:
            continue
        key = candidates[int(rng.integers(0, len(candidates)))]
        pair = pairs[key]
        idx = pair.active_idx()
        if idx is None:
            continue
        pair.failed_units.add(idx)
        pair.repair_eta_s[idx] = env.now + ARM_REPAIR_S
        pair.failover_count += 1
        pair.last_failover_t_s = env.now

        # If BOTH units are now down, start the downtime timer for
        # this pair.
        if len(pair.failed_units) >= 2 and pair.both_down_since_s is None:
            pair.both_down_since_s = env.now

        if event_sink is not None:
            event_sink.append({
                "t_s": env.now,
                "kind": "arm_fail",
                "pair": pair.name,
                "unit": pair.units[idx],
                "took_over": pair.units[1 - idx]
                              if (1 - idx) not in pair.failed_units else None,
            })

        # Schedule a repair as a sub-process so the failed unit comes
        # back to STANDBY after `ARM_REPAIR_S`.
        env.process(_repair_after(env, pair, idx, ARM_REPAIR_S, event_sink))


def _repair_after(env, pair: ArmPair, idx: int, delay_s: float,
                  event_sink: list[Any] | None):
    yield env.timeout(delay_s)
    # Closing out a both-down window — accumulate the elapsed downtime
    # before clearing the failed flag.
    if len(pair.failed_units) >= 2 and pair.both_down_since_s is not None:
        pair.total_downtime_s += env.now - pair.both_down_since_s
        pair.both_down_since_s = None
    pair.failed_units.discard(idx)
    pair.repair_eta_s.pop(idx, None)
    if event_sink is not None:
        event_sink.append({
            "t_s": env.now,
            "kind": "arm_repair",
            "pair": pair.name,
            "unit": pair.units[idx],
        })
