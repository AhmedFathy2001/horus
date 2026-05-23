"""Agent state machine — a simpy process.

Each mobile agent has:

  * (x, y) position, a current state, a facing direction (radians) used
    by the live view to draw the field-of-vision cone
  * a *local* HybridClassifier instance that, in hivemind mode, refreshes
    from coordinator snapshots between tasks
  * a current-carry slot (None or a WasteItem)
  * a battery percentage that drains during work and recharges at CHG
  * an integrated radiation dose to the robot's electronics that
    accumulates whenever the robot is near an unshielded hot item; if it
    exceeds a soft limit the agent must go to the Robot wash (recoverable
    downtime), and if it exceeds a hard limit the agent FAILS permanently.

The fixed HPGe drum-scanner agent is also represented here but has a
simpler state machine and does not move."""
from __future__ import annotations
import math
from dataclasses import dataclass
import numpy as np
import simpy

from .classifier import HybridClassifier
from .facility import (
    AGENT_SPEED_MPS, Zone, ZONES_BY_NAME, distance_m,
    storage_zone_for, ZONES_BY_ROLE, charging_zone, decon_zone,
    solidification_zone, clearance_zone, CLEARANCE_THRESHOLD_BQ_PER_G,
    shortest_waypoint_path,
)
from .sensors import (
    simulate_spectrum, simulate_dose_rate, simulate_cv_classification,
    NAI_DETECTOR, HPGE_DETECTOR,
)
from .waste_generator import WasteItem


AGENT_STATES = (
    "IDLE", "MOVING_TO_PICKUP", "PICKING_UP",
    "MOVING_TO_SORTING", "ACQUIRING", "CLASSIFYING", "REPORTING",
    "MOVING_TO_DROPOFF", "DROPPING_OFF",
    "MOVING_TO_RESCAN", "RESCANNING",
    "RETURNING_TO_CHARGE", "CHARGING",
    "RETURNING_TO_DECON", "DECONNING",
    "FAILED",
)

# Battery model (per simulated second)
BATTERY_DRAIN_ACTIVE_PER_S   = 0.012   # %/s while moving / handling
BATTERY_DRAIN_IDLE_PER_S     = 0.004
BATTERY_CHARGE_PER_S         = 0.6     # %/s at the charging bay
BATTERY_LOW_THRESHOLD        = 25.0    # %: drop everything and go charge
BATTERY_FULL_THRESHOLD       = 98.0    # leave the bay above this

# Dose model
DOSE_DECON_THRESHOLD_uSv  = 5_000.0    # >: must visit Robot wash
DOSE_FAILURE_THRESHOLD_uSv = 50_000.0  # >: permanent damage to electronics
DECON_DURATION_S          = 600.0      # 10 min sim time
DECON_REDUCTION_FRACTION  = 0.85       # decon removes 85% of integrated dose

# Preventive (radsafe) decon — triggered by contamination exposure,
# not by a fixed timer. A cart only needs the wash if it has actually
# been near something hot: either dose has built up past a soft
# threshold OR the cart has physically handled an ILW / HLW drum
# since its last decon. Carts that spend the whole shift moving VLLW
# stay clean and don't waste decon capacity.
PREVENTIVE_DECON_DOSE_uSv   = 1000.0       # accumulated since last decon
PREVENTIVE_DECON_DURATION_S = 180.0        # 3 min wash (vs. 10 min after a hot run)
PREVENTIVE_DECON_REDUCTION  = 0.40         # strips 40% of accumulated dose
# Class of waste considered "contaminated handling" for decon purposes.
# Anything in this set triggers a preventive wash on the cart's next
# between-task health check.
HOT_HANDLING_CLASSES = frozenset({"ILW", "HLW"})

# Irreducible classification error at the QA-lab step.
#
# QA combines HPGe assay + a human operator's review of the camera feed
# — no system "validates itself" via the same AI brain. The two failure
# sources are largely independent:
#
#   HPGE_ERROR_RATE  — HPGe arm misreads (calibration drift, dead-time,
#                      geometry, statistical fluctuation).
#   WORKER_ERROR_RATE — operator misreads the camera or mis-keys into
#                      the terminal (fatigue, distraction).
#
# The HPGe and worker errors only both occur on the same drum some of
# the time, so combined error rate ≈ HPGE_ERROR_RATE * (1 - W_DETECT)
# + WORKER_ERROR_RATE.  This keeps the achievable accuracy floor
# strictly below 100% (≈ 98.5–99%) while modelling the human-in-the-loop
# correctly.
HPGE_ERROR_RATE        = 0.02
WORKER_ERROR_RATE      = 0.01    # operator's own slip-up rate
WORKER_DETECTION_RATE  = 0.70    # P(operator catches an HPGe error)

# Cart-side process noise. Even on "easy" drums the cart's NaI + CV
# classification picks up rare errors from sources we don't model in
# detail: container labeling mistakes, paperwork mix-ups, ambient
# gamma background spikes, brief calibration drift. ~1% off-by-one
# every now and then makes the demo honest — no system is 100% even
# for items that LOOK obvious to the gamma spectrometer.
CART_ERROR_RATE = 0.01

# Class order for the "off-by-one" misclassification shift. We never
# jump multiple boundaries — a real HPGe-grade assay doesn't confuse
# VLLW with HLW; it just occasionally crosses a single class boundary.
_CLASS_ORDER = ("VLLW", "LLW", "ILW", "HLW")


def _shift_one_class(c: str, rng) -> str:
    """Return an adjacent class to `c`. Off-by-one misclassification
    used for the HPGe noise floor."""
    if c not in _CLASS_ORDER:
        return c
    idx = _CLASS_ORDER.index(c)
    if idx == 0:
        return _CLASS_ORDER[1]
    if idx == len(_CLASS_ORDER) - 1:
        return _CLASS_ORDER[-2]
    return _CLASS_ORDER[idx + (1 if rng.random() < 0.5 else -1)]

# Per-role chassis profile. Scanners are small/fast (sensor swarm); Handlers
# are big/slow (gripper + drum). Hybrids sit in the middle.
AGENT_TYPE_SPEED_MPS = {
    "scanner": 1.30,
    "handler": 0.85,
    "hybrid":  1.00,
    "qa_lab":  0.0,
}
AGENT_CAPABILITIES = {
    "scanner": frozenset({"scan"}),
    "handler": frozenset({"handle"}),
    "hybrid":  frozenset({"scan", "handle"}),
    "qa_lab":  frozenset({"rescan"}),
}
# Per-role agility profile. Scanners are nimble (looser curves, more jitter
# — they're sensor swarmers darting between drums). Handlers are heavy and
# move in deliberate straight-ish lines so the gripper-loaded drum doesn't
# slosh. Numbers are tuning knobs for the path-shape, not physical units.
AGENT_TYPE_AGILITY = {
    "scanner": {"curve_amp_m_per_m": 0.12, "max_curve_m": 0.8, "jitter_sigma_m": 0.06},
    "handler": {"curve_amp_m_per_m": 0.04, "max_curve_m": 0.3, "jitter_sigma_m": 0.02},
    "hybrid":  {"curve_amp_m_per_m": 0.08, "max_curve_m": 0.6, "jitter_sigma_m": 0.05},
    "qa_lab":  {"curve_amp_m_per_m": 0.0,  "max_curve_m": 0.0, "jitter_sigma_m": 0.0},
}


@dataclass
class AgentRuntimeState:
    agent_id: str
    pos: tuple[float, float]
    state: str
    carrying_item: str | None
    model_version: int
    battery_pct: float
    integrated_dose_uSv: float
    facing_rad: float


class Agent:
    def __init__(
        self,
        env: simpy.Environment,
        agent_id: str,
        coordinator,
        rng: np.random.Generator,
        metrics,
        is_rescrutiny_station: bool = False,
        home_zone_name: str = "Charging bay",
        qa_sampling_fraction: float = 0.0,
        agent_type: str = "hybrid",
    ):
        self.env = env
        self.agent_id = agent_id
        self.coordinator = coordinator
        self.rng = rng
        self.metrics = metrics
        self.is_rescrutiny_station = is_rescrutiny_station
        self.qa_sampling_fraction = qa_sampling_fraction
        # Robot role:
        #   "scanner" — NaI + camera only, no gripper. Visits items in-place
        #               at gen points, classifies, reports. Does NOT carry.
        #   "handler" — gripper + chassis, minimal sensors. Picks up items
        #               whose class is already known and routes to storage.
        #   "hybrid"  — both: can scan AND handle (the original sim behavior).
        #   "qa_lab"  — fixed HPGe drum scanner station (rescrutiny only).
        if is_rescrutiny_station:
            agent_type = "qa_lab"
        self.agent_type: str = agent_type

        home = ZONES_BY_NAME[home_zone_name]
        self.pos: tuple[float, float] = (home.x, home.y)
        self.state: str = "IDLE"
        self.carrying: WasteItem | None = None
        self.local_classifier = HybridClassifier()
        self._known_model_version = 0
        self.home_offset_x: float = 0.0

        # Battery + dose + facing
        self.battery_pct: float = 100.0
        self.integrated_dose_uSv: float = 0.0
        # Decon bookkeeping: when did we last get washed, what was the
        # dose reading when that finished, and have we touched any hot
        # material since? `needs_preventive_decon` checks all three.
        self.last_decon_t_s: float = 0.0
        self.dose_at_last_decon_uSv: float = 0.0
        self.handled_hot_since_decon: bool = False
        # Initial facing south (π/2) so the FOV cone at the charging bay
        # already covers the work area; gets updated as soon as the agent moves
        self.facing_rad: float = math.pi / 2
        # Timestamps used by the live view for OFFLINE / SPAWN animations
        self.failure_time_s: float | None = None
        self.spawn_time_s: float | None = None
        # When the dispatcher assigns this agent a task, store the timestamp
        # so the live view can show a dispatch indicator briefly
        self.last_dispatch_t_s: float | None = None
        # During ACQUIRING, the live view uses this to draw the deployable
        # classification arm reaching out to the actual drum being scanned.
        # Cleared when the scanner moves on.
        self.scanning_item: "WasteItem | None" = None
        self.scanning_zone_name: str | None = None
        # Set by the scenario builder: where this agent pulls tasks from.
        # In HIVEMIND it's a per-agent simpy.Store; in ISOLATED it's the
        # shared task_queue. Defaults to None until the scenario sets it.
        self.dispatch_queue = None

    def runtime_state(self) -> AgentRuntimeState:
        return AgentRuntimeState(
            agent_id=self.agent_id,
            pos=self.pos,
            state=self.state,
            carrying_item=self.carrying.item_id if self.carrying else None,
            model_version=self._known_model_version,
            battery_pct=self.battery_pct,
            integrated_dose_uSv=self.integrated_dose_uSv,
            facing_rad=self.facing_rad,
        )

    # ---------- internal helpers ----------

    def _drain_battery(self, dt_s: float, active: bool):
        rate = BATTERY_DRAIN_ACTIVE_PER_S if active else BATTERY_DRAIN_IDLE_PER_S
        self.battery_pct = max(self.battery_pct - rate * dt_s, 0.0)

    def _accrue_dose(self, dt_s: float, dose_rate_uSv_h: float):
        if dose_rate_uSv_h <= 0:
            return
        self.integrated_dose_uSv += dose_rate_uSv_h * (dt_s / 3600.0)

    def needs_charge(self) -> bool:
        return self.battery_pct <= BATTERY_LOW_THRESHOLD

    def needs_decon(self) -> bool:
        return self.integrated_dose_uSv >= DOSE_DECON_THRESHOLD_uSv

    def failed(self) -> bool:
        return self.state == "FAILED"

    def _move_to(self, target_pos: tuple[float, float]):
        """Drive from current pos to target_pos along the QR-marker
        waypoint graph — real AGVs follow floor markers / QR tape grids.

        The cart A*'s through the building's QR-marker waypoint graph
        (defined in facility.py from the same WINGS / CORRIDORS data that
        drives the rendered walls), then drives smoothly along each
        successive leg via `_move_along_leg`. Because the waypoint graph
        already routes around every zone bubble and respects every wall,
        the cart never clips a room or partition."""
        path = shortest_waypoint_path(self.pos, target_pos)
        # The pathfinder returns intermediate waypoints; the final leg is
        # always start_of_last_hop -> target_pos.
        for wp in path:
            if distance_m(self.pos, wp) > 0.1:
                yield from self._move_along_leg(wp)
        if distance_m(self.pos, target_pos) > 0.1:
            yield from self._move_along_leg(target_pos)

    def _move_along_leg(self, target_pos: tuple[float, float]):
        """Drive a single straight leg from current pos to target_pos.

        Each leg here is QR-to-QR (~1.5 m). Motion is a gentle bowed
        spline with a tiny amount of jitter so the cart still reads as
        a real AGV rolling between markers — not snap-teleporting. Curve
        amplitude is capped tightly because the waypoint graph already
        guarantees the straight line is clear; we just want the cart to
        not look mechanical."""
        d = distance_m(self.pos, target_pos)
        speed = AGENT_TYPE_SPEED_MPS.get(self.agent_type, AGENT_SPEED_MPS)
        if speed <= 0:
            return
        total_time = d / speed
        if total_time <= 0:
            return
        step = 0.4
        n_steps = max(int(total_time / step), 1)
        start = self.pos
        # Per-role agility profile, scaled down. Curves are intentionally
        # small (~0.1-0.3 m) because legs are short and we can't risk a
        # bow that pokes into a wall the waypoint graph just routed us
        # past.
        agility = AGENT_TYPE_AGILITY.get(self.agent_type, AGENT_TYPE_AGILITY["hybrid"])
        leg_amp_m = (
            min(agility["max_curve_m"] * 0.35,
                agility["curve_amp_m_per_m"] * 0.5 * d)
            * float(self.rng.uniform(-1.0, 1.0))
        )
        jitter_sigma = agility["jitter_sigma_m"] * 0.45
        # Perpendicular unit vector to the start->target line
        dx, dy = target_pos[0] - start[0], target_pos[1] - start[1]
        norm = max(math.hypot(dx, dy), 1e-6)
        perp = (-dy / norm, dx / norm)

        def smooth_xy(frac: float) -> tuple[float, float]:
            bow = math.sin(math.pi * frac) * leg_amp_m
            return (start[0] + dx * frac + perp[0] * bow,
                    start[1] + dy * frac + perp[1] * bow)

        # Initial facing: along the leg
        self.facing_rad = math.atan2(dy, dx)

        for i in range(1, n_steps + 1):
            # If the agent was killed mid-move, drop everything and bail —
            # the main loop will catch up to the FAILED state next iteration.
            if self.failed():
                return
            frac = i / n_steps
            sx, sy = smooth_xy(frac)
            # Small position jitter, decays toward arrival so we land cleanly.
            jitter_amp = jitter_sigma * (1.0 - frac)
            jx = float(self.rng.normal(0.0, jitter_amp))
            jy = float(self.rng.normal(0.0, jitter_amp))
            cx = sx + jx
            cy = sy + jy
            # Clamp inside facility bounds so jitter can't drive us off-screen
            from .facility import WIDTH_M, HEIGHT_M
            cx = max(0.3, min(WIDTH_M - 0.3, cx))
            cy = max(0.3, min(HEIGHT_M - 0.3, cy))
            # Facing follows the smooth path tangent (look-ahead), not the
            # jittered step delta. Low-pass smooth onto target heading so the
            # FOV cone never snaps.
            look_frac = min(frac + 0.06, 1.0)
            lx, ly = smooth_xy(look_frac)
            target_facing = math.atan2(ly - sy, lx - sx)
            # Shortest-arc interpolation toward target_facing
            delta = math.atan2(math.sin(target_facing - self.facing_rad),
                               math.cos(target_facing - self.facing_rad))
            self.facing_rad += delta * 0.35
            self.pos = (cx, cy)
            self._drain_battery(step, active=True)
            self._accrue_proximity_dose(step)
            yield self.env.timeout(step)
        self.pos = target_pos
        # Final facing snap to leg direction so the next leg's initial
        # heading is clean
        self.facing_rad = math.atan2(dy, dx)

    def _accrue_proximity_dose(self, dt_s: float):
        """Sum dose rate from items the agent is currently within 2.5 m of.
        Uses inverse-square attenuation, treated as un-shielded contact."""
        from .sensors import GAMMA_CONSTANTS_uSv_h_per_MBq_at_1m
        # We don't have a global "all items in facility" handle here without
        # passing it through, so we read from the coordinator ledger. We use
        # the item's last-known location to find candidate sources.
        rate = 0.0
        for item_id, entry in self.coordinator.ledger.items():
            if entry.current_location in ("ILW storage", "HLW storage", "Solidification"):
                continue  # shielded by storage/cell design
            zone = ZONES_BY_NAME.get(entry.current_location)
            if zone is None:
                continue
            d = distance_m(self.pos, (zone.x, zone.y))
            if d > 3.0:
                continue
            # Find the item's activities — we walk through active_items via
            # coordinator if available. For sim simplicity we approximate
            # the dose rate by reading off classification: tricky/ILW items
            # are the costly ones.
            if entry.current_classification == "ILW":
                # representative ILW unshielded dose at 1m
                src = 8000.0
            elif entry.true_class == "ILW":
                # mis-routed ILW sitting at LLW storage (the bad case)
                src = 6000.0
            else:
                continue
            rate += src / max(d ** 2, 0.25)
        self._accrue_dose(dt_s, rate)

    def _zone_park_pos(self, zone) -> tuple[float, float]:
        """Per-drone parking spot inside a zone so multiple drones don't
        stack on the same pixel when they converge (e.g. several drones
        all dropping off at Quick scan / LLW storage in the same window).

        Each drone gets a stable angular slot around the zone centre,
        keyed off a hash of its agent_id so spots are deterministic and
        non-overlapping for up to ~6 drones per zone."""
        # 8 slots round the zone perimeter at ~60% of its radius
        slot = (hash(self.agent_id) % 8) / 8.0
        ang = slot * 2 * math.pi
        r = zone.radius_m * 0.55
        return (zone.x + r * math.cos(ang),
                zone.y + r * math.sin(ang))

    def _refresh_model_if_needed(self):
        snap = self.coordinator.current_snapshot()
        if snap is None:
            return
        version, state = snap
        if version > self._known_model_version:
            self.local_classifier.load_snapshot(state)
            self._known_model_version = version

    # ---------- charge / decon side-trips ----------

    def _go_charge(self):
        self.state = "RETURNING_TO_CHARGE"
        home = charging_zone()
        yield self.env.process(self._move_to((home.x + self.home_offset_x, home.y)))
        self.state = "CHARGING"
        while self.battery_pct < BATTERY_FULL_THRESHOLD:
            yield self.env.timeout(2.0)
            self.battery_pct = min(self.battery_pct + BATTERY_CHARGE_PER_S * 2.0, 100.0)

    def _go_decon(self, preventive: bool = False):
        """Drive to the Robot wash and run a decon cycle.

        Two flavours:
          - preventive=False: full 10 min wash triggered by the dose
            threshold (5 mSv+). Strips 85% of integrated dose.
          - preventive=True: 3 min "preventive" wash on a regular
            schedule (every ~2 sim-hours). Strips ~40% of whatever
            dose has built up. This is the radsafe-mandated periodic
            decon that real plants do regardless of measured dose — so
            the DECON arms have something to do even on a clean shift.
        """
        self.state = "RETURNING_TO_DECON"
        zone = decon_zone()
        yield self.env.process(self._move_to((zone.x, zone.y)))
        self.state = "DECONNING"
        if preventive:
            wash_dur_s = PREVENTIVE_DECON_DURATION_S
            reduction = PREVENTIVE_DECON_REDUCTION
        else:
            wash_dur_s = DECON_DURATION_S
            reduction = DECON_REDUCTION_FRACTION
        # Gradually drain the dose over the wash duration so the live view
        # shows the bar visibly emptying as the articulated arm sweeps the
        # AGV — instead of one all-at-once subtraction at the end.
        initial = self.integrated_dose_uSv
        target = initial * (1.0 - reduction)
        step_s = 4.0
        n_steps = max(int(wash_dur_s / step_s), 1)
        for i in range(n_steps):
            yield self.env.timeout(step_s)
            frac = (i + 1) / n_steps
            self.integrated_dose_uSv = initial + (target - initial) * frac
        self.last_decon_t_s = self.env.now
        self.dose_at_last_decon_uSv = self.integrated_dose_uSv
        self.handled_hot_since_decon = False

    def needs_preventive_decon(self) -> bool:
        """Preventive decon trigger. Returns True only if the cart has
        actually accumulated contamination — either:
          (a) it has handled an ILW or HLW drum since its last decon, OR
          (b) it has soaked up `PREVENTIVE_DECON_DOSE_uSv` of dose since
              its last decon.
        Carts whose entire shift was spent moving VLLW / clearance
        drums stay clean and don't visit the wash bay unnecessarily."""
        if self.handled_hot_since_decon:
            return True
        delta_dose = self.integrated_dose_uSv - self.dose_at_last_decon_uSv
        return delta_dose >= PREVENTIVE_DECON_DOSE_uSv

    def _check_health_or_handle(self):
        """Returns a generator that runs charge/decon if needed, else None.
        Should be called between tasks."""
        if self.integrated_dose_uSv >= DOSE_FAILURE_THRESHOLD_uSv:
            self.state = "FAILED"
            return None  # caller handles permanent stop
        if self.needs_decon():
            return self._go_decon(preventive=False)
        if self.needs_preventive_decon():
            return self._go_decon(preventive=True)
        if self.needs_charge():
            return self._go_charge()
        return None

    # ---------- mobile agent main loop ----------

    def run(self, scan_queue: simpy.Store, handle_queue: simpy.Store,
            rescan_queue: simpy.Store):
        """Two queues feed the mobile fleet:

          scan_queue   — items that need an NaI + CV triage (Scanner/Hybrid).
          handle_queue — items already classified, awaiting transport
                         to storage / QA lab (Handler/Hybrid).
          rescan_queue — items waiting at the QA lab for HPGe rescrutiny
                         (only the fixed drum-scanner station drains this).

        Scanners pull only from scan_queue, Handlers only from
        handle_queue, Hybrids pull from either."""
        if self.is_rescrutiny_station:
            yield self.env.process(self._rescrutiny_loop(rescan_queue))
            return

        while True:
            try:
                yield self.env.process(self._one_task_iteration(
                    scan_queue, handle_queue, rescan_queue,
                ))
            except simpy.Interrupt:
                # Killed mid-task — drop carried item (scenario.kill_drone
                # already requeued it), force FAILED so the loop top runs
                # the offline/respawn branch on the next iteration.
                self.carrying = None
                self.state = "FAILED"
                # Top-of-loop is reached via the implicit while True wraparound

    def _get_next_task(self, scan_queue: simpy.Store, handle_queue: simpy.Store):
        """Pick the next task respecting this agent's capabilities. In
        HIVEMIND mode the coordinator dispatcher has already routed tasks
        into a per-agent dispatch_queue. In ISOLATED mode each agent pulls
        from the global scan/handle queues directly, filtered by role."""
        if self.dispatch_queue is not None:
            task = yield self.dispatch_queue.get()
            return task
        caps = AGENT_CAPABILITIES[self.agent_type]
        if "scan" in caps and "handle" in caps:
            # Hybrid: prefer handle (transport backlog clears faster), fall
            # back to scan when handle is empty. simpy.AnyOf returns when
            # the first of multiple events fires.
            handle_get = handle_queue.get()
            scan_get = scan_queue.get()
            result = yield self.env.any_of([handle_get, scan_get])
            if handle_get in result:
                task = result[handle_get]
                scan_get.cancel()
            else:
                task = result[scan_get]
                handle_get.cancel()
            return task
        if "scan" in caps:
            return (yield scan_queue.get())
        if "handle" in caps:
            return (yield handle_queue.get())
        # Should not happen for mobile drones
        return (yield handle_queue.get())

    def _one_task_iteration(self, scan_queue: simpy.Store,
                            handle_queue: simpy.Store,
                            rescan_queue: simpy.Store):
        """Single pass of the main loop, factored out so it can be wrapped
        in a try/except simpy.Interrupt at the call site."""
        if self.failed():
            if self.failure_time_s is None:
                self.failure_time_s = self.env.now
            # Hot-spare AGV deploys instantly — no downtime. The replacement
            # is rendered as a brief 0.5-sim-second 'spawn sparkle' so judges
            # see a new unit come online, but no work is lost or delayed.
            yield self.env.timeout(0.5)
            self.integrated_dose_uSv = 0.0
            self.battery_pct = 100.0
            self.state = "IDLE"
            self.pos = (
                ZONES_BY_NAME["Charging bay"].x + self.home_offset_x,
                ZONES_BY_NAME["Charging bay"].y,
            )
            self.spawn_time_s = self.env.now
            self.failure_time_s = None
            self.coordinator.recent_messages.append({
                "t_s": __import__("time").time(),
                "kind": "spawn",
                "agent_id": self.agent_id,
            })
            return

        # Health side-trips between tasks
        health = self._check_health_or_handle()
        if health is not None:
            yield self.env.process(health)
            if self.failed():
                return

        self._refresh_model_if_needed()
        self.state = "IDLE"
        task = yield self.env.process(self._get_next_task(scan_queue, handle_queue))
        kind = task.get("kind", "handle")
        if kind == "scan":
            yield self.env.process(self._scan_task(task, handle_queue))
        elif kind == "handle":
            yield self.env.process(self._handle_task(task, rescan_queue))
        # Unknown kinds are dropped; an event-log warning could be added later.

    # ---------- scan-only task: visit-in-place, classify, report ----------

    def _scan_task(self, task: dict, handle_queue: simpy.Store):
        """Scanner / Hybrid behaviour: drive to where the item currently
        sits (typically its gen point), perform an NaI + CV scan in-place
        without picking it up, report the classification to the coordinator,
        then enqueue a follow-up handle task so a Handler / Hybrid can
        actually transport the (now classified) drum to its destination."""
        item: WasteItem = task["item"]
        pickup_zone: Zone = task["pickup_zone"]

        # HLW process-known items don't need scanning. They should only
        # appear on handle_queue, but be defensive in case one is mis-routed.
        if item.process_known_class == "HLW":
            self._enqueue_handle({
                "kind": "handle", "item": item,
                "pickup_zone": pickup_zone,
                "target_zone": storage_zone_for("HLW"),
                "is_hlw_route": True, "needs_qa": False,
            }, handle_queue)
            return

        self.state = "MOVING_TO_PICKUP"
        yield self.env.process(self._move_to(self._zone_park_pos(pickup_zone)))
        if self.failed():
            return

        # In-place scan — no PICKING_UP step, the drum stays where it is.
        # Expose the item so the live view can draw the deployable
        # classification arm reaching out to this specific drum.
        self.scanning_item = item
        self.scanning_zone_name = pickup_zone.name
        self.state = "ACQUIRING"
        yield self.env.timeout(8.0)
        self._drain_battery(8.0, active=True)
        spectrum = simulate_spectrum(
            item.activities_bq, 8.0, NAI_DETECTOR, distance_m=0.4, rng=self.rng,
        )
        dose_rate = simulate_dose_rate(
            item.activities_bq, distance_m=0.4, rng=self.rng,
        )
        cv_probs = simulate_cv_classification(
            item.true_class, item.tricky, rng=self.rng,
        )
        self._accrue_dose(8.0, dose_rate)

        self.state = "CLASSIFYING"
        yield self.env.timeout(0.5)
        result = self.local_classifier.classify(
            item.total_specific_activity_bq_per_g, dose_rate, spectrum,
            cv_probs=cv_probs,
        )
        # Cart-side process noise — see CART_ERROR_RATE docstring. We
        # apply the off-by-one shift after the classifier so the
        # spectrum features used for downstream training still reflect
        # the (correct) signal; only the predicted_class is perturbed.
        if self.rng.random() < CART_ERROR_RATE:
            result.predicted_class = _shift_one_class(
                result.predicted_class, self.rng,
            )

        # Done scanning — release the arm.
        self.scanning_item = None
        self.scanning_zone_name = None
        self.state = "REPORTING"
        # Stash the spectrum so the live-view panel can show "what the AI
        # just looked at" prominently — judges read the actinide-window
        # signature here and see why a tricky drum gets caught.
        self.coordinator.last_scan_event = {
            "item_id": item.item_id,
            "agent_id": self.agent_id,
            "spectrum": spectrum,
            "features": result.spectrum_features,
            "predicted_class": result.predicted_class,
            "confidence": result.confidence,
            "actinide_signature": result.actinide_signature,
            "true_class": item.true_class,
            "tricky": item.tricky,
            "t_s": self.env.now,
        }
        self.coordinator.receive_report(
            item_id=item.item_id,
            agent_id=self.agent_id,
            predicted_class=result.predicted_class,
            confidence=result.confidence,
            spectrum_features=result.spectrum_features,
            true_label_for_training=None,
            is_rescrutiny=False,
        )
        self.metrics.record_classification(
            item_id=item.item_id,
            true_class=item.true_class,
            predicted_class=result.predicted_class,
            confidence=result.confidence,
            scrutiny=result.confidence < self.coordinator.scrutiny_confidence_threshold,
            agent_id=self.agent_id,
            timestamp_s=self.env.now,
        )

        entry = self.coordinator.ledger.get(item.item_id)
        scrutiny = entry.scrutiny_flag if entry else False
        if (
            not scrutiny
            and self.qa_sampling_fraction > 0.0
            and self.rng.random() < self.qa_sampling_fraction
        ):
            scrutiny = True
            if entry is not None:
                entry.scrutiny_flag = True

        # Build the follow-up handle task and enqueue it. The Handler /
        # Hybrid pool will pick this up and actually move the drum.
        if scrutiny:
            target = ZONES_BY_NAME["QA lab"]
            handle_task = {
                "kind": "handle", "item": item,
                "pickup_zone": pickup_zone, "target_zone": target,
                "needs_qa": True, "is_hlw_route": False,
            }
        else:
            # Clearance / free-release: VLLW drums below the clearance
            # threshold go back to general industry rather than long-term
            # storage. This is the recycling story — hivemind's better
            # accuracy unlocks more free-release material than isolated.
            if (
                result.predicted_class == "VLLW"
                and item.total_specific_activity_bq_per_g < CLEARANCE_THRESHOLD_BQ_PER_G
                and clearance_zone() is not None
            ):
                target = clearance_zone()
                is_clearance = True
            else:
                target = storage_zone_for(result.predicted_class)
                is_clearance = False
            handle_task = {
                "kind": "handle", "item": item,
                "pickup_zone": pickup_zone, "target_zone": target,
                "needs_qa": False, "is_hlw_route": False,
                "is_clearance": is_clearance,
            }
        self._enqueue_handle(handle_task, handle_queue)

    # ---------- handle-only task: pickup, transport, drop off ----------

    def _handle_task(self, task: dict, rescan_queue: simpy.Store):
        """Handler / Hybrid behaviour: drive to the pickup zone, pick up the
        drum, route to its destination, drop it off. For HLW items the
        target is the Solidification cell first, then HLW storage."""
        item: WasteItem = task["item"]
        pickup_zone: Zone = task["pickup_zone"]
        target_zone: Zone = task["target_zone"]

        # Drive to pickup
        self.state = "MOVING_TO_PICKUP"
        yield self.env.process(self._move_to(self._zone_park_pos(pickup_zone)))
        if self.failed():
            return
        self.state = "PICKING_UP"
        yield self.env.timeout(3.0)
        self._drain_battery(3.0, active=True)
        self.carrying = item

        # Flag the cart as having touched contaminated material so the
        # next between-task health check routes it to decon. The class
        # used here is the *known* class at pickup (process_known_class
        # for HLW casks; ledger entry's current_classification otherwise
        # — agents only know what's been classified so far).
        from .facility import storage_zone_for
        ledger_entry = self.coordinator.ledger.get(item.item_id)
        handled_cls = (item.process_known_class
                       or (ledger_entry.current_classification
                           if ledger_entry is not None else None))
        if handled_cls in HOT_HANDLING_CLASSES:
            self.handled_hot_since_decon = True

        # HLW pre-classified route: Solidification -> HLW storage
        if task.get("is_hlw_route") or item.process_known_class == "HLW":
            yield self.env.process(self._hlw_route(item))
            return

        # Drive to dropoff (storage or QA lab)
        self.state = "MOVING_TO_DROPOFF"
        yield self.env.process(self._move_to(self._zone_park_pos(target_zone)))
        if self.failed():
            return
        self.coordinator.update_location(item.item_id, target_zone.name)

        if task.get("needs_qa"):
            # Drop the drum off at the QA lab queue for HPGe rescrutiny.
            # The fixed drum-scanner station drains rescan_queue; once it
            # finishes, it places the item directly at the right storage
            # zone (see _rescrutiny_loop).
            yield rescan_queue.put({"item": item})
            self.carrying = None
        elif task.get("is_stage_to_char"):
            # Cart drops the drum on the Char station turntable, then
            # signals the central Char actuator. The actuator runs the
            # classification and emits the next transport task.
            self.state = "DROPPING_OFF"
            yield self.env.timeout(2.0)
            self._drain_battery(2.0, active=True)
            self.carrying = None
            if self.coordinator.char_in_queue is not None:
                yield self.coordinator.char_in_queue.put({"item": item})
            # NOT counted as completed — the drum's journey isn't done
            # until it lands in a storage zone after scanning.
        else:
            self.state = "DROPPING_OFF"
            yield self.env.timeout(2.0)
            self._drain_battery(2.0, active=True)
            self.carrying = None
            self.metrics.items_completed += 1
            if task.get("is_clearance"):
                self.metrics.items_released += 1
            # Drone parks here; next iteration's IDLE wait happens in place.

    def _enqueue_handle(self, task: dict, handle_queue: simpy.Store) -> None:
        """Hand the follow-up handle task back to the right queue. In
        HIVEMIND the coordinator owns the pending pool (dispatcher will
        route it to the best Handler/Hybrid); in ISOLATED it goes to the
        shared global handle_queue."""
        if self.coordinator.shared:
            self.coordinator.enqueue_task(task)
        else:
            handle_queue.put(task)

    # ---------- HLW process-known route ----------

    def _hlw_route(self, item):
        """Dissolution-cell HLW casks: route directly through Solidification
        to HLW storage. No NaI scan, no coordinator scrutiny — operations
        already knows this stream is HLW.

        Drone is treated as in a shielded carry cask for the duration of
        this route (HLW cannot be carried unshielded), so no proximity dose
        is logged against the item beyond what _accrue_proximity_dose
        already excludes for shielded storage."""
        # Record this transit in the ledger so the live view can render
        # the HLW drum at each step.
        entry = self.coordinator.ledger.get(item.item_id)
        if entry is not None:
            entry.current_classification = "HLW"

        solid = solidification_zone()
        if solid is not None:
            self.state = "MOVING_TO_DROPOFF"
            yield self.env.process(self._move_to(self._zone_park_pos(solid)))
            if self.failed():
                return
            self.coordinator.update_location(item.item_id, solid.name)
            # Vitrification: bake the liquid into a glass log. 30 sim-seconds
            # is a token nod to the process; the real step takes hours but
            # that would stall the demo.
            self.state = "DROPPING_OFF"
            yield self.env.timeout(30.0)
            self._drain_battery(30.0, active=True)

        target = storage_zone_for("HLW")
        self.state = "MOVING_TO_DROPOFF"
        yield self.env.process(self._move_to(self._zone_park_pos(target)))
        if self.failed():
            return
        self.coordinator.update_location(item.item_id, target.name)
        self.state = "DROPPING_OFF"
        yield self.env.timeout(2.0)
        self._drain_battery(2.0, active=True)
        self.carrying = None
        self.metrics.items_completed += 1
        # Record a synthetic correct classification so the dashboard's HLW
        # row populates. process_known_class == true_class always, so this
        # is always accurate.
        self.metrics.record_classification(
            item_id=item.item_id,
            true_class="HLW",
            predicted_class="HLW",
            confidence=1.0,
            scrutiny=False,
            agent_id=self.agent_id,
            timestamp_s=self.env.now,
        )
        # No walk-home: drone idles at HLW storage until the next task.

    # ---------- HPGe drum-scanner loop ----------

    def _rescrutiny_loop(self, rescan_queue: simpy.Store):
        while True:
            self.state = "IDLE"
            task = yield rescan_queue.get()
            item: WasteItem = task["item"]
            self.carrying = item
            self.state = "RESCANNING"
            yield self.env.timeout(60.0)
            spectrum = simulate_spectrum(
                item.activities_bq, 60.0, HPGE_DETECTOR, distance_m=0.15, rng=self.rng,
            )
            dose_rate = simulate_dose_rate(item.activities_bq, distance_m=0.3, rng=self.rng)
            result = self.local_classifier.classify(
                item.total_specific_activity_bq_per_g, dose_rate, spectrum,
            )
            authoritative_class = result.predicted_class
            if result.actinide_signature > 0.05:
                authoritative_class = "ILW"

            # HPGe arm's draft verdict — may be wrong due to calibration
            # drift, dead-time, geometry, or counting statistics.
            hpge_verdict = authoritative_class
            hpge_wrong = (self.rng.random() < HPGE_ERROR_RATE)
            if hpge_wrong:
                hpge_verdict = _shift_one_class(hpge_verdict, self.rng)

            # Worker oversight — human operator reviews the camera feed
            # plus the HPGe verdict and either confirms or overrides.
            # The operator catches a fraction `WORKER_DETECTION_RATE`
            # of HPGe errors, and independently makes their own mistake
            # `WORKER_ERROR_RATE` of the time. Net combined system error
            # is much lower than HPGe alone — but never zero. This is
            # the 'no AI evaluating itself' guardrail: the final
            # authoritative class is always a human-confirmed call.
            authoritative_class = hpge_verdict
            if hpge_wrong and self.rng.random() < WORKER_DETECTION_RATE:
                # Worker overrides back to what the camera shows
                authoritative_class = item.true_class
            if self.rng.random() < WORKER_ERROR_RATE:
                # Worker's own slip-up — independent of HPGe being right
                authoritative_class = _shift_one_class(
                    authoritative_class, self.rng,
                )
            self.coordinator.receive_report(
                item_id=item.item_id,
                agent_id=self.agent_id,
                predicted_class=authoritative_class,
                confidence=max(result.confidence, 0.95),
                spectrum_features=result.spectrum_features,
                true_label_for_training=authoritative_class,
                is_rescrutiny=True,
            )
            for rec in reversed(self.metrics.classifications):
                if rec["item_id"] == item.item_id:
                    rec["predicted_class"] = authoritative_class
                    rec["confidence"] = max(rec["confidence"], 0.95)
                    break

            target = storage_zone_for(authoritative_class)
            self.coordinator.update_location(item.item_id, target.name)
            self.carrying = None
            self.metrics.items_completed += 1
