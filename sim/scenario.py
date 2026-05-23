"""Builds and runs a full scenario: facility, agents, coordinator, waste
generator process, dose-tick process. Returns the populated RunMetrics."""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
import simpy

from .agent import Agent
from .coordinator import Coordinator
from .facility import (
    ZONES_BY_NAME, generation_points, distance_m, worker_zones,
    POOL_SOURCE_NAME,
)
from .metrics import RunMetrics
from .waste_generator import WasteStream


def _top_nuclides(activities_bq: dict[str, float], n: int = 3) -> list[str]:
    """Return the names of the n nuclides with the highest activity in
    this item. Used by the live view to render a per-drum nuclide-
    composition strip so judges can spot the actinide-spiked masquerade."""
    if not activities_bq:
        return []
    ranked = sorted(activities_bq.items(), key=lambda kv: kv[1], reverse=True)
    return [name for name, _ in ranked[:n]]


@dataclass
class WorkerNPC:
    """A human operator wandering inside the Worker area. Their dose is
    accumulated by the existing metrics.record_dose_tick path (which already
    keys off the worker-zone centroid); the NPC is what makes that dose
    visible on screen — judges see a person standing near a hot drum."""
    worker_id: str
    zone_name: str
    pos: tuple[float, float]
    target: tuple[float, float]
    speed_mps: float = 0.35
    integrated_dose_uSv: float = 0.0


@dataclass
class TickContext:
    """Handles passed to on_tick callbacks so the live view can both render
    the current state and inject actions back into the running sim."""
    env: simpy.Environment
    agents: list
    metrics: RunMetrics
    coord: Coordinator
    active_items: dict
    stream: WasteStream
    handle_queue: simpy.Store
    rescan_queue: simpy.Store
    cfg: "ScenarioConfig"
    workers: list = field(default_factory=list)
    paused: bool = False
    sim_duration_s: float = 0.0
    # Sim-seconds to advance per on_tick call. The view writes to this to
    # implement speed control; the default is set to a sensible value for the
    # default 30 FPS frame rate.
    sim_seconds_per_frame: float = 2.0
    # Redundancy state — per-arm primary/standby + failure events. The
    # view reads from these to render which arm is active vs failed and
    # to log failover banners.
    arm_pairs: dict = field(default_factory=dict)
    arm_events: list = field(default_factory=list)
    # End-of-shift haulage events — every shift_duration_s an offsite
    # truck collects the storage zones and ships everything offsite.
    # The live view reads from this list to log shipments and animate
    # the storage rooms emptying.
    shift_events: list = field(default_factory=list)

    def inject_tricky_item(self) -> str | None:
        """Inject a tricky (actinide-spiked) waste item at a random
        contact-handling generation point (skip upstream PUREX stages —
        they don't produce tricky-masquerade items). Returns the new id."""
        gen_points = [z for z in generation_points() if z.produces_class is None]
        if not gen_points:
            return None
        zone = gen_points[self.stream.rng.integers(0, len(gen_points))]
        item = self.stream.generate(zone.name, self.env.now, force_tricky=True)
        self.coord.register_item(item.item_id, item.true_class, zone.name,
                                 tricky=item.tricky,
                                 top_nuclides=_top_nuclides(item.activities_bq))
        self.active_items[item.item_id] = {
            "item": item, "pos": (zone.x, zone.y), "shielded": False,
        }
        # Tricky items go through the same Char-station pipeline as
        # regular drums: cart stages drum to Char, brain classifies, brain
        # dispatches the next leg. (In isolated mode this returns a deferred
        # handle_queue.put(); calling it from a UI handler triggers the Store
        # immediately, which is what we want.)
        _enqueue_initial_task(self.coord, self.handle_queue, item, zone)
        return item.item_id

    def failover_coordinator(self) -> bool:
        """Promote the hot-standby coordinator to primary. Zero data loss
        because PRIMARY and STANDBY share the same ledger + training pool
        (they're aliases of the same Coordinator object in the sim — in a
        real plant they'd be replicating over a heartbeat link). We bump a
        failover counter that the live view uses to flash a banner."""
        import time as _time
        self.coord.failover_count += 1
        self.coord.last_failover_t_s = self.env.now
        self.coord.decision_log.append({
            "t_s": _time.time(),
            "kind": "retrain",
            "text": f"FAILOVER #{self.coord.failover_count} — standby promoted (0 data loss)",
        })
        return True

    def force_retrain(self) -> bool:
        """Trigger an immediate retrain. Only meaningful in hivemind mode
        and only if there is at least some training data."""
        if not self.coord.shared:
            return False
        if len(self.coord._training_labels) < 5:
            return False
        self.coord._retrain()
        return True

    def kill_drone(self, drone_idx_1based: int) -> str | None:
        """Force-fail a mobile drone (simulates catastrophic dose damage).
        The agent's run loop notices and starts the auto-replacement
        timer. Returns the killed agent_id, or None if invalid index."""
        mobile = [a for a in self.agents if not a.is_rescrutiny_station]
        if not (1 <= drone_idx_1based <= len(mobile)):
            return None
        a = mobile[drone_idx_1based - 1]
        if a.state == "FAILED":
            return None  # already dead
        # Simulate critical dose dump and mark failure time so the live
        # view can start the OFFLINE countdown without waiting for the
        # agent's run loop to notice on its next iteration.
        from .agent import DOSE_FAILURE_THRESHOLD_uSv
        a.integrated_dose_uSv = DOSE_FAILURE_THRESHOLD_uSv + 1000
        a.state = "FAILED"
        a.failure_time_s = self.env.now
        # Drop any item the agent was carrying — and any tasks already
        # queued for this specific drone — back into the queue so another
        # cart can pick the work up.
        from .facility import ZONES_BY_NAME
        recovered: list[dict] = []
        if a.carrying is not None:
            pickup_zone = ZONES_BY_NAME.get(a.carrying.generation_point)
            if pickup_zone is not None:
                # The dead cart was carrying a drum, so this is a transport
                # (handle) task. Re-derive the target.
                entry = self.coord.ledger.get(a.carrying.item_id)
                cls = entry.current_classification if entry else None
                target_zone = None
                if a.carrying.process_known_class == "HLW":
                    target_zone = ZONES_BY_NAME.get("HLW storage")
                    is_hlw = True
                else:
                    is_hlw = False
                    from .facility import storage_zone_for
                    if cls in ("VLLW", "LLW", "ILW"):
                        target_zone = storage_zone_for(cls)
                if target_zone is not None:
                    recovered.append({
                        "kind": "handle",
                        "item": a.carrying,
                        "pickup_zone": pickup_zone,
                        "target_zone": target_zone,
                        "needs_qa": False,
                        "is_hlw_route": is_hlw,
                    })
            a.carrying = None
        # Drain the agent's dispatch queue (only meaningful in hivemind —
        # in isolated, the queue is shared and shouldn't be drained)
        if self.coord.shared and a.dispatch_queue is not None:
            while a.dispatch_queue.items:
                recovered.append(a.dispatch_queue.items.pop(0))
        for t in recovered:
            if self.coord.shared:
                self.coord.enqueue_task(t)
            else:
                self.handle_queue.put(t)
        # Tag the event in the coordinator's wire trace
        self.coord.recent_messages.append({
            "t_s": __import__("time").time(),
            "kind": "kill",
            "agent_id": a.agent_id,
        })
        # Interrupt the agent's currently-running simpy process so the
        # FAILED state takes effect *immediately* rather than waiting for
        # the agent to finish whatever yield it's currently on.
        proc = getattr(a, "proc", None)
        if proc is not None and proc.is_alive:
            try:
                proc.interrupt("killed")
            except RuntimeError:
                pass  # already interrupted / not currently yielding
        return a.agent_id


@dataclass
class ScenarioConfig:
    mode: str              # "isolated" or "hivemind"
    seed: int
    # Robot fleet composition. Every mobile cart is a transport AGV;
    # Handlers and Hybrids differ only by chassis speed. n_mobile_agents is
    # the legacy single-knob default — when set, it builds n_mobile_agents
    # Hybrid carts. Pass explicit n_handlers/n_hybrids to mix.
    n_mobile_agents: int = 3
    n_handlers: int = 0
    n_hybrids: int = 0
    sim_duration_s: float = 8 * 3600.0     # 8-hour shift simulated time
    waste_interarrival_mean_s: float = 45.0    # avg one item every ~45s so
                                                # all mobile drones stay busy
    tricky_fraction: float = 0.25
    pretrain_samples: int = 30
    rescrutiny_confidence_threshold: float = 0.70
    # Fraction of items pulled to HPGe rescrutiny independent of confidence
    # (random QA sampling, the way real plants generate trusted-label data).
    # Real plants typically sample 1-10%; we use 5%, which is enough for the
    # hivemind to learn the spike pattern but rare enough that the isolated
    # mode does not get rescued by QA alone.
    qa_sampling_fraction: float = 0.05
    # Wireless packet-drop probability on drone -> queen reports. A small
    # value (~3%) makes the network simulation visible without significantly
    # affecting throughput or accuracy.
    network_drop_probability: float = 0.03
    # Operating shift length. Every `shift_duration_s` sim-seconds, an
    # offsite haulage truck visits every storage zone, collects all
    # finalized drums, and ships them to permanent disposal. This stops
    # storage from piling up forever in long-run sims (months / years)
    # and lets memory stay bounded.
    shift_duration_s: float = 8 * 3600.0
    # Hard cap on the in-memory classification log — older items are
    # aggregated into the cumulative accuracy stat and dropped. Keeps
    # multi-month runs from growing unbounded.
    max_classifications_in_memory: int = 10_000

    def fleet_composition(self) -> list[str]:
        """Return the list of agent_type strings for the mobile fleet.

        With centralized classification at the Char Station, every cart is a
        transport AGV — Handlers and Hybrids differ only by chassis speed.
        """
        explicit = self.n_handlers + self.n_hybrids
        if explicit > 0:
            return ["handler"] * self.n_handlers + ["hybrid"] * self.n_hybrids
        return ["hybrid"] * self.n_mobile_agents


# Representative unshielded dose rate (µSv/h at 1 m) by waste class, used for
# the worker-dose metric. Deliberately a small bounded lookup rather than the
# raw activity-derived rate: the demo's dose signal must come from HOW MANY hot
# drums sit unshielded (i.e. how many ILW drums got mis-filed into the
# low-level store), not from one drum's heavy activity tail. The old
# activity-derived rate had a multi-order-of-magnitude tail that let a single
# mis-routed hot drum blow the cumulative dose up to physically absurd values
# (tens of millions of µSv) and swamp the isolated-vs-hivemind comparison.
REP_DOSE_RATE_uSv_h: dict[str, float] = {
    "VLLW": 0.5,
    "LLW":  15.0,
    "ILW":  4000.0,
    "HLW":  0.0,     # HLW is always in a shielded cask / vault — never exposes
}


def _representative_dose_rate(true_class: str) -> float:
    """Bounded per-class worker-exposure rate (µSv/h at 1 m). See
    REP_DOSE_RATE_uSv_h. The hazard tracks the item's TRUE class — a mis-filed
    ILW drum is dangerous precisely because it is really ILW even though the
    paperwork says LLW."""
    return REP_DOSE_RATE_uSv_h.get(true_class, 0.0)


def _enqueue_initial_task(coordinator: Coordinator,
                          handle_queue: simpy.Store, item, pickup_zone):
    """Push a freshly-generated item into the right queue.

    Workflow (new centralized-brain model):
      - HLW process-known items: handler routes directly to HLW storage
        via the Solidification cell (forced sub-route inside _handle_task).
      - Everything else: handler ferries the drum to the Char Station
        turntable, drops it, the Char actuator picks it up off
        coord.char_in_queue, classifies, and emits the next handle task
        for the drum's onward trip to storage (or QA lab if low-conf).

    Carts no longer scan in-place at gen points — the cart is just a
    dumb transport. Classification happens at the Char Station only."""
    from .facility import storage_zone_for as _sz, char_station_zone
    if item.process_known_class == "HLW":
        task = {
            "kind": "handle", "item": item,
            "pickup_zone": pickup_zone,
            "target_zone": _sz("HLW"),
            "is_hlw_route": True, "needs_qa": False,
        }
    else:
        # Char station is always present; route the drum there for the
        # central NaI + CV scan. The actuator emits the onward task.
        char_z = char_station_zone()
        task = {
            "kind": "handle", "item": item,
            "pickup_zone": pickup_zone,
            "target_zone": char_z,
            "is_stage_to_char": True,
            "is_hlw_route": False,
            "needs_qa": False,
        }
    if coordinator.shared:
        coordinator.enqueue_task(task)
        return None
    return handle_queue.put(task)


def _char_station_actuator_proc(
    env: simpy.Environment,
    coord: Coordinator,
    metrics: RunMetrics,
    rng: np.random.Generator,
    handle_queue: simpy.Store,
    qa_sampling_fraction: float,
):
    """Drum Characterisation Station — centralized classifier actuator.

    Carts drop drums on the Char turntable, then put a record into
    ``coord.char_in_queue``. This process drains that queue, runs a NaI +
    CV scan on each drum, reports the classification to the brain, and
    emits the follow-up transport task (Char → storage, or Char → QA lab
    if the drum was flagged low-confidence).

    This is the *only* place where ordinary (non-HLW) drum classification
    happens. Carts are dumb transports — the brain owns the decision."""
    from .sensors import (
        simulate_spectrum, simulate_dose_rate,
        simulate_cv_classification, NAI_DETECTOR,
    )
    from .facility import (
        storage_zone_for, clearance_zone, ZONES_BY_NAME,
        CLEARANCE_THRESHOLD_BQ_PER_G,
    )
    char_zone = ZONES_BY_NAME["Classifier"]
    while True:
        task = yield coord.char_in_queue.get()
        item = task["item"]
        coord.char_scanning_item_id = item.item_id
        # NaI integration (8 s sim) on the turntable
        yield env.timeout(8.0)
        spectrum = simulate_spectrum(
            item.activities_bq, 8.0, NAI_DETECTOR,
            distance_m=0.4, rng=rng,
        )
        dose_rate = simulate_dose_rate(
            item.activities_bq, distance_m=0.4, rng=rng,
        )
        cv_probs = simulate_cv_classification(
            item.true_class, item.tricky, rng=rng,
        )
        result = coord.shared_classifier.classify(
            item.total_specific_activity_bq_per_g,
            dose_rate, spectrum, cv_probs=cv_probs,
        )
        coord.last_scan_event = {
            "item_id": item.item_id,
            "agent_id": "char-A",
            "spectrum": spectrum,
            "features": result.spectrum_features,
            "predicted_class": result.predicted_class,
            "confidence": result.confidence,
            "actinide_signature": result.actinide_signature,
            "true_class": item.true_class,
            "tricky": item.tricky,
            "t_s": env.now,
        }
        coord.receive_report(
            item_id=item.item_id,
            agent_id="char-A",
            predicted_class=result.predicted_class,
            confidence=result.confidence,
            spectrum_features=result.spectrum_features,
            true_label_for_training=None,
            is_rescrutiny=False,
        )
        metrics.record_classification(
            item_id=item.item_id,
            true_class=item.true_class,
            predicted_class=result.predicted_class,
            confidence=result.confidence,
            scrutiny=result.confidence < coord.scrutiny_confidence_threshold,
            agent_id="char-A",
            timestamp_s=env.now,
        )

        entry = coord.ledger.get(item.item_id)
        scrutiny = entry.scrutiny_flag if entry else False
        if (
            not scrutiny
            and qa_sampling_fraction > 0.0
            and rng.random() < qa_sampling_fraction
        ):
            scrutiny = True
            if entry is not None:
                entry.scrutiny_flag = True

        # Follow-up handle task: Char → QA lab (if flagged) or Char → storage
        if scrutiny:
            target = ZONES_BY_NAME["QA lab"]
            handle_task = {
                "kind": "handle", "item": item,
                "pickup_zone": char_zone, "target_zone": target,
                "needs_qa": True, "is_hlw_route": False,
                "is_clearance": False,
            }
        else:
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
                "pickup_zone": char_zone, "target_zone": target,
                "needs_qa": False, "is_hlw_route": False,
                "is_clearance": is_clearance,
            }
        if coord.shared:
            coord.enqueue_task(handle_task)
        else:
            yield handle_queue.put(handle_task)
        coord.char_scanning_item_id = None


def _waste_generator_proc(
    env: simpy.Environment,
    stream: WasteStream,
    handle_queue: simpy.Store,
    coordinator: Coordinator,
    cfg: ScenarioConfig,
    active_items: dict,
    rng: np.random.Generator,
):
    """Legacy contact-handling generator. Produces statistical-mix waste at
    the original three sources (Cleanup ops / Maintenance / Plant samples).
    These are where the tricky-actinide masquerade lives and where the
    hivemind-vs-isolated demo headline comes from."""
    contact_zones = [z for z in generation_points()
                     if z.produces_class is None]
    while True:
        wait = rng.exponential(cfg.waste_interarrival_mean_s)
        yield env.timeout(wait)
        if env.now >= cfg.sim_duration_s:
            return
        if not contact_zones:
            return
        zone = contact_zones[rng.integers(0, len(contact_zones))]
        item = stream.generate(zone.name, env.now)
        coordinator.register_item(item.item_id, item.true_class, zone.name,
                                  tricky=item.tricky,
                                  top_nuclides=_top_nuclides(item.activities_bq))
        active_items[item.item_id] = {
            "item": item,
            "pos": (zone.x, zone.y),
            "shielded": False,
        }
        put_event = _enqueue_initial_task(coordinator, handle_queue,
                                          item, zone)
        if put_event is not None:
            yield put_event


def _process_stage_generator_proc(
    env: simpy.Environment,
    zone,
    stream: WasteStream,
    handle_queue: simpy.Store,
    coordinator: Coordinator,
    cfg: ScenarioConfig,
    active_items: dict,
    rng: np.random.Generator,
    mean_interarrival_s: float,
    process_known: bool,
):
    """One process stage (Shearing / Dissolution / HLW Concentration ...)
    producing its typed waste stream on its own cadence.

    Stages that produce HLW are flagged process_known=True so drones route
    them directly to Solidification without NaI scanning. Stages that
    produce activation-product waste (Shearing) are NOT process-known —
    the activity level still needs measuring, and tricky-actinide drums
    can appear here too."""
    while True:
        wait = rng.exponential(mean_interarrival_s)
        yield env.timeout(wait)
        if env.now >= cfg.sim_duration_s:
            return
        forced = zone.produces_class
        item = stream.generate(
            zone.name, env.now,
            forced_class=forced,
            process_known=process_known,
        )
        coordinator.register_item(item.item_id, item.true_class, zone.name,
                                  tricky=item.tricky,
                                  top_nuclides=_top_nuclides(item.activities_bq))
        # Shearing-cell events drive the Spent Fuel Pool's FHM crane —
        # one assembly lifted from the pool per Shearing event.
        if zone.name == "Shearing cell":
            coordinator.last_shearing_event_sim_s = env.now
        active_items[item.item_id] = {
            "item": item,
            "pos": (zone.x, zone.y),
            "shielded": process_known,  # HLW casks are shielded in transit
        }
        put_event = _enqueue_initial_task(coordinator, handle_queue,
                                          item, zone)
        if put_event is not None:
            yield put_event


def _dispatcher_proc(
    env: simpy.Environment,
    coordinator: Coordinator,
    agents: list,
):
    """Hivemind dispatcher: continuously picks up pending transport tasks and
    assigns each to the best-scoring idle cart (nearest / most charged / least
    irradiated — see Coordinator.score_agent_for_task). If no idle cart is
    available the task stays in the pending pool. Every mobile cart is a
    transport AGV, so any idle one can take any task."""
    # Tight poll interval so the dispatcher reacts quickly to respawned
    # carts and freshly-enqueued tasks. At 1x sim speed the user notices
    # any > ~0.5 wall-sec lag between a cart respawn and its first move.
    POLL_S = 0.2
    while True:
        # Wait until there's at least one pending task
        while not coordinator.pending_tasks:
            yield env.timeout(POLL_S)
        idle_pool = [
            a for a in agents
            if (not a.is_rescrutiny_station)
            and a.state == "IDLE"
            and not a.failed()
        ]
        if not idle_pool:
            yield env.timeout(POLL_S)
            continue
        # Assign the first pending task to the best-scoring idle cart.
        task = coordinator.pending_tasks[0]
        chosen, _ = coordinator.assign_task(idle_pool, task)
        if chosen is None:
            yield env.timeout(POLL_S)
            continue
        coordinator.pending_tasks.pop(0)
        chosen.last_dispatch_t_s = env.now
        yield chosen.dispatch_queue.put(task)


def _shipping_proc(
    env: simpy.Environment,
    coordinator: Coordinator,
    metrics: RunMetrics,
    active_items: dict,
):
    """Periodic outbound shipping. Real plants ship low-activity waste off
    site frequently (VLLW + LLW weekly-ish) and intermediate waste rarely
    (ILW into interim storage for years before deep disposal). Without
    this, our storage bins crater after a couple of sim-hours.

    Schedule (sim time):
      - every 6 sim-hours: ship out a chunk of VLLW + LLW drums (most cleared
        for off-site disposal)
      - every 24 sim-hours: ship out a smaller ILW chunk to interim store
      - every 72 sim-hours: ship out HLW canisters to deep-storage staging

    Shipments register in metrics.shipments so the live view can pulse a
    truck icon next to the affected storage room briefly."""
    import time as _time
    cycles = [
        # (period_s, class, max_drums_per_shipment)
        (6 * 3600.0,  "VLLW", 10),
        (6 * 3600.0,  "LLW",  10),
        (24 * 3600.0, "ILW",  4),
        (72 * 3600.0, "HLW",  2),
    ]
    next_times = {cls: period for period, cls, _ in cycles}
    cycle_lookup = {cls: (period, n) for period, cls, n in cycles}
    while True:
        # Pick the soonest scheduled cycle
        cls, due = min(next_times.items(), key=lambda kv: kv[1])
        if due > env.now:
            yield env.timeout(due - env.now)
        period, max_n = cycle_lookup[cls]
        # Find drums currently in this storage (only items at the matching
        # storage zone — mis-routed drums don't get shipped out, so the
        # demo's accuracy story stays visible).
        storage_name = f"{cls} storage"
        candidates = [
            e for e in coordinator.ledger.values()
            if e.current_location == storage_name
            and e.true_class == cls
        ]
        # Ship oldest first
        candidates.sort(key=lambda e: e.item_id)
        shipped_ids = []
        for entry in candidates[:max_n]:
            entry.current_location = "shipped"
            entry.audit_trail.append({
                "t": _time.time(),
                "event": "shipped_off_site",
                "class": cls,
            })
            active_items.pop(entry.item_id, None)
            shipped_ids.append(entry.item_id)
        if shipped_ids:
            metrics.items_shipped += len(shipped_ids)
            metrics.shipments.append({
                "t_s": env.now,
                "class": cls,
                "n": len(shipped_ids),
            })
            coordinator.decision_log.append({
                "t_s": _time.time(),
                "kind": "shipment",
                "text": f"shipped {len(shipped_ids)} {cls} drums off-site",
            })
        next_times[cls] = env.now + period


def _worker_wander_proc(
    env: simpy.Environment,
    workers: list,
    rng: np.random.Generator,
    coordinator: Coordinator,
):
    """Drift each worker NPC toward a random target inside their zone, pick
    a fresh target on arrival or after a short rest, repeat. Also accumulate
    per-worker integrated dose from nearby unshielded items so judges can
    watch the worker dose creep up when a mis-routed ILW drum is nearby."""
    tick_s = 0.5
    while True:
        yield env.timeout(tick_s)
        for w in workers:
            zone = ZONES_BY_NAME.get(w.zone_name)
            if zone is None:
                continue
            # Move toward target
            dx = w.target[0] - w.pos[0]
            dy = w.target[1] - w.pos[1]
            d = (dx * dx + dy * dy) ** 0.5
            step = w.speed_mps * tick_s
            if d <= step or d < 0.05:
                # Reached target — pick a new one inside the zone (rejection
                # sample to stay within the circle, with a small idle pause).
                if rng.random() < 0.35:
                    # 35% chance to "rest" this tick (no new target yet)
                    continue
                for _ in range(8):
                    rx = float(rng.uniform(zone.x - zone.radius_m * 0.7,
                                           zone.x + zone.radius_m * 0.7))
                    ry = float(rng.uniform(zone.y - zone.radius_m * 0.7,
                                           zone.y + zone.radius_m * 0.7))
                    if (rx - zone.x) ** 2 + (ry - zone.y) ** 2 <= (zone.radius_m * 0.75) ** 2:
                        w.target = (rx, ry)
                        break
            else:
                w.pos = (w.pos[0] + dx / d * step, w.pos[1] + dy / d * step)

            # Accumulate per-worker proximity dose from non-shielded items
            # within ~3 m. Matches the agent's _accrue_proximity_dose model
            # so the visualisation stays consistent with the global metric.
            rate = 0.0
            for entry in coordinator.ledger.values():
                if entry.current_location in ("ILW storage", "HLW storage"):
                    continue
                zone_i = ZONES_BY_NAME.get(entry.current_location)
                if zone_i is None:
                    continue
                d2 = ((w.pos[0] - zone_i.x) ** 2 + (w.pos[1] - zone_i.y) ** 2) ** 0.5
                if d2 > 3.0:
                    continue
                if entry.current_classification == "ILW" or entry.true_class == "ILW":
                    src = 6000.0
                else:
                    continue
                rate += src / max(d2 ** 2, 0.25)
            w.integrated_dose_uSv += rate * (tick_s / 3600.0)


def _dose_tick_proc(
    env: simpy.Environment,
    metrics: RunMetrics,
    coordinator: Coordinator,
    active_items: dict,
    agents: list,
    cfg: ScenarioConfig,
    tick_s: float = 30.0,
    inspection_period_s: float = 3600.0,
):
    """Every tick_s simulated seconds, accumulate worker dose from two
    sources:

      (a) Distance-attenuated dose from items in motion / on benches,
          using inverse-square geometry to the operator workspace.

      (b) A periodic "worker inspection" at 1m contact distance for every
          item currently in LLW or VLLW storage. This is the model's
          mechanism for charging the worker-dose budget when an ILW item
          has been mis-routed to LLW storage: it sits there looking like
          LLW, and the worker handles it at close range during routine
          inspection. ILW storage is assumed to be shielded; ILW correctly
          routed there contributes nothing.
    """
    last_inspection_s = env.now
    while True:
        yield env.timeout(tick_s)
        do_inspection = (env.now - last_inspection_s) >= inspection_period_s
        if do_inspection:
            last_inspection_s = env.now

        active = []
        inspection_dose = 0.0
        for item_id, info in list(active_items.items()):
            item = info["item"]
            entry = coordinator.ledger.get(item_id)
            if entry is None:
                continue
            # If the item has reached its TRUE final storage, it's done from
            # the dose perspective. (Mis-routed items stay in the loop and
            # keep contributing exposure during worker inspections.)
            true_storage_name = {
                "VLLW": "VLLW storage", "LLW": "LLW storage",
                "ILW": "ILW storage", "HLW": "HLW storage",
            }[item.true_class]
            # Items released for clearance are also "done" from the dose
            # perspective — they've left the controlled area.
            if (
                entry.current_location == true_storage_name
                or entry.current_location == "Free release"
            ):
                active_items.pop(item_id, None)
                continue

            pos = info["pos"]
            for ag in agents:
                if ag.carrying is not None and ag.carrying.item_id == item_id:
                    pos = ag.pos
                    break
            else:
                zone = ZONES_BY_NAME.get(entry.current_location)
                if zone is not None:
                    pos = (zone.x, zone.y)

            shielded = False
            if entry.current_classification == "ILW" and entry.current_location in (
                "QA lab", "ILW storage",
            ):
                shielded = True
            if entry.current_location == "ILW storage":
                shielded = True
            # HLW casks are always in a shielded carry cask through
            # transit and the solidification cell, then sit shielded in
            # HLW storage. They never expose workers in this sim.
            if (
                item.process_known_class == "HLW"
                or entry.current_location in ("HLW storage", "Solidification")
            ):
                shielded = True

            dose_rate_uSv_h = _representative_dose_rate(item.true_class)
            active.append({
                "pos": pos,
                "dose_rate_uSv_h": dose_rate_uSv_h,
                "shielded": shielded,
            })

            # Periodic worker inspection at 1m for items at non-shielded
            # storage. This is where mis-routed ILW items pay the dose
            # price: a worker walks up to what they think is LLW.
            if do_inspection and entry.current_location in ("LLW storage", "VLLW storage"):
                # 1 minute of contact at 1m
                inspection_dose += dose_rate_uSv_h * (60.0 / 3600.0)

        metrics.record_dose_tick(tick_s, active, env.now)
        if inspection_dose > 0:
            metrics.cumulative_dose_uSv += inspection_dose
            metrics.dose_history.append((env.now, metrics.cumulative_dose_uSv))
        metrics.update_accuracy_history(env.now)


def _shift_cycle_proc(
    env,
    coordinator: Coordinator,
    metrics: RunMetrics,
    active_items: dict,
    workers: list,
    cfg: ScenarioConfig,
    event_sink: list,
):
    """Shift bookkeeping that keeps the sim bounded over months / years.

    The per-class outbound-shipping process (`_shipping_proc`) is the
    realistic story for what physically leaves the plant: VLLW/LLW go
    every 6 h, ILW every 24 h, HLW every 72 h. Once a drum is shipped,
    its ledger entry's current_location is set to "shipped".

    Every `shift_duration_s` sim-seconds we then do the OPS bookkeeping
    that keeps memory bounded over a long run:

      1) Garbage-collect ledger entries that have been shipped — they're
         gone from the plant, no point keeping the record in RAM (a real
         plant archives them to a separate database).
      2) Trim the classification log to the most recent
         `cfg.max_classifications_in_memory` items. Accuracy + confusion
         matrix are intentionally computed over this rolling window;
         the cumulative totals (items_shipped, dose) are kept whole.
      3) Worker rotation — the previous shift's operators clock out
         with their integrated dose recorded; the next shift comes in
         with fresh exposure budgets.

    This is what lets `--sim-hours 720` (one month) or 8760 (one year)
    run without unbounded growth in the ledger or the per-tick dose loop."""
    shift_dur = cfg.shift_duration_s
    if shift_dur <= 0:
        return
    shift_idx = 0
    while True:
        yield env.timeout(shift_dur)
        shift_idx += 1

        # 1) GC the ledger of items already shipped.
        gc_count = 0
        for item_id in list(coordinator.ledger.keys()):
            entry = coordinator.ledger[item_id]
            if entry.current_location == "shipped":
                coordinator.ledger.pop(item_id, None)
                active_items.pop(item_id, None)
                gc_count += 1

        # 2) Trim the in-memory classification log.
        cap = max(cfg.max_classifications_in_memory, 1000)
        trimmed = 0
        if len(metrics.classifications) > cap:
            trimmed = len(metrics.classifications) - cap
            metrics.classifications = metrics.classifications[-cap:]

        # 3) Rotate workers — outgoing shift's dose is captured before
        # we zero them out for the next shift.
        outgoing_dose = sum(getattr(w, "integrated_dose_uSv", 0.0)
                             for w in workers)
        for w in workers:
            w.integrated_dose_uSv = 0.0

        event_sink.append({
            "t_s": env.now,
            "kind": "shift_end",
            "shift_idx": shift_idx,
            "gc_count": gc_count,
            "trimmed_logs": trimmed,
            "outgoing_worker_dose": outgoing_dose,
        })


def _pretrain_shared(
    coordinator: Coordinator,
    stream: WasteStream,
    rng: np.random.Generator,
    n: int,
) -> None:
    """Pre-load coordinator's training set with n samples in NaI feature
    space (the same space mobile agents observe at inference time).

    Pretraining uses non-tricky items only — the actinide spike pattern is
    learned during the run from in-run HPGe rescrutiny labels, which is what
    we want the demo to show."""
    from .sensors import simulate_spectrum, NAI_DETECTOR
    from .classifier import extract_spectrum_features

    feats, labels = [], []
    for _ in range(n):
        item = stream.generate("Cleanup ops", 0.0)
        spec = simulate_spectrum(item.activities_bq, 10.0, NAI_DETECTOR, distance_m=0.3, rng=rng)
        feats.append(extract_spectrum_features(spec))
        labels.append(item.true_class)
    coordinator._training_features.extend(feats)
    coordinator._training_labels.extend(labels)
    coordinator._retrain()


def run_scenario(
    cfg: ScenarioConfig,
    on_tick=None,
) -> tuple[RunMetrics, Coordinator, list]:
    """Run a scenario synchronously. on_tick(env, agents, metrics) is called
    every simulated second for the live visualization. Return metrics +
    coordinator + agent list for the dashboard to inspect."""
    rng = np.random.default_rng(cfg.seed)
    env = simpy.Environment()
    coord = Coordinator(
        shared=(cfg.mode == "hivemind"),
        scrutiny_confidence_threshold=cfg.rescrutiny_confidence_threshold,
        rng=np.random.default_rng(cfg.seed + 5555),
    )
    coord.packet_drop_probability = cfg.network_drop_probability
    metrics = RunMetrics(mode=cfg.mode)

    # Pretrain shared classifier (hivemind only) using a separate RNG so the
    # main waste stream isn't perturbed.
    pretrain_rng = np.random.default_rng(cfg.seed + 1001)
    pretrain_stream = WasteStream(pretrain_rng, tricky_fraction=0.0)
    if cfg.mode == "hivemind":
        _pretrain_shared(coord, pretrain_stream, pretrain_rng, cfg.pretrain_samples)
    else:
        # Isolated mode: each agent gets a private classifier trained on a
        # tiny disjoint slice. They won't share -> no collective improvement.
        pass

    # Build waste stream for the actual run
    stream = WasteStream(rng, tricky_fraction=cfg.tricky_fraction)

    # In ISOLATED mode the mobile fleet pulls transport tasks from one shared
    # handle_queue. In HIVEMIND mode each cart gets its own dispatch queue and
    # the coordinator's dispatcher routes each task to the best cart.
    handle_queue = simpy.Store(env)
    rescan_queue = simpy.Store(env)
    # Char station ingress queue — carts drop drums on the turntable and
    # push a record here; the Char actuator process drains it. The brain
    # holds the handle so any process can access it.
    char_in_queue = simpy.Store(env)
    coord.char_in_queue = char_in_queue

    agents: list[Agent] = []
    home = ZONES_BY_NAME["Charging bay"]
    # Resolve the fleet composition (Handler / Hybrid transport carts).
    fleet = cfg.fleet_composition()
    n_total = len(fleet)
    if n_total == 0:
        # Failsafe: at least one hybrid so the scenario still runs
        fleet = ["hybrid"]
        n_total = 1
    # Adaptive spacing so however many agents you ask for fit cleanly inside
    # the charging bay zone without their labels colliding.
    if n_total > 1:
        usable = (home.radius_m - 0.4) * 2.0
        spacing = min(1.7, usable / (n_total - 1))
    else:
        spacing = 0.0
    # Per-role index counters so IDs come out handler-1 / hybrid-1
    role_idx = {"handler": 0, "hybrid": 0}
    for i, role in enumerate(fleet):
        role_idx[role] += 1
        agent_id = f"{role}-{role_idx[role]}"
        a = Agent(
            env, agent_id, coord, rng=np.random.default_rng(cfg.seed + 100 + i),
            metrics=metrics, is_rescrutiny_station=False,
            qa_sampling_fraction=cfg.qa_sampling_fraction,
            agent_type=role,
        )
        offset = (i - (n_total - 1) / 2.0) * spacing
        a.pos = (home.x + offset, home.y)
        a.home_offset_x = offset
        # Per-agent dispatch queue in hivemind; None in isolated (the cart
        # pulls transport tasks straight from the shared handle_queue).
        if cfg.mode == "hivemind":
            a.dispatch_queue = simpy.Store(env)
        else:
            a.dispatch_queue = None
        agents.append(a)

    # HPGe drum scanners — TWO redundant stations (HPGe-A primary +
    # HPGe-B standby). Both drain the same rescan_queue, so whichever is
    # free picks up the next drum. Visible redundancy in the QA lab room.
    for j, qa_id in enumerate(("hpge-a", "hpge-b")):
        drum = Agent(
            env, qa_id, coord,
            rng=np.random.default_rng(cfg.seed + 999 + j),
            metrics=metrics, is_rescrutiny_station=True,
            home_zone_name="QA lab",
        )
        agents.append(drum)

    # Bookkeeping struct for the dose-tick process
    active_items: dict = {}

    # Worker NPCs — 3 humans drifting through the Worker area. Their dose is
    # tracked separately (visual) but the global cumulative_dose_uSv metric
    # comes from the existing _dose_tick_proc, so the headline number stays
    # comparable across versions.
    workers: list = []
    worker_rng = np.random.default_rng(cfg.seed + 7777)
    for wz in worker_zones():
        for i in range(3):
            angle = (i / 3.0) * 2 * np.pi
            r0 = wz.radius_m * 0.45
            pos = (wz.x + r0 * float(np.cos(angle)), wz.y + r0 * float(np.sin(angle)))
            workers.append(WorkerNPC(
                worker_id=f"op-{len(workers)+1}",
                zone_name=wz.name,
                pos=pos,
                target=pos,
            ))

    # Launch processes — store the process handle on the agent so an
    # interactive kill can interrupt the running task immediately.
    for a in agents:
        a.proc = env.process(a.run(handle_queue, rescan_queue))
    env.process(_waste_generator_proc(env, stream, handle_queue,
                                       coord, cfg, active_items, rng))
    env.process(_dose_tick_proc(env, metrics, coord, active_items, agents, cfg))
    # Spin up one generator process per upstream PUREX stage so each
    # produces its typed waste stream on its own cadence. Cadences are
    # deliberately slower than the legacy contact-handling stream so the
    # tricky-actinide demo headline is still driven by Cleanup/Maint/Sampl.
    upstream_cadence_s = {
        "Shearing cell":      max(cfg.waste_interarrival_mean_s * 3.0, 120.0),
        "Dissolution cell":   max(cfg.waste_interarrival_mean_s * 4.0, 240.0),
        "HLW concentration":  max(cfg.waste_interarrival_mean_s * 5.0, 360.0),
    }
    upstream_stream_rng = np.random.default_rng(cfg.seed + 2024)
    upstream_stream = WasteStream(upstream_stream_rng, tricky_fraction=cfg.tricky_fraction * 0.5)
    # NOTE: per-stage RNG seeds are derived from a stable enumeration index,
    # NOT from hash(z.name) — Python salts string hashing per process, which
    # made `--seed N` produce a different scenario on every launch.
    for stage_i, z in enumerate(generation_points()):
        if z.produces_class is None:
            continue  # legacy contact-handling sources, handled above
        cadence = upstream_cadence_s.get(z.name, cfg.waste_interarrival_mean_s * 4.0)
        is_hlw = (z.produces_class == "HLW")
        env.process(_process_stage_generator_proc(
            env, z, upstream_stream, handle_queue, coord, cfg,
            active_items,
            rng=np.random.default_rng(cfg.seed + 3000 + stage_i),
            mean_interarrival_s=cadence,
            process_known=is_hlw,
        ))
    # Spent Fuel Pool waste source — filters / sludge / contaminated tools, a
    # mixed contact-waste stream on a slow cadence. This is what makes the
    # plant's visual origin (the pool) actually feed the Classifier, instead
    # of being decorative. produces_class is None -> normal mixed stream that
    # goes through the central scan like the legacy contact sources.
    pool_zone = ZONES_BY_NAME.get(POOL_SOURCE_NAME)
    if pool_zone is not None:
        env.process(_process_stage_generator_proc(
            env, pool_zone, upstream_stream, handle_queue, coord, cfg,
            active_items,
            rng=np.random.default_rng(cfg.seed + 3900),
            mean_interarrival_s=max(cfg.waste_interarrival_mean_s * 2.5, 150.0),
            process_known=False,
        ))
    if workers:
        env.process(_worker_wander_proc(env, workers, worker_rng, coord))
    env.process(_shipping_proc(env, coord, metrics, active_items))
    # Central classifier actuator at the Char station. Drains
    # coord.char_in_queue (drums dropped there by carts), scans, reports
    # to the brain, and emits the next transport task.
    env.process(_char_station_actuator_proc(
        env, coord, metrics,
        rng=np.random.default_rng(cfg.seed + 4242),
        handle_queue=handle_queue,
        qa_sampling_fraction=cfg.qa_sampling_fraction,
    ))
    if cfg.mode == "hivemind":
        env.process(_dispatcher_proc(env, coord,
                                     [a for a in agents if not a.is_rescrutiny_station]))

    # Per-arm redundancy — every robotic arm in the plant comes as a
    # primary + standby pair. The failure process fails the active unit
    # of one pair at a time on an exponential MTBF; a repair process
    # restores it ARM_REPAIR_S later. Throughput is unaffected because
    # the standby takes over instantly.
    from .redundancy import make_default_pairs, run_arm_failure_process
    arm_pairs = make_default_pairs()
    arm_events: list = []
    env.process(run_arm_failure_process(
        env, arm_pairs,
        np.random.default_rng(cfg.seed + 9999),
        event_sink=arm_events,
    ))

    # Shift cycle bookkeeping — every shift_duration_s, GC shipped
    # ledger entries, trim the in-memory classification log, and rotate
    # workers (new shift starts with fresh dose budgets). The actual
    # shipping of drums offsite is handled by `_shipping_proc` above on
    # its own per-class cadence. This combined design keeps memory
    # bounded over months/years sims.
    shift_events: list = []
    env.process(_shift_cycle_proc(
        env, coord, metrics, active_items, workers, cfg, shift_events,
    ))

    if on_tick is None:
        env.run(until=cfg.sim_duration_s)
    else:
        ctx = TickContext(
            env=env, agents=agents, metrics=metrics, coord=coord,
            active_items=active_items, stream=stream,
            handle_queue=handle_queue, rescan_queue=rescan_queue,
            cfg=cfg, workers=workers, sim_duration_s=cfg.sim_duration_s,
            arm_pairs=arm_pairs, arm_events=arm_events,
            shift_events=shift_events,
        )
        while env.now < cfg.sim_duration_s:
            if not ctx.paused and ctx.sim_seconds_per_frame > 0:
                target = min(env.now + ctx.sim_seconds_per_frame, cfg.sim_duration_s)
                env.run(until=target)
            cont = on_tick(ctx)
            if cont is False:
                break

    metrics.sim_duration_s = env.now
    return metrics, coord, agents
