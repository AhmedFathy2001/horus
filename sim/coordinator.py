"""Coordinator ("queen") — owner of the authoritative inventory ledger, drift
detection, model updates pushed to agents, and the scrutiny stigmergy flag.

The redundant hot-standby is mentioned in the README as a production gap; here
we have a single coordinator object."""
from __future__ import annotations
import copy
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Deque
import numpy as np
import pandas as pd

from .classifier import HybridClassifier, derive_actinide_threshold


@dataclass
class InventoryEntry:
    item_id: str
    true_class: str           # known only to god-view; not used in classification logic
    current_location: str
    current_classification: str | None
    confidence: float | None
    scrutiny_flag: bool
    audit_trail: list[dict] = field(default_factory=list)
    # Visual / ground-truth markers used by the live view to highlight
    # actinide-spiked drums. Set at register_item() time.
    tricky: bool = False
    # Top dominant nuclides in this item by activity (ordered, up to 3).
    # Lets the live view render a nuclide-composition strip on each drum
    # so judges can see *what's actually inside* — e.g. a tricky drum
    # visibly carrying an Am-241 stripe even though its gross activity
    # says LLW.
    top_nuclides: list[str] = field(default_factory=list)


class Coordinator:
    """Lives on the central server. Agents push reports; coordinator decides
    when to retrain and broadcasts model snapshots.

    Two modes:
      * shared=True  (hivemind): aggregates classifications, retrains, pushes
        snapshot, raises scrutiny flags.
      * shared=False (isolated): agents own their own private classifiers,
        coordinator only tracks ledger + final routing decisions.
    """

    def __init__(
        self,
        shared: bool,
        retrain_every_n_reports: int = 12,
        scrutiny_confidence_threshold: float = 0.70,
    ):
        self.shared = shared
        self.retrain_every_n_reports = retrain_every_n_reports
        self.scrutiny_confidence_threshold = scrutiny_confidence_threshold
        self.ledger: dict[str, InventoryEntry] = {}
        # Aggregated labeled training data for the shared classifier
        self._training_features: deque = deque(maxlen=2000)
        self._training_labels: deque = deque(maxlen=2000)
        self.shared_classifier = HybridClassifier()
        self._reports_since_retrain = 0
        # Model snapshot version; agents check this to know if they need to refresh
        self.model_version: int = 0
        self._snapshot_cache: dict | None = None
        # Telemetry
        self.retrain_events: list[dict] = []
        self.last_retrain_duration_s: float = 0.0
        # Lightweight wire trace for the live view to animate. Each entry:
        # {"t_s": float, "kind": "report"|"snapshot", "agent_id": str|None}
        # ("snapshot" entries are broadcasts and have agent_id=None).
        self.recent_messages: Deque[dict] = deque(maxlen=200)
        # Live activity status for the queen panel
        self.activity_status: str = "idle"
        self.activity_until_ms: int = 0
        # Packet drop probability on incoming reports (wireless link sim).
        # The simulated drop = the report never makes it to the ledger.
        self.packet_drop_probability: float = 0.0
        self.dropped_reports: int = 0
        # Dispatcher state — only used in hivemind mode. The queen owns the
        # pending-task list and decides which drone gets each item.
        self.pending_tasks: list[dict] = []
        self.dispatch_log: list[dict] = []
        # Telemetry for the live view's "queen activity" line
        self.last_dispatch_decision: dict | None = None
        # The most recent scanner classification event. The live view reads
        # this to render a "what the AI is looking at right now" spectrum
        # panel: full NaI spectrum + photopeak windows + actinide signature
        # + verdict + confidence.
        self.last_scan_event: dict | None = None
        # Hot-standby flag — the second server rack inside the COORD room
        # mirrors all the state of the primary in real time. Pressing K
        # in the live view triggers a failover: the standby takes over
        # with zero data loss (ledger + training data are this object's
        # own attributes; failover is purely a visual annotation).
        self.failover_count: int = 0
        self.last_failover_t_s: float | None = None
        # High-signal coordinator decisions for the right-side decision
        # feed (classify verdicts, scrutiny flags, retrains, threshold
        # updates, dispatch picks). Separate from the wire-trace
        # recent_messages so judges read a clean "what the AI just did".
        self.decision_log: Deque[dict] = deque(maxlen=120)
        # Sim-time of the most recent Shearing-cell waste-item generation.
        # The live view uses this to drive the Spent Fuel Pool's FHM crane:
        # each Shearing event corresponds to a fresh fuel assembly being
        # lifted from the pool, so the crane runs one delivery cycle
        # starting from this timestamp.
        self.last_shearing_event_sim_s: float | None = None
        # The Char Station (drum characterisation room) is the centralized
        # classifier. Carts drop drums on its turntable; the actuator
        # process picks them up off this queue, scans, and reports.
        # Set by scenario builder before processes start.
        self.char_in_queue = None
        # Item-id currently being scanned at the Char station (for the
        # live view's "SCANNING DRUM" banner). None when idle.
        self.char_scanning_item_id: str | None = None

    # ---------- Inventory ledger ----------

    def register_item(self, item_id: str, true_class: str, location: str,
                      tricky: bool = False,
                      top_nuclides: list[str] | None = None) -> None:
        self.ledger[item_id] = InventoryEntry(
            item_id=item_id,
            true_class=true_class,
            current_location=location,
            current_classification=None,
            confidence=None,
            scrutiny_flag=False,
            audit_trail=[{"t": time.time(), "event": "generated", "location": location}],
            tricky=tricky,
            top_nuclides=top_nuclides or [],
        )

    def update_location(self, item_id: str, location: str) -> None:
        if item_id not in self.ledger:
            return
        self.ledger[item_id].current_location = location
        self.ledger[item_id].audit_trail.append(
            {"t": time.time(), "event": "moved", "location": location}
        )

    # ---------- Classification reports ----------

    def receive_report(
        self,
        item_id: str,
        agent_id: str,
        predicted_class: str,
        confidence: float,
        spectrum_features: np.ndarray,
        true_label_for_training: str | None = None,
        is_rescrutiny: bool = False,
    ) -> None:
        """Agent reports a classification. In hivemind mode we use this to
        accumulate training data and trigger retraining.

        Federated-learning pattern: mobile agents (NaI) submit their feature
        vectors; the HPGe rescrutiny station later submits the *trusted label*
        for items it sees. The coordinator stitches the two together so the
        model trains on (NaI features, HPGe-trusted label) pairs — meaning the
        threshold learned is in the same feature space the mobile agents use
        at inference time."""
        entry = self.ledger.get(item_id)
        if entry is None:
            return
        # Wireless link sim: occasionally a report is dropped before it
        # reaches the queen. Rescrutiny reports are sent over a higher-
        # priority wired link in real plants and assumed reliable here.
        if (
            not is_rescrutiny
            and self.packet_drop_probability > 0.0
            and np.random.random() < self.packet_drop_probability
        ):
            self.recent_messages.append({
                "t_s": time.time(),
                "kind": "dropped",
                "agent_id": agent_id,
            })
            self.dropped_reports += 1
            return
        # Record the wire event so the live view can animate the report line
        self.recent_messages.append({
            "t_s": time.time(),
            "kind": "report",
            "agent_id": agent_id,
            "is_rescrutiny": is_rescrutiny,
        })
        self.activity_status = (
            f"writing report from {agent_id}" if not is_rescrutiny
            else f"recording QA-lab verdict from {agent_id}"
        )
        # Stash the most recent mobile-agent (NaI) feature vector so the
        # rescrutiny report can pick it up and pair it with the trusted label.
        if not is_rescrutiny:
            entry.audit_trail.append({
                "_internal_nai_features": spectrum_features.tolist(),
            })
        # The drum scanner provides high-confidence labels we can trust as
        # ground truth for *training* purposes — in a real plant this would be
        # qualified manual assay or sample-based labs. We pass true_class only
        # when the report comes from the HPGe drum scanner (is_rescrutiny=True).
        if is_rescrutiny:
            # Trust the rescrutiny result more than the initial classification
            entry.current_classification = predicted_class
            entry.confidence = confidence
            entry.scrutiny_flag = False  # cleared after rescrutiny
            entry.audit_trail.append({
                "t": time.time(),
                "event": "reclassified_by_HPGe",
                "agent": agent_id,
                "class": predicted_class,
                "confidence": confidence,
            })
            self.decision_log.append({
                "t_s": time.time(),
                "kind": "rescan_verdict",
                "text": f"HPGe re-class {item_id} -> {predicted_class}",
            })
        else:
            entry.current_classification = predicted_class
            entry.confidence = confidence
            entry.audit_trail.append({
                "t": time.time(),
                "event": "classified",
                "agent": agent_id,
                "class": predicted_class,
                "confidence": confidence,
            })
            self.decision_log.append({
                "t_s": time.time(),
                "kind": "verdict",
                "text": f"{agent_id} → {item_id}: {predicted_class} @ {confidence:.2f}",
            })
            if confidence < self.scrutiny_confidence_threshold:
                entry.scrutiny_flag = True
                self.decision_log.append({
                    "t_s": time.time(),
                    "kind": "scrutiny",
                    "text": f"flagged {item_id} (conf {confidence:.2f}) → HPGe",
                })

        if self.shared and true_label_for_training is not None:
            # Pull the original NaI feature vector this mobile agent measured
            # for this item, if there is one. That's what we want to train on,
            # so the learned threshold is in the same feature space mobile
            # agents see at inference time.
            mobile_features = None
            for ev in reversed(entry.audit_trail):
                if "_internal_nai_features" in ev:
                    mobile_features = np.asarray(ev["_internal_nai_features"])
                    break
            if mobile_features is None:
                # Pure HPGe-only path (no upstream NaI measurement) — skip.
                return
            self._training_features.append(mobile_features)
            self._training_labels.append(true_label_for_training)
            self._reports_since_retrain += 1
            if self._reports_since_retrain >= self.retrain_every_n_reports:
                self._retrain()

    def _retrain(self) -> None:
        if len(self._training_labels) < 5:
            return
        t0 = time.perf_counter()
        self.activity_status = "retraining classifier"
        X = np.asarray(self._training_features)
        y = list(self._training_labels)
        actinide_thr = derive_actinide_threshold(X, y)
        self.shared_classifier.fit(X, y, actinide_threshold=actinide_thr)
        self.last_retrain_duration_s = time.perf_counter() - t0
        self.activity_status = f"broadcasting v{self.model_version + 1} snapshot"
        self.model_version += 1
        self._snapshot_cache = None
        self._reports_since_retrain = 0
        self.retrain_events.append({
            "version": self.model_version,
            "n_samples": len(y),
            "duration_s": self.last_retrain_duration_s,
            "actinide_threshold": actinide_thr,
        })
        thr_s = f"{actinide_thr:.4f}" if actinide_thr is not None else "None"
        self.decision_log.append({
            "t_s": time.time(),
            "kind": "retrain",
            "text": f"retrain v{self.model_version}  n={len(y)}  thr={thr_s}",
        })
        # Broadcast snapshot event for the live view
        self.recent_messages.append({
            "t_s": time.time(),
            "kind": "snapshot",
            "agent_id": None,
            "version": self.model_version,
        })

    # ---------- Model push to agents ----------

    def current_snapshot(self) -> tuple[int, dict] | None:
        if not self.shared or not self.shared_classifier._trained:
            return None
        if self._snapshot_cache is None:
            self._snapshot_cache = self.shared_classifier.state_snapshot()
        return self.model_version, copy.deepcopy(self._snapshot_cache)

    # ---------- Task dispatch (hivemind only) ----------

    def enqueue_task(self, task: dict) -> None:
        """Producer side: waste generator drops a new task into the queen's
        pending pool. The dispatcher process picks it up."""
        self.pending_tasks.append(task)

    def score_agent_for_task(self, agent, task) -> float:
        """How desirable is `agent` for `task`? Higher is better.

        We combine three signals:
          * distance to the pickup zone (closer is better)
          * battery % (more is better, scaled small)
          * accumulated dose (less is better, avoid sending hot drones to ILW)
        """
        from .facility import distance_m
        pickup_zone = task["pickup_zone"]
        d = distance_m(agent.pos, (pickup_zone.x, pickup_zone.y))
        # Anchor the scoring so distance dominates by ~10x other factors
        return (
            -d
            + (agent.battery_pct / 100.0) * 3.0
            - (agent.integrated_dose_uSv / 1000.0) * 0.5
        )

    def assign_task(self, candidates: list, task: dict):
        """Pick the best candidate for the task, log the decision, and return
        (chosen_agent, justification_str). Caller is responsible for pushing
        the task to the chosen agent's dispatch queue."""
        if not candidates:
            return None, "no candidates"
        scored = [(self.score_agent_for_task(a, task), a) for a in candidates]
        scored.sort(reverse=True, key=lambda x: x[0])
        best_score, best_agent = scored[0]
        from .facility import distance_m
        pickup = task["pickup_zone"]
        d = distance_m(best_agent.pos, (pickup.x, pickup.y))
        justification = (
            f"{d:.1f}m, batt {best_agent.battery_pct:.0f}%, "
            f"dose {best_agent.integrated_dose_uSv:.0f}µSv"
        )
        self.last_dispatch_decision = {
            "agent_id": best_agent.agent_id,
            "item_id": task["item"].item_id,
            "pickup": pickup.name,
            "justification": justification,
        }
        self.recent_messages.append({
            "t_s": time.time(),
            "kind": "dispatch",
            "agent_id": best_agent.agent_id,
            "pickup_pos": (pickup.x, pickup.y),
            "item_id": task["item"].item_id,
            "justification": justification,
        })
        kind_s = task.get("kind", "handle")
        self.decision_log.append({
            "t_s": time.time(),
            "kind": "dispatch",
            "text": f"{kind_s}: {best_agent.agent_id} ← {task['item'].item_id} ({justification})",
        })
        self.activity_status = f"dispatching to {best_agent.agent_id}"
        return best_agent, justification

    # ---------- Ledger export ----------

    def ledger_dataframe(self) -> pd.DataFrame:
        rows = []
        for e in self.ledger.values():
            rows.append({
                "item_id": e.item_id,
                "true_class": e.true_class,
                "location": e.current_location,
                "classification": e.current_classification,
                "confidence": e.confidence,
                "scrutiny": e.scrutiny_flag,
                "audit_events": len(e.audit_trail),
            })
        return pd.DataFrame(rows)
