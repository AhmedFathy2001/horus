# Radwaste Hivemind Sim

A multi-agent simulation of robotic radioactive-waste classification at a
nuclear power plant. A central coordinator ("queen") learns from the
high-resolution QA lab and pushes a learned actinide-spike threshold
out to a fleet of mobile NaI-equipped agents ("drones") so they can catch
tricky waste items that look like LLW by activity but are really ILW by
composition.

The facility models a simplified end-to-end PUREX reprocessing pipeline:
upstream stages (Spent Fuel Receipt → Shearing → Dissolution → Solvent
Extraction → HLW Concentration → Solidification) feed the robot fleet,
which sorts and routes drums into VLLW / LLW / ILW / HLW storage. HLW
casks from the Dissolution / HLW-Concentration cells are pre-classified
by process knowledge and follow a forced sub-route through Solidification
to HLW storage; the classification challenge lives on the legacy
contact-handling sources (Cleanup ops / Maintenance / Plant samples)
where drone NaI scans must catch the actinide-masquerade trick.

The point of this sim is to make one architectural claim demo-able on a
laptop in under a minute: **coordinated classification with collective
learning measurably outperforms isolated single-agent classification on
classification accuracy, with no penalty in throughput.**

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python run_demo.py                 # default: compare isolated vs hivemind
```

You'll get:

1. A live pygame window showing the facility, agents, the coordinator panel
   (model version, retrains, learned threshold), and a progress bar — for
   each mode in turn.
2. A matplotlib dashboard with side-by-side accuracy / dose / throughput
   bars, confusion matrices, the coordinator's learning curve, and dose
   accumulation over time.
3. A terminal headline summarizing the comparison.

### Flags

| flag                      | default  | notes |
|---------------------------|----------|-------|
| `--mode {compare,isolated,hivemind}` | `compare` | compare runs both |
| `--seed N`                | `1`      | seed `1` is the cleanest demo scenario |
| `--sim-hours H`           | `8`      | one operating shift |
| `--interarrival-s S`      | `180`    | mean Poisson interarrival |
| `--agents N`              | `3`      | number of mobile sorting agents |
| `--tricky-fraction F`     | `0.25`   | fraction of items with actinide-spike trick |
| `--qa-fraction F`         | `0.05`   | random items sent to HPGe regardless of confidence |
| `--no-live`               | off      | skip pygame window |
| `--no-dashboard`          | off      | skip matplotlib charts |
| `--save-dashboard path`   | —        | also save dashboard PNG to disk |
| `--audit-csv path`        | —        | dump coordinator's ledger as CSV |

## What you should see

### Live view
- **Five mobile agents** by default (`drone-1` … `drone-5`) shuttle between
  generation points, the Triage station, and storage. Their colour
  indicates state: green = idle, blue = moving, orange = acquiring,
  magenta = classifying, cyan = charging, red = decon.
- A **field-of-vision cone** is drawn in front of each active drone — a
  60° wedge that follows the direction of motion, indicating which part
  of the facility the on-board camera + sensors can see at any moment.
- One agent (`drum-scanner-1`) is anchored at the HPGe QA lab and
  re-classifies items the mobile fleet flagged (low-confidence) or that
  came in via random QA sampling.
- A **QUEEN node** at top-center represents the coordinator. Reports from
  drones to the queen and snapshot broadcasts from the queen to all drones
  appear as **animated wire-trace lines** with a travelling dot, so you can
  see the actual coordination happening in real time.
- Zone labels appear below each circle; the short tag inside the circle
  identifies the zone at a glance (DECON, MAINT, RCS, SORT, HPGe, OPS,
  CHG, QUEEN, DECON*, VLLW/LLW/ILW).
- Storage zones show a count badge of items currently inside. Generation
  points show a queue-depth badge when items are waiting for pickup.
- Mobile agents carry a class-coloured chip next to their body when
  holding an item (gray = unclassified, green = VLLW, yellow = LLW,
  red = ILW).
- Under each drone are two thin bars: **battery** (green→amber→red as it
  drains) and **integrated radiation dose** (rises toward red as the
  electronics get cooked).
- The right-hand panel has four sections: coordinator state, live metrics,
  **fleet health** (battery + dose per agent), and a scrolling event log
  highlighting retrains, scrutiny flags, charging trips, decon trips, and
  any permanent failures.

### Robot health model
Real robots in nuclear environments wear out. The sim models this:

- **Battery** drains during work (~0.012%/sim-s active, ~0.004%/sim-s idle).
  Below 25%, the agent drops everything and goes back to the charging bay
  (state: `RETURNING_TO_CHARGE` → `CHARGING`). Recharges at 0.6%/s.
- **Integrated dose** accumulates whenever the agent is within ~3 m of an
  unshielded hot item (inverse-square attenuation). Mis-routed ILW at the
  LLW storage zone is the worst exposure source — robots brushing past
  pick up dose.
- Above 5 mSv cumulative, the agent must visit the **Robot wash** (state:
  `RETURNING_TO_DECON` → `DECONNING`). Decon takes 10 sim-minutes and
  removes 85% of the integrated dose.
- Above 50 mSv cumulative, the agent **permanently fails** (state: `FAILED`,
  visualised with an X through the body). The remaining fleet picks up the
  slack — which is where the hivemind's shared task queue shines.

### Item routing flow
Each waste item follows this path:

```
gen point  →  picked up by drone  →  Triage station
                                      │
                       (NaI + dose + CV classification)
                                      │
                          ┌───────────┴───────────┐
                  high confidence            low confidence
                          │                       │
                          ▼                       ▼
                  → storage zone           → QA lab (HPGe)
                                                  │
                                       authoritative re-class
                                                  │
                                                  ▼
                                          → storage zone
```

Most items go straight from Triage to storage. An item gets routed to the
QA lab if **any** of these happens at Triage:

- The gamma-derived class sits right at a class boundary (e.g. specific
  activity within ~0.1 decade of the VLLW/LLW or LLW/ILW threshold).
- The CV classifier disagrees with the gamma+rules verdict on a normal
  item (CV is intentionally fooled by the actinide-spike trick, so it
  won't trigger on tricky items — those are caught by the actinide
  threshold instead).
- The agent's local ML head disagrees with the rules class.
- Random QA sampling (5% by default — adjustable via `--qa-fraction`).

Typically ~15-30% of items get routed via the QA lab.

### Computer vision
Each mobile agent runs a lightweight camera + ML head alongside the gamma
spectrometer. The CV head classifies items by container type / packaging
with a noisy confusion matrix (~75–88% per class on normal items). It is
**deliberately fooled by the actinide-spike trick** (the trick *is* a
mis-packaged LLW container), so CV alone can't catch tricks — but when CV
disagrees with the gamma-rules verdict on a normal item, the agent drops
confidence below the scrutiny threshold and routes the item to HPGe. This
gives the hivemind another path to discover edge cases.

### Live controls
Once the pygame window is focused:

| key       | action |
|-----------|--------|
| `SPACE`   | pause / resume the sim |
| `+` / `-` | speed up / slow down (`30x` → `60x` → `120x` → `240x` → `480x` → `960x`) |
| `T`       | inject a tricky actinide-spiked item at a random generation point — watch a hivemind agent catch it once the threshold has been learned, or watch an isolated agent route it to LLW storage |
| `R`       | force the coordinator to retrain immediately (hivemind only); useful right after pressing `T` a few times to fast-forward the learning loop |
| `Q` / `ESC` | quit the current run (the next mode in `compare` will still start) |

### Dashboard
- **Headline bars** (top row): accuracy, cumulative worker dose, throughput.
  Hivemind should beat isolated on accuracy by 5–10 percentage points.
- **Confusion matrices**: in isolated mode you'll typically see a chunky
  `(true=ILW → predicted=LLW)` cell. In hivemind, that cell empties out
  after the coordinator pushes the threshold.
- **Coordinator learning curve**: shows the actinide threshold being derived
  and refined as more HPGe rescrutiny labels come in.

### Honest caveats
- The **dose metric is noisy** — it's dominated by the in-transit handling
  of high-activity normal ILW items, which is identical in both modes.
  The hivemind advantage on dose comes from misrouted ILW items sitting
  unshielded at LLW storage and being exposed during periodic worker
  inspections. Across seeds, hivemind dose is at or below isolated, but
  the gap is variance-bound. **Average across several seeds** for a robust
  read; default seed `1` shows the cleanest single-shot.
- The **accuracy gap is consistent** across seeds.
- Throughput is essentially the same in both modes (which is the point:
  the hivemind doesn't cost you items/hour).

## Architecture

```
                ┌────────────────────────────────────────────┐
                │  Coordinator ("queen")                     │
                │   • inventory ledger (pandas)              │
                │   • aggregate training corpus              │
                │   • learns actinide-spike threshold        │
                │   • broadcasts model snapshots             │
                │   • drift detection on rescrutiny gap      │
                └────────────┬───────────────────────┬───────┘
                  reports    │                       │  snapshots
                ┌────────────▼──┐  ┌────────────────▼──────────┐
                │ mobile agents │  │ QA lab agent (HPGe) │
                │ (NaI 2"x2")   │  │  authoritative re-class.  │
                └───────────────┘  └───────────────────────────┘
```

### Two-tier classifier
1. **Rules engine** (auditable, regulator-friendly): hard thresholds on
   specific activity (Bq/g) and surface dose rate (µSv/h) from
   `data/iaea_thresholds.json`. Documented inline.
2. **Learned actinide-spike threshold**: a single scalar derived by the
   coordinator from HPGe rescrutiny labels. Mobile agents apply it: if rules
   say LLW or VLLW but the normalized counts in actinide photopeak windows
   exceed the threshold, escalate to ILW.

The discriminating learned quantity is intentionally one number — easy to
demo ("look, threshold goes from `None` to `0.04`, accuracy jumps"), easy
to compute from a handful of trusted labels, easy to explain to a judge who
doesn't want to wade through k-NN internals.

A small k-NN ML head is also kept for confidence estimation (it shrinks the
mobile agent's confidence when it disagrees with the rules class, which
triggers HPGe rescrutiny and feeds more training data).

### Why HPGe gets to be the oracle
Mobile agents carry NaI(Tl) 2"×2" scintillators (~7% FWHM at 662 keV). The
fixed QA lab uses HPGe (~0.2% FWHM at 1332 keV) and takes a 60-s
integration at close geometry. HPGe is the same kind of qualified
instrument a plant would use for shipping/disposal characterization, so we
treat its calls as authoritative for training labels. This is the
federated-learning pattern: field instruments train against the lab.

### Stigmergy
Items the mobile fleet classifies with low confidence get a `scrutiny_flag`
set in the coordinator's ledger. The flag persists; downstream stations
(QA lab) treat flagged items with higher-integration assays. This is
the "indirect coordination via the environment" piece the spec asked for.

## Physics — what's real, what's simplified

**Real:**
- Radionuclide gamma lines and intensities (Co-60, Cs-137/134, Co-58,
  Mn-54, Am-241, U-235, U-238, Pu-239, etc.) from public nuclear data.
- NaI(Tl) ~7% FWHM at 662 keV; HPGe ~0.2% FWHM at 1332 keV. Resolution
  scales as `sqrt(E)`.
- Poisson counting statistics applied per channel.
- Inverse-square distance scaling for dose rate and detector geometry.
- Specific gamma-ray dose constants (µSv/h per MBq at 1 m) for the
  major nuclides.

**Simplified:**
- Compton continuum is a smoothed step up to the Klein-Nishina edge — not
  a full transport calculation.
- Surface contamination probe is a stub (alpha/beta count rate proxy).
- No detector dead-time modeling beyond a constant fraction.
- No shielding self-attenuation inside the waste matrix.
- Beta dose ignored (we only score photon dose).
- Pathfinding is direct-line travel, not A* around obstacles. The facility
  has no walls.
- HLW *handling* is in scope as a forced-route transit task; the chemistry
  that produces HLW (PUREX dissolution / extraction) is represented as
  named upstream stages with their own waste-generation cadence but is
  not chemically modelled. The robot sees HLW as "pre-typed liquid casks
  that must pass through Solidification before storage", which is the
  operationally-relevant part for a sorting-fleet simulation.

**Things a production deployment would add that this sim doesn't:**
- ROS 2 / DDS for actual robot comms (this sim uses Python in-process).
- OPC UA for coordinator ↔ plant I&C integration.
- A data diode between the safety-class side and the conventional side
  (IEC 60709 / IEC 62859).
- Redundant hot-standby coordinator. The sim has one queen.
- Safety-class qualification for any classification decision used in
  regulatory documentation (IEC 61513, IEEE 7-4.3.2 for ML, RG 1.152).
- Real HPGe integration with cryostat handling, energy/efficiency
  calibration drift tracking, ANSI N42.14-style QA.

## Code layout

```
run_demo.py                     # single-command entrypoint
requirements.txt
data/iaea_thresholds.json       # documented class boundaries
sim/
  radionuclides.py              # nuclide data, half-lives, gamma lines, waste profiles
  sensors.py                    # gamma spectrometer, dose-rate meter, contamination probe
  waste_generator.py            # synthetic items with ground-truth compositions
  facility.py                   # 2D zones, direct-line travel
  classifier.py                 # rules + learned actinide threshold + k-NN
  coordinator.py                # ledger, retrain, threshold derivation, snapshots
  agent.py                      # state machine for mobile + drum-scanner agents
  metrics.py                    # accuracy / dose / throughput tracking
  scenario.py                   # builds and runs end-to-end with simpy
viz/
  pygame_view.py                # live 2D facility view + coordinator panel
  dashboard.py                  # end-of-run matplotlib comparison
tests/
  test_classifier.py            # photopeak landing tests, threshold derivation, snapshot roundtrip
```

## Running the tests

```bash
python -m pytest -q tests/
```

The tests verify:
- Co-60 and Cs-137 photopeaks land at the correct energies on NaI.
- HPGe photopeaks are sharper than NaI for the same source.
- The rules engine respects the boundaries in `iaea_thresholds.json`.
- The actinide signature is higher for actinide sources than for Cs-137.
- The classifier doesn't override anything until the coordinator pushes a
  threshold (the isolated story).
- Once the threshold is set, tricky items get correctly escalated.
- The coordinator → agent snapshot roundtrip preserves class predictions.

## Tuning the demo

If the headline gap isn't visible on your machine:

- **Increase tricky fraction** (`--tricky-fraction 0.40`) — more cases where
  the hivemind's learned override matters.
- **Decrease QA sampling** (`--qa-fraction 0.02`) — fewer free rescues for
  the isolated mode.
- **Longer run** (`--sim-hours 12`) — more retrains, more time for the
  hivemind to learn.
- **More agents** (`--agents 5`) — more diverse classifications feeding the
  coordinator.

The defaults are picked so that, with `--seed 1`, the demo runs in under 90
seconds wall-clock and the dashboard tells a clean story.
