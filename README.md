# HORUS — Hivemind for Onboard Radiological Understanding & Sorting

Pantomath · Hackatom 2026 · Egypt

A multi-agent simulation of robotic radioactive-waste classification at a
nuclear power plant. A central **coordinator** learns from the
high-resolution QA lab and pushes a learned actinide-spike threshold
into a centralized drum-characterization station so a fleet of mobile
**cart agents** (transport AGVs) can keep the line moving and catch
tricky waste items that look like LLW by activity but are really ILW by
composition. Classification happens once, centrally, at the Char Station —
the carts are dumb transports that ferry drums to it and route the verdicts
onward.

The facility models a simplified end-to-end PUREX reprocessing pipeline:
upstream stages (Spent Fuel Receipt → Shearing → Dissolution → Solvent
Extraction → HLW Concentration → Solidification) feed the robot fleet,
which sorts and routes drums into VLLW / LLW / ILW / HLW storage. HLW
casks from the Dissolution / HLW-Concentration cells are pre-classified
by process knowledge and follow a forced sub-route through Solidification
to HLW storage; the classification challenge lives on the legacy
contact-handling sources (Cleanup ops / Maintenance / Plant samples)
where the centralized NaI scan must catch the actinide-masquerade trick.

The point of this sim is to make one architectural claim demo-able on a
laptop in under a minute: **a classifier with the coordinator's collective
learning loop (hivemind) measurably outperforms the same classifier with that
loop switched off (isolated) on classification accuracy, with no penalty in
throughput.**

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python run_demo.py                 # default: compare isolated vs hivemind
```

You'll get:

1. A live pygame window showing the facility, the cart fleet, the
   coordinator panel (model version, retrains, learned threshold), and
   a progress bar — for each mode in turn.
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
| `--interarrival-s S`      | `45`     | mean Poisson interarrival |
| `--handlers N`            | `5`      | gripper transport AGVs |
| `--hybrids N`             | `1`      | transport AGVs with a quicker chassis |
| `--agents N`              | `0`      | legacy single-knob fleet size (all-hybrid); ignored when role flags are set |
| `--tricky-fraction F`     | `0.25`   | fraction of items with actinide-spike trick |
| `--qa-fraction F`         | `0.05`   | random items sent to HPGe regardless of confidence |
| `--net-drop F`            | `0.03`   | wireless packet-drop probability on agent → coordinator reports |
| `--no-live`               | off      | skip pygame window |
| `--no-dashboard`          | off      | skip matplotlib charts |
| `--save-dashboard path`   | —        | also save dashboard PNG to disk |
| `--audit-csv path`        | —        | dump coordinator's ledger as CSV |

> Default fleet is **6 cart-style AGVs**: `handler-1`…`handler-5` plus
> `hybrid-1`. With centralized characterization at the Char Station, every
> cart is a pure transport AGV (Handlers and Hybrids differ only by chassis
> speed) — there is no roaming "scanner" role.

## What you should see

### Live view
- **Cart fleet** (default 6) shuttles between generation points, the
  central **Drum Characterization Station** (zone tag `CLASSIFY`), and
  storage. Agent colour indicates state: green = idle, blue = moving,
  cyan = charging, red = decon. Carts don't classify — they transport.
- The **Spent Fuel Pool** (`POOL`) is both the visual origin of the plant
  and a real waste source: pool filters / sludge / contaminated tools flow
  from it to the Char Station like any other generation cell.
- One fixed station agent (`char-A`) is anchored at the HPGe **QA lab**
  and re-classifies items the central classifier flagged low-confidence
  or that came in via random QA sampling.
- A **Coordinator pillar** (tag `AI BRAIN`) sits between the upstream
  pipeline and the cart aisles. Agent → coordinator reports and
  coordinator → fleet snapshot broadcasts appear as **animated wire-trace
  lines with a travelling dot**, so you can see the actual coordination
  happening in real time. Dropped packets (default ~3%) appear as faded
  lines that never complete.
- Zone labels appear below each circle; the short tag inside the circle
  identifies the zone at a glance (`POOL`, `SHEAR`, `DISS`, `SOLV`,
  `HLWC`, `VITR`, `OFFGAS`, `ACID`, `SOLTRT`, `AI BRAIN`, `CHG`,
  `CLASSIFY`, `CTRL`, `QA`, `WASH`, `CLEAN`, `MAINT`, `SAMPL`, `CLEAR`,
  `VLLW`/`LLW`/`ILW`/`HLW`).
- Storage zones show a count badge of items currently inside. Generation
  points show a queue-depth badge when items are waiting for pickup.
- Carts carry a class-coloured chip next to their body when holding an
  item (gray = unclassified, green = VLLW, yellow = LLW, red = ILW).
- Under each cart are two thin bars: **battery** (green→amber→red as it
  drains) and **integrated radiation dose** (rises toward red as the
  electronics get cooked).
- The right-hand panel has four sections: coordinator state, live metrics,
  **fleet health** (battery + dose per agent), and a scrolling event log
  highlighting retrains, scrutiny flags, charging trips, decon trips,
  failovers, and any permanent failures.

### Robot health model
Real robots in nuclear environments wear out. The sim models this:

- **Battery** drains during work (~0.012%/sim-s active, ~0.004%/sim-s idle).
  Below 25%, the agent drops everything and goes back to the charging bay
  (state: `RETURNING_TO_CHARGE` → `CHARGING`). Recharges at 0.6%/s.
- **Integrated dose** accumulates whenever the agent is within ~3 m of an
  unshielded hot item (inverse-square attenuation). Mis-routed ILW at the
  LLW storage zone is the worst exposure source — carts brushing past
  pick up dose.
- Above 5 mSv cumulative, the agent must visit the **Robot wash** (state:
  `RETURNING_TO_DECON` → `DECONNING`). Decon takes 10 sim-minutes and
  removes 85% of the integrated dose. A lighter **preventive wash**
  (3 min, 40% reduction) fires after 1 mSv of dose accrued since the
  last decon, so wash arms have something to do on a clean shift.
- Above 50 mSv cumulative, the agent **permanently fails** (state:
  `FAILED`, visualised with an X through the body). A hot-spare
  **replacement spawns automatically** ~30 sim-seconds later — the
  hivemind's shared task queue keeps the line running while the new
  cart drives in.

### Item routing flow
Each waste item follows this path:

```
gen point  →  picked up by cart  →  Char Station (CLASSIFY)
                                      │
                       (NaI + dose + CV classification by the brain)
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

Classification happens **once, centrally, at the Char Station** — carts
are dumb transports. HLW drums from the Dissolution / HLW-Concentration
cells skip the Char station and follow a forced sub-route through
Solidification straight to HLW storage (their class is process-known).

An item gets routed onward to the QA lab if **any** of these happens at
the Char station:

- The gamma-derived class sits right at a class boundary (e.g. specific
  activity within ~0.1 decade of the VLLW/LLW or LLW/ILW threshold).
- The CV classifier disagrees with the gamma+rules verdict on a normal
  item (CV is intentionally fooled by the actinide-spike trick, so it
  won't trigger on tricky items — those are caught by the actinide
  threshold instead).
- The k-NN ML head disagrees with the rules class.
- Random QA sampling (5% by default — adjustable via `--qa-fraction`).

Typically ~15-30% of items get routed via the QA lab.

### Computer vision
The Char station runs a lightweight camera + ML head alongside the gamma
spectrometer. The CV head classifies items by container type / packaging
with a noisy confusion matrix (~75–88% per class on normal items). It is
**deliberately fooled by the actinide-spike trick** (the trick *is* a
mis-packaged LLW container), so CV alone can't catch tricks — but when CV
disagrees with the gamma-rules verdict on a normal item, confidence
drops below the scrutiny threshold and the item routes to HPGe. This
gives the hivemind another path to discover edge cases.

### Live controls
Once the pygame window is focused:

| key       | action |
|-----------|--------|
| `SPACE`   | pause / resume the sim |
| `+` / `-` | speed up / slow down (`30x` → `60x` → `120x` → `240x` → `480x` → `960x`) |
| `T`       | inject a tricky actinide-spiked item at a random generation point — watch the hivemind catch it once the threshold has been learned, or watch isolated mode route it to LLW storage |
| `R`       | force the coordinator to retrain immediately (hivemind only); useful right after pressing `T` a few times to fast-forward the learning loop |
| `1` … `5` | kill cart `1`…`5` (simulates catastrophic dose damage); a hot-spare deploys automatically |
| `K`       | kill the coordinator → hot-standby promotes with zero data loss; failover banner flashes |
| `F`       | cycle through preset fleet compositions (apply with `X`) |
| `[` / `]` | shrink / grow the cart fleet by one (apply with `X`) |
| `X`       | restart the current mode from t=0 with the new fleet |
| `M`       | swap mode (hivemind ⇄ isolated) and restart |
| `H`       | toggle help overlay |
| `Q` / `ESC` | quit the current run (the next mode in `compare` will still start) |

### Dashboard
- **Headline bars** (top row): accuracy, cumulative worker dose, throughput.
  Hivemind beats isolated on accuracy on every seed; on the clean seeds the
  gap is ~10–14 percentage points (seed `1`: ~+12 pp), smaller on noisier
  ones.
- **Confusion matrices**: in isolated mode you'll typically see a chunky
  `(true=ILW → predicted=LLW)` cell. In hivemind, that cell empties out
  after the coordinator pushes the threshold.
- **Coordinator learning curve**: shows the actinide threshold being derived
  and refined as more HPGe rescrutiny labels come in.

### Honest caveats
- The **dose metric is a bounded model, not a transport calculation.** Worker
  dose is charged from hot drums (ILW) sitting unshielded in the low-level
  store because they were mis-filed — each waste class has a representative
  exposure rate, so the dose signal tracks *how many* drums are mis-routed,
  not one drum's raw activity. Hivemind mis-routes far fewer, so its dose is
  **at or below** isolated across seeds — often 20–50% lower, occasionally
  about equal. Average across a few seeds for a robust read; seed `1` is the
  cleanest single-shot. Every run is now fully reproducible for a given
  `--seed` (the upstream RNG used to be salted by Python's per-process string
  hashing — fixed).
- The **accuracy gap is consistent in sign** (hivemind always wins) but its
  size varies by seed, ~3–14 pp on a short shift and wider on a full one.
- Throughput is essentially the same in both modes (which is the point:
  the hivemind doesn't cost you items/hour).

## Architecture

```
                ┌────────────────────────────────────────────┐
                │  Coordinator (AI BRAIN)                    │
                │   • inventory ledger (pandas)              │
                │   • aggregate training corpus              │
                │   • learns actinide-spike threshold        │
                │   • broadcasts model snapshots             │
                │   • drift detection on rescrutiny gap      │
                │   • hot-standby with zero-data-loss failover│
                └────┬───────────┬───────────────────────┬───┘
              reports│   snapshots│                       │
            ┌────────▼────┐  ┌────▼─────────────┐  ┌─────▼─────────┐
            │ Char Station│  │ Cart fleet       │  │ QA lab agent  │
            │ (CLASSIFY)  │  │ transport AGVs   │  │ (HPGe, char-A)│
            │ central NaI │  │ (handler /       │  │ authoritative │
            │ + CV head   │  │  hybrid)         │  │ re-class.     │
            └─────────────┘  └──────────────────┘  └───────────────┘
```

### Two-tier classifier
1. **Rules engine** (auditable, regulator-friendly): hard thresholds on
   specific activity (Bq/g) and surface dose rate (µSv/h) from
   `data/iaea_thresholds.json`. Documented inline.
2. **Learned actinide-spike threshold**: a single scalar derived by the
   coordinator from HPGe rescrutiny labels. The central classifier applies
   it: if rules say LLW or VLLW but the normalized counts in actinide
   photopeak windows exceed the threshold, escalate to ILW.

The discriminating learned quantity is intentionally one number — easy to
demo ("look, threshold goes from `None` to `0.04`, accuracy jumps"), easy
to compute from a handful of trusted labels, easy to explain to a judge
who doesn't want to wade through k-NN internals.

A small k-NN ML head is also kept for confidence estimation (it shrinks
the central classifier's confidence when it disagrees with the rules
class, which triggers HPGe rescrutiny and feeds more training data).

### Why HPGe gets to be the oracle
The Char station carries a NaI(Tl) 2"×2" scintillator (~7% FWHM at
662 keV). The fixed QA lab uses HPGe (~0.2% FWHM at 1332 keV) and takes a
60-s integration at close geometry. HPGe is the same kind of qualified
instrument a plant would use for shipping/disposal characterization, so
we treat its calls as authoritative for training labels. This is the
federated-learning pattern: field instruments train against the lab.

### Stigmergy
Items the central classifier flags with low confidence get a
`scrutiny_flag` set in the coordinator's ledger. The flag persists;
downstream stations (QA lab) treat flagged items with higher-integration
assays. This is the "indirect coordination via the environment" piece
the spec asked for.

### Coordinator resilience
The coordinator runs as a **primary + hot-standby pair** sharing the
ledger and training pool. Pressing `K` in the live view triggers a
failover: the standby is promoted instantly with zero data loss and the
sim keeps running. A small fraction of agent reports drop on the
wireless link (default 3%, `--net-drop`) to make network behaviour
visible — dropped messages render as faded animated lines that never
arrive at the brain.

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
- ROS 2 / DDS for actual robot comms (this sim uses Python in-process
  with a tunable packet-drop rate).
- OPC UA for coordinator ↔ plant I&C integration.
- A data diode between the safety-class side and the conventional side
  (IEC 60709 / IEC 62859).
- Real replicated hot-standby coordinator over a heartbeat link (the sim
  models the failover semantics with a counter, not two separate
  processes).
- Safety-class qualification for any classification decision used in
  regulatory documentation (IEC 61513, IEEE 7-4.3.2 for ML, RG 1.152).
- Real HPGe integration with cryostat handling, energy/efficiency
  calibration drift tracking, ANSI N42.14-style QA.

## Code layout

```
run_demo.py                     # single-command entrypoint
requirements.txt
horus.spec                      # PyInstaller spec (builds HORUS.app / horus.exe)
data/iaea_thresholds.json       # documented class boundaries
sim/
  radionuclides.py              # nuclide data, half-lives, gamma lines, waste profiles
  sensors.py                    # gamma spectrometer, dose-rate meter, contamination probe
  waste_generator.py            # synthetic items with ground-truth compositions
  facility.py                   # 2D zones, direct-line travel
  classifier.py                 # rules + learned actinide threshold + k-NN
  coordinator.py                # ledger, retrain, threshold derivation, snapshots, failover
  agent.py                      # state machine for transport carts (handler/hybrid) + QA station
  metrics.py                    # accuracy / dose / throughput tracking
  scenario.py                   # builds and runs end-to-end with simpy; Char station actuator
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
- The coordinator → classifier snapshot roundtrip preserves class
  predictions.

## Tuning the demo

If the headline gap isn't visible on your machine:

- **Increase tricky fraction** (`--tricky-fraction 0.40`) — more cases where
  the hivemind's learned override matters.
- **Decrease QA sampling** (`--qa-fraction 0.02`) — fewer free rescues for
  the isolated mode.
- **Longer run** (`--sim-hours 12`) — more retrains, more time for the
  hivemind to learn.
- **Bigger fleet** (`--handlers 5 --hybrids 2`) — more transport bandwidth
  feeding the Char station, more rescrutiny labels per shift.

The defaults are picked so that, with `--seed 1`, the demo runs in under 90
seconds wall-clock and the dashboard tells a clean story.
