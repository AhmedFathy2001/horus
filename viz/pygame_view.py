"""Live 2D facility view. Designed for hackathon demo:

  * Zones are large, bordered, and labeled with their role.
  * Storage zones show a count badge of items currently inside.
  * Generation points show a queue depth badge (items waiting for pickup).
  * Agents are color-coded by state and carry a class-colored chip when
    holding an item.
  * Coordinator panel on the right is broken into Coordinator / Live
    metrics / Event log sections.
  * Interactive controls (pause, speed, inject tricky, force retrain)
    are listed in the footer.

The on_tick(ctx) entrypoint is called from sim.scenario.run_scenario with a
TickContext that bundles the running sim's state and exposes
inject_tricky_item() / force_retrain() helpers.
"""
from __future__ import annotations
import math
import os
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import pygame

from sim.facility import (
    ZONES, ZONES_BY_NAME, WIDTH_M, HEIGHT_M,
)

ICON_PATH = Path(__file__).parent.parent / "assets" / "horus_icon.png"


# --- Layout ---------------------------------------------------------------

# Pixels per metre. Tuned so the wider (50 m) facility plus the right-hand
# panel still fits comfortably on a 1080p display.
PX_PER_M = 22
HEADER_H = 60
FOOTER_H = 96
PANEL_W = 460
FACILITY_W = int(WIDTH_M * PX_PER_M)
FACILITY_H = int(HEIGHT_M * PX_PER_M)
WINDOW_W = FACILITY_W + PANEL_W
WINDOW_H = FACILITY_H + HEADER_H + FOOTER_H

# Wireless network range (queen broadcasts reach this far in metres). Drones
# outside the ring are visualized as out-of-coverage.
WIRELESS_RANGE_M = 22.0

# Activity-status banner duration after each report (ms)
COORD_STATUS_TTL_MS = 900

# --- Colors ---------------------------------------------------------------

BG          = (22, 25, 31)
PANEL_BG    = (28, 32, 40)
GRID        = (45, 50, 60)
TEXT        = (225, 228, 235)
DIM_TEXT    = (140, 148, 160)
HI_TEXT     = (255, 255, 255)
ACCENT      = (95, 200, 255)
WARN        = (255, 180, 80)
DANGER      = (240, 92, 92)
OK          = (130, 220, 130)

# Class colors — match the dashboard
# Plain-English captions for agent states. Keeps the label pill readable
# to a non-engineer audience: an AGV doesn't say "ACQUIRING", it says
# "scanning gamma" — the actual physics step a nuclear pro recognises.
STATE_CAPTION = {
    "IDLE":                "idle",
    "MOVING_TO_PICKUP":    "→ pickup",
    "PICKING_UP":          "loading drum",
    "MOVING_TO_SORTING":   "→ scan station",
    "ACQUIRING":           "scanning gamma",
    "CLASSIFYING":         "matching peaks",
    "REPORTING":           "→ coordinator",
    "MOVING_TO_DROPOFF":   "→ dropoff",
    "DROPPING_OFF":        "unloading",
    "MOVING_TO_RESCAN":    "→ QA lab",
    "RESCANNING":          "HPGe rescan",
    "RETURNING_TO_CHARGE": "→ charging bay",
    "CHARGING":            "charging",
    "RETURNING_TO_DECON":  "→ decon",
    "DECONNING":           "decontaminating",
    "FAILED":              "OFFLINE",
}


CLASS_COLOR = {
    "VLLW": (180, 220, 160),
    "LLW":  (240, 210, 110),
    "ILW":  (235, 110, 95),
    "HLW":  (200, 70, 70),
    None:   (120, 125, 140),
}

# Per-nuclide tint used by the drum-contents strip. Each drum carries a
# thin coloured stripe per dominant nuclide so the actinide-spike trick
# is *visibly* present on tricky drums (purple = Am-241 / Pu-239).
NUCLIDE_COLOR = {
    "Cs-137": (120, 220, 240),   # cyan
    "Cs-134": (130, 200, 235),   # paler cyan
    "Co-60":  (240, 215, 100),   # warm yellow
    "Co-58":  (240, 195, 130),   # softer yellow
    "Mn-54":  (220, 170, 110),   # tan
    "Sr-90":  (140, 220, 140),   # green
    "I-131":  (250, 165, 165),   # pink
    "Fe-55":  (180, 170, 200),
    "Ni-63":  (170, 170, 200),
    "U-235":  (200, 140, 220),   # purple — actinide
    "U-238":  (180, 130, 200),   # purple — actinide
    "Pu-239": (210, 110, 230),   # bright purple — actinide
    "Am-241": (220, 130, 240),   # bright purple — actinide
}
# Quick lookup: which nuclides are actinides (for the spike-detection
# story). Used to highlight actinide stripes more boldly.
ACTINIDE_NUCLIDES = frozenset({"U-235", "U-238", "Pu-239", "Am-241"})

# Agent state colors
STATE_COLOR = {
    "IDLE":                  (110, 200, 130),
    "MOVING_TO_PICKUP":      (110, 180, 230),
    "PICKING_UP":            (160, 200, 230),
    "MOVING_TO_SORTING":     (110, 180, 230),
    "ACQUIRING":             (230, 170, 90),
    "CLASSIFYING":           (200, 130, 220),
    "REPORTING":             (220, 140, 200),
    "MOVING_TO_DROPOFF":     (110, 180, 230),
    "DROPPING_OFF":          (160, 200, 230),
    "MOVING_TO_RESCAN":      (250, 160, 100),
    "RESCANNING":            (255, 200, 120),
    "RETURNING_TO_CHARGE":   (150, 200, 230),
    "CHARGING":              (120, 220, 255),
    "RETURNING_TO_DECON":    (255, 130, 130),
    "DECONNING":             (255, 100, 100),
    "FAILED":                (90, 90, 95),
}

# Role colors for zone outlines
ROLE_OUTLINE = {
    "generation":     (170, 170, 200),
    "sorting":        (130, 210, 130),
    "drum_scanner":   (110, 180, 230),
    "char_station":   (140, 220, 200),
    "fuel_pool":      (90, 160, 220),
    "storage":        (200, 200, 200),
    "worker":         (255, 200, 80),
    "charging":       (120, 180, 230),
    "decon":          (255, 130, 130),
    "coordinator":    (95, 200, 255),
    "process":        (180, 160, 220),
    "solidification": (190, 200, 230),
    "clearance":      (130, 220, 150),
}

# Per-AGENT-TYPE chassis profile used by the pygame renderer.
#   length_m / width_m : chassis dimensions in world metres (the chassis is
#                        drawn as a rotated rounded rectangle of this size).
#   mast               : Scanner-style sensor mast + camera dome on top.
#   fork               : Handler-style gripper fork prongs at the front.
#   accent             : role-identifying colour band painted across the
#                        chassis so judges can spot roles at a glance.
AGENT_PROFILE = {
    "scanner": {
        "length_m": 0.65, "width_m": 0.50,
        "mast": True, "fork": False,
        "accent": (110, 200, 255),   # cyan — "sensors"
    },
    "handler": {
        "length_m": 1.00, "width_m": 0.70,
        "mast": False, "fork": True,
        "accent": (255, 180, 90),    # orange — "muscle"
    },
    "hybrid": {
        "length_m": 0.80, "width_m": 0.60,
        "mast": True, "fork": True,
        "accent": (160, 220, 130),   # green — "generalist"
    },
    "qa_lab": {
        "length_m": 0.90, "width_m": 0.70,
        "mast": True, "fork": False,
        "accent": (180, 220, 250),
    },
}

# AGVs are supporting elements now (humans teleop fine in real plants).
# The visual stars are the AI panels: live spectrum, decision feed, decon
# arm. Slightly dim the state colors so chassis don't fight the spectrum.
def _dim(rgb: tuple, factor: float = 0.78) -> tuple:
    return (int(rgb[0] * factor), int(rgb[1] * factor), int(rgb[2] * factor))

# Comm-line animation duration
COMM_ANIM_MS = 700


# --- Event log ------------------------------------------------------------

@dataclass
class LogEvent:
    t_sim_s: float
    text: str
    color: tuple = TEXT


# --- LiveView -------------------------------------------------------------

@dataclass
class LiveView:
    title: str = "HORUS — Hivemind for Onboard Radiological Understanding & Sorting"
    target_fps: int = 30

    def __post_init__(self):
        os.environ.setdefault("SDL_VIDEO_CENTERED", "1")
        pygame.init()
        if ICON_PATH.exists():
            try:
                pygame.display.set_icon(pygame.image.load(str(ICON_PATH)))
            except pygame.error:
                pass
        self.screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
        pygame.display.set_caption(self.title)
        self.font_xs   = pygame.font.SysFont("Menlo, Consolas, monospace", 11)
        self.font_s    = pygame.font.SysFont("Menlo, Consolas, monospace", 13)
        self.font_m    = pygame.font.SysFont("Menlo, Consolas, monospace", 14, bold=True)
        self.font_l    = pygame.font.SysFont("Menlo, Consolas, monospace", 18, bold=True)
        self.font_xl   = pygame.font.SysFont("Menlo, Consolas, monospace", 24, bold=True)
        self.clock = pygame.time.Clock()
        self.closed = False
        self.mode_label = ""
        # Multipliers on real-time. 1x = wall-clock; useful for showing
        # judges every individual drum hand-off step. Higher levels skip
        # through to retrains + storage fills faster. Press +/- to cycle.
        self._speed_levels = [1, 2, 5, 10, 30, 60, 120, 240, 480, 960]
        self._speed_idx = 0  # start at 1x so judges can follow every step;
                              # press + to speed through to retrains
        # event log
        self.events: deque[LogEvent] = deque(maxlen=14)
        # Cached cursors for "what's new" diffing
        self._last_n_classifications = 0
        self._last_n_retrains = 0
        self._last_ledger_size = 0
        # Flash effect on coordinator panel when retrain occurs
        self._retrain_flash_until_ms: int = 0
        # Highlight an injected item briefly
        self._inject_flash_item: str | None = None
        self._inject_flash_until_ms: int = 0
        # Active comm animations + index of last seen coord message
        self._active_comms: list[dict] = []
        self._last_seen_msg_idx: int = 0
        # When the queen broadcasts a model snapshot, every drone should
        # briefly glow cyan to show they all received the same update.
        # Stored as a wall-clock deadline.
        self._hive_sync_until_ms: int = 0
        # User-driven restart / fleet-tuning state. The scenario runner
        # checks `restart_requested` at the end of each tick and re-runs
        # the current mode with the live fleet config when set.
        self.restart_requested: bool = False
        self.help_open: bool = False
        self.live_fleet: dict[str, int] = {"scanner": 2, "handler": 3, "hybrid": 1}
        # Snapshot of `live_fleet` as it was when the current mode started.
        # `live_fleet` becomes the *pending* config when the user adjusts
        # it with F / [ / ]; we render a header banner whenever the two
        # diverge so the user can see their change is staged for the
        # next X-restart.
        self.applied_fleet: dict[str, int] = dict(self.live_fleet)
        # When True, run_demo.py swaps the mode (hivemind <-> isolated)
        # and restarts. Set by the M-key.
        self.mode_switch_requested: bool = False
        # Wall-clock deadline for a "FAILOVER" banner flash on the coord
        self._coord_failover_until_ms: int = 0

    # ---------- public ----------

    def reset_for_mode(self, mode: str):
        self.mode_label = mode
        self.events.clear()
        self._last_n_classifications = 0
        self._last_n_retrains = 0
        self._last_ledger_size = 0
        self._active_comms.clear()
        self._last_seen_msg_idx = 0
        # New scenario started — the live_fleet config is now the active
        # one, so the 'pending fleet change' banner disappears.
        self.applied_fleet = dict(self.live_fleet)
        if hasattr(self, "_prev_agent_states"):
            del self._prev_agent_states
        self.events.append(LogEvent(0.0, f"--- starting {mode.upper()} mode ---", ACCENT))

    @property
    def sim_speedup(self) -> int:
        return self._speed_levels[self._speed_idx]

    # Preset fleet compositions for the F-cycle key.
    _FLEET_PRESETS = [
        {"scanner": 1, "handler": 2, "hybrid": 1},   # minimal
        {"scanner": 2, "handler": 3, "hybrid": 1},   # default
        {"scanner": 3, "handler": 4, "hybrid": 2},   # large
        {"scanner": 4, "handler": 5, "hybrid": 2},   # max
        {"scanner": 0, "handler": 0, "hybrid": 5},   # all-hybrid (legacy)
    ]

    def _cycle_fleet_preset(self) -> None:
        # Find the current preset index, advance one
        current = (self.live_fleet["scanner"],
                   self.live_fleet["handler"],
                   self.live_fleet["hybrid"])
        idx = 0
        for i, p in enumerate(self._FLEET_PRESETS):
            if (p["scanner"], p["handler"], p["hybrid"]) == current:
                idx = i
                break
        nxt = self._FLEET_PRESETS[(idx + 1) % len(self._FLEET_PRESETS)]
        self.live_fleet = dict(nxt)

    def _adjust_fleet_size(self, delta: int) -> None:
        """Grow or shrink the cart fleet by one. Redistributes the new
        total as 70% handlers + 30% hybrids (mix that runs well with
        the centralized-classifier model). Clamped 1..12."""
        total = (self.live_fleet["scanner"]
                 + self.live_fleet["handler"]
                 + self.live_fleet["hybrid"])
        total = max(1, min(12, total + delta))
        n_handlers = max(1, int(round(total * 0.7)))
        n_hybrids = max(0, total - n_handlers)
        self.live_fleet = {
            "scanner": 0,
            "handler": n_handlers,
            "hybrid": n_hybrids,
        }

    def on_tick(self, ctx) -> bool:
        """Called by scenario.run_scenario once per simulated step.
        Returns False when the user has quit, otherwise True."""
        self._pump_events(ctx)
        if self.closed:
            return False
        # Drive sim-step size from the current speed level so the user's
        # +/- presses actually slow / speed up the world.
        ctx.sim_seconds_per_frame = self.sim_speedup / self.target_fps
        self.clock.tick(self.target_fps)
        self._collect_events(ctx)
        self._draw(ctx)
        pygame.display.flip()
        return True

    def close(self):
        pygame.quit()

    # ---------- input ----------

    def _pump_events(self, ctx):
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                self.closed = True
            elif e.type == pygame.KEYDOWN:
                if e.key in (pygame.K_q, pygame.K_ESCAPE):
                    self.closed = True
                elif e.key == pygame.K_SPACE:
                    ctx.paused = not ctx.paused
                    self.events.append(LogEvent(
                        ctx.env.now,
                        "[PAUSED]" if ctx.paused else "[RESUMED]",
                        WARN if ctx.paused else OK,
                    ))
                elif e.key in (pygame.K_EQUALS, pygame.K_PLUS, pygame.K_KP_PLUS):
                    self._speed_idx = min(self._speed_idx + 1, len(self._speed_levels) - 1)
                elif e.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    self._speed_idx = max(self._speed_idx - 1, 0)
                elif e.key == pygame.K_t:
                    item_id = ctx.inject_tricky_item()
                    if item_id is not None:
                        self._inject_flash_item = item_id
                        self._inject_flash_until_ms = pygame.time.get_ticks() + 4000
                        self.events.append(LogEvent(
                            ctx.env.now,
                            f"[USER] injected tricky item {item_id}",
                            DANGER,
                        ))
                elif e.key in (pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4, pygame.K_5):
                    idx = {pygame.K_1: 1, pygame.K_2: 2, pygame.K_3: 3,
                           pygame.K_4: 4, pygame.K_5: 5}[e.key]
                    killed = ctx.kill_drone(idx)
                    if killed:
                        self.events.append(LogEvent(
                            ctx.env.now,
                            f"[USER] killed {killed} — hot-spare deploying (zero downtime)",
                            DANGER,
                        ))
                elif e.key == pygame.K_r:
                    ok = ctx.force_retrain()
                    if ok:
                        self.events.append(LogEvent(
                            ctx.env.now,
                            f"[USER] forced retrain -> v{ctx.coord.model_version}",
                            ACCENT,
                        ))
                        self._retrain_flash_until_ms = pygame.time.get_ticks() + 600
                elif e.key == pygame.K_h:
                    # Toggle help overlay
                    self.help_open = not self.help_open
                elif e.key == pygame.K_x:
                    # Full restart of the current mode from t=0
                    self.restart_requested = True
                    self.closed = True
                    self.events.append(LogEvent(
                        ctx.env.now, "[USER] restarting…", ACCENT,
                    ))
                elif e.key == pygame.K_f:
                    # Cycle through preset fleet compositions
                    self._cycle_fleet_preset()
                    self.events.append(LogEvent(
                        ctx.env.now,
                        f"[USER] fleet preset → "
                        f"{self.live_fleet['scanner']}sc / "
                        f"{self.live_fleet['handler']}hd / "
                        f"{self.live_fleet['hybrid']}hy "
                        "(press X to apply)",
                        ACCENT,
                    ))
                elif e.key in (pygame.K_RIGHTBRACKET, pygame.K_LEFTBRACKET):
                    # ] grows the cart fleet by 1; [ shrinks it by 1.
                    # Redistributes: 70% handlers + 30% hybrids (the
                    # mix that gives reasonable throughput with the
                    # centralized-classifier model). Clamped 1..12.
                    delta = 1 if e.key == pygame.K_RIGHTBRACKET else -1
                    self._adjust_fleet_size(delta)
                    self.events.append(LogEvent(
                        ctx.env.now,
                        f"[USER] fleet size → "
                        f"{self.live_fleet['handler']}hd / "
                        f"{self.live_fleet['hybrid']}hy "
                        f"(total {self.live_fleet['handler'] + self.live_fleet['hybrid']}, "
                        "press X to apply)",
                        ACCENT,
                    ))
                elif e.key == pygame.K_k:
                    # K = Kill the coordinator → hot-standby promotes
                    ok = ctx.failover_coordinator()
                    if ok:
                        self._coord_failover_until_ms = pygame.time.get_ticks() + 2400
                        self.events.append(LogEvent(
                            ctx.env.now,
                            f"[USER] coordinator failover #{ctx.coord.failover_count} "
                            "— standby promoted, 0 data loss",
                            ACCENT,
                        ))
                elif e.key == pygame.K_m:
                    # M = swap mode (hivemind <-> isolated) and restart
                    self.mode_switch_requested = True
                    self.restart_requested = True
                    self.closed = True
                    self.events.append(LogEvent(
                        ctx.env.now,
                        f"[USER] switching mode → "
                        f"{'isolated' if self.mode_label == 'hivemind' else 'hivemind'}",
                        ACCENT,
                    ))

    # ---------- event collection ----------

    def _collect_comm_messages(self, ctx):
        """Promote new coordinator messages into active animations and
        prune expired ones."""
        msgs = ctx.coord.recent_messages
        # recent_messages is a deque — index using its current contents
        msg_list = list(msgs)
        if self._last_seen_msg_idx > len(msg_list):
            self._last_seen_msg_idx = 0
        new_msgs = msg_list[self._last_seen_msg_idx:]
        self._last_seen_msg_idx = len(msg_list)
        now_ms = pygame.time.get_ticks()
        agents_by_id = {a.agent_id: a for a in ctx.agents}
        for m in new_msgs:
            if m["kind"] == "report":
                agent = agents_by_id.get(m["agent_id"])
                if agent is None:
                    continue
                self._active_comms.append({
                    "kind": "report",
                    "start_pos": agent.pos,
                    "end_pos": (ZONES_BY_NAME["Coordinator"].x,
                                ZONES_BY_NAME["Coordinator"].y),
                    "expires_ms": now_ms + COMM_ANIM_MS,
                    "color": (255, 200, 110) if m.get("is_rescrutiny") else (160, 200, 230),
                })
            elif m["kind"] == "snapshot":
                # Broadcast: one line per mobile agent + arm the hive-sync
                # glow so every drone visibly receives the same update.
                src = (ZONES_BY_NAME["Coordinator"].x, ZONES_BY_NAME["Coordinator"].y)
                for a in ctx.agents:
                    if a.is_rescrutiny_station:
                        continue
                    self._active_comms.append({
                        "kind": "snapshot",
                        "start_pos": src,
                        "end_pos": a.pos,
                        "expires_ms": now_ms + COMM_ANIM_MS,
                        "color": (95, 200, 255),
                    })
                self._hive_sync_until_ms = now_ms + 1800
            elif m["kind"] == "dropped":
                # Visualize the dropped packet — a faded line + red X at midpoint
                agent = agents_by_id.get(m["agent_id"])
                if agent is not None:
                    self._active_comms.append({
                        "kind": "dropped",
                        "start_pos": agent.pos,
                        "end_pos": (ZONES_BY_NAME["Coordinator"].x,
                                    ZONES_BY_NAME["Coordinator"].y),
                        "expires_ms": now_ms + COMM_ANIM_MS,
                        "color": (255, 90, 90),
                    })
                    self.events.append(LogEvent(
                        ctx.env.now,
                        f"[NET] dropped report from {m['agent_id']}",
                        DANGER,
                    ))
            elif m["kind"] == "kill":
                self.events.append(LogEvent(
                    ctx.env.now,
                    f"[FAIL] {m['agent_id']} — replacement dispatching in 30s",
                    DANGER,
                ))
            elif m["kind"] == "spawn":
                self.events.append(LogEvent(
                    ctx.env.now,
                    f"[SPAWN] {m['agent_id']} replacement online",
                    OK,
                ))
            elif m["kind"] == "dispatch":
                # Queen → drone → pickup as a 2-segment animated line.
                agent = agents_by_id.get(m["agent_id"])
                if agent is not None:
                    self._active_comms.append({
                        "kind": "dispatch",
                        "start_pos": (ZONES_BY_NAME["Coordinator"].x,
                                      ZONES_BY_NAME["Coordinator"].y),
                        "mid_pos": agent.pos,
                        "end_pos": m["pickup_pos"],
                        "expires_ms": now_ms + COMM_ANIM_MS + 400,
                        "color": (180, 220, 130),
                    })
                self.events.append(LogEvent(
                    ctx.env.now,
                    f"[COORD] → {m['agent_id']}: pickup {m['item_id']} ({m['justification']})",
                    (180, 220, 130),
                ))
        # Prune expired
        self._active_comms = [c for c in self._active_comms if c["expires_ms"] > now_ms]

    def _draw_comm_lines(self, ctx):
        """Translucent fading line from start_pos → end_pos for each active
        comm. Reports use a warm color, snapshot broadcasts use the coord
        accent color."""
        if not self._active_comms:
            return
        layer = pygame.Surface((FACILITY_W, FACILITY_H), pygame.SRCALPHA)
        now_ms = pygame.time.get_ticks()
        for c in self._active_comms:
            duration_ms = COMM_ANIM_MS + (400 if c["kind"] == "dispatch" else 0)
            remaining_ms = max(c["expires_ms"] - now_ms, 0)
            alpha = int(255 * remaining_ms / duration_ms)
            sx, sy = self.world_to_screen(c["start_pos"])
            ex, ey = self.world_to_screen(c["end_pos"])
            sxl, syl = sx, sy - HEADER_H
            exl, eyl = ex, ey - HEADER_H
            color = (*c["color"], alpha)
            if c["kind"] == "dispatch":
                # 2-segment line: queen → drone → pickup, with a travelling
                # dot that walks segment 1 then segment 2.
                mx, my = self.world_to_screen(c["mid_pos"])
                mxl, myl = mx, my - HEADER_H
                pygame.draw.line(layer, color, (sxl, syl), (mxl, myl), 2)
                pygame.draw.line(layer, color, (mxl, myl), (exl, eyl), 2)
                t = 1.0 - remaining_ms / duration_ms
                if t < 0.5:
                    tt = t * 2
                    dx = sxl + (mxl - sxl) * tt
                    dy = syl + (myl - syl) * tt
                else:
                    tt = (t - 0.5) * 2
                    dx = mxl + (exl - mxl) * tt
                    dy = myl + (eyl - myl) * tt
                pygame.draw.circle(layer, (*c["color"], min(255, alpha + 60)),
                                   (int(dx), int(dy)), 5)
                continue
            if c["kind"] == "dropped":
                # Dashed faded line + red X at midpoint to signal a drop
                for seg in range(8):
                    if seg % 2 == 1:
                        continue
                    fa = seg / 8
                    fb = (seg + 1) / 8
                    pa = (int(sxl + (exl - sxl) * fa), int(syl + (eyl - syl) * fa))
                    pb = (int(sxl + (exl - sxl) * fb), int(syl + (eyl - syl) * fb))
                    pygame.draw.line(layer, color, pa, pb, 2)
                mx = int(sxl + (exl - sxl) * 0.5)
                my = int(syl + (eyl - syl) * 0.5)
                pygame.draw.line(layer, (255, 90, 90, alpha), (mx - 5, my - 5), (mx + 5, my + 5), 2)
                pygame.draw.line(layer, (255, 90, 90, alpha), (mx + 5, my - 5), (mx - 5, my + 5), 2)
                continue
            pygame.draw.line(layer, color, (sxl, syl), (exl, eyl), 2)
            t = 1.0 - remaining_ms / COMM_ANIM_MS
            dx = sxl + (exl - sxl) * t
            dy = syl + (eyl - syl) * t
            pygame.draw.circle(layer, (*c["color"], min(255, alpha + 60)),
                               (int(dx), int(dy)), 4)
        self.screen.blit(layer, (0, HEADER_H))

    def _collect_events(self, ctx):
        # Promote any new coordinator messages into visual animations
        self._collect_comm_messages(ctx)
        # Per-agent state transitions worth surfacing
        if not hasattr(self, "_prev_agent_states"):
            self._prev_agent_states = {a.agent_id: a.state for a in ctx.agents}
        for a in ctx.agents:
            prev = self._prev_agent_states.get(a.agent_id)
            if prev != a.state:
                interesting = {
                    "CHARGING":          ("[BATT] {} reached charging bay".format(a.agent_id), ACCENT),
                    "DECONNING":         ("[DECON] {} undergoing decontamination ({:.0f} µSv)".format(a.agent_id, a.integrated_dose_uSv), DANGER),
                    "FAILED":            ("[FAIL] {} permanent damage ({:.0f} µSv cumulative)".format(a.agent_id, a.integrated_dose_uSv), DANGER),
                    "RETURNING_TO_CHARGE": ("[BATT] {} returning to charge ({:.0f}%)".format(a.agent_id, a.battery_pct), WARN),
                    "RETURNING_TO_DECON":  ("[DECON] {} needs decon ({:.0f} µSv)".format(a.agent_id, a.integrated_dose_uSv), WARN),
                }
                if a.state in interesting:
                    msg, col = interesting[a.state]
                    self.events.append(LogEvent(ctx.env.now, msg, col))
                self._prev_agent_states[a.agent_id] = a.state

        # Detect new retrain events
        n_retrains = len(ctx.coord.retrain_events)
        if n_retrains > self._last_n_retrains:
            for ev in ctx.coord.retrain_events[self._last_n_retrains:]:
                thr = ev.get("actinide_threshold")
                thr_s = f"{thr:.4f}" if thr is not None else "None"
                self.events.append(LogEvent(
                    ctx.env.now,
                    f"[COORD] retrain v{ev['version']}  n={ev['n_samples']}  thr={thr_s}",
                    ACCENT,
                ))
            self._last_n_retrains = n_retrains
            self._retrain_flash_until_ms = pygame.time.get_ticks() + 500

        # Detect new classifications (only log notable ones to avoid spam)
        cls = ctx.metrics.classifications
        if len(cls) > self._last_n_classifications:
            for c in cls[self._last_n_classifications:]:
                if c["true_class"] != c["predicted_class"]:
                    self.events.append(LogEvent(
                        ctx.env.now,
                        f"[MISS] {c['item_id']} {c['agent_id']}: "
                        f"true {c['true_class']} -> pred {c['predicted_class']}",
                        DANGER,
                    ))
                elif c["scrutiny"]:
                    self.events.append(LogEvent(
                        ctx.env.now,
                        f"[FLAG] {c['item_id']} {c['agent_id']}: "
                        f"low-conf {c['confidence']:.2f}, sent to QA lab",
                        WARN,
                    ))
            self._last_n_classifications = len(cls)

        # End-of-shift bookkeeping events — surfaces ledger GC + worker
        # rotation. The actual shipping log lines come from existing
        # metrics.shipments via the dispatcher; this just flags the
        # shift boundary so long-run viewers see the rhythm.
        if not hasattr(self, "_last_shift_event_idx"):
            self._last_shift_event_idx = 0
        shift_events = getattr(ctx, "shift_events", None) or []
        if len(shift_events) > self._last_shift_event_idx:
            for ev in shift_events[self._last_shift_event_idx:]:
                if ev["kind"] == "shift_end":
                    gc = ev["gc_count"]
                    out_dose = ev["outgoing_worker_dose"]
                    self.events.append(LogEvent(
                        ctx.env.now,
                        f"[SHIFT {ev['shift_idx']}] end-of-shift — "
                        f"workers rotated ({out_dose:.0f} µSv carried out), "
                        f"GC {gc} entries",
                        ACCENT,
                    ))
            self._last_shift_event_idx = len(shift_events)

        # Arm failure / repair events. Surface each one as a log line
        # so judges can see "primary failed → standby took over,
        # zero downtime" in real time. We track our own index into
        # ctx.arm_events so we don't re-log on every tick.
        if not hasattr(self, "_last_arm_event_idx"):
            self._last_arm_event_idx = 0
        arm_events = getattr(ctx, "arm_events", None) or []
        if len(arm_events) > self._last_arm_event_idx:
            for ev in arm_events[self._last_arm_event_idx:]:
                if ev["kind"] == "arm_fail":
                    took = ev.get("took_over")
                    if took is None:
                        self.events.append(LogEvent(
                            ctx.env.now,
                            f"[ARM] {ev['unit']} failed — BOTH UNITS DOWN",
                            DANGER,
                        ))
                    else:
                        self.events.append(LogEvent(
                            ctx.env.now,
                            f"[ARM] {ev['unit']} failed → "
                            f"{took} took over (0 downtime)",
                            ACCENT,
                        ))
                elif ev["kind"] == "arm_repair":
                    self.events.append(LogEvent(
                        ctx.env.now,
                        f"[ARM] {ev['unit']} back online (standby)",
                        OK,
                    ))
            self._last_arm_event_idx = len(arm_events)

        # Newly generated items (just to make the activity feel live)
        if len(ctx.coord.ledger) > self._last_ledger_size:
            n_new = len(ctx.coord.ledger) - self._last_ledger_size
            if n_new > 0 and len(self.events) < 5:
                # Only log on slow days when log is sparse
                self.events.append(LogEvent(
                    ctx.env.now,
                    f"[GEN] {n_new} new item(s) registered",
                    DIM_TEXT,
                ))
            self._last_ledger_size = len(ctx.coord.ledger)

    # ---------- drawing ----------

    @staticmethod
    def world_to_screen(pos):
        return int(pos[0] * PX_PER_M), int(pos[1] * PX_PER_M) + HEADER_H

    def _draw(self, ctx):
        self.screen.fill(BG)
        self._draw_header(ctx)
        self._draw_facility(ctx)
        self._draw_panel(ctx)
        self._draw_footer(ctx)

    def _draw_header(self, ctx):
        # Mode chip
        mode_col = (110, 180, 230) if self.mode_label == "hivemind" else (210, 140, 80)
        title = self.font_xl.render("Radwaste hivemind sim", True, HI_TEXT)
        self.screen.blit(title, (16, 14))
        mode_w = self.font_l.size(f"  {self.mode_label.upper()}  ")[0]
        pygame.draw.rect(self.screen, mode_col, (title.get_width() + 30, 18, mode_w + 8, 26), border_radius=5)
        mode_txt = self.font_l.render(f"  {self.mode_label.upper()}  ", True, (20, 22, 28))
        self.screen.blit(mode_txt, (title.get_width() + 34, 22))

        # Multi-scale time display on the right of the header. A nuclear
        # facility tracks dose / inspection schedules on multiple cadences
        # — shift, daily, weekly, monthly. We surface them all so judges
        # can see where in the operating cycle we are.
        sim_s = ctx.env.now
        shift_h = sim_s / 3600.0
        # We treat one operating "day" as a full 8-hour shift completing,
        # so an 8h sim run = one nuclear shift. Day / week / month tick
        # over from that base.
        day = int(shift_h // 8)
        week = day // 5
        month = day // 22
        # Speed + run/pause state
        speed_txt = self.font_s.render(
            f"speed {self.sim_speedup}x  "
            f"|  {'PAUSED' if ctx.paused else 'RUNNING'}",
            True, TEXT if not ctx.paused else WARN,
        )
        # Two-line clock so we can fit everything without overflowing
        clock_line1 = self.font_m.render(
            f"sim t = {sim_s/60:5.1f} min  ({shift_h:.2f} h)",
            True, TEXT,
        )
        clock_line2 = self.font_xs.render(
            f"shift day {day+1}  |  week {week+1}  |  month {month+1}",
            True, DIM_TEXT,
        )
        x_right = WINDOW_W - 16
        self.screen.blit(clock_line1, (x_right - clock_line1.get_width(), 14))
        self.screen.blit(clock_line2, (x_right - clock_line2.get_width(), 32))
        self.screen.blit(speed_txt, (x_right - speed_txt.get_width(), 46))

        # Pending fleet-change banner. When the user has cycled F or
        # used [/] but not yet restarted with X, surface the staged
        # config so they can see their input registered.
        if self.live_fleet != self.applied_fleet:
            pending_msg = (
                f"FLEET PENDING — staged: "
                f"{self.live_fleet['scanner']}sc / "
                f"{self.live_fleet['handler']}hd / "
                f"{self.live_fleet['hybrid']}hy   "
                f"(press X to apply)"
            )
            pend = self.font_s.render(pending_msg, True, (20, 22, 28))
            pad_x, pad_y = 10, 4
            bx = WINDOW_W // 2 - (pend.get_width() // 2) - pad_x
            by = 16
            pygame.draw.rect(
                self.screen, (255, 200, 80),
                (bx, by, pend.get_width() + 2 * pad_x,
                 pend.get_height() + 2 * pad_y),
                border_radius=4,
            )
            self.screen.blit(pend, (bx + pad_x, by + pad_y))

    def _draw_facility(self, ctx):
        # Facility frame
        rect = pygame.Rect(0, HEADER_H, FACILITY_W, FACILITY_H)
        pygame.draw.rect(self.screen, (16, 18, 22), rect)
        # Subtle grid
        for gx in range(0, FACILITY_W, PX_PER_M * 2):
            pygame.draw.line(self.screen, GRID, (gx, HEADER_H), (gx, HEADER_H + FACILITY_H), 1)
        for gy in range(0, FACILITY_H, PX_PER_M * 2):
            pygame.draw.line(self.screen, GRID, (0, HEADER_H + gy), (FACILITY_W, HEADER_H + gy), 1)
        pygame.draw.rect(self.screen, (60, 65, 75), rect, 2)

        self._draw_building_envelope(ctx)
        self._draw_wireless_range(ctx)
        self._draw_floor_markers(ctx)
        self._draw_process_chain(ctx)
        self._draw_drum_flow_arrows(ctx)
        self._draw_hive_web(ctx)
        self._draw_zones(ctx)
        self._draw_waste_items(ctx)
        self._draw_workers(ctx)
        self._draw_agents(ctx)
        self._draw_decon_arm(ctx)
        self._draw_qa_lab_arm(ctx)
        self._draw_loader_arms(ctx)
        # Zone callouts removed — they overlapped with zone names underneath
        # each room and with each other. The compartment tabs inside each
        # room + the zone-name labels are enough.
        self._draw_learning_loop_strip(ctx)

    def _draw_hive_web(self, ctx):
        """In HIVEMIND mode, draw a persistent (but very subtle) link line
        from each connected mobile agent to the QUEEN. Failed/offline
        drones are not connected. This is the visual representation of
        the hive — even when no message is being actively sent, every
        living drone is part of the network."""
        if self.mode_label != "hivemind":
            return
        queen = ZONES_BY_NAME.get("Coordinator")
        if queen is None:
            return
        qx, qy = self.world_to_screen((queen.x, queen.y))
        layer = pygame.Surface((FACILITY_W, FACILITY_H), pygame.SRCALPHA)
        pulse = 0.5 + 0.5 * math.sin(pygame.time.get_ticks() / 700)
        base_alpha = int(40 + 20 * pulse)
        for a in ctx.agents:
            if a.is_rescrutiny_station:
                continue
            if a.state == "FAILED":
                continue
            ax, ay = self.world_to_screen(a.pos)
            pygame.draw.line(
                layer, (95, 200, 255, base_alpha),
                (qx, qy - HEADER_H), (ax, ay - HEADER_H), 1,
            )
            # Tiny antenna pulse dot at the agent end
            pygame.draw.circle(
                layer, (95, 200, 255, base_alpha + 50),
                (ax, ay - HEADER_H), 2, 0,
            )
        self.screen.blit(layer, (0, HEADER_H))

    def _draw_wireless_range(self, ctx):
        """No-op. The wireless coverage ring was visually dominant on the
        new wider facility layout and conveyed little information beyond
        what the persistent hive-web lines already show. Left as a stub
        in case we want to bring it back as a smaller per-zone indicator."""
        return

    # Building layout shared with sim/facility.py — same WINGS / CORRIDORS
    # data feeds the rendered walls AND the AGV waypoint graph, so the
    # cart never goes through a wall the user can see. The "label" string
    # for each wing is added here as display-only metadata.
    _WING_LABELS = {
        "REPROCESSING":   "REPROCESSING WING",
        "AI BRAIN":       "AI BRAIN ROOM",
        "SORT/CLASSIFY":  "CLASSIFICATION HALL",
        "QA":             "QA LAB",
        "DECON":          "DECON BAY",
        "STORAGE":        "STORAGE WING",
        "CONTROL":        "CONTROL WING",
        "LEGACY":         "LEGACY WASTE",
    }

    @property
    def _BUILDING_WINGS(self):
        from sim.facility import WINGS as _W
        return [(n, x0, y0, x1, y1, self._WING_LABELS.get(n, n))
                for (n, x0, y0, x1, y1) in _W]

    @property
    def _CORRIDORS(self):
        from sim.facility import CORRIDORS as _C
        return list(_C)

    def _draw_building_envelope(self, ctx):
        """Plant building structure — outer perimeter wall + internal
        wing partition walls. Each wing is a labeled room with concrete
        walls; corridors between wings are open so carts can pass.

        Walls render in two passes: dark concrete fill on top of the
        existing dark floor, then a brighter top-edge line that gives
        them a 3D feel. Wing tags float at the wing centroid as faint
        room labels so judges can read 'this is the storage wing' /
        'this is the classification hall' / etc."""
        wall_layer = pygame.Surface(
            (FACILITY_W, FACILITY_H), pygame.SRCALPHA,
        )
        floor_layer = pygame.Surface(
            (FACILITY_W, FACILITY_H), pygame.SRCALPHA,
        )

        # 1) Wing-floor tint (slightly lighter than the void around)
        wing_floor_col = (32, 38, 50, 90)
        for _name, x0, y0, x1, y1, _label in self._BUILDING_WINGS:
            sx0, sy0 = self.world_to_screen((x0, y0))
            sx1, sy1 = self.world_to_screen((x1, y1))
            pygame.draw.rect(
                floor_layer, wing_floor_col,
                (sx0, sy0 - HEADER_H, sx1 - sx0, sy1 - sy0),
            )

        # 2) Corridor lane tint (even lighter — the marked travel route)
        corridor_col = (60, 80, 110, 110)
        for x0, y0, x1, y1 in self._CORRIDORS:
            sx0, sy0 = self.world_to_screen((x0, y0))
            sx1, sy1 = self.world_to_screen((x1, y1))
            pygame.draw.rect(
                floor_layer, corridor_col,
                (sx0, sy0 - HEADER_H, sx1 - sx0, sy1 - sy0),
            )

        self.screen.blit(floor_layer, (0, HEADER_H))

        # 3) Walls — drawn as four sides per wing, BUT skip wall segments
        # that would block a corridor. We do this by checking each wall
        # segment against each corridor rect — if they overlap, render
        # the wall only on the non-overlapping portion.
        wall_col = (88, 95, 110)
        wall_hi  = (140, 150, 170)
        wall_thickness = 3

        def draw_horizontal_wall(x0_m, x1_m, y_m):
            # Returns segments of [x0, x1] minus all corridor overlaps
            segments = [(x0_m, x1_m)]
            for cx0, cy0, cx1, cy1 in self._CORRIDORS:
                # Only relevant if the corridor crosses this y
                if not (cy0 <= y_m <= cy1):
                    continue
                new_segs = []
                for s_x0, s_x1 in segments:
                    if cx1 <= s_x0 or cx0 >= s_x1:
                        new_segs.append((s_x0, s_x1))
                    else:
                        if cx0 > s_x0:
                            new_segs.append((s_x0, cx0))
                        if cx1 < s_x1:
                            new_segs.append((cx1, s_x1))
                segments = new_segs
            for s_x0, s_x1 in segments:
                if s_x1 - s_x0 < 0.15:
                    continue
                a = self.world_to_screen((s_x0, y_m))
                b = self.world_to_screen((s_x1, y_m))
                pygame.draw.line(self.screen, wall_col,
                                 a, b, wall_thickness)
                pygame.draw.line(self.screen, wall_hi,
                                 a, (b[0], a[1] - 0), 1)

        def draw_vertical_wall(y0_m, y1_m, x_m):
            segments = [(y0_m, y1_m)]
            for cx0, cy0, cx1, cy1 in self._CORRIDORS:
                if not (cx0 <= x_m <= cx1):
                    continue
                new_segs = []
                for s_y0, s_y1 in segments:
                    if cy1 <= s_y0 or cy0 >= s_y1:
                        new_segs.append((s_y0, s_y1))
                    else:
                        if cy0 > s_y0:
                            new_segs.append((s_y0, cy0))
                        if cy1 < s_y1:
                            new_segs.append((cy1, s_y1))
                segments = new_segs
            for s_y0, s_y1 in segments:
                if s_y1 - s_y0 < 0.15:
                    continue
                a = self.world_to_screen((x_m, s_y0))
                b = self.world_to_screen((x_m, s_y1))
                pygame.draw.line(self.screen, wall_col,
                                 a, b, wall_thickness)

        for _name, x0, y0, x1, y1, _label in self._BUILDING_WINGS:
            draw_horizontal_wall(x0, x1, y0)  # top
            draw_horizontal_wall(x0, x1, y1)  # bottom
            draw_vertical_wall(y0, y1, x0)    # left
            draw_vertical_wall(y0, y1, x1)    # right

        # 4) Wing tags — faint room labels at the top of each wing
        for _name, x0, y0, x1, y1, label in self._BUILDING_WINGS:
            cx = (x0 + x1) / 2.0
            sx, sy = self.world_to_screen((cx, y0))
            tag = self.font_xs.render(label, True, (140, 160, 200))
            tag_bg = pygame.Surface(
                (tag.get_width() + 6, tag.get_height() + 2),
                pygame.SRCALPHA,
            )
            tag_bg.fill((20, 24, 34, 200))
            self.screen.blit(tag_bg,
                             (sx - tag.get_width() // 2 - 3,
                              sy - tag.get_height() - 4))
            self.screen.blit(tag,
                             (sx - tag.get_width() // 2,
                              sy - tag.get_height() - 3))

    def _draw_drum_flow_arrows(self, ctx):
        """Faint animated arrows from every waste-emitting cell into the
        Char Station. Tells the 'every drum goes through one classifier'
        story without needing words. Drawn very subtle so they don't fight
        with the upstream PUREX chain or the hive-web."""
        char_z = ZONES_BY_NAME.get("Classifier")
        if char_z is None:
            return
        char_pos = (char_z.x, char_z.y - char_z.radius_m * 0.6)
        now_ms = pygame.time.get_ticks()
        layer = pygame.Surface((FACILITY_W, FACILITY_H), pygame.SRCALPHA)
        # Find waste-emitting cells: generation role only (process /
        # solidification / pool are not direct drum sources for our pipeline)
        for z in ZONES:
            if z.role != "generation":
                continue
            src_pos = (z.x, z.y + z.radius_m * 0.6)
            sx, sy = self.world_to_screen(src_pos)
            ex, ey = self.world_to_screen(char_pos)
            sxl, syl = sx, sy - HEADER_H
            exl, eyl = ex, ey - HEADER_H
            # Dashed faint line so it doesn't dominate
            dx, dy = exl - sxl, eyl - syl
            d = math.hypot(dx, dy)
            if d < 8:
                continue
            ux, uy = dx / d, dy / d
            segs = 16
            for i in range(segs):
                if i % 3 == 0:
                    continue   # gap every 3 segs for a dashed look
                fa = i / segs
                fb = (i + 1) / segs
                pa = (sxl + dx * fa, syl + dy * fa)
                pb = (sxl + dx * fb, syl + dy * fb)
                pygame.draw.line(layer, (140, 200, 180, 70), pa, pb, 1)
            # Travelling dot along the line — communicates "flow"
            t = ((now_ms + hash(z.name) * 37) % 3500) / 3500
            dxp = sxl + dx * t
            dyp = syl + dy * t
            pygame.draw.circle(layer, (180, 230, 210, 200),
                               (int(dxp), int(dyp)), 2, 0)
        self.screen.blit(layer, (0, HEADER_H))

    def _draw_floor_markers(self, ctx):
        """QR-code floor markers, dual purpose:

          1) **Navigation waypoints** — carts A* through this exact set of
             markers (defined in `facility.waypoints()`); the rendered
             glyphs are literally the nodes of the routing graph.
          2) **Telemetry beacons** — when a cart rolls over a marker, it
             scans the QR, decodes (marker_id, x_world, y_world), and
             pings the coordinator with its current position. The brain
             builds its live cart-position map from these pings rather
             than from magic shared memory.

        Idle markers are dim; markers currently being scanned by a cart
        pulse cyan with a faint trace toward the AI BRAIN to suggest the
        telemetry signal. The faint thin lines linking adjacent markers
        are the actual graph edges — judges can see the routing network
        the carts move on."""
        from sim.facility import waypoints, waypoint_edges
        markers = waypoints()

        cart_positions = [
            a.pos for a in ctx.agents
            if not a.is_rescrutiny_station and a.state != "FAILED"
        ]

        coord_zone = ZONES_BY_NAME.get("Coordinator")
        coord_screen = self.world_to_screen(
            (coord_zone.x, coord_zone.y)
        ) if coord_zone is not None else None

        # 1) Faint network lines on a translucent layer — these are the
        # waypoint graph edges. Drawn first so QR glyphs sit on top.
        net_layer = pygame.Surface((FACILITY_W, FACILITY_H), pygame.SRCALPHA)
        edges = waypoint_edges()
        seen: set[tuple[int, int]] = set()
        for i, nbrs in edges.items():
            for j, _cost in nbrs:
                key = (i, j) if i < j else (j, i)
                if key in seen:
                    continue
                seen.add(key)
                ax, ay = self.world_to_screen(markers[i])
                bx, by = self.world_to_screen(markers[j])
                pygame.draw.line(
                    net_layer, (60, 110, 160, 60),
                    (ax, ay - HEADER_H), (bx, by - HEADER_H), 1,
                )
        self.screen.blit(net_layer, (0, HEADER_H))

        # 2) Markers themselves.
        active_markers: list[tuple[int, int]] = []
        dim_layer = pygame.Surface((FACILITY_W, FACILITY_H), pygame.SRCALPHA)
        for (x, y) in markers:
            scanned = False
            for cx, cy in cart_positions:
                if (cx - x) ** 2 + (cy - y) ** 2 < 0.55 ** 2:
                    scanned = True
                    break
            sx, sy = self.world_to_screen((x, y))
            if scanned:
                active_markers.append((sx, sy))
                self._draw_qr_glyph(self.screen, sx, sy,
                                    (160, 240, 255), bright=True)
            else:
                self._draw_qr_glyph(dim_layer, sx, sy - HEADER_H,
                                    (130, 180, 220, 180), bright=False)
        self.screen.blit(dim_layer, (0, HEADER_H))

        # Telemetry trace: thin faint line from each active marker to the
        # AI BRAIN, with a small travelling dot. Communicates "ping sent".
        if coord_screen is not None and active_markers:
            trace_layer = pygame.Surface(
                (FACILITY_W, FACILITY_H), pygame.SRCALPHA,
            )
            cxs, cys = coord_screen
            now_ms = pygame.time.get_ticks()
            for (msx, msy) in active_markers:
                ax, ay = msx, msy - HEADER_H
                bx, by = cxs, cys - HEADER_H
                pygame.draw.line(trace_layer, (110, 200, 255, 90),
                                 (ax, ay), (bx, by), 1)
                # Travelling dot
                t = (now_ms % 600) / 600
                dx = ax + (bx - ax) * t
                dy = ay + (by - ay) * t
                pygame.draw.circle(trace_layer, (140, 220, 255, 200),
                                   (int(dx), int(dy)), 2, 0)
            self.screen.blit(trace_layer, (0, HEADER_H))

    def _draw_qr_glyph(self, surf, cx: int, cy: int, color, bright: bool):
        """Mini QR-code-style glyph: three corner finder squares + a
        deterministic interior pattern. 10×10 footprint — chunky enough
        to read clearly as a floor marker at demo distance."""
        size = 10
        x0 = cx - size // 2
        y0 = cy - size // 2
        # Background tile so the QR pops off the floor
        bg_alpha = 220 if bright else 130
        if len(color) == 4:
            bg_col = (color[0] // 3, color[1] // 3, color[2] // 3,
                       min(color[3] + 30, 255))
        else:
            bg_col = (color[0] // 3, color[1] // 3, color[2] // 3)
        pygame.draw.rect(surf, bg_col, (x0 - 1, y0 - 1, size + 2, size + 2))
        # Three finder squares (top-left, top-right, bottom-left) — 3x3
        # each, with a hollow centre, the recognisable QR-code corners.
        for ox, oy in [(0, 0), (size - 3, 0), (0, size - 3)]:
            pygame.draw.rect(surf, color, (x0 + ox, y0 + oy, 3, 3))
            # Hollow centre — drawn in the background colour so the
            # finder square reads as a square ring (true QR look).
            pygame.draw.rect(surf, bg_col, (x0 + ox + 1, y0 + oy + 1, 1, 1))
        # Deterministic interior data pixels (4x4 in the centre)
        seed = (cx * 73856093) ^ (cy * 19349663)
        for i in range(4):
            for j in range(4):
                if ((seed >> (i * 4 + j)) & 1):
                    pygame.draw.rect(surf, color,
                                     (x0 + 3 + i, y0 + 3 + j, 1, 1))
        if bright:
            # Outer pulse ring for the active (currently-scanned) state
            pygame.draw.rect(surf, color, (x0 - 2, y0 - 2, size + 4, size + 4), 1)

    def _draw_zones(self, ctx):
        """Render each zone as a rectangular industrial room with concrete
        walls, a tiled interior floor, and role-specific markings (hazard
        stripes for hot zones, equipment silhouettes inside)."""
        gen_queues = self._compute_gen_queues(ctx)
        storage_counts = self._compute_storage_counts(ctx)

        for z in ZONES:
            cx, cy = self.world_to_screen((z.x, z.y))
            r = int(z.radius_m * PX_PER_M)
            # Room dimensions — wider than tall for most rooms, square for
            # storage / process cells.
            if z.role in ("storage", "clearance"):
                rw, rh = int(r * 2.2), int(r * 2.4)
            elif z.role == "charging":
                rw, rh = int(r * 1.8), int(r * 1.8)
            elif z.role in ("generation", "process", "solidification"):
                rw, rh = int(r * 2.0), int(r * 2.0)
            elif z.role == "drum_scanner":
                rw, rh = int(r * 2.4), int(r * 2.2)
            elif z.role == "char_station":
                # Wide room — the gantry runs along its full width so the
                # back-and-forth scan motion is obvious. Slightly shorter
                # vertically so the room reads as a "characterisation cell".
                rw, rh = int(r * 2.6), int(r * 2.0)
            elif z.role == "fuel_pool":
                # Pool is a tall water-filled basin. Render slightly taller
                # than wide so the depth reads as 'underwater storage'.
                rw, rh = int(r * 2.0), int(r * 2.4)
            elif z.role == "decon":
                rw, rh = int(r * 2.4), int(r * 2.0)
            elif z.role == "worker":
                rw, rh = int(r * 2.0), int(r * 2.0)
            elif z.role == "coordinator":
                # Wider than tall — the AI brain holds three labeled
                # compartments side-by-side: CLASSIFIER, KNOWLEDGE BASE,
                # LEARNING LOOP. Width pulled in so it doesn't crowd the
                # storage stacks on the right.
                rw, rh = int(r * 2.3), int(r * 1.7)
            else:
                rw, rh = int(r * 2.0), int(r * 2.0)
            x0, y0 = cx - rw // 2, cy - rh // 2
            self._draw_room(x0, y0, rw, rh, z, ctx)

            # Count badge for storage zones (top-right corner)
            if z.role == "storage":
                count = storage_counts.get(z.storage_class, 0)
                self._draw_badge(x0 + rw, y0,
                                 str(count), CLASS_COLOR.get(z.storage_class, (200, 200, 200)))

            # Queue depth for generation points
            if z.role == "generation":
                q = gen_queues.get(z.name, 0)
                if q > 0:
                    self._draw_badge(x0 + rw, y0, str(q), WARN)

            # Clearance zone exit arrow
            if z.role == "clearance":
                ax_start = x0 + rw + 4
                ax_end = x0 + rw + 28
                arrow_col = (130, 220, 150)
                pygame.draw.line(self.screen, arrow_col,
                                 (ax_start, cy), (ax_end, cy), 2)
                pygame.draw.polygon(self.screen, arrow_col, [
                    (ax_end, cy),
                    (ax_end - 6, cy - 4),
                    (ax_end - 6, cy + 4),
                ])
                exit_lbl = self.font_xs.render("FREE", True, arrow_col)
                self.screen.blit(exit_lbl, (ax_end + 2, cy - 11))
                exit_lbl2 = self.font_xs.render("RELEASE", True, arrow_col)
                self.screen.blit(exit_lbl2, (ax_end + 2, cy + 1))

            # QA-lab progress arc while HPGe is integrating
            if z.role == "drum_scanner":
                drum_agents = [a for a in ctx.agents if a.is_rescrutiny_station]
                if drum_agents and drum_agents[0].state == "RESCANNING":
                    angle = (pygame.time.get_ticks() // 4) % 360
                    arc_rect = pygame.Rect(x0 - 4, y0 - 4, rw + 8, rh + 8)
                    pygame.draw.arc(self.screen, ACCENT, arc_rect,
                                    math.radians(angle), math.radians(angle + 90), 3)

    def _draw_room(self, x: int, y: int, w: int, h: int, z, ctx):
        """One rectangular industrial room: concrete walls, solid floor,
        role-specific accents and equipment silhouettes.

        Hazard stripes are reserved for the *truly* hot zones (ILW storage,
        HLW storage, Robot wash). Process cells get a thin coloured wall
        instead — the previous 'stripes on everything' look was visually
        overwhelming."""
        # Solid floor (no tile grid) — calmer, lets the equipment + drums
        # inside the room be the visual focus.
        floor_col = (
            int(z.color[0] * 0.40 + 16),
            int(z.color[1] * 0.40 + 16),
            int(z.color[2] * 0.40 + 20),
        )

        # Every room gets a consistent concrete wall around it. The
        # 'hot zone' distinction is layered INSIDE as a thin hazard
        # chevron — that way no two adjacent hot rooms can have their
        # warning stripes merge into each other.
        wall_col_mid = (105, 110, 122)
        wall_col_dark = (45, 50, 60)
        wall_col_light = (150, 155, 170)
        wall_w = 3
        pygame.draw.rect(self.screen, wall_col_dark,
                         (x - wall_w, y - wall_w, w + 2 * wall_w, h + 2 * wall_w))
        pygame.draw.rect(self.screen, wall_col_mid,
                         (x - wall_w + 1, y - wall_w + 1,
                          w + 2 * wall_w - 2, h + 2 * wall_w - 2))
        pygame.draw.rect(self.screen, floor_col, (x, y, w, h))
        pygame.draw.line(self.screen, wall_col_light, (x - 1, y - 1), (x + w + 1, y - 1), 1)

        hot = (
            (z.role == "storage" and z.storage_class in ("ILW", "HLW"))
            or z.role == "decon"
        )
        if hot:
            self._draw_hazard_stripes(x, y, w, h)

        # Role-specific accent: outer outline by role colour, drawn last
        outline_col = ROLE_OUTLINE.get(z.role, (200, 200, 200))
        pygame.draw.rect(self.screen, outline_col, (x - 1, y - 1, w + 2, h + 2), 1)

        # Role-specific equipment silhouettes inside the room
        self._draw_room_equipment(x, y, w, h, z, ctx)

        # Coordinator: pulsing glow around the room
        if z.role == "coordinator":
            pulse_t = (pygame.time.get_ticks() % 1600) / 1600
            glow_pad = 4 + int(6 * abs(pulse_t - 0.5))
            glow_alpha = int(110 - 60 * abs(pulse_t - 0.5))
            glow = pygame.Surface((w + 2 * glow_pad, h + 2 * glow_pad), pygame.SRCALPHA)
            pygame.draw.rect(glow, (95, 200, 255, glow_alpha),
                             (0, 0, w + 2 * glow_pad, h + 2 * glow_pad), 3)
            self.screen.blit(glow, (x - glow_pad, y - glow_pad))

        # Worker zone: pulsing inner highlight
        if z.role == "worker":
            pulse = 1 + int(1 * abs((pygame.time.get_ticks() % 1200) - 600) / 600)
            pygame.draw.rect(self.screen, (255, 230, 120),
                             (x + 4 + pulse, y + 4 + pulse,
                              w - 8 - 2 * pulse, h - 8 - 2 * pulse), 1)

        # Room label tag at the top-left of the room (cleaner than centred)
        short = z.short or z.name[:5]
        tag = self.font_m.render(short, True, HI_TEXT)
        tag_bg = pygame.Surface((tag.get_width() + 6, tag.get_height() + 2), pygame.SRCALPHA)
        tag_bg.fill((12, 14, 20, 200))
        self.screen.blit(tag_bg, (x + 2, y + 2))
        self.screen.blit(tag, (x + 5, y + 3))

        # Full zone name UNDER the room
        full = self.font_xs.render(z.name, True, TEXT)
        self.screen.blit(full, (x + w // 2 - full.get_width() // 2, y + h + 4))

    def _draw_fuel_pool_interior(self, x: int, y: int, w: int, h: int, ctx):
        """Spent Fuel Pool: a water-filled basin where spent fuel assemblies
        cool for ~5 years before being lifted to Shearing. We render:

          - Deep water body with horizontal surface ripples
          - Underwater fuel-rack grid storing visible assemblies
          - Bridge-crane FUEL HANDLING MACHINE: horizontal rail at the top
            of the room, a sliding trolley, and a telescoping mast +
            grapple that lowers into the pool to pick up an assembly. The
            assembly being lifted is drawn riding back up with the mast.
          - A small submerged inspection manipulator anchored at the
            right wall, slowly sweeping the bottom of the pool.

        Animation loops on an 8-second cycle so the pool always reads as
        'live operation'."""
        now_ms = pygame.time.get_ticks()
        # Water surface starts a bit below the top of the room (the bridge
        # crane lives above water)
        crane_band_h = 12
        water_y = y + crane_band_h
        water_h = h - crane_band_h
        # Solid water column — slightly different shade from the room floor
        # so judges read 'this is water, not concrete'
        for band_i in range(water_h):
            t = band_i / max(water_h - 1, 1)
            # Lighter at the top (surface lit), darker at depth
            r = int(30 + (1 - t) * 30)
            g = int(75 + (1 - t) * 40)
            b = int(120 + (1 - t) * 50)
            pygame.draw.line(self.screen, (r, g, b),
                             (x + 1, water_y + band_i),
                             (x + w - 1, water_y + band_i), 1)
        # Animated surface ripples (3 horizontal squiggly lines)
        for i in range(3):
            ripple_y = water_y + 2 + i * 2
            phase = now_ms / 250.0 + i * 1.3
            for rx in range(x + 2, x + w - 2, 4):
                offset = int(math.sin(phase + rx * 0.18) * 1.2)
                pygame.draw.line(
                    self.screen, (190, 230, 255),
                    (rx, ripple_y + offset),
                    (rx + 2, ripple_y + offset), 1,
                )

        # --- Underwater fuel rack: 4x3 grid of slots with visible assemblies
        rack_left = x + 8
        rack_top = water_y + 14
        rack_right = x + w - 10
        rack_bot = y + h - 8
        rack_w = rack_right - rack_left
        rack_h = rack_bot - rack_top
        cols, rows = 4, 3
        slot_w = rack_w // cols
        slot_h = rack_h // rows
        for r_i in range(rows):
            for c_i in range(cols):
                sx = rack_left + c_i * slot_w
                sy = rack_top + r_i * slot_h
                # Slot outline
                pygame.draw.rect(self.screen, (15, 35, 60),
                                 (sx + 1, sy + 1, slot_w - 2, slot_h - 2), 1)
                # Assembly inside most slots — long thin rectangle
                # Skip a couple to suggest "in use / being moved"
                if (r_i, c_i) in {(0, 0), (1, 2)}:
                    continue
                asm_w = max(3, (slot_w - 6))
                asm_h = max(3, (slot_h - 6))
                asm_col = (180, 165, 110)  # uranium-yellow tint
                pygame.draw.rect(self.screen, asm_col,
                                 (sx + (slot_w - asm_w) // 2,
                                  sy + (slot_h - asm_h) // 2,
                                  asm_w, asm_h))
                pygame.draw.rect(self.screen, (50, 50, 30),
                                 (sx + (slot_w - asm_w) // 2,
                                  sy + (slot_h - asm_h) // 2,
                                  asm_w, asm_h), 1)
                # Subtle Cerenkov-blue glow on the top
                pygame.draw.line(self.screen, (140, 220, 255),
                                 (sx + (slot_w - asm_w) // 2 + 1,
                                  sy + (slot_h - asm_h) // 2 + 1),
                                 (sx + (slot_w - asm_w) // 2 + asm_w - 1,
                                  sy + (slot_h - asm_h) // 2 + 1), 1)

        # --- Fuel Handling Machine: bridge + trolley + telescoping mast
        # The trolley slides horizontally along a rail at the top of the
        # room. The mast hangs from the trolley and telescopes down into
        # the water, grabs an assembly, lifts it back up, and traverses
        # toward the right edge (where the assembly continues on to
        # Shearing).
        #
        # The cycle is tied to actual Shearing-cell events: each Shearing
        # waste item corresponds to one assembly being lifted from the
        # pool. Between events the FHM parks with its mast retracted.
        coord_obj = ctx.coord
        env_now = ctx.env.now
        last_shear = getattr(coord_obj, "last_shearing_event_sim_s", None)
        cycle_duration_s = 90.0   # one FHM delivery cycle, in sim-seconds
        if last_shear is None:
            cycle_t = 1.0    # never had a Shearing event — park
        else:
            elapsed = env_now - last_shear
            if elapsed < 0 or elapsed >= cycle_duration_s:
                cycle_t = 1.0   # idle / parked between deliveries
            else:
                cycle_t = elapsed / cycle_duration_s
        is_parked = cycle_t >= 0.999

        # Bridge rail across the top of the room (above the water)
        rail_y = y + 4
        pygame.draw.rect(self.screen, (120, 130, 150),
                         (x + 2, rail_y, w - 4, 3))
        pygame.draw.rect(self.screen, (10, 12, 16),
                         (x + 2, rail_y, w - 4, 3), 1)

        # Trolley horizontal position: parks over the empty slot (col 0)
        # for the first half of the cycle, then traverses right to drop the
        # assembly off at the room's edge (toward Shearing) for the second
        # half. Simple ease-in-out via cosine.
        slot_col0_cx = rack_left + slot_w * 0 + slot_w // 2
        slot_col2_cx = rack_left + slot_w * 2 + slot_w // 2
        if cycle_t < 0.5:
            # Hover over slot then traverse to slot_col2_cx (where the
            # second empty slot is — the one we're "pulling from")
            phase = cycle_t / 0.5
            tx = slot_col0_cx + (slot_col2_cx - slot_col0_cx) * (
                0.5 - 0.5 * math.cos(phase * math.pi)
            )
        else:
            phase = (cycle_t - 0.5) / 0.5
            # Traverse back to the right edge (exit toward Shearing)
            exit_x = x + w - 12
            tx = slot_col2_cx + (exit_x - slot_col2_cx) * (
                0.5 - 0.5 * math.cos(phase * math.pi)
            )
        trolley_x = int(tx)
        # Trolley body
        pygame.draw.rect(self.screen, (140, 150, 175),
                         (trolley_x - 7, rail_y + 3, 14, 6))
        pygame.draw.rect(self.screen, (10, 12, 16),
                         (trolley_x - 7, rail_y + 3, 14, 6), 1)

        # Mast: descends into the water with a grapple at the tip. The
        # mast length varies through the cycle so we see the dip + lift.
        if cycle_t < 0.5:
            # Descend (0..0.2), grab (0.2..0.3), ascend (0.3..0.5)
            if cycle_t < 0.2:
                mast_t = cycle_t / 0.2
            elif cycle_t < 0.3:
                mast_t = 1.0
            else:
                mast_t = 1.0 - (cycle_t - 0.3) / 0.2
            # Carrying assembly only when ascending or near bottom
            carrying = cycle_t >= 0.2
        else:
            # Traversing to dropoff; mast partially extended carrying assembly
            mast_t = 0.45
            carrying = (cycle_t < 0.95)
        max_mast_len = (y + h - 6) - (rail_y + 9)
        mast_len = int(max_mast_len * mast_t)
        # Mast (thin grey column)
        pygame.draw.rect(self.screen, (180, 190, 210),
                         (trolley_x - 2, rail_y + 9, 4, mast_len))
        pygame.draw.rect(self.screen, (10, 12, 16),
                         (trolley_x - 2, rail_y + 9, 4, mast_len), 1)
        # Grapple at the tip
        grapple_y = rail_y + 9 + mast_len
        pygame.draw.rect(self.screen, (200, 210, 230),
                         (trolley_x - 5, grapple_y - 1, 10, 4))
        pygame.draw.rect(self.screen, (10, 12, 16),
                         (trolley_x - 5, grapple_y - 1, 10, 4), 1)
        # If carrying, draw a fuel-assembly icon hanging from the grapple
        if carrying:
            asm_w_px = max(4, slot_w - 8)
            asm_h_px = max(6, slot_h - 6)
            ay = grapple_y + 3
            pygame.draw.rect(self.screen, (200, 180, 120),
                             (trolley_x - asm_w_px // 2, ay,
                              asm_w_px, asm_h_px))
            pygame.draw.rect(self.screen, (50, 50, 30),
                             (trolley_x - asm_w_px // 2, ay,
                              asm_w_px, asm_h_px), 1)
            # Cerenkov glow on the top edge
            pygame.draw.line(self.screen, (140, 220, 255),
                             (trolley_x - asm_w_px // 2 + 1, ay + 1),
                             (trolley_x + asm_w_px // 2 - 1, ay + 1), 1)

        # "FHM" tag above the rail
        fhm = self.font_xs.render("FHM crane", True, (180, 220, 255))
        self.screen.blit(fhm, (x + 4, rail_y - fhm.get_height() - 1))

        # --- Submerged inspection manipulator: anchored on the right wall,
        # arm reaches into the lower-right of the pool and slowly sweeps.
        sub_anchor_x = x + w - 4
        sub_anchor_y = water_y + 4
        sweep = math.sin(now_ms / 1100.0)
        sub_L1 = int(slot_w * 1.4)
        sub_L2 = int(slot_h * 1.4)
        # Target sweeps along the bottom of the pool
        sub_target_x = sub_anchor_x - 14 - int(sweep * 10)
        sub_target_y = y + h - 8
        # 2-link IK
        sdx_arm = sub_target_x - sub_anchor_x
        sdy_arm = sub_target_y - sub_anchor_y
        sd_arm = math.hypot(sdx_arm, sdy_arm)
        sd_arm = max(min(sd_arm, sub_L1 + sub_L2 - 1),
                     abs(sub_L1 - sub_L2) + 1)
        cos_a = (sub_L1 ** 2 + sd_arm ** 2 - sub_L2 ** 2) / (2 * sub_L1 * sd_arm)
        cos_a = max(-1.0, min(1.0, cos_a))
        theta = math.atan2(sdy_arm, sdx_arm) - math.acos(cos_a)
        elbow_x = sub_anchor_x + sub_L1 * math.cos(theta)
        elbow_y = sub_anchor_y + sub_L1 * math.sin(theta)
        sub_arm_col = (220, 230, 240)
        pygame.draw.line(self.screen, sub_arm_col,
                         (sub_anchor_x, sub_anchor_y),
                         (int(elbow_x), int(elbow_y)), 2)
        pygame.draw.line(self.screen, sub_arm_col,
                         (int(elbow_x), int(elbow_y)),
                         (sub_target_x, sub_target_y), 2)
        # Joints
        pygame.draw.circle(self.screen, (60, 100, 140),
                           (sub_anchor_x, sub_anchor_y), 3, 0)
        pygame.draw.circle(self.screen, (60, 100, 140),
                           (int(elbow_x), int(elbow_y)), 2, 0)
        # Sensor head (small light)
        pygame.draw.circle(self.screen, (255, 220, 130),
                           (sub_target_x, sub_target_y), 2, 0)

    def _draw_char_station_interior(self, x: int, y: int, w: int, h: int, ctx):
        """Drum Characterisation Station interior.

        Layout reflects a real characterisation cell:
          - TWO overhead scanner gantries running parallel rails across
            the top of the room. Each carries an NaI + camera head and
            paces back and forth independently. Two gantries means the
            cell can scan multiple drums in parallel and also have
            redundancy if one gantry's drive belt jams.
          - THREE characterisation turntables across the floor:
            INTAKE → MEASURE → DISPATCH. A drum lands on INTAKE, gets
            rotated and read by the gantry at MEASURE, then exits via
            DISPATCH. Three stations means three drums can be in
            flight at once.

        Drums currently in the room are drawn by the regular
        waste-items layer at the zone centroid — the turntable visuals
        just make it read as 'on a station being characterised' rather
        than 'sitting loose on the floor'."""
        now_ms = pygame.time.get_ticks()

        # Two parallel gantry rails across the top of the room.
        rail_col = (110, 130, 160)
        rail_ys = [y + 6, y + 14]
        for r_idx, rail_y in enumerate(rail_ys):
            pygame.draw.rect(self.screen, rail_col, (x + 6, rail_y, w - 12, 2))
            pygame.draw.rect(self.screen, (10, 12, 16),
                             (x + 6, rail_y, w - 12, 2), 1)
            # Sliding scanner head — paces back and forth on the rail.
            # The two heads are offset out of phase so they don't move
            # in lockstep.
            phase = (math.sin(now_ms / 700.0 + r_idx * 1.4) + 1.0) / 2.0
            head_x = x + 12 + int((w - 24) * phase)
            head_w = 14
            head_h = 8
            pygame.draw.rect(self.screen, (60, 90, 130),
                             (head_x - head_w // 2, rail_y + 2, head_w, head_h))
            pygame.draw.rect(self.screen, (10, 12, 16),
                             (head_x - head_w // 2, rail_y + 2, head_w, head_h), 1)
            pygame.draw.circle(self.screen, (140, 220, 255),
                               (head_x - 3, rail_y + 2 + head_h - 1), 2, 0)
            pygame.draw.circle(self.screen, (255, 200, 110),
                               (head_x + 3, rail_y + 2 + head_h - 1), 2, 0)
            # Gantry label tag (left edge of each rail)
            g_label = self.font_xs.render(
                f"gantry {r_idx + 1}", True, (160, 220, 255),
            )
            self.screen.blit(g_label, (x + 6, rail_y - g_label.get_height() - 1))

        # Three characterisation turntables across the bottom of the room.
        char_zone = ZONES_BY_NAME.get("Classifier")
        has_drum = False
        if char_zone is not None:
            for entry in ctx.coord.ledger.values():
                if entry.current_location == "Classifier":
                    has_drum = True
                    break
        turntable_y = y + h - 14
        tt_radius = 8
        stations = (
            ("INTAKE",   0.22),
            ("MEASURE",  0.50),
            ("DISPATCH", 0.78),
        )
        for idx, (label, fx) in enumerate(stations):
            tt_x = x + int(w * fx)
            # Plate
            ring_col = (140, 220, 200) if label == "MEASURE" else (200, 170, 100)
            pygame.draw.circle(self.screen, (40, 50, 70), (tt_x, turntable_y),
                               tt_radius, 0)
            pygame.draw.circle(self.screen, ring_col, (tt_x, turntable_y),
                               tt_radius, 2)
            # Spinning tick when MEASURE has a drum
            if label == "MEASURE" and has_drum:
                ang = now_ms / 600.0
                tx = tt_x + int((tt_radius - 2) * math.cos(ang))
                ty = turntable_y + int((tt_radius - 2) * math.sin(ang))
                pygame.draw.line(self.screen, (220, 240, 255),
                                 (tt_x, turntable_y), (tx, ty), 2)
            # Tag above the turntable
            t = self.font_xs.render(label, True, ring_col)
            self.screen.blit(t,
                             (tt_x - t.get_width() // 2,
                              turntable_y - tt_radius - t.get_height() - 1))

        # Conveyor link between adjacent stations — short arrow segment
        # between each pair, suggests the drum flows INTAKE → MEASURE →
        # DISPATCH.
        conv_y = turntable_y
        for i in range(len(stations) - 1):
            x1 = x + int(w * stations[i][1]) + tt_radius
            x2 = x + int(w * stations[i + 1][1]) - tt_radius
            pygame.draw.line(self.screen, (130, 150, 180),
                             (x1, conv_y), (x2, conv_y), 1)
            pygame.draw.polygon(self.screen, (130, 150, 180), [
                (x2, conv_y),
                (x2 - 4, conv_y - 3),
                (x2 - 4, conv_y + 3),
            ])

        # SCANNING banner when a drum is on a turntable
        if has_drum:
            banner_y = rail_ys[0] - 14
            banner = pygame.Surface((w - 8, 12), pygame.SRCALPHA)
            banner.fill((20, 80, 100, 160))
            self.screen.blit(banner, (x + 4, banner_y))
            txt = self.font_xs.render("SCANNING DRUM", True, (160, 240, 255))
            self.screen.blit(txt, (x + w // 2 - txt.get_width() // 2,
                                    banner_y + 1))

    def _draw_kb_section_label(self, x: int, y: int, w: int, text: str, color: tuple):
        """Small tab-style section label drawn at the top of each AI-brain
        compartment so judges can read 'CLASSIFIER', 'KNOWLEDGE BASE',
        'LEARNING LOOP' from across the room."""
        lbl = self.font_xs.render(text, True, color)
        tab_w = min(w, lbl.get_width() + 6)
        tab_h = 10
        pygame.draw.rect(self.screen, (16, 20, 30),
                         (x, y, tab_w, tab_h))
        pygame.draw.rect(self.screen, color,
                         (x, y, tab_w, tab_h), 1)
        self.screen.blit(lbl, (x + (tab_w - lbl.get_width()) // 2, y - 1))

    def _draw_coordinator_interior(self, x: int, y: int, w: int, h: int, ctx):
        """The AI brain. We split the room into three labeled compartments
        so a first-time viewer can read the system at a glance:

          [ CLASSIFIER ×2 ][ KNOWLEDGE BASE ][ RESCAN LOOP ]

        Compartment 1 (CLASSIFIER): Twin live server racks side-by-side
        marked PRIMARY + STANDBY. Tells the redundancy story without
        words — there are visibly two of them and they pulse in lock-step.

        Compartment 2 (KNOWLEDGE BASE): Stacked horizontal "shelves" that
        each represent one tier of the data warehouse — NaI feature
        vectors, HPGe-labelled rows, the active training buffer, and the
        ledger of model versions. Each shelf has a fill bar that grows
        with the live counts, so judges literally watch the warehouse
        fill up over the shift.

        Compartment 3 (RESCAN LOOP): A small circular arrow indicator
        plus the LEARNED scalar (actinide threshold) and the live
        rescan / retrain counters. This is the visible 'learning' part.
        """
        now_ms = pygame.time.get_ticks()
        coord = ctx.coord

        # The compartment tabs (CLASSIFIER / KNOWLEDGE BASE / LEARNING LOOP)
        # tell the same story a title bar would, so we skip the title bar to
        # save vertical space and prevent label overlap with the zone-name
        # text drawn below the room.
        inner_y = y + 4
        inner_h = h - 8
        col_pad = 3
        col_w = (w - 4 - 2 * col_pad) // 3
        c1_x = x + 2
        c2_x = c1_x + col_w + col_pad
        c3_x = c2_x + col_w + col_pad
        # Compartment dividers (subtle vertical lines)
        for cx_line in (c2_x - col_pad // 2, c3_x - col_pad // 2):
            pygame.draw.line(self.screen, (40, 55, 75),
                             (cx_line, inner_y), (cx_line, inner_y + inner_h), 1)

        # ----- Compartment 1: CLASSIFIER (twin racks) -----
        self._draw_kb_section_label(c1_x, inner_y, col_w, "CLASSIFIER", (140, 220, 255))
        racks_y = inner_y + 11
        racks_h = inner_h - 13
        rack_w = (col_w - 8) // 2
        for idx in range(2):
            rx = c1_x + 3 + idx * (rack_w + 2)
            base = (28, 36, 52) if idx == 0 else (40, 36, 28)
            pygame.draw.rect(self.screen, base, (rx, racks_y, rack_w, racks_h))
            pygame.draw.rect(self.screen, (10, 12, 16),
                             (rx, racks_y, rack_w, racks_h), 1)
            phase = now_ms // 110 + idx * 5
            led_on = (110, 200, 255) if idx == 0 else (255, 180, 90)
            led_off = (30, 50, 80) if idx == 0 else (60, 50, 30)
            for ly in range(racks_y + 3, racks_y + racks_h - 3, 3):
                on = (phase + ly) % 4 != 0
                pygame.draw.line(self.screen, led_on if on else led_off,
                                 (rx + 2, ly), (rx + rack_w - 2, ly), 1)
            # Single-letter mark on the rack face: P = primary, S = standby.
            # Colour (cyan / amber) tells the same story, the letter is just
            # the accessibility hint.
            mark = self.font_xs.render("P" if idx == 0 else "S", True,
                                        (140, 220, 255) if idx == 0 else (255, 200, 120))
            self.screen.blit(mark,
                             (rx + rack_w // 2 - mark.get_width() // 2,
                              racks_y + racks_h // 2 - mark.get_height() // 2))

        # ----- Compartment 2: KNOWLEDGE BASE (warehouse shelves) -----
        self._draw_kb_section_label(c2_x, inner_y, col_w, "KNOWLEDGE BASE", (180, 220, 200))
        kb_y = inner_y + 11
        kb_h = inner_h - 13
        # Four shelves stacked: NaI features / HPGe labels / training buffer / models
        n_features = len(coord._training_features)
        n_labels = len(coord._training_labels)
        n_models = len(coord.retrain_events)
        buf_n = coord._reports_since_retrain
        buf_cap = max(1, coord.retrain_every_n_reports)
        # Each shelf: a thin coloured bar with a tiny label + count number
        shelves = [
            ("NaI scans",    min(1.0, n_features / 200.0), (140, 200, 240), n_features),
            ("HPGe labels",  min(1.0, n_labels   / 200.0), (200, 140, 230), n_labels),
            ("train buf",    min(1.0, buf_n / buf_cap),    (130, 230, 200), f"{buf_n}/{buf_cap}"),
            ("models",       min(1.0, n_models / 10.0),    (255, 200, 120), n_models),
        ]
        shelf_h = max(7, kb_h // len(shelves) - 1)
        sx = c2_x + 3
        sw = col_w - 6
        for i, (name, frac, col, count) in enumerate(shelves):
            sy = kb_y + i * (shelf_h + 1)
            # Shelf back
            pygame.draw.rect(self.screen, (16, 22, 32), (sx, sy, sw, shelf_h))
            pygame.draw.rect(self.screen, (50, 70, 95), (sx, sy, sw, shelf_h), 1)
            # Fill bar
            fill_w = int((sw - 2) * frac)
            if fill_w > 0:
                pygame.draw.rect(self.screen, col, (sx + 1, sy + 1, fill_w, shelf_h - 2))
            # Caption + count (only if there's vertical space)
            if shelf_h >= 9:
                cap = self.font_xs.render(name, True, (200, 220, 240))
                cnt = self.font_xs.render(str(count), True, (220, 230, 240))
                self.screen.blit(cap, (sx + 3, sy + max(0, (shelf_h - cap.get_height()) // 2)))
                self.screen.blit(cnt, (sx + sw - cnt.get_width() - 3,
                                        sy + max(0, (shelf_h - cnt.get_height()) // 2)))

        # ----- Compartment 3: RESCAN LOOP (learning indicator) -----
        self._draw_kb_section_label(c3_x, inner_y, col_w, "LEARNING LOOP", (255, 200, 140))
        loop_y = inner_y + 11
        loop_h = inner_h - 13
        # Top half: animated circular arrow loop
        loop_cx = c3_x + col_w // 2
        loop_cy = loop_y + loop_h // 3
        radius = min(col_w // 2 - 4, loop_h // 3)
        if radius >= 6:
            # The loop animation rotates a bright arc around a faint ring,
            # signalling the live scan→flag→HPGe→label→retrain cycle.
            ring_layer = pygame.Surface((radius * 2 + 4, radius * 2 + 4),
                                         pygame.SRCALPHA)
            cx0 = radius + 2
            cy0 = radius + 2
            pygame.draw.circle(ring_layer, (90, 120, 150, 130),
                               (cx0, cy0), radius, 2)
            # Bright moving arc
            sweep = (now_ms / 4) % 360
            for a_deg in range(int(sweep), int(sweep) + 110, 6):
                ang = math.radians(a_deg)
                px = cx0 + (radius) * math.cos(ang)
                py = cy0 + (radius) * math.sin(ang)
                pygame.draw.circle(ring_layer, (255, 210, 130, 220),
                                   (int(px), int(py)), 2, 0)
            # Arrowhead at the leading edge
            head_ang = math.radians(sweep + 110)
            hx = cx0 + radius * math.cos(head_ang)
            hy = cy0 + radius * math.sin(head_ang)
            pygame.draw.circle(ring_layer, (255, 230, 160, 250),
                               (int(hx), int(hy)), 3, 0)
            self.screen.blit(ring_layer,
                             (loop_cx - radius - 2, loop_cy - radius - 2))
            # Centre label — the learned scalar (actinide threshold)
            thr = coord.shared_classifier.actinide_threshold
            if thr is not None:
                thr_label = self.font_s.render(f"{thr:.3f}", True, (160, 240, 200))
            else:
                thr_label = self.font_s.render("—", True, (140, 150, 170))
            self.screen.blit(thr_label,
                             (loop_cx - thr_label.get_width() // 2,
                              loop_cy - thr_label.get_height() // 2))
        # Bottom half: model version + retrain counter + threshold label
        bot_y = loop_y + max(int(loop_h * 0.62), 2 * radius + 4)
        ver_surf = self.font_m.render(f"v{coord.model_version}",
                                       True, (180, 230, 255))
        self.screen.blit(ver_surf, (c3_x + 4, bot_y))
        sub = self.font_xs.render(
            f"thr  ·  retrains {n_models}", True, (160, 200, 230),
        )
        self.screen.blit(sub, (c3_x + 4, bot_y + ver_surf.get_height() - 1))

        # --- TRAINING overlay (briefly when the model just retrained) ---
        if now_ms < self._retrain_flash_until_ms + 2200:
            # Recent retrain — flash a TRAINING banner across the room
            remaining = max(0, (self._retrain_flash_until_ms + 2200) - now_ms)
            alpha = min(220, int(220 * remaining / 2200))
            if alpha > 0:
                banner = pygame.Surface((w - 8, 22), pygame.SRCALPHA)
                banner.fill((30, 60, 100, alpha))
                pygame.draw.rect(banner, (140, 220, 255, alpha),
                                 (0, 0, w - 8, 22), 2)
                self.screen.blit(banner, (x + 4, y + h // 2 - 11))
                txt = self.font_m.render(
                    f"TRAINING v{coord.model_version}", True, (220, 240, 255),
                )
                self.screen.blit(txt, (x + w // 2 - txt.get_width() // 2,
                                        y + h // 2 - txt.get_height() // 2))

        # --- BROADCAST overlay when a snapshot was just pushed ---
        if now_ms < self._hive_sync_until_ms:
            remaining = self._hive_sync_until_ms - now_ms
            alpha = min(220, int(220 * remaining / 1800))
            banner = pygame.Surface((w - 8, 18), pygame.SRCALPHA)
            banner.fill((20, 90, 80, alpha))
            self.screen.blit(banner, (x + 4, y + 4))
            txt = self.font_s.render("BROADCAST → AGENTS", True, (160, 240, 220))
            self.screen.blit(txt, (x + w // 2 - txt.get_width() // 2, y + 7))

        # --- FAILOVER overlay (K-key) — flashes a 'STANDBY PROMOTED' banner
        if now_ms < self._coord_failover_until_ms:
            remaining = self._coord_failover_until_ms - now_ms
            alpha = min(240, int(240 * remaining / 2400))
            banner = pygame.Surface((w - 8, 22), pygame.SRCALPHA)
            banner.fill((110, 60, 20, alpha))
            pygame.draw.rect(banner, (255, 200, 120, alpha),
                             (0, 0, w - 8, 22), 2)
            self.screen.blit(banner, (x + 4, y + h // 2 - 11))
            txt = self.font_m.render(
                f"FAILOVER #{coord.failover_count} — 0 data loss",
                True, (255, 230, 180),
            )
            self.screen.blit(txt, (x + w // 2 - txt.get_width() // 2,
                                    y + h // 2 - txt.get_height() // 2))

    def _draw_hazard_stripes(self, x: int, y: int, w: int, h: int):
        """Yellow-black diagonal hazard chevron drawn INSIDE the room's
        perimeter — the standard industrial 'radiation hot zone' marking.

        The chevron is clipped to a `stripe_w`-pixel band just inside the
        room rectangle, so adjacent hot rooms never have stripe-frames
        bleeding into each other or onto labels."""
        stripe_w = 4
        stripe_period = 10
        yellow = (220, 180, 30)
        black = (18, 20, 28)

        # Backing band — dark frame just inside the room edge.
        band_rects = [
            pygame.Rect(x, y, w, stripe_w),                          # top
            pygame.Rect(x, y + h - stripe_w, w, stripe_w),           # bottom
            pygame.Rect(x, y, stripe_w, h),                          # left
            pygame.Rect(x + w - stripe_w, y, stripe_w, h),           # right
        ]
        for r in band_rects:
            pygame.draw.rect(self.screen, black, r)

        prev_clip = self.screen.get_clip()
        room_clip = pygame.Rect(x, y, w, h)
        for r in band_rects:
            band_clip = r.clip(room_clip)
            self.screen.set_clip(band_clip)
            # True 45° yellow slashes: every `stripe_period` pixels along
            # the horizontal axis, a `stripe_w`-wide parallelogram that
            # shifts down-right at 1:1 slope. The polygons span the whole
            # room height; pygame's clip rect keeps each band clean.
            for i in range(-h - stripe_w, w + h, stripe_period):
                pts = [
                    (x + i,                       y),
                    (x + i + stripe_w,            y),
                    (x + i + stripe_w + h,        y + h),
                    (x + i + h,                   y + h),
                ]
                pygame.draw.polygon(self.screen, yellow, pts)
        self.screen.set_clip(prev_clip)

    def _draw_room_equipment(self, x: int, y: int, w: int, h: int, z, ctx):
        """Per-role equipment silhouettes painted at the back of each room
        so each station reads as a specific function instead of an empty
        box."""
        if z.role == "drum_scanner":
            # HPGe cabinet silhouette at the back
            cab_w, cab_h = 20, 14
            cab_x = x + w - cab_w - 4
            cab_y = y + 3
            pygame.draw.rect(self.screen, (45, 50, 65), (cab_x, cab_y, cab_w, cab_h))
            pygame.draw.rect(self.screen, (15, 18, 28), (cab_x, cab_y, cab_w, cab_h), 1)
            # Cabinet vents
            for vy in (cab_y + 3, cab_y + 6, cab_y + 9):
                pygame.draw.line(self.screen, (20, 22, 30),
                                 (cab_x + 3, vy), (cab_x + cab_w - 3, vy), 1)
        elif z.role == "decon":
            # Drainage grating at the bottom
            grate_y = y + h - 8
            for gx in range(x + 4, x + w - 4, 4):
                pygame.draw.line(self.screen, (35, 40, 50),
                                 (gx, grate_y), (gx, grate_y + 6), 1)
            # Spray nozzle stubs on the ceiling
            for nx in (x + w // 3, x + 2 * w // 3):
                pygame.draw.rect(self.screen, (60, 65, 80), (nx - 2, y + 1, 4, 4))
        elif z.role == "charging":
            # Charging contact pads on the floor
            for px in (x + w // 4, x + w // 2, x + 3 * w // 4):
                pygame.draw.rect(self.screen, (80, 130, 180),
                                 (px - 4, y + h - 6, 8, 3))
                pygame.draw.rect(self.screen, (10, 12, 16),
                                 (px - 4, y + h - 6, 8, 3), 1)
        elif z.role == "solidification":
            # Vitrification crucible silhouette
            crucible_x = x + w // 2 - 6
            crucible_y = y + h // 2 - 4
            pygame.draw.rect(self.screen, (180, 90, 70), (crucible_x, crucible_y, 12, 8))
            pygame.draw.rect(self.screen, (10, 12, 16), (crucible_x, crucible_y, 12, 8), 1)
            # Heating coils underneath
            for ix in range(crucible_x + 1, crucible_x + 11, 2):
                pygame.draw.line(self.screen, (220, 110, 60),
                                 (ix, crucible_y + 8), (ix, crucible_y + 11), 1)
        elif z.role == "coordinator":
            self._draw_coordinator_interior(x, y, w, h, ctx)
        elif z.role == "char_station":
            self._draw_char_station_interior(x, y, w, h, ctx)
        elif z.role == "fuel_pool":
            self._draw_fuel_pool_interior(x, y, w, h, ctx)
        elif z.role == "worker":
            # Lab bench silhouette across the bottom
            pygame.draw.rect(self.screen, (95, 90, 60),
                             (x + 4, y + h - 8, w - 8, 4))
            pygame.draw.rect(self.screen, (15, 16, 22),
                             (x + 4, y + h - 8, w - 8, 4), 1)
        elif z.role == "generation":
            # Drum-output hopper at the bottom-right corner — visually
            # marks this room as a *waste source*. Process cells without
            # this hopper (POOL, SOLV, VITR) are transit only. The
            # hopper is a small chute + a half-drum poking out of it.
            hop_w, hop_h = 18, 12
            hx = x + w - hop_w - 3
            hy = y + h - hop_h - 3
            # Hopper body (dark trapezoid funneling down to the spout)
            pygame.draw.polygon(self.screen, (75, 80, 95), [
                (hx, hy),
                (hx + hop_w, hy),
                (hx + hop_w - 4, hy + hop_h),
                (hx + 4, hy + hop_h),
            ])
            pygame.draw.polygon(self.screen, (10, 12, 16), [
                (hx, hy),
                (hx + hop_w, hy),
                (hx + hop_w - 4, hy + hop_h),
                (hx + 4, hy + hop_h),
            ], 1)
            # Spout
            pygame.draw.rect(self.screen, (60, 65, 80),
                             (hx + hop_w // 2 - 2, hy + hop_h, 4, 3))
            # Half-drum just emerging — uses the produced-class colour so
            # judges can read 'this room outputs LLW' / 'this outputs HLW'.
            out_class = z.produces_class
            drum_col = CLASS_COLOR.get(out_class, CLASS_COLOR[None])
            pygame.draw.ellipse(self.screen, drum_col,
                                (hx + hop_w // 2 - 4, hy + hop_h + 1, 8, 5))
            pygame.draw.ellipse(self.screen, (10, 12, 16),
                                (hx + hop_w // 2 - 4, hy + hop_h + 1, 8, 5), 1)
            # Tiny "OUT" label under the hopper to make it unmistakable
            tag = self.font_xs.render("OUT", True, (180, 200, 220))
            self.screen.blit(tag, (hx + hop_w // 2 - tag.get_width() // 2,
                                    hy - tag.get_height() - 1))
        elif z.role == "process":
            # Pipe / valve silhouette across the top — suggests process plumbing
            pipe_y = y + 5
            pygame.draw.rect(self.screen, (110, 115, 130),
                             (x + 2, pipe_y, w - 4, 4))
            for vx in (x + w // 3, x + 2 * w // 3):
                pygame.draw.rect(self.screen, (170, 100, 60),
                                 (vx - 2, pipe_y - 2, 4, 8))

    def _compute_gen_queues(self, ctx) -> dict[str, int]:
        """Items waiting for pickup, grouped by their pickup zone. Counts
        across both the scan queue (untriaged) and the handle queue
        (classified but not yet transported)."""
        counts: dict[str, int] = {}
        for task in list(ctx.scan_queue.items) + list(ctx.handle_queue.items):
            zone = task["pickup_zone"]
            counts[zone.name] = counts.get(zone.name, 0) + 1
        return counts

    def _compute_storage_counts(self, ctx) -> dict[str, int]:
        counts: dict[str, int] = {"VLLW": 0, "LLW": 0, "ILW": 0, "HLW": 0}
        for e in ctx.coord.ledger.values():
            if e.current_location == "VLLW storage":
                counts["VLLW"] += 1
            elif e.current_location == "LLW storage":
                counts["LLW"] += 1
            elif e.current_location == "ILW storage":
                counts["ILW"] += 1
            elif e.current_location == "HLW storage":
                counts["HLW"] += 1
        return counts

    def _draw_process_chain(self, ctx):
        """Stylised flow-diagram arrows linking the upstream PUREX stages
        (Spent Fuel Receipt -> Shearing -> Dissolution -> Solvent extraction
        -> HLW concentration -> Solidification) and the HLW transit edge
        from Solidification down to HLW storage. Animated dashes ride along
        each arrow so the chain reads as 'material moving through process'
        rather than static geometry.

        Modelled on the research-paper PUREX flow figure: the boxes are the
        same as the upstream zones below, the arrows make their ordering
        explicit so judges can follow the source-to-storage path."""
        chain = [
            "Spent fuel pool", "Shearing cell", "Dissolution cell",
            "Solvent extraction", "HLW concentration", "Solidification",
        ]
        layer = pygame.Surface((FACILITY_W, FACILITY_H), pygame.SRCALPHA)
        # Process-chain arrow colour: muted lavender, intentionally dim so the
        # PUREX chain reads as background context. The visual centrepiece is
        # the AI BRAIN room — the chain just tells you where waste comes from.
        col_main = (160, 155, 200, 130)
        col_dot = (200, 195, 230, 200)
        col_hlw = (200, 130, 130, 150)
        now_ms = pygame.time.get_ticks()
        # Stage-to-stage arrows
        for src_name, dst_name in zip(chain, chain[1:]):
            sz = ZONES_BY_NAME.get(src_name)
            dz = ZONES_BY_NAME.get(dst_name)
            if sz is None or dz is None:
                continue
            self._draw_flow_arrow(layer, (sz.x, sz.y), (dz.x, dz.y),
                                  sz.radius_m, dz.radius_m,
                                  col_main, col_dot, now_ms,
                                  dot_period_ms=2200)
        # Solidification -> HLW storage (the only forced transit route in
        # the robot fleet; drawn dashed/red to call it out).
        sol = ZONES_BY_NAME.get("Solidification")
        hlw = ZONES_BY_NAME.get("HLW storage")
        if sol is not None and hlw is not None:
            self._draw_flow_arrow(layer, (sol.x, sol.y), (hlw.x, hlw.y),
                                  sol.radius_m, hlw.radius_m,
                                  col_hlw, (240, 180, 180, 255), now_ms,
                                  dot_period_ms=3200, dashed=True)
        self.screen.blit(layer, (0, HEADER_H))

    def _draw_flow_arrow(
        self, surface, src_world, dst_world,
        src_r_m: float, dst_r_m: float,
        line_color, dot_color, now_ms: int,
        dot_period_ms: int = 2000, dashed: bool = False,
    ):
        """Arrow from src to dst (world coords), starting/ending at each
        zone's edge so the line doesn't clip into the circle. Animated dot
        rides 0 -> 1 along the segment on a `dot_period_ms` cycle."""
        sx, sy = self.world_to_screen(src_world)
        ex, ey = self.world_to_screen(dst_world)
        sxl, syl = sx, sy - HEADER_H
        exl, eyl = ex, ey - HEADER_H
        dx, dy = exl - sxl, eyl - syl
        d = math.hypot(dx, dy)
        if d < 1.0:
            return
        ux, uy = dx / d, dy / d
        # Offset so endpoints sit at the zone edge, not the centre
        sxo = sxl + ux * (src_r_m * PX_PER_M + 2)
        syo = syl + uy * (src_r_m * PX_PER_M + 2)
        exo = exl - ux * (dst_r_m * PX_PER_M + 2)
        eyo = eyl - uy * (dst_r_m * PX_PER_M + 2)
        # Draw line (dashed for HLW transit, solid for upstream chain)
        if dashed:
            segs = 14
            for i in range(segs):
                if i % 2 == 1:
                    continue
                fa = i / segs
                fb = (i + 1) / segs
                pa = (sxo + (exo - sxo) * fa, syo + (eyo - syo) * fa)
                pb = (sxo + (exo - sxo) * fb, syo + (eyo - syo) * fb)
                pygame.draw.line(surface, line_color, pa, pb, 2)
        else:
            pygame.draw.line(surface, line_color, (sxo, syo), (exo, eyo), 2)
        # Arrowhead at destination end
        ang = math.atan2(eyo - syo, exo - sxo)
        head_len = 9
        head_half = 5
        ax1 = exo - head_len * math.cos(ang) + head_half * math.sin(ang)
        ay1 = eyo - head_len * math.sin(ang) - head_half * math.cos(ang)
        ax2 = exo - head_len * math.cos(ang) - head_half * math.sin(ang)
        ay2 = eyo - head_len * math.sin(ang) + head_half * math.cos(ang)
        pygame.draw.polygon(surface, line_color, [(exo, eyo), (ax1, ay1), (ax2, ay2)])
        # Travelling dot
        t = (now_ms % dot_period_ms) / dot_period_ms
        dxp = sxo + (exo - sxo) * t
        dyp = syo + (eyo - syo) * t
        pygame.draw.circle(surface, dot_color, (int(dxp), int(dyp)), 3, 0)

    def _draw_decon_arm(self, ctx):
        """Two articulated decontamination arms anchored on either side of
        the Robot wash zone — visible redundancy. The arm closer to the
        AGV being deconned does the spray sweep; the second arm parks in
        a "STANDBY" pose with a faint amber tag so judges see both arms
        and understand one is a hot spare.

        When the active arm sweeps the wash nozzle over the AGV the dose
        bar drains in parallel (see Agent._go_decon — dose is interpolated
        linearly over wash duration). The standby arm idles."""
        wash = ZONES_BY_NAME.get("Robot wash")
        if wash is None:
            return

        target_agent = None
        for a in ctx.agents:
            if getattr(a, "state", "") == "DECONNING":
                target_agent = a
                break
        now_ms = pygame.time.get_ticks()

        # Two shoulder anchors: top-left and top-right of the wash bay.
        anchors = [
            ("DECON-A",
             self.world_to_screen((wash.x - wash.radius_m * 0.85,
                                    wash.y - wash.radius_m * 0.4))),
            ("DECON-B",
             self.world_to_screen((wash.x + wash.radius_m * 0.85,
                                    wash.y - wash.radius_m * 0.4))),
        ]
        L1_px = wash.radius_m * 0.8 * PX_PER_M
        L2_px = wash.radius_m * 0.8 * PX_PER_M

        # Which unit is active comes from the redundancy state — whichever
        # arm currently leads (primary by default, standby if the primary
        # is in a repair window).
        pair = (ctx.arm_pairs or {}).get("DECON")
        leader_idx = pair.active_idx() if pair is not None else 0
        if leader_idx is None:
            leader_idx = 0
        failed_set = set(pair.failed_units) if pair is not None else set()

        for idx, (name, (sx, sy)) in enumerate(anchors):
            is_failed = idx in failed_set
            active = (target_agent is not None
                      and idx == leader_idx and not is_failed)
            if active:
                tx, ty = self.world_to_screen(target_agent.pos)
                sweep_phase = math.sin(now_ms / 350.0 + idx * 0.6)
                dx, dy = tx - sx, ty - sy
                d = max(math.hypot(dx, dy), 1.0)
                px, py = -dy / d, dx / d
                sweep_offset_px = sweep_phase * 18
                target_x = tx + px * sweep_offset_px
                target_y = ty + py * sweep_offset_px - 4
            else:
                # Idle housekeeping sweep — slow lazy arc over the empty
                # wash bay floor so the arm always looks alive instead of
                # frozen. Centred over the bay centre, narrower amplitude
                # than active mode. Slightly offset per-arm so the two
                # arms don't sweep in perfect lockstep.
                bay_x = wash.x * PX_PER_M
                bay_y = wash.y * PX_PER_M + HEADER_H
                sweep = math.sin(now_ms / 1300.0 + idx * math.pi / 2)
                lateral = math.cos(now_ms / 1700.0 + idx * 0.7)
                side = -1 if idx == 0 else 1
                target_x = bay_x + side * 10 + sweep * 8
                target_y = bay_y + 6 + lateral * 4

            dx, dy = target_x - sx, target_y - sy
            d = math.hypot(dx, dy)
            d = max(min(d, L1_px + L2_px - 1), abs(L1_px - L2_px) + 1)
            cos_a = (L1_px ** 2 + d ** 2 - L2_px ** 2) / (2 * L1_px * d)
            cos_a = max(-1.0, min(1.0, cos_a))
            # Mirror the elbow on the right-hand arm so it bends away from
            # the wash bay's interior cleanly.
            side = -1 if idx == 0 else 1
            theta_shoulder = math.atan2(dy, dx) - side * math.acos(cos_a)
            ex = sx + L1_px * math.cos(theta_shoulder)
            ey = sy + L1_px * math.sin(theta_shoulder)

            arm_col = (220, 230, 240) if active else (140, 150, 165)
            joint_col = (60, 90, 130) if active else (50, 60, 80)
            # Pedestal base
            ped_h = 14
            pygame.draw.rect(self.screen, (55, 60, 75),
                             (int(sx) - 5, int(sy) - 1, 10, ped_h))
            pygame.draw.rect(self.screen, (10, 12, 16),
                             (int(sx) - 5, int(sy) - 1, 10, ped_h), 1)
            # Upper + forearm
            pygame.draw.line(self.screen, arm_col, (sx, sy), (ex, ey), 5)
            pygame.draw.line(self.screen, arm_col, (ex, ey), (target_x, target_y), 4)
            # Joints
            pygame.draw.circle(self.screen, joint_col, (int(sx), int(sy)), 5, 0)
            pygame.draw.circle(self.screen, (10, 12, 16), (int(sx), int(sy)), 5, 1)
            pygame.draw.circle(self.screen, joint_col, (int(ex), int(ey)), 4, 0)
            pygame.draw.circle(self.screen, (10, 12, 16), (int(ex), int(ey)), 4, 1)
            # End-effector wash nozzle
            nozzle_col = (110, 200, 255) if active else (110, 120, 140)
            pygame.draw.circle(self.screen, nozzle_col,
                               (int(target_x), int(target_y)), 5, 0)
            pygame.draw.circle(self.screen, (10, 12, 16),
                               (int(target_x), int(target_y)), 5, 1)
            pygame.draw.polygon(self.screen, nozzle_col, [
                (target_x, target_y),
                (target_x - 3, target_y + 5),
                (target_x + 3, target_y + 5),
            ])

            # Per-arm label tag — short name only. Cyan = active, amber =
            # standby, red strike-through = failed (in repair).
            if is_failed:
                tag_col = (240, 100, 110)
                label = f"{name} FAILED"
            elif active:
                tag_col = (140, 220, 255)
                label = name
            else:
                tag_col = (220, 170, 90)
                label = name
            tag = self.font_xs.render(label, True, tag_col)
            tag_bg = pygame.Surface((tag.get_width() + 6, tag.get_height() + 2),
                                      pygame.SRCALPHA)
            tag_bg.fill((12, 14, 20, 220))
            self.screen.blit(tag_bg,
                             (int(sx) - tag.get_width() // 2 - 3, int(sy) - 22))
            self.screen.blit(tag, (int(sx) - tag.get_width() // 2, int(sy) - 21))
            if is_failed:
                # Diagonal red strike through the pedestal so the failure
                # is unmistakable at a glance.
                pygame.draw.line(self.screen, (240, 100, 110),
                                 (int(sx) - 8, int(sy) - 4),
                                 (int(sx) + 8, int(sy) + 12), 2)

            if active and target_agent is not None:
                # Spray cone toward the AGV from this arm's nozzle.
                sdx = (target_agent.pos[0] * PX_PER_M) - target_x
                sdy = (target_agent.pos[1] * PX_PER_M + HEADER_H) - target_y
                sd = max(math.hypot(sdx, sdy), 1.0)
                ux, uy = sdx / sd, sdy / sd
                spray_layer = pygame.Surface((FACILITY_W, FACILITY_H), pygame.SRCALPHA)
                for i in range(20):
                    phase = ((now_ms / 28.0) + i * 3) % 24
                    spread = (i - 10) * 0.05
                    rx = -uy * spread + ux
                    ry = ux * spread + uy
                    px_pos = target_x + rx * phase * 1.4
                    py_pos = target_y + ry * phase * 1.4
                    alpha = max(0, int(220 - phase * 9))
                    r = 2 if i % 4 == 0 else 1
                    pygame.draw.circle(
                        spray_layer, (140, 220, 255, alpha),
                        (int(px_pos), int(py_pos - HEADER_H)), r, 0,
                    )
                self.screen.blit(spray_layer, (0, HEADER_H))
                # Pulsing active ring around the AGV
                agv_x, agv_y = self.world_to_screen(target_agent.pos)
                pulse = 0.5 + 0.5 * math.sin(now_ms / 300.0)
                ring_r = int(16 + 4 * pulse)
                ring = pygame.Surface((ring_r * 2 + 4, ring_r * 2 + 4), pygame.SRCALPHA)
                pygame.draw.circle(ring, (140, 220, 255, int(120 + 80 * pulse)),
                                   (ring_r + 2, ring_r + 2), ring_r, 2)
                self.screen.blit(ring, (agv_x - ring_r - 2, agv_y - ring_r - 2))

    def _draw_qa_lab_arm(self, ctx):
        """Two redundant HPGe stations sitting side-by-side in the QA lab:
        HPGe-A (primary) on the left and HPGe-B (standby) on the right.
        Each station has its own lead-shielded detector head, articulated
        manipulator arm, and labelled tag. When RESCANNING is active the
        primary station performs the assay; the standby holds a parked
        pose with an amber STANDBY tag.

        This is the most prominent classification visual in the sim — the
        AI's authoritative oracle, made into TWO physical robots so the
        redundancy story reads at a glance."""
        lab = ZONES_BY_NAME.get("QA lab")
        if lab is None:
            return
        qa_agent = None
        for a in ctx.agents:
            if getattr(a, "agent_type", "") == "qa_lab":
                qa_agent = a
                break
        rescanning = (qa_agent is not None and qa_agent.state == "RESCANNING"
                      and qa_agent.carrying is not None)
        now_ms = pygame.time.get_ticks()

        # Active arm comes from the redundancy state — the primary leads
        # unless it's currently in a repair window, then the standby
        # takes over.
        pair = (ctx.arm_pairs or {}).get("HPGe")
        leader_idx = pair.active_idx() if pair is not None else 0
        if leader_idx is None:
            leader_idx = 0
        failed_set = set(pair.failed_units) if pair is not None else set()

        # Two stations, mirrored across the lab centreline. `idx` 0 = A,
        # `idx` 1 = B. Whichever pair.active_idx() points at gets the
        # work when an item is currently being rescanned.
        stations = [
            {
                "idx": 0,
                "name": "HPGe-A",
                "anchor": self.world_to_screen((lab.x - lab.radius_m * 0.62,
                                                 lab.y - lab.radius_m * 0.4)),
                "det":    self.world_to_screen((lab.x - lab.radius_m * 0.35,
                                                 lab.y + lab.radius_m * 0.25)),
                "side":   +1,
            },
            {
                "idx": 1,
                "name": "HPGe-B",
                "anchor": self.world_to_screen((lab.x + lab.radius_m * 0.62,
                                                 lab.y - lab.radius_m * 0.4)),
                "det":    self.world_to_screen((lab.x + lab.radius_m * 0.35,
                                                 lab.y + lab.radius_m * 0.25)),
                "side":   -1,
            },
        ]
        for st in stations:
            is_failed = st["idx"] in failed_set
            st["failed"] = is_failed
            st["active"] = (rescanning and st["idx"] == leader_idx
                            and not is_failed)
        L1_px = lab.radius_m * 0.55 * PX_PER_M
        L2_px = lab.radius_m * 0.55 * PX_PER_M

        for st in stations:
            sx, sy = st["anchor"]
            det_cx, det_cy = st["det"]
            active = st["active"]
            failed = st["failed"]

            # --- Detector body ---
            det_w, det_h = 22, 16
            pygame.draw.rect(self.screen, (55, 60, 75),
                             (det_cx - det_w // 2, det_cy - det_h // 2, det_w, det_h))
            pygame.draw.rect(self.screen, (10, 12, 16),
                             (det_cx - det_w // 2, det_cy - det_h // 2, det_w, det_h), 1)
            # Detector aperture facing up
            pygame.draw.rect(self.screen, (15, 18, 28),
                             (det_cx - 5, det_cy - det_h // 2 - 3, 10, 4))
            # Station label inside the detector body
            if failed:
                lbl_col = (240, 100, 110)
            elif active:
                lbl_col = (140, 220, 255)
            else:
                lbl_col = (200, 170, 100)
            lbl = self.font_xs.render(st["name"], True, lbl_col)
            self.screen.blit(lbl, (det_cx - lbl.get_width() // 2, det_cy - 4))
            if failed:
                # Red diagonal strike across the detector body.
                pygame.draw.line(self.screen, (240, 100, 110),
                                 (det_cx - det_w // 2,
                                  det_cy - det_h // 2),
                                 (det_cx + det_w // 2,
                                  det_cy + det_h // 2), 2)
            if active:
                pulse = 0.5 + 0.5 * math.sin(now_ms / 180.0)
                pygame.draw.circle(self.screen, (130, 230, 255),
                                   (det_cx + det_w // 2 - 4,
                                    det_cy - det_h // 2 + 4),
                                   2 + int(2 * pulse), 0)

            # --- Arm target ---
            if active:
                target_x = det_cx
                target_y = det_cy - 20
                target_y += int(math.sin(now_ms / 280.0))
            else:
                # Idle calibration sweep — the arm slowly paces back and
                # forth over its own detector head, suggesting a
                # housekeeping / re-calibration routine. Each station
                # paces independently so they aren't in lockstep.
                phase = math.sin(now_ms / 1400.0 + st["side"] * 0.7)
                target_x = det_cx + phase * 8
                target_y = det_cy - 14 + math.cos(now_ms / 1600.0) * 3

            dx, dy = target_x - sx, target_y - sy
            d = math.hypot(dx, dy)
            d = max(min(d, L1_px + L2_px - 1), abs(L1_px - L2_px) + 1)
            cos_a = (L1_px ** 2 + d ** 2 - L2_px ** 2) / (2 * L1_px * d)
            cos_a = max(-1.0, min(1.0, cos_a))
            theta_shoulder = math.atan2(dy, dx) - st["side"] * math.acos(cos_a)
            ex = sx + L1_px * math.cos(theta_shoulder)
            ey = sy + L1_px * math.sin(theta_shoulder)

            arm_col = (220, 230, 240) if active else (140, 150, 165)
            joint_col = (60, 90, 130) if active else (60, 60, 80)
            # Pedestal
            ped_h = 14
            pygame.draw.rect(self.screen, (55, 60, 75),
                             (int(sx) - 5, int(sy) - 1, 10, ped_h))
            pygame.draw.rect(self.screen, (10, 12, 16),
                             (int(sx) - 5, int(sy) - 1, 10, ped_h), 1)
            # Arm segments
            pygame.draw.line(self.screen, arm_col, (sx, sy), (ex, ey), 5)
            pygame.draw.line(self.screen, arm_col, (ex, ey), (target_x, target_y), 4)
            pygame.draw.circle(self.screen, joint_col, (int(sx), int(sy)), 5, 0)
            pygame.draw.circle(self.screen, (10, 12, 16), (int(sx), int(sy)), 5, 1)
            pygame.draw.circle(self.screen, joint_col, (int(ex), int(ey)), 4, 0)
            pygame.draw.circle(self.screen, (10, 12, 16), (int(ex), int(ey)), 4, 1)

            if active and qa_agent.carrying is not None:
                entry = ctx.coord.ledger.get(qa_agent.carrying.item_id)
                if entry is not None:
                    # Match the small "process container" sprite size
                    # used elsewhere in the pipeline.
                    drum_w, drum_h = 14, 20
                    halo_r = 16 + int(3 * math.sin(now_ms / 250.0))
                    halo = pygame.Surface((halo_r * 2 + 4, halo_r * 2 + 4),
                                           pygame.SRCALPHA)
                    pygame.draw.circle(halo, (140, 220, 255, 90),
                                       (halo_r + 2, halo_r + 2), halo_r, 0)
                    self.screen.blit(halo,
                                     (int(target_x) - halo_r - 2,
                                      int(target_y) - halo_r + 4))
                    dx_d = int(target_x) - drum_w // 2
                    dy_d = int(target_y) - drum_h // 2
                    self._draw_drum(
                        dx_d, dy_d, drum_w, drum_h, entry,
                        pulse=0.5 + 0.5 * math.sin(now_ms / 240),
                        now_ms=now_ms,
                    )
                    grip_col = (110, 200, 255)
                    pygame.draw.line(self.screen, grip_col,
                                     (target_x - 6, target_y - drum_h // 2),
                                     (target_x - 6, target_y - drum_h // 2 + 7), 3)
                    pygame.draw.line(self.screen, grip_col,
                                     (target_x + 6, target_y - drum_h // 2),
                                     (target_x + 6, target_y - drum_h // 2 + 7), 3)

            # Per-station label tag above the pedestal — just the short
            # name. Colour (cyan = active, amber = standby, red = failed)
            # carries the role distinction without long text.
            if failed:
                tag_col = (240, 100, 110)
            elif active:
                tag_col = (140, 220, 255)
            else:
                tag_col = (220, 170, 90)
            tag = self.font_xs.render(st["name"], True, tag_col)
            tag_bg = pygame.Surface((tag.get_width() + 6, tag.get_height() + 2),
                                     pygame.SRCALPHA)
            tag_bg.fill((12, 14, 20, 220))
            self.screen.blit(tag_bg,
                             (int(sx) - tag.get_width() // 2 - 3, int(sy) - 22))
            self.screen.blit(tag, (int(sx) - tag.get_width() // 2, int(sy) - 21))

        # Worker oversight station — every HPGe assay is reviewed by a
        # human at a control console before the verdict is logged as
        # authoritative. The AI cannot validate itself. This sits inside
        # the QA lab room near the bottom-centre, with a small monitor +
        # operator icon. When a rescan is happening, the monitor shows
        # the AI's pending verdict and an animated "REVIEWING" pulse.
        self._draw_qa_worker_oversight(ctx, lab, qa_agent, rescanning, now_ms)

    def _draw_qa_worker_oversight(self, ctx, lab, qa_agent, rescanning, now_ms):
        """Human-in-the-loop verification at the QA lab.

        A worker sits at a console between HPGe-A and HPGe-B, reviewing
        every assay against the camera feed. The on-console monitor
        shows the AI's pending class call and a small green/red badge
        once the worker has confirmed or overridden it. This is the
        'no AI evaluating itself' guardrail — combined HPGe+worker
        error is lower than HPGe alone, but it's never zero (workers
        also misread, mis-key, get fatigued).
        """
        # Anchor: bottom-centre of QA lab room
        ox, oy = self.world_to_screen((lab.x, lab.y + lab.radius_m * 0.55))

        # Console body
        c_w, c_h = 22, 12
        pygame.draw.rect(self.screen, (60, 70, 90),
                         (ox - c_w // 2, oy - c_h // 2, c_w, c_h))
        pygame.draw.rect(self.screen, (10, 12, 16),
                         (ox - c_w // 2, oy - c_h // 2, c_w, c_h), 1)

        # Monitor screen on top of console — shows either pending verdict
        # while a rescan is happening, or 'IDLE' between drums.
        m_w, m_h = 18, 8
        m_x = ox - m_w // 2
        m_y = oy - c_h // 2 - m_h - 1
        # Screen background
        screen_bg = (20, 50, 30)
        if rescanning and qa_agent is not None and qa_agent.carrying is not None:
            # Pulse while reviewing
            t = (math.sin(now_ms / 200.0) + 1.0) / 2.0
            screen_bg = (
                int(20 + 80 * t),
                int(60 + 100 * t),
                int(30 + 80 * t),
            )
        pygame.draw.rect(self.screen, screen_bg, (m_x, m_y, m_w, m_h))
        pygame.draw.rect(self.screen, (10, 12, 16), (m_x, m_y, m_w, m_h), 1)

        # Tiny "REVIEW" or "IDLE" text in the screen — too small to render
        # the word so we use 2-3 colour squares as a glyph proxy.
        if rescanning:
            # Three scrolling pixels suggest a live camera/data feed
            for i in range(3):
                phase = (now_ms // 150 + i) % 4
                px = m_x + 2 + phase * 4
                pygame.draw.rect(self.screen, (180, 240, 200),
                                 (px, m_y + 3, 2, 2))
        else:
            # Single dot, idle indicator
            pygame.draw.rect(self.screen, (90, 130, 100),
                             (m_x + m_w // 2 - 1, m_y + 3, 2, 2))

        # Worker icon — sits in front of the console facing the screen
        wx, wy = ox, oy + 8
        # Body (small chair + torso triangle)
        pygame.draw.polygon(self.screen, (235, 220, 200), [
            (wx, wy - 1),
            (wx - 4, wy + 5),
            (wx + 4, wy + 5),
        ])
        pygame.draw.polygon(self.screen, (10, 12, 16), [
            (wx, wy - 1),
            (wx - 4, wy + 5),
            (wx + 4, wy + 5),
        ], 1)
        # Head
        pygame.draw.circle(self.screen, (245, 225, 195), (wx, wy - 4), 3, 0)
        pygame.draw.circle(self.screen, (10, 12, 16), (wx, wy - 4), 3, 1)

        # "OPERATOR" tag below the worker
        tag = self.font_xs.render("operator", True, (220, 180, 90))
        tag_bg = pygame.Surface((tag.get_width() + 4, tag.get_height() + 1),
                                pygame.SRCALPHA)
        tag_bg.fill((12, 14, 20, 220))
        self.screen.blit(tag_bg,
                         (ox - tag.get_width() // 2 - 2, oy + 14))
        self.screen.blit(tag, (ox - tag.get_width() // 2, oy + 14))

        # Camera-link line from HPGe stations to the monitor — a faint
        # cable / data link, so the visual story reads "the operator is
        # seeing what each HPGe sees".
        layer = pygame.Surface((FACILITY_W, FACILITY_H), pygame.SRCALPHA)
        for offset_x in (-int(lab.radius_m * 0.35 * PX_PER_M),
                          int(lab.radius_m * 0.35 * PX_PER_M)):
            sx = ox + offset_x
            sy = oy - int(lab.radius_m * 0.65 * PX_PER_M)
            pygame.draw.line(
                layer, (110, 180, 220, 100),
                (sx, sy - HEADER_H), (ox, m_y - HEADER_H), 1,
            )
        self.screen.blit(layer, (0, HEADER_H))

    def _draw_loader_arms(self, ctx):
        """Small AI-controlled loader arms at every drum hand-off point —
        storage shelves, the Char station, the QA lab. These do the
        physical lift on/off the cart so a 200L drum doesn't magically
        levitate from cart to shelf.

        Each loader arm is anchored at its zone's perimeter. When a cart
        is currently DROPPING_OFF or PICKING_UP at that zone, the arm
        extends to the cart, grabs the drum, and lifts it into the zone.
        Otherwise it parks in a folded pose."""
        # (zone_name, anchor_side: -1 = left edge / +1 = right edge,
        #  active_states_for_this_zone)
        loader_zones = [
            ("VLLW storage",   -1, ("DROPPING_OFF",)),
            ("LLW storage",    -1, ("DROPPING_OFF",)),
            ("ILW storage",    -1, ("DROPPING_OFF",)),
            ("HLW storage",    -1, ("DROPPING_OFF",)),
            ("Free release",   -1, ("DROPPING_OFF",)),
            ("Classifier",   +1, ("DROPPING_OFF", "PICKING_UP")),
        ]
        now_ms = pygame.time.get_ticks()
        for zone_name, side, active_states in loader_zones:
            z = ZONES_BY_NAME.get(zone_name)
            if z is None:
                continue
            # Find a nearby cart in an active state for this zone
            target_agent = None
            for a in ctx.agents:
                if a.is_rescrutiny_station:
                    continue
                if a.state not in active_states:
                    continue
                if math.hypot(a.pos[0] - z.x, a.pos[1] - z.y) > z.radius_m + 1.5:
                    continue
                target_agent = a
                break

            # Anchor: top corner of the zone on the cart-facing side
            anchor_world = (
                z.x + side * z.radius_m * 0.85,
                z.y - z.radius_m * 0.55,
            )
            sx, sy = self.world_to_screen(anchor_world)
            L1_px = z.radius_m * 0.7 * PX_PER_M
            L2_px = z.radius_m * 0.7 * PX_PER_M

            active = target_agent is not None
            if active:
                tx, ty = self.world_to_screen(target_agent.pos)
                # Add a small bob so the arm visibly handles the drum
                ty -= 4 + int(math.sin(now_ms / 200.0) * 1.5)
                target_x = tx
                target_y = ty
            else:
                # Parked pose: folded inward toward zone centre
                target_x = sx - side * L1_px * 0.4
                target_y = sy + L1_px * 0.2

            dx, dy = target_x - sx, target_y - sy
            d = math.hypot(dx, dy)
            d = max(min(d, L1_px + L2_px - 1), abs(L1_px - L2_px) + 1)
            cos_a = (L1_px ** 2 + d ** 2 - L2_px ** 2) / (2 * L1_px * d)
            cos_a = max(-1.0, min(1.0, cos_a))
            # Choose elbow direction so the arm bends *into* the zone
            elbow_sign = side
            theta_shoulder = math.atan2(dy, dx) + elbow_sign * math.acos(cos_a)
            ex = sx + L1_px * math.cos(theta_shoulder)
            ey = sy + L1_px * math.sin(theta_shoulder)

            arm_col = (210, 220, 235) if active else (130, 140, 155)
            joint_col = (60, 90, 130) if active else (60, 70, 90)
            # Compact pedestal
            ped_h = 9
            pygame.draw.rect(self.screen, (60, 65, 80),
                             (int(sx) - 3, int(sy) - 1, 6, ped_h))
            pygame.draw.rect(self.screen, (10, 12, 16),
                             (int(sx) - 3, int(sy) - 1, 6, ped_h), 1)
            # Arm segments
            pygame.draw.line(self.screen, arm_col,
                             (sx, sy), (int(ex), int(ey)), 3)
            pygame.draw.line(self.screen, arm_col,
                             (int(ex), int(ey)), (int(target_x), int(target_y)), 3)
            # Joints
            pygame.draw.circle(self.screen, joint_col, (int(sx), int(sy)), 3, 0)
            pygame.draw.circle(self.screen, joint_col, (int(ex), int(ey)), 2, 0)
            # Gripper at the tip — two small prongs when active
            if active:
                # Open / close phase
                gop = (1 + math.sin(now_ms / 280.0)) * 0.5
                spread = 3 + int(gop * 2)
                pygame.draw.line(self.screen, (160, 220, 255),
                                 (int(target_x) - spread, int(target_y) - 3),
                                 (int(target_x) - spread, int(target_y) + 3), 2)
                pygame.draw.line(self.screen, (160, 220, 255),
                                 (int(target_x) + spread, int(target_y) - 3),
                                 (int(target_x) + spread, int(target_y) + 3), 2)
            else:
                # Tiny passive gripper head
                pygame.draw.rect(self.screen, (150, 160, 175),
                                 (int(target_x) - 2, int(target_y) - 2, 4, 4))

    def _draw_zone_callouts(self, ctx):
        """Plain-English purpose callouts pinned to the four hero stations so
        a first-time viewer can read the system at a glance:

          - AI BRAIN room → 'classifier + knowledge base'
          - QA lab        → 'redundant HPGe assay  ×2'
          - Robot wash    → 'redundant decon arms  ×2'
          - Charging bay  → 'AGV charging'

        Callouts are translucent so they don't fight with the room
        interiors, and pinned ABOVE each zone (or BELOW for the top-row
        Coordinator)."""
        callouts = [
            ("Coordinator",
             "CLASSIFIER + KNOWLEDGE BASE",
             (140, 220, 255), "below"),
            ("Spent fuel pool",
             "SPENT FUEL POOL  ·  FHM crane + submerged manipulator",
             (140, 200, 240), "below"),
            ("Classifier",
             "DRUM CHARACTERISATION  ·  turntable + NaI + CV",
             (140, 220, 200), "above"),
            ("QA lab",
             "REDUNDANT HPGe  ×2",
             (200, 150, 230), "above"),
            ("Robot wash",
             "REDUNDANT DECON ARMS  ×2",
             (140, 220, 255), "above"),
        ]
        for zone_name, text, color, side in callouts:
            z = ZONES_BY_NAME.get(zone_name)
            if z is None:
                continue
            cx, cy = self.world_to_screen((z.x, z.y))
            r_px = int(z.radius_m * PX_PER_M)
            # Determine room extent so the callout sits cleanly outside it
            if z.role == "coordinator":
                rw, rh = int(z.radius_m * 2.9 * PX_PER_M), int(z.radius_m * 1.7 * PX_PER_M)
            elif z.role == "decon":
                rw, rh = int(z.radius_m * 2.4 * PX_PER_M), int(z.radius_m * 2.0 * PX_PER_M)
            elif z.role == "drum_scanner":
                rw, rh = int(z.radius_m * 2.4 * PX_PER_M), int(z.radius_m * 2.2 * PX_PER_M)
            elif z.role == "char_station":
                rw, rh = int(z.radius_m * 2.6 * PX_PER_M), int(z.radius_m * 2.0 * PX_PER_M)
            elif z.role == "fuel_pool":
                rw, rh = int(z.radius_m * 2.0 * PX_PER_M), int(z.radius_m * 2.4 * PX_PER_M)
            else:
                rw, rh = int(z.radius_m * 2.0 * PX_PER_M), int(z.radius_m * 2.0 * PX_PER_M)
            text_surf = self.font_xs.render(text, True, color)
            pad_x, pad_y = 6, 2
            bw = text_surf.get_width() + 2 * pad_x
            bh = text_surf.get_height() + 2 * pad_y
            bx = cx - bw // 2
            if side == "above":
                by = cy - rh // 2 - bh - 14
            else:
                by = cy + rh // 2 + 14
            # Background pill
            bg = pygame.Surface((bw, bh), pygame.SRCALPHA)
            bg.fill((10, 14, 22, 220))
            self.screen.blit(bg, (bx, by))
            pygame.draw.rect(self.screen, color, (bx, by, bw, bh), 1, border_radius=3)
            self.screen.blit(text_surf, (bx + pad_x, by + pad_y))
            # Thin tail line from pill to the room edge, so the eye links
            # the callout to its station.
            tail_y_start = by + (bh if side == "above" else 0)
            tail_y_end   = cy + (-rh // 2 if side == "above" else rh // 2)
            tail_layer = pygame.Surface((4, abs(tail_y_end - tail_y_start) + 2),
                                         pygame.SRCALPHA)
            for ty in range(0, tail_layer.get_height(), 4):
                pygame.draw.rect(tail_layer, (*color, 140), (1, ty, 2, 2))
            self.screen.blit(tail_layer,
                             (cx - 2, min(tail_y_start, tail_y_end)))

    # The 5-stage closed-loop pipeline that the AI runs every iteration.
    # Static labels — the run-time "which step is active" indicator is
    # already covered by the rotating arc inside the COORD's LEARNING LOOP
    # compartment, so this strip is a *narrative* annotation rather than
    # a live indicator. Keeps the eye from needing to chase a moving dot
    # while still spelling out what the system does.
    _LEARNING_LOOP_STAGES = [
        ("1", "scan",         (110, 200, 255)),
        ("2", "low-conf flag",(255, 200, 100)),
        ("3", "HPGe re-class",(200, 130, 230)),
        ("4", "train + retrain", (130, 230, 200)),
        ("5", "broadcast → fleet", (95, 200, 255)),
    ]

    def _draw_learning_loop_strip(self, ctx):
        """Static labelled pipeline rendered at the very bottom of the
        facility view: shows the closed learning loop in 5 numbered
        stages with arrows between. Self-narrating system diagram so a
        judge doesn't need the README to follow what they're looking at."""
        strip_h = 22
        margin = 8
        x0 = margin
        y0 = HEADER_H + FACILITY_H - strip_h - 4
        w = FACILITY_W - 2 * margin
        # Faint background bar
        bg = pygame.Surface((w, strip_h), pygame.SRCALPHA)
        bg.fill((10, 14, 22, 215))
        self.screen.blit(bg, (x0, y0))
        pygame.draw.rect(self.screen, (60, 90, 130, 255),
                         (x0, y0, w, strip_h), 1)
        # Title pill at the left edge
        title = self.font_xs.render("LEARNING LOOP", True, (180, 230, 255))
        self.screen.blit(title, (x0 + 8, y0 + (strip_h - title.get_height()) // 2))
        # Stages laid out evenly after the title
        stage_start = x0 + 8 + title.get_width() + 14
        stage_end = x0 + w - 8
        n = len(self._LEARNING_LOOP_STAGES)
        # Compute total width consumed by stages + arrows
        stage_w = (stage_end - stage_start) // n
        for i, (num, label, color) in enumerate(self._LEARNING_LOOP_STAGES):
            sx = stage_start + i * stage_w
            # Stage chip: number circle + label
            cy = y0 + strip_h // 2
            pygame.draw.circle(self.screen, color, (sx + 9, cy), 8, 0)
            pygame.draw.circle(self.screen, (10, 12, 16), (sx + 9, cy), 8, 1)
            num_surf = self.font_xs.render(num, True, (10, 14, 22))
            self.screen.blit(num_surf,
                             (sx + 9 - num_surf.get_width() // 2,
                              cy - num_surf.get_height() // 2))
            lbl_surf = self.font_xs.render(label, True, (220, 230, 240))
            self.screen.blit(lbl_surf,
                             (sx + 22, cy - lbl_surf.get_height() // 2))
            # Arrow to next stage
            if i < n - 1:
                ax = sx + stage_w - 14
                ay = cy
                pygame.draw.line(self.screen, (140, 180, 210),
                                 (ax, ay), (ax + 10, ay), 1)
                pygame.draw.polygon(self.screen, (140, 180, 210),
                                    [(ax + 10, ay),
                                     (ax + 6, ay - 3),
                                     (ax + 6, ay + 3)])
        # Closing loop hint at the far right: curved arrow back to stage 1
        loop_x = x0 + w - 4
        loop_y = y0 + strip_h // 2
        # Small ↺ glyph
        glyph = self.font_xs.render("↺", True, (180, 220, 255))
        self.screen.blit(glyph,
                         (loop_x - glyph.get_width(),
                          loop_y - glyph.get_height() // 2))

    def _draw_workers(self, ctx):
        """Render the worker NPCs as tiny human icons (head + body) drifting
        inside the Worker area. Their integrated dose tints the body from
        pale to orange to red as exposure rises."""
        workers = getattr(ctx, "workers", None) or []
        for w in workers:
            cx, cy = self.world_to_screen(w.pos)
            # Tint by integrated dose (cap at the agent decon threshold for
            # the colour ramp — workers don't get deconned, this is purely
            # visual feedback).
            dose = max(0.0, w.integrated_dose_uSv)
            if dose < 500:
                body_col = (235, 220, 200)
            elif dose < 2500:
                body_col = (250, 200, 130)
            else:
                body_col = (240, 110, 100)
            head_col = (245, 225, 195)
            # Head
            pygame.draw.circle(self.screen, head_col, (cx, cy - 5), 3, 0)
            pygame.draw.circle(self.screen, (10, 12, 16), (cx, cy - 5), 3, 1)
            # Body — a small triangle
            pygame.draw.polygon(self.screen, body_col, [
                (cx, cy - 1),
                (cx - 4, cy + 5),
                (cx + 4, cy + 5),
            ])
            pygame.draw.polygon(self.screen, (10, 12, 16), [
                (cx, cy - 1),
                (cx - 4, cy + 5),
                (cx + 4, cy + 5),
            ], 1)

    def _draw_waste_items(self, ctx):
        """Render every tracked waste drum at its current location so judges
        can literally see waste flowing through the facility. Carried items
        are skipped — the agent draws its own carry chip.

        Items are stacked in a deterministic grid inside each zone (sorted by
        item_id) so the same drum stays in the same slot frame-to-frame.
        Tricky/actinide-spiked drums get a faint red pulse halo."""
        # Items currently being carried by a drone — drawn by the agent layer
        carried_ids = {
            a.carrying.item_id for a in ctx.agents
            if a.carrying is not None
        }
        # Group visible items by zone name
        by_zone: dict[str, list] = {}
        for entry in ctx.coord.ledger.values():
            if entry.item_id in carried_ids:
                continue
            loc = entry.current_location
            if loc not in ZONES_BY_NAME:
                continue
            by_zone.setdefault(loc, []).append(entry)
        # Stable order
        for entries in by_zone.values():
            entries.sort(key=lambda e: e.item_id)

        now_ms = pygame.time.get_ticks()
        pulse = 0.5 + 0.5 * math.sin(now_ms / 320)

        for zname, entries in by_zone.items():
            z = ZONES_BY_NAME[zname]
            zcx, zcy = self.world_to_screen((z.x, z.y))
            zr_px = int(z.radius_m * PX_PER_M)
            # Drum sizing tells the "pipeline handles smaller process
            # containers" story: storage drums are the big 200L final
            # canisters; everywhere else (gen points, Char station,
            # clearance) is smaller process containers (~30-60L) that
            # carry the fragments / samples / hulls through our sorting
            # pipeline and only get repacked into 200L drums at storage.
            if z.role == "storage":
                cols = 4
                drum_w, drum_h = 18, 26
            else:
                cols = 3
                drum_w, drum_h = 13, 19
            pad = 3
            n_visible = min(len(entries), cols * 4)
            # Compute the bounding rect for the drum block so we can frame
            # it with an industrial-bin shape (open-top box) underneath.
            n_rows = max(1, math.ceil(n_visible / cols)) if n_visible else 0
            block_w = cols * (drum_w + pad) - pad
            block_h = max(drum_h, n_rows * (drum_h + pad) - pad)
            ax = zcx - block_w // 2
            base_y = zcy + zr_px - 14 if z.role == "storage" else zcy + zr_px - 8
            # Top of the block (since rows stack upward from base_y).
            top_y = base_y - (n_rows - 1) * (drum_h + pad) if n_rows else base_y
            if n_visible > 0 and z.role in ("generation", "storage", "clearance"):
                self._draw_bin_frame(
                    ax - 4, top_y - 4,
                    block_w + 8, block_h + 8,
                    accent=ROLE_OUTLINE.get(z.role, (200, 200, 200)),
                )
            for i, entry in enumerate(entries[:n_visible]):
                col = i % cols
                row = i // cols
                ay = base_y - row * (drum_h + pad)
                dx = ax + col * (drum_w + pad)
                dy = ay
                self._draw_drum(dx, dy, drum_w, drum_h, entry, pulse, now_ms)
            # If we capped, indicate "+N more" overflow
            if len(entries) > n_visible:
                extra = self.font_xs.render(f"+{len(entries) - n_visible}", True, TEXT)
                self.screen.blit(extra, (zcx + zr_px - extra.get_width() - 2,
                                          zcy - zr_px + 2))

    def _draw_bin_frame(self, x: int, y: int, w: int, h: int, accent: tuple):
        """Industrial open-top waste-bin shape framing a drum block:
        chunky left + right walls, a darker base plate, an accent-coloured
        top lip, and a drop shadow. Made to read as 'this is a physical
        container of waste at this station' from across the room — not a
        faint outline."""
        bin_col_dark = (32, 36, 44)
        bin_col_mid = (76, 82, 96)
        bin_col_light = (130, 138, 158)
        # Drop shadow under the bin
        shadow = pygame.Surface((w + 12, 10), pygame.SRCALPHA)
        pygame.draw.ellipse(shadow, (0, 0, 0, 140), (0, 0, w + 12, 10))
        self.screen.blit(shadow, (x - 6, y + h - 1))
        wall_w = 5
        # Left wall (with mid-tone face + dark outline + light inner edge)
        pygame.draw.rect(self.screen, bin_col_mid, (x - wall_w, y - 2, wall_w, h + 4))
        pygame.draw.rect(self.screen, bin_col_dark, (x - wall_w, y - 2, wall_w, h + 4), 1)
        pygame.draw.line(self.screen, bin_col_light,
                         (x - 1, y - 2), (x - 1, y + h + 2), 1)
        # Right wall
        pygame.draw.rect(self.screen, bin_col_mid, (x + w, y - 2, wall_w, h + 4))
        pygame.draw.rect(self.screen, bin_col_dark, (x + w, y - 2, wall_w, h + 4), 1)
        pygame.draw.line(self.screen, bin_col_light,
                         (x + w, y - 2), (x + w, y + h + 2), 1)
        # Base plate (chunky lip across the bottom)
        pygame.draw.rect(self.screen, bin_col_mid, (x - wall_w, y + h, w + 2 * wall_w, 4))
        pygame.draw.rect(self.screen, bin_col_dark, (x - wall_w, y + h, w + 2 * wall_w, 4), 1)
        # Top accent strip — coloured by the zone's role so green = clearance,
        # lavender = source, light grey = storage. Makes the bin identity
        # readable at a glance.
        pygame.draw.rect(self.screen, accent, (x - wall_w, y - 4, w + 2 * wall_w, 3))
        pygame.draw.rect(self.screen, bin_col_dark, (x - wall_w, y - 4, w + 2 * wall_w, 3), 1)

    def _draw_drum(self, x: int, y: int, w: int, h: int, entry, pulse: float, now_ms: int):
        """One drum icon, drawn as a cylindrical waste drum:

          - top ellipse (lid) brighter than the body
          - class-coloured body with thin dark outline
          - a horizontal label band in the middle (IAEA-style coloured band)
          - a thin bottom rim shadow

        Tricky/actinide-spiked drums get a pulsing red halo behind them and
        a tiny radiation trefoil glyph painted on the label band so the
        actinide-masquerade item is obvious on screen."""
        # Tricky halo behind the drum body — additive blend for a glow feel
        if entry.tricky:
            halo_r = int(max(w, h) * 0.7 + 3 + 3 * pulse)
            halo_alpha = int(110 + 90 * pulse)
            halo = pygame.Surface((halo_r * 2 + 2, halo_r * 2 + 2), pygame.SRCALPHA)
            pygame.draw.circle(halo, (255, 90, 90, halo_alpha),
                               (halo_r + 1, halo_r + 1), halo_r, 0)
            self.screen.blit(halo, (x + w // 2 - halo_r - 1,
                                     y + h // 2 - halo_r - 1),
                              special_flags=pygame.BLEND_RGBA_ADD)
        # Body color: predicted (post-classify) takes priority, otherwise true.
        # Pre-scan items at gen-points show in dim "unclassified" grey so the
        # transition gen-point -> scan -> coloured-drum is visible.
        cls = entry.current_classification
        if cls is None:
            body = CLASS_COLOR[None]
        else:
            body = CLASS_COLOR.get(cls, CLASS_COLOR[None])
        light = (min(body[0] + 35, 255), min(body[1] + 35, 255), min(body[2] + 35, 255))
        dark  = (max(body[0] - 50, 0),   max(body[1] - 50, 0),   max(body[2] - 50, 0))
        outline = (10, 12, 16)
        # Drum body
        pygame.draw.rect(self.screen, body, (x, y + 2, w, h - 2), border_radius=1)
        # Top lid (ellipse) sits a hair above the body for a 3D feel
        pygame.draw.ellipse(self.screen, light, (x, y - 1, w, 6))
        pygame.draw.ellipse(self.screen, outline, (x, y - 1, w, 6), 1)
        # --- Drum-contents strip: 3 thin nuclide-coloured bands stacked
        # in the lower half of the drum. Lets judges read what's actually
        # inside — a tricky drum visibly carries a purple actinide band
        # even when its gross activity / class colour says LLW.
        nuclides = entry.top_nuclides[:3]
        strip_top = y + h // 2 + 1
        strip_h = max(1, (y + h - 3) - strip_top - 1)
        if nuclides:
            band_h = max(1, strip_h // len(nuclides))
            for i, nuc in enumerate(nuclides):
                col = NUCLIDE_COLOR.get(nuc, (200, 200, 200))
                by = strip_top + i * band_h
                pygame.draw.rect(self.screen, col, (x + 1, by, w - 2, band_h))
                # Actinide bands get a thin dark border so they POP visually
                if nuc in ACTINIDE_NUCLIDES:
                    pygame.draw.rect(self.screen, (60, 0, 70),
                                     (x + 1, by, w - 2, band_h), 1)
        # Bottom shadow rim
        pygame.draw.rect(self.screen, dark, (x, y + h - 3, w, 2))
        # Outline last so it sits crisp
        pygame.draw.rect(self.screen, outline, (x, y + 2, w, h - 2), 1, border_radius=1)
        # Radiation trefoil on the lid for tricky / high-class drums
        if entry.tricky or (cls in ("ILW", "HLW")):
            tx = x + w // 2
            ty = y + 1
            trefoil_col = (40, 40, 40) if cls in ("VLLW", "LLW") else (255, 240, 100)
            pygame.draw.circle(self.screen, trefoil_col, (tx, ty), 1, 0)
            for a in (0, 2 * math.pi / 3, 4 * math.pi / 3):
                px = tx + int(2.2 * math.cos(a - math.pi / 2))
                py = ty + int(2.2 * math.sin(a - math.pi / 2))
                pygame.draw.circle(self.screen, trefoil_col, (px, py), 1, 0)
        # Scrutiny flag: small yellow chevron above the drum
        if entry.scrutiny_flag:
            cx = x + w // 2
            pygame.draw.polygon(self.screen, (255, 220, 110), [
                (cx, y - 4), (cx - 3, y - 1), (cx + 3, y - 1),
            ])

    def _draw_badge(self, x: int, y: int, text: str, color: tuple, fg=(20, 22, 28)):
        w = max(20, self.font_s.size(text)[0] + 10)
        pygame.draw.rect(self.screen, color, (x - w // 2, y - 9, w, 18), border_radius=9)
        pygame.draw.rect(self.screen, (20, 22, 28), (x - w // 2, y - 9, w, 18), 1, border_radius=9)
        t = self.font_s.render(text, True, fg)
        self.screen.blit(t, (x - t.get_width() // 2, y - t.get_height() // 2))

    def _draw_agent_chassis(self, a, cx: int, cy: int, state_color: tuple):
        """Render the agent body as a role-specific AGV chassis facing the
        agent's heading. Scanners are small with a sensor mast; Handlers
        are bigger with forklift prongs; Hybrids carry both. The chassis
        is tinted by the agent's current state (so a moving handler still
        shows its 'moving' colour) but always carries a role-coloured
        accent band so judges can tell what each drone is at a glance."""
        prof = AGENT_PROFILE.get(a.agent_type, AGENT_PROFILE["hybrid"])
        ang = a.facing_rad
        L_px = prof["length_m"] * PX_PER_M / 2.0
        W_px = prof["width_m"] * PX_PER_M / 2.0
        cos_a, sin_a = math.cos(ang), math.sin(ang)

        def rot(lx: float, ly: float) -> tuple[float, float]:
            return (cx + lx * cos_a - ly * sin_a,
                    cy + lx * sin_a + ly * cos_a)

        # Chassis corners — front is +lx (along facing direction).
        fl = rot(L_px,  -W_px)
        fr = rot(L_px,   W_px)
        br = rot(-L_px,  W_px)
        bl = rot(-L_px, -W_px)
        outline = (10, 12, 16)
        # Wheels — four small filled circles at the corners along the side
        wheel_col = (38, 42, 50)
        for fwd in (-1, 1):
            for side in (-1, 1):
                wx, wy = rot(L_px * 0.55 * fwd, W_px * 1.0 * side)
                pygame.draw.circle(self.screen, wheel_col, (int(wx), int(wy)), 2, 0)
        # Body (chassis polygon). Slightly darker on the back half to
        # suggest depth.
        body_dark = (max(state_color[0] - 30, 0),
                     max(state_color[1] - 30, 0),
                     max(state_color[2] - 30, 0))
        pygame.draw.polygon(self.screen, body_dark, [bl, br, rot(0, W_px), rot(0, -W_px)])
        pygame.draw.polygon(self.screen, state_color, [rot(0, -W_px), rot(0, W_px), fr, fl])
        pygame.draw.polygon(self.screen, outline, [fl, fr, br, bl], 1)
        # Role accent band: a thin stripe across the chassis width about
        # 30% back from the front.
        accent = prof["accent"]
        band_a = rot(L_px * 0.25, -W_px * 0.9)
        band_b = rot(L_px * 0.25,  W_px * 0.9)
        pygame.draw.line(self.screen, accent, band_a, band_b, 3)

        # Sensor mast (scanners + hybrids + qa lab): a small pole rising
        # above the chassis with a camera dome at the top.
        if prof["mast"]:
            mast_base = rot(L_px * 0.1, 0)
            mast_top  = rot(L_px * 0.1 + W_px * 1.4, 0)
            pygame.draw.line(self.screen, (210, 215, 230), mast_base, mast_top, 2)
            dome_col = (110, 200, 255)
            pygame.draw.circle(self.screen, dome_col,
                               (int(mast_top[0]), int(mast_top[1])), 3, 0)
            pygame.draw.circle(self.screen, outline,
                               (int(mast_top[0]), int(mast_top[1])), 3, 1)

        # Forklift prongs (handlers + hybrids): two short prongs extending
        # forward from the chassis front.
        if prof["fork"]:
            for side in (-1, 1):
                base = rot(L_px * 0.95, W_px * 0.5 * side)
                tip  = rot(L_px * 1.5,  W_px * 0.5 * side)
                pygame.draw.line(self.screen, (200, 200, 180), base, tip, 3)

        # Subtle headlight pair on the front of the chassis when moving
        # (suggests the cart is actively traversing).
        if a.state.startswith("MOVING_") or a.state == "ACQUIRING":
            for side in (-1, 1):
                hl = rot(L_px * 0.95, W_px * 0.6 * side)
                pygame.draw.circle(self.screen, (255, 240, 200),
                                   (int(hl[0]), int(hl[1])), 1, 0)

        # No operator icon — carts are REMOTE-CONTROLLED (teleoperated
        # from the control room), not ridden. A small radio-comm beacon
        # at the back communicates "this thing is taking commands from
        # the AI brain" — a pulsing antenna chip.
        if a.agent_type != "qa_lab" and a.state != "FAILED":
            ant_x, ant_y = rot(-L_px * 0.5, 0)
            pulse = 0.5 + 0.5 * math.sin(pygame.time.get_ticks() / 380.0
                                         + hash(a.agent_id) % 7)
            beacon_col = (110, 220, 255) if pulse > 0.5 else (60, 130, 180)
            pygame.draw.circle(self.screen, beacon_col,
                               (int(ant_x), int(ant_y) - 1), 2, 0)
            pygame.draw.circle(self.screen, outline,
                               (int(ant_x), int(ant_y) - 1), 2, 1)
            # Tiny antenna whisker rising from the beacon
            wx, wy = rot(-L_px * 0.5, 0)
            wx_tip, wy_tip = rot(-L_px * 0.5, 0)
            pygame.draw.line(self.screen, (180, 210, 230),
                             (int(wx), int(wy) - 3),
                             (int(wx_tip), int(wy_tip) - 6), 1)

        # Per-cart deployable classification arm removed: classification
        # now happens centrally at the Char Station gantry, not on board
        # the carts. Carts are dumb transports.

    def _draw_agents(self, ctx):
        # Comm lines drawn behind the agent bodies (FOV cones were removed —
        # the state pill, carry chip, and deployable classification arm
        # already convey what each AGV is doing, the cones were just clutter).
        self._draw_comm_lines(ctx)

        # Track recent positions for AGVs carrying drums, so we can draw a
        # short fading motion trail behind them in transit. Trail dots are
        # only painted for handlers/hybrids actually carrying — visually
        # selling "this drum is being moved across the facility" without
        # adding clutter for idle agents.
        if not hasattr(self, "_carry_trails"):
            self._carry_trails: dict[str, list] = {}
        for a in ctx.agents:
            if a.carrying is not None and a.state.startswith("MOVING_"):
                trail = self._carry_trails.setdefault(a.agent_id, [])
                cx_t, cy_t = self.world_to_screen(a.pos)
                if not trail or (abs(trail[-1][0] - cx_t) + abs(trail[-1][1] - cy_t)) > 6:
                    trail.append((cx_t, cy_t))
                if len(trail) > 10:
                    del trail[0:len(trail) - 10]
            else:
                self._carry_trails.pop(a.agent_id, None)
        # Render trails behind everything else
        for agent_id, trail in list(self._carry_trails.items()):
            for i, (tx, ty) in enumerate(trail[:-1]):
                alpha = int(160 * (i / max(len(trail) - 1, 1)))
                size = 2 + i // 3
                dot = pygame.Surface((size * 2 + 2, size * 2 + 2), pygame.SRCALPHA)
                pygame.draw.circle(dot, (200, 220, 255, alpha),
                                   (size + 1, size + 1), size, 0)
                self.screen.blit(dot, (tx - size - 1, ty - size - 1))

        for a in ctx.agents:
            cx, cy = self.world_to_screen(a.pos)
            color = STATE_COLOR.get(a.state, (180, 180, 200))
            radius = 11 if a.is_rescrutiny_station else 10

            # SPAWN sparkle ring: visible for 4 sim-seconds after spawn
            if a.spawn_time_s is not None and (ctx.env.now - a.spawn_time_s) < 4.0:
                t = (ctx.env.now - a.spawn_time_s) / 4.0
                ring_r = int(radius + 8 + 16 * (1 - t))
                alpha = int(220 * (1 - t))
                sp = pygame.Surface((ring_r * 2 + 4, ring_r * 2 + 4), pygame.SRCALPHA)
                pygame.draw.circle(sp, (130, 255, 170, alpha),
                                   (ring_r + 2, ring_r + 2), ring_r, 3)
                self.screen.blit(sp, (cx - ring_r - 2, cy - ring_r - 2))

            # HIVE SYNC glow: every drone glows cyan for ~1.8s after a queen
            # snapshot broadcast — the visual "all drones got the update".
            now_ms = pygame.time.get_ticks()
            if (
                not a.is_rescrutiny_station
                and a.state != "FAILED"
                and now_ms < self._hive_sync_until_ms
            ):
                remaining = (self._hive_sync_until_ms - now_ms) / 1800
                ring_r = radius + 6
                alpha = int(170 * remaining)
                sp = pygame.Surface((ring_r * 2 + 4, ring_r * 2 + 4), pygame.SRCALPHA)
                pygame.draw.circle(sp, (95, 200, 255, alpha),
                                   (ring_r + 2, ring_r + 2), ring_r, 2)
                self.screen.blit(sp, (cx - ring_r - 2, cy - ring_r - 2))

            self._draw_agent_chassis(a, cx, cy, _dim(color))

            # FAILED agents: dramatic OFFLINE indicator + countdown
            if a.state == "FAILED":
                # Gray out the body
                dim = pygame.Surface((radius * 2 + 4, radius * 2 + 4), pygame.SRCALPHA)
                pygame.draw.circle(dim, (30, 30, 35, 200),
                                   (radius + 2, radius + 2), radius, 0)
                self.screen.blit(dim, (cx - radius - 2, cy - radius - 2))
                # Big red X
                off = radius - 1
                pygame.draw.line(self.screen, (240, 90, 90),
                                 (cx - off, cy - off), (cx + off, cy + off), 3)
                pygame.draw.line(self.screen, (240, 90, 90),
                                 (cx + off, cy - off), (cx - off, cy + off), 3)
                # OFFLINE pill — hot-spare replacement is instant (0.5s),
                # so this pill flashes briefly rather than counting down.
                txt = "OFFLINE — hot-spare deploying"
                ts = self.font_xs.render(txt, True, DANGER)
                pill_w = ts.get_width() + 8
                pill_h = ts.get_height() + 4
                px = cx - pill_w // 2
                py = cy - radius - pill_h - 4
                pbg = pygame.Surface((pill_w, pill_h), pygame.SRCALPHA)
                pbg.fill((30, 10, 12, 230))
                self.screen.blit(pbg, (px, py))
                pygame.draw.rect(self.screen, DANGER, (px, py, pill_w, pill_h), 1, border_radius=3)
                self.screen.blit(ts, (px + 4, py + 2))
                # Skip the normal label pill below — the OFFLINE pill replaces it
                continue

            # Carried drum — render the full drum sprite (class color +
            # nuclide bands + trefoil) sitting on top of the chassis so
            # judges can literally watch a drum cross the screen instead
            # of a tiny chip. Position: just above the chassis, slightly
            # offset along the facing direction so it looks "loaded".
            if a.carrying is not None:
                entry = ctx.coord.ledger.get(a.carrying.item_id)
                if entry is not None:
                    # Smaller "process container" sprite for in-transit
                    # items — these aren't 200L drums, they're the
                    # fragments/samples/sealed casks that flow through
                    # the sort pipeline. 200L drums only live at storage.
                    drum_w, drum_h = 16, 22
                    # Sit the drum above the chassis. Slight bob while moving
                    # so the carry visibly reads as a held / lifted drum.
                    dx = cx - drum_w // 2
                    dy = cy - drum_h - 2
                    if a.state in ("MOVING_TO_DROPOFF", "MOVING_TO_PICKUP"):
                        dy += int(math.sin(pygame.time.get_ticks() / 160.0) * 2)
                    # Faint "lifted" shadow underneath the drum
                    shadow = pygame.Surface((drum_w + 4, 4), pygame.SRCALPHA)
                    pygame.draw.ellipse(shadow, (0, 0, 0, 120),
                                        (0, 0, drum_w + 4, 4))
                    self.screen.blit(shadow, (dx - 2, cy + 4))
                    self._draw_drum(dx, dy, drum_w, drum_h, entry,
                                    pulse=0.5 + 0.5 * math.sin(pygame.time.get_ticks() / 220),
                                    now_ms=pygame.time.get_ticks())
                    # Two short gripper "fingers" rising from the chassis
                    # forks up to the drum's lower body.
                    pygame.draw.line(self.screen, (230, 230, 240),
                                     (cx - 6, cy - 2),
                                     (cx - 6, dy + drum_h - 4), 2)
                    pygame.draw.line(self.screen, (230, 230, 240),
                                     (cx + 6, cy - 2),
                                     (cx + 6, dy + drum_h - 4), 2)

            # Label pill above the agent. While IDLE we use a *tiny* compact
            # pill (just "s1"/"h2"/"y3"/"ql1") so several idle agents at the
            # charging bay don't overlap each other; while active we use a
            # two-line pill with the full ID + state.
            if a.state == "IDLE":
                # Short ID by role prefix: scanner-N -> sN, handler-N -> hN,
                # hybrid-N -> yN, qa-lab-N -> qlN.
                prefix_map = {
                    "scanner-": "s",
                    "handler-": "h",
                    "hybrid-":  "y",
                    "qa-lab-":  "ql",
                }
                short_id = None
                for prefix, abbr in prefix_map.items():
                    if a.agent_id.startswith(prefix):
                        short_id = abbr + a.agent_id.split("-", 1)[1]
                        break
                if short_id is None:
                    short_id = a.agent_id[:4]
                id_surf = self.font_xs.render(short_id, True, HI_TEXT)
                pill_w = id_surf.get_width() + 6
                pill_h = id_surf.get_height() + 2
                pill_x = cx - pill_w // 2
                pill_y = cy - radius - pill_h - 2
                pill_bg = pygame.Surface((pill_w, pill_h), pygame.SRCALPHA)
                pill_bg.fill((12, 14, 20, 200))
                self.screen.blit(pill_bg, (pill_x, pill_y))
                pygame.draw.rect(self.screen, color, (pill_x, pill_y, pill_w, pill_h), 1, border_radius=3)
                self.screen.blit(id_surf, (pill_x + 3, pill_y + 1))
            else:
                id_surf = self.font_m.render(a.agent_id, True, HI_TEXT)
                state_text = STATE_CAPTION.get(a.state, a.state.replace("_", " ").lower())
                state_surf = self.font_xs.render(state_text, True, DIM_TEXT)
                pill_w = max(id_surf.get_width(), state_surf.get_width()) + 10
                pill_h = id_surf.get_height() + state_surf.get_height() + 6
                pill_x = cx - pill_w // 2
                pill_y = cy - radius - pill_h - 4
                pill_bg = pygame.Surface((pill_w, pill_h), pygame.SRCALPHA)
                pill_bg.fill((12, 14, 20, 220))
                self.screen.blit(pill_bg, (pill_x, pill_y))
                pygame.draw.rect(self.screen, color, (pill_x, pill_y, pill_w, pill_h), 1, border_radius=3)
                self.screen.blit(id_surf,    (pill_x + pill_w // 2 - id_surf.get_width() // 2,    pill_y + 2))
                self.screen.blit(state_surf, (pill_x + pill_w // 2 - state_surf.get_width() // 2, pill_y + 2 + id_surf.get_height()))

            # Highlight an injected item briefly with a pulsing ring
            if (
                self._inject_flash_item
                and a.carrying is not None
                and a.carrying.item_id == self._inject_flash_item
                and pygame.time.get_ticks() < self._inject_flash_until_ms
            ):
                pulse = radius + 5 + int(4 * abs((pygame.time.get_ticks() % 400) - 200) / 200)
                pygame.draw.circle(self.screen, DANGER, (cx, cy), pulse, 2)

            # Battery + dose bars stacked under the body
            if not a.is_rescrutiny_station:
                self._draw_health_bars(cx, cy + radius + 4, a)

            # Live sensor widgets when the agent is mid-classification.
            # ACQUIRING shows the gamma + camera streams; CLASSIFYING shows
            # a "thinking" pulse — these make it visually obvious what the
            # robot is doing on board.
            if a.state == "ACQUIRING":
                self._draw_sensor_widget(cx, cy, a)
                self._draw_classification_callout(a, cx, cy, "Scanning NaI gamma…")
            elif a.state == "CLASSIFYING":
                self._draw_thinking_pulse(cx, cy)
                self._draw_classification_callout(a, cx, cy, "Matching photopeaks…")
            elif a.state == "REPORTING":
                self._draw_classification_callout(a, cx, cy, "Reporting → coordinator")
            elif a.state == "RESCANNING":
                self._draw_sensor_widget(cx, cy, a, hpge=True)
                self._draw_classification_callout(a, cx, cy,
                                                  "HPGe 60s integration", hpge=True)

    def _draw_classification_callout(self, a, cx: int, cy: int,
                                     caption: str, hpge: bool = False):
        """Floating plain-English callout under an AGV during the
        scan/classify/report phases. Tells judges *what* the agent is
        doing in a sentence, so the visual is self-narrating.

        On classifiers (mobile + QA-lab) we also surface the specific
        item-id being processed, since that links the on-screen agent
        to the row in the spectrum panel + decision feed."""
        item_id = None
        if a.is_rescrutiny_station and a.carrying is not None:
            item_id = a.carrying.item_id
        elif getattr(a, "scanning_item", None) is not None:
            item_id = a.scanning_item.item_id
        accent = (140, 220, 255) if hpge else (200, 230, 255)
        # Compose the callout text
        text = caption
        if item_id:
            text = f"{caption}  ({item_id})"
        text_surf = self.font_xs.render(text, True, accent)
        pad_x = 5
        pad_y = 2
        bw = text_surf.get_width() + 2 * pad_x
        bh = text_surf.get_height() + 2 * pad_y
        bx = cx - bw // 2
        by = cy + 28  # below the AGV body, clear of health bars
        # Translucent background
        bg = pygame.Surface((bw, bh), pygame.SRCALPHA)
        bg.fill((12, 14, 20, 220))
        self.screen.blit(bg, (bx, by))
        pygame.draw.rect(self.screen, accent, (bx, by, bw, bh), 1, border_radius=2)
        self.screen.blit(text_surf, (bx + pad_x, by + pad_y))

    def _draw_sensor_widget(self, cx: int, cy: int, a, hpge: bool = False):
        """Mini sensor panel rendered next to the agent during ACQUIRING.

        It shows a *very stylized* spectrum (vertical bars at the photopeak
        windows we actually use as features) plus a camera-eye icon for the
        CV stream. The widget is decorative — the actual gamma spectrum is
        simulated in sensors.py and used by the classifier — but it makes
        on-board sensing visible. Bars wobble per frame so the user reads
        it as "live data streaming in" rather than a static image."""
        wx = cx + 28
        wy = cy - 20
        w, h = 60, 36
        bg = pygame.Surface((w, h), pygame.SRCALPHA)
        bg.fill((12, 14, 20, 220))
        self.screen.blit(bg, (wx, wy))
        det_color = (110, 200, 255) if hpge else (255, 180, 90)
        pygame.draw.rect(self.screen, det_color, (wx, wy, w, h), 1, border_radius=3)
        label = self.font_xs.render("HPGe" if hpge else "NaI", True, det_color)
        self.screen.blit(label, (wx + 3, wy + 1))
        # Photopeak bars — pseudo-random heights based on agent id + frame
        seed = (hash(a.agent_id) + pygame.time.get_ticks() // 90) & 0xFFFF
        rng_state = seed
        n_bars = 8
        bar_w = (w - 8) // n_bars
        base_y = wy + h - 2
        for i in range(n_bars):
            rng_state = (rng_state * 1103515245 + 12345) & 0x7FFFFFFF
            r = (rng_state % 100) / 100.0
            bh = int(2 + (h - 14) * (0.25 + 0.75 * r))
            x = wx + 4 + i * bar_w
            col = det_color if i not in (3, 6) else (220, 100, 220)
            pygame.draw.rect(self.screen, col,
                             (x, base_y - bh, max(bar_w - 1, 1), bh))
        # Tiny camera-eye on the right edge to suggest CV concurrently active
        if not hpge:
            ex = wx + w + 4
            ey = wy + h // 2
            pygame.draw.circle(self.screen, (200, 200, 220), (ex, ey), 5, 1)
            pygame.draw.circle(self.screen, (200, 200, 220), (ex, ey), 2, 0)
            blink = (pygame.time.get_ticks() // 250) % 4 == 0
            if blink:
                pygame.draw.line(self.screen, (200, 200, 220),
                                 (ex - 5, ey), (ex + 5, ey), 1)

    def _draw_spectrum_panel(self, ctx, x: int, y: int, w: int) -> int:
        """Big NaI spectrum chart in the right-side panel. Shows the most
        recent classifier-eye view: spectrum bars (bucketed to ~64), the
        photopeak windows the classifier reads as features (highlighted),
        the actinide windows (purple, larger highlight), and a bottom
        line with the verdict + confidence + actinide signature.

        This is the visual centrepiece of 'what the AI is doing': judges
        can literally see the spike-window counts the classifier reads."""
        ev = ctx.coord.last_scan_event
        chart_h = 60
        if ev is None or "spectrum" not in ev:
            # Empty placeholder
            pygame.draw.rect(self.screen, (16, 18, 22), (x, y, w, chart_h))
            pygame.draw.rect(self.screen, GRID, (x, y, w, chart_h), 1)
            t = self.font_xs.render("waiting for first scan…", True, DIM_TEXT)
            self.screen.blit(t, (x + 6, y + chart_h // 2 - t.get_height() // 2))
            return y + chart_h + 4
        from sim.sensors import CHANNEL_CENTERS, N_CHANNELS
        from sim.classifier import PHOTOPEAK_WINDOWS, ACTINIDE_WINDOW_NAMES
        spectrum = ev["spectrum"]
        # Bucket 1024 channels into ~ chart-width bars
        n_bars = min(int(w * 0.9), 96)
        bucket = max(1, N_CHANNELS // n_bars)
        bars = []
        e_keV = []
        for i in range(n_bars):
            lo = i * bucket
            hi = min(lo + bucket, N_CHANNELS)
            bars.append(float(spectrum[lo:hi].sum()))
            e_keV.append(float(CHANNEL_CENTERS[lo:hi].mean()))
        peak = max(bars + [1.0])
        # Background
        pygame.draw.rect(self.screen, (12, 14, 20), (x, y, w, chart_h))
        pygame.draw.rect(self.screen, (60, 65, 75), (x, y, w, chart_h), 1)
        # Highlight photopeak windows (faint behind bars)
        e_min, e_max = 0.0, float(CHANNEL_CENTERS[-1])
        def e_to_x(e_keV_val: float) -> int:
            return int(x + (e_keV_val - e_min) / max(e_max - e_min, 1e-6) * w)
        for name, (lo_keV, hi_keV) in PHOTOPEAK_WINDOWS.items():
            xa = e_to_x(lo_keV)
            xb = e_to_x(hi_keV)
            if name in ACTINIDE_WINDOW_NAMES:
                # Actinide windows: stronger purple highlight
                tint = pygame.Surface((max(xb - xa, 1), chart_h), pygame.SRCALPHA)
                tint.fill((200, 120, 230, 55))
                self.screen.blit(tint, (xa, y))
                pygame.draw.line(self.screen, (200, 120, 230, 255),
                                 (xa, y), (xa, y + chart_h), 1)
            else:
                tint = pygame.Surface((max(xb - xa, 1), chart_h), pygame.SRCALPHA)
                tint.fill((110, 200, 255, 24))
                self.screen.blit(tint, (xa, y))
        # Bars
        bar_w = max(1, w // n_bars)
        for i, v in enumerate(bars):
            bh = int((v / peak) * (chart_h - 4))
            bx = x + i * bar_w
            by = y + chart_h - bh - 1
            # Bar colour: brighter in actinide windows
            in_actinide = any(
                e_keV[i] >= lo and e_keV[i] < hi
                for n, (lo, hi) in PHOTOPEAK_WINDOWS.items() if n in ACTINIDE_WINDOW_NAMES
            )
            col = (200, 120, 230) if in_actinide else (140, 200, 240)
            pygame.draw.rect(self.screen, col, (bx, by, max(bar_w - 1, 1), bh))
        # Verdict line below the chart
        ly = y + chart_h + 3
        verdict_class = ev["predicted_class"]
        verdict_col = CLASS_COLOR.get(verdict_class, TEXT)
        cls_surf = self.font_m.render(verdict_class, True, verdict_col)
        conf_surf = self.font_xs.render(f"@ {ev['confidence']:.2f}", True, DIM_TEXT)
        sig_surf = self.font_xs.render(
            f"act sig {ev['actinide_signature']:.4f}", True, (200, 120, 230)
        )
        agent_surf = self.font_xs.render(
            f"{ev['agent_id']} → {ev['item_id']}", True, DIM_TEXT,
        )
        self.screen.blit(cls_surf, (x, ly))
        self.screen.blit(conf_surf, (x + cls_surf.get_width() + 6, ly + 3))
        self.screen.blit(sig_surf,  (x + cls_surf.get_width() + 70, ly + 3))
        self.screen.blit(agent_surf, (x, ly + cls_surf.get_height() + 2))
        # Truth annotation (god-view, helps demo): a small "actual:" tag if
        # the verdict was wrong, or "✓" if it agrees.
        true_class = ev.get("true_class")
        if true_class and true_class != verdict_class:
            warn = self.font_xs.render(
                f"actual {true_class}{' (tricky)' if ev.get('tricky') else ''}",
                True, DANGER,
            )
            self.screen.blit(warn, (x + w - warn.get_width(), ly + 3))
        elif true_class:
            ok = self.font_xs.render("✓", True, OK)
            self.screen.blit(ok, (x + w - ok.get_width() - 2, ly + 3))
        return ly + cls_surf.get_height() + self.font_xs.get_linesize() + 4

    def _draw_decision_feed(self, ctx, x: int, y: int, w: int, max_lines: int) -> int:
        """Recent coordinator decisions: verdicts / scrutiny flags / retrains /
        dispatches. Colour-coded by kind so the AI's reasoning is readable."""
        feed = list(ctx.coord.decision_log)[-max_lines:][::-1]
        if not feed:
            t = self.font_xs.render("no decisions yet", True, DIM_TEXT)
            self.screen.blit(t, (x, y))
            return y + 14
        kind_col = {
            "verdict":        TEXT,
            "scrutiny":       WARN,
            "retrain":        ACCENT,
            "rescan_verdict": (200, 120, 230),
            "dispatch":       (180, 220, 130),
        }
        for d in feed:
            col = kind_col.get(d.get("kind"), TEXT)
            text = d.get("text", "")
            # Truncate to fit
            while self.font_xs.size(text)[0] > w and len(text) > 4:
                text = text[:-2]
            s = self.font_xs.render(text, True, col)
            self.screen.blit(s, (x, y))
            y += 13
        return y + 2

    def _draw_thinking_pulse(self, cx: int, cy: int):
        """Three dots pulsing in sequence, drawn above the agent body —
        the universal 'thinking' affordance."""
        t = pygame.time.get_ticks()
        for i in range(3):
            phase = (t / 200 + i) % 3
            scale = 0.4 + 0.6 * max(0.0, 1.0 - abs(phase - 1.5))
            size = int(2 + 3 * scale)
            x = cx + (i - 1) * 9
            y = cy - 30
            pygame.draw.circle(self.screen, (200, 130, 220), (x, y), size, 0)

    def _draw_health_bars(self, cx: int, cy: int, a):
        """Two thin bars: battery (green→amber→red) and integrated dose
        (transparent → red). Stacked vertically under the agent body."""
        bar_w = 28
        bar_h = 3
        # Battery
        battery = max(0.0, min(a.battery_pct, 100.0)) / 100.0
        bcol = OK if battery > 0.5 else (WARN if battery > 0.2 else DANGER)
        pygame.draw.rect(self.screen, (35, 38, 45),
                         (cx - bar_w // 2 - 1, cy - 1, bar_w + 2, bar_h + 2))
        pygame.draw.rect(self.screen, bcol,
                         (cx - bar_w // 2, cy, int(bar_w * battery), bar_h))
        # Dose (clamp to failure threshold for the bar fill)
        from sim.agent import DOSE_FAILURE_THRESHOLD_uSv, DOSE_DECON_THRESHOLD_uSv
        dose_frac = min(a.integrated_dose_uSv / DOSE_FAILURE_THRESHOLD_uSv, 1.0)
        dcol = (
            DANGER if a.integrated_dose_uSv > DOSE_DECON_THRESHOLD_uSv
            else WARN if a.integrated_dose_uSv > DOSE_DECON_THRESHOLD_uSv * 0.5
            else (130, 200, 130)
        )
        cy2 = cy + bar_h + 2
        pygame.draw.rect(self.screen, (35, 38, 45),
                         (cx - bar_w // 2 - 1, cy2 - 1, bar_w + 2, bar_h + 2))
        pygame.draw.rect(self.screen, dcol,
                         (cx - bar_w // 2, cy2, int(bar_w * dose_frac), bar_h))

    def _draw_panel(self, ctx):
        x0 = FACILITY_W
        # Panel background
        pygame.draw.rect(self.screen, PANEL_BG, (x0, HEADER_H, PANEL_W, FACILITY_H))
        pygame.draw.line(self.screen, (60, 65, 75), (x0, HEADER_H), (x0, HEADER_H + FACILITY_H), 1)

        # --- 1) Coordinator section (condensed) -----------------------------
        flash = pygame.time.get_ticks() < self._retrain_flash_until_ms
        head_color = WARN if flash else ACCENT
        y = HEADER_H + 10
        head = self.font_l.render("◆ COORDINATOR", True, head_color)
        self.screen.blit(head, (x0 + 14, y)); y += 22

        thr = ctx.coord.shared_classifier.actinide_threshold
        thr_txt = "—" if thr is None else f"{thr:.4f}"
        thr_color = OK if thr is not None else DIM_TEXT
        activity = ctx.coord.activity_status
        if pygame.time.get_ticks() > getattr(self, "_coord_status_until_ms", 0):
            if any(c["kind"] in ("report", "snapshot") and c["expires_ms"] > pygame.time.get_ticks()
                   for c in self._active_comms):
                pass
            else:
                activity = "idle"
        else:
            self._coord_status_until_ms = pygame.time.get_ticks() + COORD_STATUS_TTL_MS
        activity_color = WARN if activity not in ("idle", "") else DIM_TEXT
        mobile = [a for a in ctx.agents if not a.is_rescrutiny_station]
        connected = sum(1 for a in mobile if a.state != "FAILED")
        pending = len(ctx.coord.pending_tasks)
        from collections import Counter
        role_counts = Counter(a.agent_type for a in mobile)
        ROLE_ABBR = {"scanner": "sc", "handler": "hd", "hybrid": "hy"}
        role_summary = " / ".join(
            f"{n}{ROLE_ABBR.get(t, t[:2])}"
            for t, n in sorted(role_counts.items()) if n > 0
        ) or "—"
        m = ctx.metrics
        lines = [
            ("mode",          self.mode_label,                              TEXT),
            ("activity",      activity,                                      activity_color),
            ("agents",        f"{connected}/{len(mobile)} ({role_summary})",
                              OK if connected == len(mobile) else WARN),
            ("model",         f"v{ctx.coord.model_version}  retrains={len(ctx.coord.retrain_events)}", TEXT),
            ("actinide thr",  thr_txt,                                       thr_color),
            ("accuracy",      f"{m.accuracy()*100:.1f}%",                    OK if m.accuracy() > 0.9 else WARN),
            ("classified / done",
                              f"{len(m.classifications)} / {m.items_completed}", TEXT),
            ("released (clearance)",
                              f"{m.items_released}", OK if m.items_released > 0 else DIM_TEXT),
            ("cum. worker dose", f"{m.cumulative_dose_uSv:.1f} µSv",         DANGER if m.cumulative_dose_uSv > 50000 else TEXT),
            ("pending / rescan", f"{pending} / {len(ctx.rescan_queue.items)}", TEXT),
            ("dropped pkts",  str(ctx.coord.dropped_reports),
                              DIM_TEXT if ctx.coord.dropped_reports == 0 else WARN),
        ]
        max_val_w = PANEL_W - 170 - 18
        for k, v, c in lines:
            kt = self.font_s.render(k, True, DIM_TEXT)
            # Truncate the value if it would overflow the panel
            while self.font_s.size(v)[0] > max_val_w and len(v) > 4:
                v = v[:-2]
            vt = self.font_s.render(v, True, c)
            self.screen.blit(kt, (x0 + 14, y))
            self.screen.blit(vt, (x0 + 170, y))
            y += 14

        # --- Redundancy snapshot — every arm pair's current status -------
        y += 4
        head = self.font_l.render("◆ REDUNDANCY", True, ACCENT)
        self.screen.blit(head, (x0 + 14, y)); y += 18
        arm_pairs = getattr(ctx, "arm_pairs", None) or {}
        coord_failovers = getattr(ctx.coord, "failover_count", 0)
        # Coordinator entry first so it reads as the top of the redundancy
        # story.
        coord_status = "primary" if coord_failovers % 2 == 0 else "standby promoted"
        coord_col = OK if coord_failovers == 0 else WARN
        kt = self.font_s.render("Coordinator", True, DIM_TEXT)
        vt = self.font_s.render(f"{coord_status}  ({coord_failovers} failovers)",
                                 True, coord_col)
        self.screen.blit(kt, (x0 + 14, y))
        self.screen.blit(vt, (x0 + 130, y))
        y += 14
        total_arm_failovers = 0
        any_down = False
        for key, pair in arm_pairs.items():
            total_arm_failovers += pair.failover_count
            active = pair.active_idx()
            if active is None:
                status = "BOTH DOWN"
                col = DANGER
                any_down = True
            elif pair.failed_units:
                # One unit down, standby is leading.
                status = f"{pair.units[active]} active (1 in repair)"
                col = WARN
            else:
                status = f"{pair.units[active]} active"
                col = OK
            kt = self.font_s.render(pair.name, True, DIM_TEXT)
            vt = self.font_s.render(status, True, col)
            # Truncate value if it overflows
            max_w = PANEL_W - 130 - 18
            while self.font_s.size(status)[0] > max_w and len(status) > 4:
                status = status[:-2]
                vt = self.font_s.render(status, True, col)
            self.screen.blit(kt, (x0 + 14, y))
            self.screen.blit(vt, (x0 + 130, y))
            y += 14

        # Total integrated downtime — sum of `total_downtime_s` across
        # all pairs, plus any currently-open BOTH-DOWN windows. Stays at
        # 0 s during runs where the standby always took over in time;
        # ticks up only on the rare "both A and B down at once" event.
        total_downtime = 0.0
        for pair in arm_pairs.values():
            total_downtime += pair.total_downtime_s
            if pair.both_down_since_s is not None:
                total_downtime += max(0.0, ctx.env.now - pair.both_down_since_s)
        if any_down:
            downtime_txt = f"{total_downtime:.1f} s — DEGRADED"
            downtime_col = DANGER
        elif total_downtime <= 0.05:
            downtime_txt = "0 s (auto-failover)"
            downtime_col = OK
        else:
            downtime_txt = f"{total_downtime:.1f} s total"
            downtime_col = WARN
        kt = self.font_s.render("arm downtime", True, DIM_TEXT)
        vt = self.font_s.render(downtime_txt, True, downtime_col)
        self.screen.blit(kt, (x0 + 14, y))
        self.screen.blit(vt, (x0 + 130, y))
        y += 14
        kt = self.font_s.render("absorbed failures", True, DIM_TEXT)
        vt = self.font_s.render(str(total_arm_failovers), True, TEXT)
        self.screen.blit(kt, (x0 + 14, y))
        self.screen.blit(vt, (x0 + 130, y))
        y += 14

        # --- 2) Live spectrum panel — "what the AI is looking at" ---------
        y += 8
        head = self.font_l.render("◆ LIVE SPECTRUM (NaI)", True, ACCENT)
        self.screen.blit(head, (x0 + 14, y)); y += 20
        y = self._draw_spectrum_panel(ctx, x0 + 14, y, PANEL_W - 28)

        # --- 3) Decision feed — "what the AI is deciding" -----------------
        y += 8
        head = self.font_l.render("◆ DECISIONS", True, ACCENT)
        self.screen.blit(head, (x0 + 14, y)); y += 18
        y = self._draw_decision_feed(ctx, x0 + 14, y, PANEL_W - 28, max_lines=6)

        # Fleet health section — per-agent battery + integrated dose
        y += 10
        head = self.font_l.render("◆ FLEET HEALTH", True, ACCENT)
        self.screen.blit(head, (x0 + 14, y)); y += 20
        from sim.agent import DOSE_DECON_THRESHOLD_uSv, DOSE_FAILURE_THRESHOLD_uSv
        for a in ctx.agents:
            if a.is_rescrutiny_station:
                continue
            id_surf = self.font_xs.render(a.agent_id, True, TEXT)
            self.screen.blit(id_surf, (x0 + 14, y))
            # Battery bar — shifted right to make room for role-prefixed
            # IDs (scanner-1 / handler-1 are wider than the legacy drone-1)
            bx = x0 + 90
            bw = 55
            battery = max(0.0, min(a.battery_pct, 100.0)) / 100.0
            bcol = OK if battery > 0.5 else (WARN if battery > 0.2 else DANGER)
            pygame.draw.rect(self.screen, GRID, (bx, y + 3, bw, 5))
            pygame.draw.rect(self.screen, bcol, (bx, y + 3, int(bw * battery), 5))
            # Dose bar next to battery
            dx = bx + bw + 10
            dw = 55
            dose_frac = min(a.integrated_dose_uSv / DOSE_FAILURE_THRESHOLD_uSv, 1.0)
            dose_col = (
                DANGER if a.integrated_dose_uSv > DOSE_DECON_THRESHOLD_uSv
                else WARN if a.integrated_dose_uSv > DOSE_DECON_THRESHOLD_uSv * 0.5
                else (130, 200, 130)
            )
            pygame.draw.rect(self.screen, GRID, (dx, y + 3, dw, 5))
            pygame.draw.rect(self.screen, dose_col, (dx, y + 3, int(dw * dose_frac), 5))
            # Optional status flag
            flag = ""
            if a.state == "FAILED":           flag = " FAILED"
            elif a.state == "CHARGING":       flag = " chg"
            elif a.state == "DECONNING":      flag = " decon"
            if flag:
                fs = self.font_xs.render(flag, True, dose_col if flag != " chg" else ACCENT)
                self.screen.blit(fs, (dx + dw + 4, y - 1))
            y += 12

        # Event log section
        y += 10
        head = self.font_l.render("◆ EVENT LOG", True, ACCENT)
        self.screen.blit(head, (x0 + 14, y)); y += 22
        # Render events bottom-up so newest is at top of the visible block
        log_lines = list(self.events)
        # Trim to fit
        available_lines = (HEADER_H + FACILITY_H - y - 12) // 15
        log_lines = log_lines[-available_lines:]
        for ev in log_lines:
            tstr = f"{ev.t_sim_s/60:6.1f}m  "
            tt = self.font_xs.render(tstr, True, DIM_TEXT)
            self.screen.blit(tt, (x0 + 14, y))
            # Truncate text to fit
            max_w = PANEL_W - 75
            text = ev.text
            while self.font_xs.size(text)[0] > max_w and len(text) > 4:
                text = text[:-2]
            mt = self.font_xs.render(text, True, ev.color)
            self.screen.blit(mt, (x0 + 14 + tt.get_width(), y))
            y += 15

    def _draw_footer(self, ctx):
        y0 = HEADER_H + FACILITY_H
        pygame.draw.rect(self.screen, (18, 20, 26), (0, y0, WINDOW_W, FOOTER_H))
        pygame.draw.line(self.screen, (60, 65, 75), (0, y0), (WINDOW_W, y0), 1)

        # Progress bar
        progress = min(ctx.env.now / max(ctx.sim_duration_s, 1.0), 1.0)
        bar_y = y0 + 12
        pygame.draw.rect(self.screen, GRID, (16, bar_y, WINDOW_W - 32, 8), border_radius=4)
        pygame.draw.rect(self.screen, ACCENT,
                         (16, bar_y, int((WINDOW_W - 32) * progress), 8), border_radius=4)
        prog_txt = self.font_xs.render(
            f"shift progress: {progress*100:.0f}%   ({ctx.env.now/3600:.2f}h / {ctx.sim_duration_s/3600:.1f}h)",
            True, DIM_TEXT,
        )
        self.screen.blit(prog_txt, (16, bar_y - 14))

        # Legend (left half of footer-below-bar)
        ly = bar_y + 18
        items = [
            ("VLLW", CLASS_COLOR["VLLW"]),
            ("LLW",  CLASS_COLOR["LLW"]),
            ("ILW",  CLASS_COLOR["ILW"]),
            ("HLW",  CLASS_COLOR["HLW"]),
            ("Scanner", AGENT_PROFILE["scanner"]["accent"]),
            ("Handler", AGENT_PROFILE["handler"]["accent"]),
            ("Hybrid",  AGENT_PROFILE["hybrid"]["accent"]),
            ("moving",        STATE_COLOR["MOVING_TO_PICKUP"]),
            ("acquiring",     STATE_COLOR["ACQUIRING"]),
            ("rescan (HPGe)", STATE_COLOR["RESCANNING"]),
        ]
        x = 16
        for label, col in items:
            pygame.draw.rect(self.screen, col, (x, ly + 3, 12, 12))
            pygame.draw.rect(self.screen, (10, 12, 16), (x, ly + 3, 12, 12), 1)
            t = self.font_xs.render(label, True, TEXT)
            self.screen.blit(t, (x + 18, ly + 2))
            x += 18 + t.get_width() + 14

        # Controls (right-aligned), two lines. Keys are listed explicitly
        # so e.g. "[" and "]" can't be misread as a single "/" key.
        ctrl_lines = [
            "SPACE: pause   +/-: speed   T: tricky   R: retrain   1-5: kill cart   K: kill coord   M: switch mode",
            "F: cycle fleet preset   [: shrink fleet   ]: grow fleet   X: apply fleet/restart   H: help   Q: quit",
        ]
        for i, line in enumerate(ctrl_lines):
            t = self.font_s.render(line, True, DIM_TEXT)
            self.screen.blit(t, (WINDOW_W - t.get_width() - 16, ly + 22 + i * 16))
