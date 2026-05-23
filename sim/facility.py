"""2D facility layout. Zones in meters; agents move with direct-line travel
at a fixed speed. Pathfinding sophistication is not the point of this sim —
what matters is that movement takes time, agents can be seen approaching
zones, and proximity-to-worker can be computed for the dose model.

The layout has three vertical bands:

  TOP (y ~ 2-5 m)  — upstream PUREX reprocessing chain:
                    Spent Fuel Pool -> Shearing -> Dissolution ->
                    Solvent Extraction -> HLW Concentration -> Solidification.
                    The arrows between them show the real PUREX material flow.
                    The Spent Fuel Pool, Shearing, Dissolution and HLW
                    Concentration cells each shed a waste stream that carts
                    ferry to the Classifier; Solvent Extraction is a liquid
                    process step and Solidification a transit step for HLW.
  MIDDLE (y ~ 7-15 m) — robot working area: Charging bay, Worker area,
                    Quick scan, QA lab, Robot wash, Coordinator pillar.
  RIGHT (x > 42 m) — storage stacks: VLLW / LLW / ILW / HLW.
  LEFT (x ~ 3 m, mid-bottom) — legacy contact-handling sources
                    (Cleanup ops / Maintenance / Plant samples).
"""
from __future__ import annotations
from dataclasses import dataclass, field

# Facility footprint, in metres. Widened from the original 36x23 to make room
# for the upstream PUREX chain across the top.
WIDTH_M = 50.0
HEIGHT_M = 25.0

AGENT_SPEED_MPS = 0.9  # ~0.9 m/s — slightly faster so trips across the wider
                       # layout still feel responsive at default sim speed.


@dataclass(frozen=True)
class Zone:
    name: str
    x: float            # centre
    y: float
    radius_m: float
    role: str           # generation | sorting | drum_scanner | storage | worker | charging | decon | process | solidification | coordinator
    color: tuple        # RGB tuple for pygame
    storage_class: str | None = None   # for storage zones: which IAEA class
    short: str = ""      # abbreviation rendered inside the zone circle
    # For "process" / generation zones, what waste class they predominantly
    # produce. None means "mixed" (the original abstract gen points).
    produces_class: str | None = None


ZONES: list[Zone] = [
    # --- Upstream PUREX reprocessing chain (top row) ---
    # Each upstream stage that *produces* waste is also a generation point
    # for the robot fleet. Solidification is a transit step (HLW only).
    # The Spent Fuel Pool is the first stop after receipt — water-cooled
    # storage where assemblies cool for ~5 years before being lifted to
    # Shearing by the overhead Fuel Handling Machine. It is ALSO a waste
    # source: pool filters, sludge, and contaminated tools are shed as a
    # mixed contact-waste stream that carts ferry to the Classifier (see
    # drum_source_zones()), so the plant's visual origin is also where the
    # sorting pipeline starts.
    Zone("Spent fuel pool",     4.0,  3.0, 2.0, "fuel_pool",     (30, 90, 140),   short="POOL"),
    Zone("Shearing cell",      10.0,  3.0, 1.3, "generation",    (200, 180, 160), short="SHEAR", produces_class="LLW"),
    Zone("Dissolution cell",   14.5,  3.0, 1.3, "generation",    (220, 160, 160), short="DISS",  produces_class="HLW"),
    Zone("Solvent extraction", 19.0,  3.0, 1.3, "process",       (190, 180, 220), short="SOLV"),
    Zone("HLW concentration",  24.0,  3.0, 1.3, "generation",    (210, 140, 140), short="HLWC",  produces_class="HLW"),
    Zone("Solidification",     29.0,  3.0, 1.4, "solidification",(150, 170, 200), short="VITR"),
    # Coordinator queen — central decision node, between upstream and the
    # storage stacks so dispatch arrows fan out clearly. This is the AI
    # brain of the facility: twin redundant classifiers, the data warehouse
    # of NaI features + HPGe labels, and the broadcaster of model snapshots
    # to the fleet. Wider than the other rooms so its three internal
    # compartments (CLASSIFIER / KNOWLEDGE BASE / RESCAN LOOP) read clearly.
    Zone("Coordinator",        38.0,  4.2, 4.0, "coordinator",   (95, 200, 255),  short="AI BRAIN"),

    # --- Mid-row: robot working area ---
    Zone("Charging bay",        6.0,  9.0, 2.6, "charging",     (90, 100, 130),   short="CHG"),
    # Classifier (Drum Characterisation Station) — the heart of the
    # classification step. Process containers/fragments land on a
    # turntable here so the overhead NaI + CV gantry can read every
    # face. Carts stage drums in, the gantry classifies, carts route
    # them out. Renamed from "Char station" for obviousness on screen.
    Zone("Classifier",         18.0,  8.5, 2.4, "char_station", (160, 200, 230),  short="CLASSIFY"),
    # Workers don't stand inside the radwaste sorting hall — they sit in
    # the control room at the side of the plant and teleoperate. Worker
    # zone moved off the cart corridor.
    Zone("Control room",        3.5, 13.0, 1.2, "worker",       (255, 200, 80),   short="CTRL"),
    # QA lab fits TWO redundant HPGe stations side-by-side (HPGe-A primary +
    # HPGe-B standby). Slightly wider radius so both rigs render cleanly.
    Zone("QA lab",             27.0, 11.0, 2.4, "drum_scanner", (120, 180, 220),  short="QA"),
    Zone("Robot wash",         37.5, 12.5, 1.5, "decon",        (255, 130, 130),  short="WASH"),

    # --- Legacy contact-handling sources (left column, lower half) ---
    # These remain so the original tricky-actinide demo keeps working: each
    # still produces a statistical VLLW/LLW/ILW mix with a configurable
    # tricky-fraction.
    # Legacy contact-handling sources — aligned in a single row at y ≈
    # 18, evenly spaced inside the LEGACY WASTE wing. Earlier layout had
    # them in an L-shape which read as cluttered; the row reads tidier
    # and matches the row pattern of the upstream PUREX chain across the
    # top.
    Zone("Cleanup ops",         3.0, 18.0, 1.3, "generation",   (160, 160, 200), short="CLEAN"),
    Zone("Maintenance",         7.0, 18.0, 1.3, "generation",   (160, 160, 200), short="MAINT"),
    Zone("Plant samples",      11.0, 18.0, 1.3, "generation",   (160, 160, 200), short="SAMPL"),

    # --- Storage stacks (right edge) ---
    # Five-tall column on x=46: CLEAR sits at the top (the cleanest
    # release path), then VLLW / LLW / ILW / HLW descending. Even
    # vertical spacing of ~4 m so the stack reads as one ordered
    # column of containment levels.
    Zone("Free release", 46.0,  3.0, 0.85, "clearance", (160, 220, 170), short="CLEAR"),
    Zone("VLLW storage", 46.0,  7.0, 1.4,  "storage", (200, 230, 200), storage_class="VLLW", short="VLLW"),
    Zone("LLW storage",  46.0, 11.0, 1.4,  "storage", (180, 220, 160), storage_class="LLW",  short="LLW"),
    Zone("ILW storage",  46.0, 15.0, 1.4,  "storage", (140, 180, 120), storage_class="ILW",  short="ILW"),
    Zone("HLW storage",  46.0, 19.5, 1.6,  "storage", (180, 130, 130), storage_class="HLW",  short="HLW"),
]

ZONES_BY_NAME = {z.name: z for z in ZONES}
ZONES_BY_ROLE: dict[str, list[Zone]] = {}
for _z in ZONES:
    ZONES_BY_ROLE.setdefault(_z.role, []).append(_z)


def distance_m(a: tuple[float, float], b: tuple[float, float]) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def travel_time_s(a: tuple[float, float], b: tuple[float, float]) -> float:
    return distance_m(a, b) / AGENT_SPEED_MPS


def storage_zone_for(iaea_class: str) -> Zone:
    for z in ZONES_BY_ROLE.get("storage", []):
        if z.storage_class == iaea_class:
            return z
    raise KeyError(f"No storage for class {iaea_class}")


def generation_points() -> list[Zone]:
    return list(ZONES_BY_ROLE.get("generation", []))


# The Spent Fuel Pool is the visual origin of the plant, so it must also be a
# real source of sorting work: pool-water filters, sludge, and contaminated
# handling tools are shed as a mixed contact-waste stream that carts ferry to
# the central Classifier (exactly like the Cleanup/Maintenance/Sample sources).
# Kept as its own zone role ("fuel_pool") so its underwater interior still
# renders; this list is what makes it a drum source for routing + the flow
# arrows.
POOL_SOURCE_NAME = "Spent fuel pool"


def drum_source_zones() -> list[Zone]:
    """Every zone that emits waste drums into the sorting pipeline (i.e. feeds
    the central Classifier): the generation-role cells plus the Spent Fuel
    Pool. Used by the live view's flow arrows and by the scenario builder."""
    out = list(ZONES_BY_ROLE.get("generation", []))
    pool = ZONES_BY_NAME.get(POOL_SOURCE_NAME)
    if pool is not None:
        out.append(pool)
    return out


def worker_zones() -> list[Zone]:
    return list(ZONES_BY_ROLE.get("worker", []))


def decon_zone() -> Zone:
    return ZONES_BY_ROLE["decon"][0]


def charging_zone() -> Zone:
    return ZONES_BY_ROLE["charging"][0]


def solidification_zone() -> Zone | None:
    zs = ZONES_BY_ROLE.get("solidification", [])
    return zs[0] if zs else None


def clearance_zone() -> Zone | None:
    zs = ZONES_BY_ROLE.get("clearance", [])
    return zs[0] if zs else None


def char_station_zone() -> Zone | None:
    """Drum Characterisation Station — where drums get scanned. Carts stage
    drums on the turntable; the station's overhead NaI + CV gantry classifies
    them centrally; then carts route them out to storage or the QA lab."""
    zs = ZONES_BY_ROLE.get("char_station", [])
    return zs[0] if zs else None


# Below this specific activity, VLLW items qualify for free release back
# to general industry (loosely tracks IAEA RS-G-1.7 clearance levels for
# common LWR nuclides).
CLEARANCE_THRESHOLD_BQ_PER_G = 1.0


# ---------------------------------------------------------------------------
# Plant building envelope — wings and corridors. The view renders walls
# from this list; the agent pathfinder treats every wall segment as a
# barrier that the line between two waypoints must not cross. The same
# WINGS / CORRIDORS data therefore drives BOTH the visible building
# structure AND the AGV routing graph — no chance of carts walking
# through a wall the user can see.
# Each wing: (name, x0, y0, x1, y1) in metres (top-left, bottom-right).
# Wings now line up with the zones they actually contain — the CLASSIFY,
# QA LAB and DECON BAY wings each surround their corresponding equipment
# zone with at least 1m of working floor on each side. Wings are
# separated by short vertical corridors (and one big horizontal corridor
# along the top + bottom of the middle row) so carts can always reach
# every wing through an explicit "door".
WINGS: list[tuple[str, float, float, float, float]] = [
    ("REPROCESSING",      0.8,   0.8, 33.0,  6.4),
    ("AI BRAIN",         33.5,   0.8, 43.0,  8.5),
    ("CLASSIFY",         10.0,   7.0, 23.0, 14.0),
    ("QA LAB",           23.6,   7.0, 32.0, 14.0),
    ("DECON BAY",        32.6,   7.0, 42.5, 14.0),
    # Storage wing extended slightly upward (y0=1.8) so the new CLEAR
    # zone at the top of the storage column has its own wall.
    ("STORAGE",          43.0,   1.8, 48.6, 22.0),
    ("CONTROL",           0.8,   7.0,  9.2, 14.0),
    ("LEGACY",            0.8,  14.6, 13.5, 23.0),
]

# Cart corridors — open lanes between wings (the cart graph passes
# through these gaps; walls aren't drawn here). Each entry:
# (x0, y0, x1, y1) in metres.
CORRIDORS: list[tuple[float, float, float, float]] = [
    # Top main corridor — between the top row and middle row.
    (0.5,   6.4, 43.0,  7.0),
    # Bottom main corridor — between the middle row and LEGACY.
    (0.5,  14.0, 13.5, 14.6),
    # Vertical "doors" between adjacent middle-row wings so a cart can
    # cross from CONTROL → CLASSIFY → QA LAB → DECON BAY → STORAGE.
    ( 9.2,  7.0,  9.8, 14.0),  # CONTROL  ↔ CLASSIFY
    (23.0,  7.0, 23.6, 14.0),  # CLASSIFY ↔ QA LAB
    (32.0,  7.0, 32.6, 14.0),  # QA LAB   ↔ DECON BAY
    (42.5,  1.8, 43.0, 22.0),  # DECON BAY / AI BRAIN ↔ STORAGE
]


def _segments_minus_corridors(
    is_horizontal: bool, fixed_coord: float, var_lo: float, var_hi: float,
) -> list[tuple[float, float]]:
    """Given one wall line and the set of corridors, return the sub-segments
    of the wall that remain (corridors punch gaps in the wall)."""
    segments = [(var_lo, var_hi)]
    for cx0, cy0, cx1, cy1 in CORRIDORS:
        if is_horizontal:
            # Wall is horizontal at y=fixed_coord, spans x in [var_lo, var_hi].
            # Corridor punches a gap if it crosses this y.
            if not (cy0 <= fixed_coord <= cy1):
                continue
            gap_lo, gap_hi = cx0, cx1
        else:
            # Wall is vertical at x=fixed_coord, spans y in [var_lo, var_hi].
            if not (cx0 <= fixed_coord <= cx1):
                continue
            gap_lo, gap_hi = cy0, cy1
        new_segs = []
        for s_lo, s_hi in segments:
            if gap_hi <= s_lo or gap_lo >= s_hi:
                new_segs.append((s_lo, s_hi))
            else:
                if gap_lo > s_lo:
                    new_segs.append((s_lo, gap_lo))
                if gap_hi < s_hi:
                    new_segs.append((gap_hi, s_hi))
        segments = new_segs
    return [(a, b) for a, b in segments if b - a > 0.05]


def _all_wall_segments() -> list[tuple[float, float, float, float]]:
    """Return every wall segment of every wing as (x0, y0, x1, y1).
    Horizontal walls have y0 == y1; vertical walls have x0 == x1.
    Corridor gaps are pre-punched out."""
    out: list[tuple[float, float, float, float]] = []
    for _name, x0, y0, x1, y1 in WINGS:
        for x_lo, x_hi in _segments_minus_corridors(True, y0, x0, x1):
            out.append((x_lo, y0, x_hi, y0))                # top
        for x_lo, x_hi in _segments_minus_corridors(True, y1, x0, x1):
            out.append((x_lo, y1, x_hi, y1))                # bottom
        for y_lo, y_hi in _segments_minus_corridors(False, x0, y0, y1):
            out.append((x0, y_lo, x0, y_hi))                # left
        for y_lo, y_hi in _segments_minus_corridors(False, x1, y0, y1):
            out.append((x1, y_lo, x1, y_hi))                # right
    return out


_WALL_SEGMENTS_CACHE: list[tuple[float, float, float, float]] | None = None


def wall_segments() -> list[tuple[float, float, float, float]]:
    global _WALL_SEGMENTS_CACHE
    if _WALL_SEGMENTS_CACHE is None:
        _WALL_SEGMENTS_CACHE = _all_wall_segments()
    return _WALL_SEGMENTS_CACHE


def _segments_intersect(
    p1: tuple[float, float], p2: tuple[float, float],
    p3: tuple[float, float], p4: tuple[float, float],
) -> bool:
    """Return True if segment p1p2 crosses segment p3p4. Endpoints touching
    are NOT counted as intersecting (so a waypoint sitting exactly on a
    wall's endpoint at a corridor mouth still connects through)."""
    def ccw(a, b, c):
        return (c[1] - a[1]) * (b[0] - a[0]) - (b[1] - a[1]) * (c[0] - a[0])
    d1 = ccw(p3, p4, p1)
    d2 = ccw(p3, p4, p2)
    d3 = ccw(p1, p2, p3)
    d4 = ccw(p1, p2, p4)
    # Strict opposite-sign check on both pairs => proper crossing.
    return ((d1 > 1e-9 and d2 < -1e-9) or (d1 < -1e-9 and d2 > 1e-9)) and \
           ((d3 > 1e-9 and d4 < -1e-9) or (d3 < -1e-9 and d4 > 1e-9))


def line_crosses_any_wall(
    p1: tuple[float, float], p2: tuple[float, float],
) -> bool:
    """True iff the straight line p1→p2 crosses any internal wing wall."""
    for x0, y0, x1, y1 in wall_segments():
        if _segments_intersect(p1, p2, (x0, y0), (x1, y1)):
            return True
    return False


def _line_intersects_circle(
    p1: tuple[float, float], p2: tuple[float, float],
    centre: tuple[float, float], radius: float,
) -> bool:
    """Strict segment-vs-circle intersection.
    Endpoint touches don't count (so a path that *ends* at a zone's
    perimeter is allowed)."""
    cx, cy = centre
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    fx, fy = p1[0] - cx, p1[1] - cy
    a = dx * dx + dy * dy
    if a < 1e-9:
        return False
    b = 2 * (fx * dx + fy * dy)
    c = fx * fx + fy * fy - radius * radius
    disc = b * b - 4 * a * c
    if disc < 0:
        return False
    disc_sq = disc ** 0.5
    t1 = (-b - disc_sq) / (2 * a)
    t2 = (-b + disc_sq) / (2 * a)
    # Strict interior crossing: any t in (eps, 1-eps).
    eps = 1e-3
    return (eps < t1 < 1 - eps) or (eps < t2 < 1 - eps) or (t1 < eps and t2 > 1 - eps)


def line_blocked(
    p1: tuple[float, float], p2: tuple[float, float],
    ignore_zone_names: tuple[str, ...] = (),
) -> bool:
    """True iff the straight line p1→p2 is not walkable: it crosses a wing
    wall OR cuts through a zone bubble (other than `ignore_zone_names`).

    Zones are treated as no-go circular obstacles for pathing, so a cart
    can never plow through the middle of e.g. the Classifier or HLW
    storage on its way to somewhere else. Carts arriving AT a zone pass
    the zone's name in `ignore_zone_names` so the final approach is
    allowed."""
    if line_crosses_any_wall(p1, p2):
        return True
    # Slightly inflate the zone radius so paths give rooms a respectful
    # berth instead of grazing their walls. 0.85× keeps the bubble inside
    # the actual room rectangle (room half-width ≈ 1.0..1.2× radius).
    for z in ZONES:
        if z.name in ignore_zone_names:
            continue
        if _line_intersects_circle(p1, p2, (z.x, z.y), z.radius_m * 0.95):
            return True
    return False


# ---------------------------------------------------------------------------
# QR-marker waypoint graph. Markers are placed on a ~3 m grid in walkable
# areas (corridors + wing interiors that don't fall inside a zone radius).
# Real warehouse AGV deployments use sparse fiducial markers, not a tag in
# every square inch — 3 m matches that look. Edges connect adjacent
# markers whose straight line does not cross a wall or a zone bubble.
# The view renders these same markers, so what you see is literally what
# the cart routes on.
WAYPOINT_SPACING_M = 3.0


def _is_inside_zone_radius(x: float, y: float, slack: float = 0.45) -> bool:
    """Is (x, y) inside any zone's exclusion bubble (zone radius + a little
    breathing room)? Used to keep QR markers off rooms. Carts arriving AT
    a zone don't need a marker inside it — they drive the last few metres
    from the nearest perimeter waypoint, with that zone's name passed in
    `ignore_zone_names` so the final approach line is allowed."""
    for z in ZONES:
        if (x - z.x) ** 2 + (y - z.y) ** 2 < (z.radius_m + slack) ** 2:
            return True
    return False


def _is_inside_wing_or_corridor(x: float, y: float) -> bool:
    for _name, x0, y0, x1, y1 in WINGS:
        if x0 - 0.05 <= x <= x1 + 0.05 and y0 - 0.05 <= y <= y1 + 0.05:
            return True
    for cx0, cy0, cx1, cy1 in CORRIDORS:
        if cx0 - 0.05 <= x <= cx1 + 0.05 and cy0 - 0.05 <= y <= cy1 + 0.05:
            return True
    return False


_WAYPOINTS_CACHE: list[tuple[float, float]] | None = None
_WAYPOINT_EDGES_CACHE: dict[int, list[tuple[int, float]]] | None = None


def waypoints() -> list[tuple[float, float]]:
    """QR floor markers as (x, y) waypoints. Cached on first call.

    Two layers of markers:

      1) A sparse 3 m grid inside every wing — the bulk of the network,
         placed where carts actually drive within rooms.
      2) Explicit anchor markers running down the centreline of each
         corridor (every 3 m). Corridors are typically ~0.5 m wide so a
         pure 3 m grid can miss them and split the graph into orphan
         wings — these anchors guarantee a marker exists at every
         corridor crossing."""
    global _WAYPOINTS_CACHE
    if _WAYPOINTS_CACHE is not None:
        return _WAYPOINTS_CACHE
    pts: list[tuple[float, float]] = []
    spacing = WAYPOINT_SPACING_M

    # Layer 1: sparse grid covering wing interiors.
    x = 1.5
    while x < WIDTH_M - 0.5:
        y = 1.5
        while y < HEIGHT_M - 0.5:
            if (_is_inside_wing_or_corridor(x, y)
                    and not _is_inside_zone_radius(x, y)):
                pts.append((x, y))
            y += spacing
        x += spacing

    # Layer 2: corridor anchors. One marker per ~3 m along each corridor
    # centreline, with extra markers near each end so wings on either
    # side have a visible "door" marker to/from the corridor.
    def _too_close(p: tuple[float, float]) -> bool:
        for q in pts:
            if (p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 < 0.8 ** 2:
                return True
        return False

    for cx0, cy0, cx1, cy1 in CORRIDORS:
        mid_x = (cx0 + cx1) / 2.0
        mid_y = (cy0 + cy1) / 2.0
        if cx1 - cx0 > cy1 - cy0:
            # Horizontal corridor: anchor every `spacing` along x.
            x = cx0 + 0.6
            while x < cx1 - 0.4:
                p = (x, mid_y)
                if not _is_inside_zone_radius(p[0], p[1]) and not _too_close(p):
                    pts.append(p)
                x += spacing
        else:
            # Vertical corridor: anchor every `spacing` along y.
            y = cy0 + 0.6
            while y < cy1 - 0.4:
                p = (mid_x, y)
                if not _is_inside_zone_radius(p[0], p[1]) and not _too_close(p):
                    pts.append(p)
                y += spacing

    _WAYPOINTS_CACHE = pts
    return pts


def waypoint_edges() -> dict[int, list[tuple[int, float]]]:
    """Adjacency list: node_idx -> list of (neighbour_idx, edge_cost).
    Two waypoints are connected if their straight line is ≤ ~2.2× spacing
    AND does not cross any wall segment or any zone bubble."""
    global _WAYPOINT_EDGES_CACHE
    if _WAYPOINT_EDGES_CACHE is not None:
        return _WAYPOINT_EDGES_CACHE
    pts = waypoints()
    max_edge = WAYPOINT_SPACING_M * 2.2
    edges: dict[int, list[tuple[int, float]]] = {i: [] for i in range(len(pts))}
    for i, (x1, y1) in enumerate(pts):
        for j in range(i + 1, len(pts)):
            x2, y2 = pts[j]
            d = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
            if d > max_edge:
                continue
            if line_blocked((x1, y1), (x2, y2)):
                continue
            edges[i].append((j, d))
            edges[j].append((i, d))
    _WAYPOINT_EDGES_CACHE = edges
    return edges


def _zone_containing(pos: tuple[float, float]) -> Zone | None:
    """Return the zone whose bubble contains pos, if any. Used to know
    whether the start point is already inside a room (in which case the
    pathfinder lets the first hop ignore that zone)."""
    for z in ZONES:
        dx = pos[0] - z.x
        dy = pos[1] - z.y
        if dx * dx + dy * dy <= (z.radius_m * 0.95) ** 2:
            return z
    return None


def nearest_waypoint(
    pos: tuple[float, float],
    ignore_zone_names: tuple[str, ...] = (),
) -> int:
    """Index of the QR marker closest to pos with a clear walkable line
    of sight (no wall AND no foreign zone bubble in between)."""
    pts = waypoints()
    best_i = -1
    best_d = float("inf")
    for i, p in enumerate(pts):
        d = (pos[0] - p[0]) ** 2 + (pos[1] - p[1]) ** 2
        if d < best_d and not line_blocked(pos, p, ignore_zone_names):
            best_d = d
            best_i = i
    # Fallback — if every candidate is blocked (rare; can happen if pos
    # sits just outside any wing), pick the absolute nearest.
    if best_i < 0:
        for i, p in enumerate(pts):
            d = (pos[0] - p[0]) ** 2 + (pos[1] - p[1]) ** 2
            if d < best_d:
                best_d = d
                best_i = i
    return best_i


def shortest_waypoint_path(
    start: tuple[float, float], goal: tuple[float, float],
) -> list[tuple[float, float]]:
    """A* from `start` to `goal` over the QR waypoint graph.

    Returns the list of intermediate waypoints to traverse (excluding
    `start`, NOT including the final goal point — caller drives from the
    last waypoint to `goal` directly).

    If `start` and `goal` already have a clear walkable line of sight,
    returns [] so the caller can just drive direct.

    Zones that contain the start or goal are ignored as obstacles, so the
    cart can exit/enter a room cleanly along the line that touches the
    room's perimeter."""
    start_zone = _zone_containing(start)
    goal_zone = _zone_containing(goal)
    ignore = tuple(z.name for z in (start_zone, goal_zone) if z is not None)

    if not line_blocked(start, goal, ignore):
        return []

    pts = waypoints()
    edges = waypoint_edges()
    if not pts:
        return []

    src = nearest_waypoint(start, ignore_zone_names=ignore)
    dst = nearest_waypoint(goal, ignore_zone_names=ignore)
    if src == dst:
        return [pts[src]]

    import heapq
    gx, gy = pts[dst]

    def heuristic(i: int) -> float:
        px, py = pts[i]
        return ((px - gx) ** 2 + (py - gy) ** 2) ** 0.5

    open_heap: list[tuple[float, int]] = [(heuristic(src), src)]
    came_from: dict[int, int] = {}
    g_score: dict[int, float] = {src: 0.0}

    while open_heap:
        _, cur = heapq.heappop(open_heap)
        if cur == dst:
            break
        for nbr, cost in edges.get(cur, ()):
            tentative = g_score[cur] + cost
            if tentative < g_score.get(nbr, float("inf")):
                came_from[nbr] = cur
                g_score[nbr] = tentative
                heapq.heappush(open_heap, (tentative + heuristic(nbr), nbr))

    # Reconstruct
    if dst not in came_from and dst != src:
        return []
    path_idx: list[int] = [dst]
    cur = dst
    while cur in came_from:
        cur = came_from[cur]
        path_idx.append(cur)
    path_idx.reverse()
    return [pts[i] for i in path_idx]
