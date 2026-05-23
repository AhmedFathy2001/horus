"""Single-command entrypoint.

  python run_demo.py --mode compare        # default: run both, show dashboard
  python run_demo.py --mode hivemind       # hivemind only
  python run_demo.py --mode isolated
  python run_demo.py --no-live             # skip pygame, dashboard only
  python run_demo.py --seed N              # reproducible scenarios
"""
from __future__ import annotations
import argparse
import sys


def parse_args():
    p = argparse.ArgumentParser(description="Radwaste hivemind classification sim")
    p.add_argument(
        "--mode",
        choices=("compare", "isolated", "hivemind"),
        default="compare",
    )
    p.add_argument("--seed", type=int, default=1,
                   help="Default 1 is the cleanest demo scenario; vary it for robustness")
    p.add_argument("--sim-hours", type=float, default=8.0)
    p.add_argument("--interarrival-s", type=float, default=45.0)
    p.add_argument("--agents", type=int, default=0,
                   help="Legacy single-knob fleet size (builds all Hybrids). "
                        "Ignored when --scanners/--handlers/--hybrids are set.")
    p.add_argument("--scanners", type=int, default=2,
                   help="Number of sensor-only Scanner AGVs in the fleet.")
    p.add_argument("--handlers", type=int, default=3,
                   help="Number of gripper-only Handler AGVs in the fleet.")
    p.add_argument("--hybrids",  type=int, default=1,
                   help="Number of full-capability Hybrid AGVs in the fleet.")
    p.add_argument("--tricky-fraction", type=float, default=0.25)
    p.add_argument("--qa-fraction", type=float, default=0.05)
    p.add_argument("--net-drop", type=float, default=0.03,
                   help="Wireless packet-drop probability on drone reports (0-1)")
    p.add_argument("--no-live", action="store_true",
                   help="Skip the pygame live window; dashboard only")
    p.add_argument("--save-dashboard", type=str, default=None,
                   help="Path to save the dashboard PNG (e.g. dashboard.png)")
    p.add_argument("--no-dashboard", action="store_true")
    p.add_argument("--audit-csv", type=str, default=None,
                   help="If set, dump the coordinator ledger as CSV here")
    return p.parse_args()


def build_cfg(args, mode: str):
    from sim.scenario import ScenarioConfig
    cfg_kwargs = dict(
        mode=mode,
        seed=args.seed,
        sim_duration_s=args.sim_hours * 3600.0,
        waste_interarrival_mean_s=args.interarrival_s,
        tricky_fraction=args.tricky_fraction,
        qa_sampling_fraction=args.qa_fraction,
        network_drop_probability=args.net_drop,
    )
    # If the user explicitly mixed the fleet, use those counts. Otherwise
    # fall back to --agents (legacy: all-hybrid).
    if args.scanners + args.handlers + args.hybrids > 0:
        cfg_kwargs["n_scanners"] = args.scanners
        cfg_kwargs["n_handlers"] = args.handlers
        cfg_kwargs["n_hybrids"] = args.hybrids
    elif args.agents > 0:
        cfg_kwargs["n_mobile_agents"] = args.agents
    return ScenarioConfig(**cfg_kwargs)


def run_one(args, mode: str, live_view=None):
    """Run one mode end-to-end. If the live view requests a restart (X-key
    in-window), re-build the scenario config from the live fleet state
    and run again. Returns the last completed scenario's metrics."""
    from sim.scenario import run_scenario
    while True:
        cfg = build_cfg(args, mode)
        # When the user has tuned the fleet live (F-cycle then X-restart),
        # honour that composition instead of the CLI-derived one.
        if live_view is not None and live_view.live_fleet is not None:
            cfg.n_scanners = live_view.live_fleet["scanner"]
            cfg.n_handlers = live_view.live_fleet["handler"]
            cfg.n_hybrids  = live_view.live_fleet["hybrid"]
            cfg.n_mobile_agents = 0  # disable legacy single-knob fallback
        on_tick = None
        if live_view is not None:
            live_view.reset_for_mode(mode)
            live_view.restart_requested = False
            live_view.closed = False

            def on_tick(ctx, _v=live_view):
                return _v.on_tick(ctx)
        print(f"\n[{mode}] running scenario seed={args.seed}, "
              f"{args.sim_hours:.1f}h simulated... "
              f"(fleet: {cfg.n_scanners}sc/{cfg.n_handlers}hd/{cfg.n_hybrids}hy)")
        metrics, coord, agents = run_scenario(cfg, on_tick=on_tick)
        s = metrics.summary()
        print(f"[{mode}]   classified : {s['n_classified']}")
        print(f"[{mode}]   accuracy   : {s['accuracy']*100:.1f}%")
        print(f"[{mode}]   dose       : {s['cumulative_dose_uSv']:.1f} µSv")
        print(f"[{mode}]   throughput : {s['throughput_per_hour']:.1f} items/h")
        print(f"[{mode}]   retrains   : {len(coord.retrain_events)}  (model versions)")
        if live_view is None or not live_view.restart_requested:
            return mode, metrics, coord, agents
        # If the user pressed M to swap modes, flip the mode and re-run.
        if getattr(live_view, "mode_switch_requested", False):
            mode = "isolated" if mode == "hivemind" else "hivemind"
            live_view.mode_switch_requested = False
        # Loop and run the (possibly different) mode again with the
        # (possibly new) fleet


def main():
    args = parse_args()

    live_view = None
    if not args.no_live:
        try:
            from viz.pygame_view import LiveView
            live_view = LiveView()
        except Exception as e:
            print(f"pygame unavailable ({e}); running headless", file=sys.stderr)
            live_view = None

    results: dict[str, tuple] = {}

    if args.mode == "compare":
        modes = ["isolated", "hivemind"]
    else:
        modes = [args.mode]

    for mode in modes:
        result = run_one(args, mode, live_view=live_view)
        actual_mode, metrics, coord, agents = result
        results[actual_mode] = (metrics, coord, agents)

    if live_view is not None:
        live_view.close()

    if args.audit_csv:
        last_coord = results[modes[-1]][1]
        df = last_coord.ledger_dataframe()
        df.to_csv(args.audit_csv, index=False)
        print(f"audit ledger written to {args.audit_csv}")

    if not args.no_dashboard and len(results) >= 1:
        try:
            from viz.dashboard import render_comparison
            print("\nrendering dashboard…")
            render_comparison(results, output_path=args.save_dashboard, show=True)
        except Exception as e:
            print(f"dashboard rendering failed: {e}", file=sys.stderr)

    # Suggest a one-line interpretation
    if "isolated" in results and "hivemind" in results:
        iso_acc = results["isolated"][0].accuracy() * 100
        hive_acc = results["hivemind"][0].accuracy() * 100
        iso_dose = results["isolated"][0].cumulative_dose_uSv
        hive_dose = results["hivemind"][0].cumulative_dose_uSv
        print("\n=== headline ===")
        print(f"  accuracy  isolated {iso_acc:.1f}%  →  hivemind {hive_acc:.1f}%   "
              f"(+{hive_acc - iso_acc:.1f} pp)")
        print(f"  dose      isolated {iso_dose:.0f}  →  hivemind {hive_dose:.0f}  µSv   "
              f"({(hive_dose-iso_dose)/max(iso_dose,1)*100:+.1f}%)")


if __name__ == "__main__":
    main()
