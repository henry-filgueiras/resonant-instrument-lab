#!/usr/bin/env python3
"""Run one simulator config and write artifacts to an output directory.

Usage:
    python scripts/run_sim.py --config PATH --out DIR [--seed N] [--summary]

The config is validated (DESIGN_V0.md §9.2) before simulation; the run
writes state.npz, events.jsonl, topology.json, audio.wav, and a frozen
config.yaml copy into the output directory (DESIGN_V0.md §9.6).

With --summary, a compact semantic regime summary is printed after the
run, driven by the detectors in sim/detectors.py and supporting stats
from sim/derived.py. No raw arrays to stdout — just verdicts, margins,
and a few physical numbers.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import yaml  # noqa: E402

from sim.config import ConfigError, load  # noqa: E402
from sim.derived import cluster_assignments, kuramoto_order, mean_phase_velocity  # noqa: E402
from sim.detectors import detect_drifting, detect_phase_locked  # noqa: E402
from sim.garden import simulate  # noqa: E402


def _fmt_sep(sep):
    if sep == float("inf"):
        return "inf"
    return f"{sep:.2f}"


def _print_summary(cfg, config_path, out_dir):
    """Print a compact human-readable regime summary for one run.

    Reads state.npz from `out_dir`, computes the current detectors and a
    couple of supporting physical statistics, and emits a short block
    on stdout. Not a labels.json writer — that schema lands later.
    """
    rate = int(cfg["run"]["control_rate_hz"])
    duration_s = float(cfg["run"]["duration_s"])
    N = int(cfg["scene"]["N"])

    with np.load(Path(out_dir) / "state.npz") as st:
        theta = st["theta"]
        phase_vel = st["phase_vel"]

    pl = detect_phase_locked(theta, rate)
    dr = detect_drifting(theta, rate)

    r = kuramoto_order(theta)
    mean_r = float(r.mean())
    tail_frames = min(rate, len(r))
    tail_r = float(r[-tail_frames:].mean())

    tail_slice = min(int(3.0 * rate), phase_vel.shape[0])
    v_tail = mean_phase_velocity(phase_vel, start_idx=-tail_slice)
    sep = cluster_assignments(v_tail, n_clusters=2).separability_ratio if N >= 2 else None

    def _line(name, result):
        if result.fired:
            win_s = result.longest_window_frames / rate
            return (
                f"  {name:<13}: FIRED   conf {result.confidence:.3f}   "
                f"longest window {win_s:.2f} s"
            )
        return f"  {name:<13}: silent"

    lines = [
        "",
        f"regime summary — {config_path}",
        f"  {duration_s:.2f} s, N={N}, control_rate={rate} Hz",
        "",
        _line("phase_locked", pl),
        _line("drifting", dr),
        "",
        f"  mean r(t)                    {mean_r:.3f}",
        f"  tail-1s r                    {tail_r:.3f}",
    ]
    if sep is not None:
        lines.append(f"  tail 2-way velocity sep.     {_fmt_sep(sep)}")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description="Run a single v0 simulator config.")
    ap.add_argument("--config", required=True, help="path to a validated run config YAML")
    ap.add_argument("--out", required=True, help="output directory (created if missing)")
    ap.add_argument("--seed", type=int, default=None,
                    help="override config.run.seed (useful for dataset generation)")
    ap.add_argument("--summary", action="store_true",
                    help="after the run, print a compact semantic regime summary to stdout")
    args = ap.parse_args()

    try:
        cfg = load(args.config)
    except FileNotFoundError as e:
        sys.exit(f"error: {e}")
    except ConfigError as e:
        sys.exit(f"error: {e}")
    except yaml.YAMLError as e:
        sys.exit(f"error: YAML parse failure in {args.config}: {e}")

    if args.seed is not None:
        if args.seed < 0:
            sys.exit(f"error: --seed must be >= 0 (got {args.seed})")
        cfg["run"]["seed"] = args.seed

    out = Path(args.out)
    simulate(cfg, out, config_path=args.config)
    print(f"ok: wrote run artifacts to {out}")

    if args.summary:
        _print_summary(cfg, args.config, out)


if __name__ == "__main__":
    main()
