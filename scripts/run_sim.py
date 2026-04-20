#!/usr/bin/env python3
"""Run one simulator config and write artifacts to an output directory.

Usage:
    python scripts/run_sim.py --config PATH --out DIR [--seed N]

The config is validated (DESIGN_V0.md §9.2) before simulation; the run
writes state.npz, events.jsonl, topology.json, audio.wav, and a frozen
config.yaml copy into the output directory (DESIGN_V0.md §9.6).

This is a v0 scaffold — dynamics are a stub (see sim/garden.py).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml  # noqa: E402

from sim.config import ConfigError, load  # noqa: E402
from sim.garden import simulate  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="Run a single v0 simulator config.")
    ap.add_argument("--config", required=True, help="path to a validated run config YAML")
    ap.add_argument("--out", required=True, help="output directory (created if missing)")
    ap.add_argument("--seed", type=int, default=None,
                    help="override config.run.seed (useful for dataset generation)")
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


if __name__ == "__main__":
    main()
