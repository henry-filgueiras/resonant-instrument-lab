#!/usr/bin/env python3
"""Run one simulator config with a single node ablated for the whole run.

Usage:
    python scripts/run_ablation.py --config PATH --out DIR --node K
                                   [--seed N] [--summary] [--summary-json]

Mirrors `scripts/run_sim.py` but routes through `sim.ablate.ablate_node`:
the resulting run directory contains the standard §9.6 artifact set
(state.npz / events.jsonl / topology.json / audio.wav / optional frozen
config.yaml) plus one extra small file — `ablation.json` — documenting
which node was ablated and under what semantics. Baseline runs never
emit `ablation.json`, so its presence is itself the marker "this is a
counterfactual bundle".

The counterfactual summary (`--summary`, `--summary-json`) is rendered
from the same shared builder that `run_sim.py` uses, so ablated and
baseline summaries are directly comparable.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml  # noqa: E402

from sim.ablate import ablate_node  # noqa: E402
from sim.config import ConfigError, load  # noqa: E402

# Reuse the single-source-of-truth summary builder from run_sim to keep
# baseline and ablated summaries on identical code paths.
from scripts.run_sim import (  # noqa: E402
    _build_summary,
    _render_summary_text,
    _write_summary_json,
)


def main():
    ap = argparse.ArgumentParser(
        description="Run a v0 simulator config with a single node ablated."
    )
    ap.add_argument("--config", required=True, help="path to a validated baseline config YAML")
    ap.add_argument("--out", required=True, help="output directory (created if missing)")
    ap.add_argument("--node", type=int, required=True,
                    help="node index to ablate (decouple + silence) for the whole run")
    ap.add_argument("--seed", type=int, default=None,
                    help="override config.run.seed (useful for dataset generation)")
    ap.add_argument("--summary", action="store_true",
                    help="after the run, print a compact semantic regime summary to stdout")
    ap.add_argument("--summary-json", action="store_true",
                    help="after the run, write summary.json into the output directory")
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

    N = int(cfg["scene"]["N"])
    if not 0 <= args.node < N:
        sys.exit(f"error: --node {args.node} out of range [0, {N})")

    out = Path(args.out)
    ablate_node(cfg, out, node=args.node, config_path=args.config)
    print(f"ok: wrote ablated (node={args.node}) run artifacts to {out}")

    if args.summary or args.summary_json:
        summary = _build_summary(cfg, args.config, out)
        if args.summary:
            print(_render_summary_text(summary))
        if args.summary_json:
            path = _write_summary_json(summary, out)
            print(f"ok: wrote {path}")


if __name__ == "__main__":
    main()
