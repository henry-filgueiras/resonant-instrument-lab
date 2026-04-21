#!/usr/bin/env python3
"""Run one simulator config and write artifacts to an output directory.

Usage:
    python scripts/run_sim.py --config PATH --out DIR [--seed N]
                              [--summary] [--summary-json]

The config is validated (DESIGN_V0.md §9.2) before simulation; the run
writes state.npz, events.jsonl, topology.json, audio.wav, and a frozen
config.yaml copy into the output directory (DESIGN_V0.md §9.6).

With --summary, a compact semantic regime summary is printed after the
run, driven by the detectors in sim/detectors.py and supporting stats
from sim/derived.py. With --summary-json, the same verdicts and stats
are also written to `summary.json` in the output directory as a
machine-readable seam for future browser / notebook consumers. Both
views are rendered from one shared summary dict so they cannot drift.
"""
import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import yaml  # noqa: E402

from sim.config import ConfigError, load  # noqa: E402
from sim.derived import cluster_assignments, kuramoto_order, mean_phase_velocity  # noqa: E402
from sim.detectors import (  # noqa: E402
    detect_drifting,
    detect_flam,
    detect_phase_beating,
    detect_phase_locked,
)
from sim.garden import simulate  # noqa: E402

SUMMARY_SCHEMA_VERSION = 1


def _json_float(x):
    """Return `x` as a JSON-safe float, or `None` if non-finite / None."""
    if x is None:
        return None
    xf = float(x)
    if not math.isfinite(xf):
        return None
    return xf


def _detector_block(result, control_rate_hz):
    return {
        "fired": bool(result.fired),
        "confidence": float(result.confidence),
        "longest_window_s": float(result.longest_window_frames) / control_rate_hz,
        "windows_s": [
            [float(s) / control_rate_hz, float(e) / control_rate_hz]
            for s, e in result.windows
        ],
    }


def _build_summary(cfg, config_path, out_dir):
    """Compute the regime summary dict consumed by both text and JSON renderers.

    Reads `state.npz` from `out_dir`, invokes the current detectors, and
    computes a small set of supporting physical stats. The returned dict
    is the JSON schema — the text renderer consumes the same object so
    the two views cannot drift.
    """
    rate = int(cfg["run"]["control_rate_hz"])
    duration_s = float(cfg["run"]["duration_s"])
    N = int(cfg["scene"]["N"])

    with np.load(Path(out_dir) / "state.npz") as st:
        theta = st["theta"]
        phase_vel = st["phase_vel"]
        pulse_fired = st["pulse_fired"]

    pl = detect_phase_locked(theta, rate)
    dr = detect_drifting(theta, rate)
    pb = detect_phase_beating(theta, phase_vel, pulse_fired, rate)
    fl = detect_flam(phase_vel, pulse_fired, rate)

    r = kuramoto_order(theta)
    mean_r = float(r.mean())
    tail_frames = min(rate, len(r))
    tail_r = float(r[-tail_frames:].mean())

    stats = {
        "mean_r": mean_r,
        "tail_1s_mean_r": tail_r,
    }

    if N >= 2:
        tail_slice = min(int(3.0 * rate), phase_vel.shape[0])
        v_tail = mean_phase_velocity(phase_vel, start_idx=-tail_slice)
        sep = cluster_assignments(v_tail, n_clusters=2).separability_ratio
        stats["tail_2way_velocity_separability"] = _json_float(sep)

    meta_name = None
    cfg_meta = cfg.get("meta")
    if isinstance(cfg_meta, dict):
        meta_name = cfg_meta.get("name")

    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "meta": {
            "config_path": str(config_path),
            "config_name": meta_name,
            "duration_s": duration_s,
            "N": N,
            "control_rate_hz": rate,
        },
        "detectors": {
            "phase_locked": _detector_block(pl, rate),
            "drifting": _detector_block(dr, rate),
            "phase_beating": _detector_block(pb, rate),
            "flam": _detector_block(fl, rate),
        },
        "stats": stats,
    }


def _fmt_sep(sep):
    if sep is None:
        return "inf"
    return f"{sep:.2f}"


def _render_summary_text(summary):
    """Build the human-readable terminal block from a summary dict."""
    meta = summary["meta"]
    det = summary["detectors"]
    stats = summary["stats"]

    def _line(name, d):
        if d["fired"]:
            return (
                f"  {name:<13}: FIRED   conf {d['confidence']:.3f}   "
                f"longest window {d['longest_window_s']:.2f} s"
            )
        return f"  {name:<13}: silent"

    lines = [
        "",
        f"regime summary — {meta['config_path']}",
        f"  {meta['duration_s']:.2f} s, N={meta['N']}, control_rate={meta['control_rate_hz']} Hz",
        "",
        _line("phase_locked", det["phase_locked"]),
        _line("drifting", det["drifting"]),
        _line("phase_beating", det["phase_beating"]),
        _line("flam", det["flam"]),
        "",
        f"  mean r(t)                    {stats['mean_r']:.3f}",
        f"  tail-1s r                    {stats['tail_1s_mean_r']:.3f}",
    ]
    if "tail_2way_velocity_separability" in stats:
        lines.append(
            f"  tail 2-way velocity sep.     "
            f"{_fmt_sep(stats['tail_2way_velocity_separability'])}"
        )
    return "\n".join(lines)


def _write_summary_json(summary, out_dir):
    path = Path(out_dir) / "summary.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")
    return path


def main():
    ap = argparse.ArgumentParser(description="Run a single v0 simulator config.")
    ap.add_argument("--config", required=True, help="path to a validated run config YAML")
    ap.add_argument("--out", required=True, help="output directory (created if missing)")
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

    out = Path(args.out)
    simulate(cfg, out, config_path=args.config)
    print(f"ok: wrote run artifacts to {out}")

    if args.summary or args.summary_json:
        summary = _build_summary(cfg, args.config, out)
        if args.summary:
            print(_render_summary_text(summary))
        if args.summary_json:
            path = _write_summary_json(summary, out)
            print(f"ok: wrote {path}")


if __name__ == "__main__":
    main()
