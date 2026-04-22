#!/usr/bin/env python3
"""Build the Intervention Atlas for one baseline run.

For one baseline `(config, run dir)` pair, enumerates a bounded set of
single interventions through the existing `sim.ablate` surface, computes
each one's resulting summary, ranks them by a single explainable score,
and writes one machine-readable artifact `atlas.json` next to the
baseline `summary.json` / `topology.json`.

Usage
-----
    python scripts/build_atlas.py --config PATH --baseline-dir DIR

The baseline dir is expected to already contain `summary.json` (the
canonical full summary, with counterfactual detectors) and the standard
§9.6 artifact bundle. `atlas.json` is written into the same directory.

Intervention families (v1)
--------------------------
Two families, single interventions only. Total 2N interventions per run
(`N` from the baseline scene).

- `ablate_node(k)` for each `k ∈ [0, N)`. Routes through
  `sim.ablate.ablate_node`. "Decouple and silence" semantics —
  see `sim/ablate.py` for the full contract.
- `nudge_node(k, delta_hz=BRITTLE_LOCK_NUDGE_HZ)` for each `k`. Routes
  through `sim.ablate.nudge_node` with the canonical `+0.25 Hz` whole-run
  shift used by the `brittle_lock` detector. "Detune in place" semantics.

No other intervention types in this pass — each new counterfactual
intervention earns one narrow seam in `sim.ablate`, and v1 atlas reuses
exactly the two seams that already exist (DESIGN_V0.md §10.1, project
rule "no broad framework expansion unless forced").

Per-intervention summary
------------------------
Each intervention is run into a `TemporaryDirectory` and its summary is
computed via `_build_summary(..., include_counterfactual=False)`. The
counterfactual detectors (`unstable_bridge`, `brittle_lock`) each run
their own O(N) inner loop, so recursing into them on every atlas
intervention would inflate the atlas to O(N²) sims per fixture.
The atlas explicitly documents itself as observational-detectors-only
(`schema_version` carries that promise via the `observational_detectors`
list); `atlas.json` does not omit counterfactual detector keys silently.

Ranking score
-------------
One simple, honest, explainable score:

    score = (#detector flips vs baseline) + |Δ tail_1s_mean_r|

- `flips` is the integer count of observational detectors whose
  fired/silent state flipped between baseline and intervention. A
  single flip dominates a stat-only delta because flips cross a named
  semantic boundary; stat deltas are continuous.
- `|Δ tail_1s_mean_r|` is in `[0, 1]` and serves both as a magnitude
  signal on no-flip interventions and as a tiebreaker between
  same-flip-count interventions.

The compared baseline is the observational subset of the baseline
summary — counterfactual detectors are excluded from the flip count
because intervention summaries do not compute them.

Each intervention carries a `score_components` block and a one-line
`score_explanation` string so the UI can show *why* the score is high
without recomputing.

Audio artifacts
---------------
Each intervention's sim produces an `audio.wav` (pulse-synthesized — see
`sim.garden._write_pulse_wav`). The atlas builder persists those WAVs
into a sibling `atlas_audio/` directory alongside `atlas.json`, one file
per intervention named `<intervention_id>.wav`, and records the relative
path on the entry's `audio_path`. The browser's A/B audio toggle plays
baseline vs selected-intervention directly from these static files with
no backend. If the sim emitted no WAV for some reason, `audio_path` is
`null` and the UI falls back to "audio unavailable".

Output schema (atlas.json, schema_version 1)
--------------------------------------------
    {
      "schema_version": 1,
      "baseline": {
        "config_path": "...",
        "config_name": "...",
        "summary_path": "summary.json",
        "topology_path": "topology.json",
        "audio_path": "audio.wav",             // null if absent
        "observational_summary": { ... }      // observational subset
      },
      "intervention_families": ["ablate_node", "nudge_node"],
      "observational_detectors": [...],       // names actually compared
      "ranking": {
        "score_formula": "(#detector flips) + |Δ tail_1s_mean_r|",
        "tiebreakers": ["score", "magnitude of stat deltas"]
      },
      "interventions": [                       // ranked, descending
        {
          "id": "nudge_n4_+0.25hz",
          "kind": "nudge_node",
          "node": 4,
          "delta_hz": 0.25,
          "label": "nudge node 4 by +0.25 Hz",
          "summary": { ... },                  // observational summary
          "audio_path": "atlas_audio/nudge_n4_+0.25hz.wav",  // null if absent
          "deltas": {
            "flips": [ {"detector":"phase_locked","from":true,"to":false}, ... ],
            "stat_changes": {
              "tail_1s_mean_r": {"from": 0.94, "to": 0.76, "delta": -0.18},
              ...
            }
          },
          "score": 1.18,
          "score_components": {"flips": 1, "abs_tail_r_delta": 0.182},
          "score_explanation": "1 detector flip (phase_locked: FIRED → silent); tail r dropped 0.18."
        },
        ...
      ]
    }
"""
import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml  # noqa: E402

from sim.ablate import ablate_node, nudge_node  # noqa: E402
from sim.config import ConfigError, load  # noqa: E402
from sim.detectors import BRITTLE_LOCK_NUDGE_HZ  # noqa: E402

# Reuse the single-source-of-truth summary builder from run_sim.
from scripts.run_sim import _build_summary  # noqa: E402

ATLAS_SCHEMA_VERSION = 1

# Subdirectory (relative to `baseline_dir`) where per-intervention audio
# WAVs are persisted. One file per intervention, named by intervention
# id. Kept as a sibling of `atlas.json` so the atlas is self-contained
# (atlas.json + atlas_audio/ move together), and so the browser can
# fetch audio by a stable relative URL resolved against the atlas URL.
ATLAS_AUDIO_SUBDIR = "atlas_audio"

# Observational detectors — the ones included in atlas intervention
# summaries and used for the flip-count score. Counterfactual detectors
# (unstable_bridge, brittle_lock) are skipped on intervention runs
# because they each run an O(N) inner loop; including them would push
# atlas cost from O(N) sims/fixture to O(N²).
OBSERVATIONAL_DETECTORS = (
    "phase_locked",
    "drifting",
    "phase_beating",
    "flam",
    "polyrhythmic",
    "dominant_cluster",
)

# Stat keys diffed per intervention. Order matters only for the JSON
# layout (sort_keys=True will reorder anyway, but readers benefit from
# a canonical short list).
STAT_DIFF_KEYS = (
    "mean_r",
    "tail_1s_mean_r",
    "tail_2way_velocity_separability",
)


def _observational_subset(summary):
    """Project a full summary onto the observational-only shape.

    Used to compare baseline (which carries all 8 detectors) against
    intervention summaries (which only carry the 6 observational ones)
    on equal footing.
    """
    out = dict(summary)
    dets = summary.get("detectors", {}) or {}
    out["detectors"] = {
        name: dets[name] for name in OBSERVATIONAL_DETECTORS if name in dets
    }
    return out


def _flip_diff(baseline_dets, intervention_dets):
    """Return list of `{detector, from, to}` for fired/silent flips.

    Both inputs are dicts of `{name: detector_block}`. Only detectors
    present in both are compared. A detector missing from either side
    is skipped (the atlas's `observational_detectors` list is the
    contract — names there will be present in both).
    """
    flips = []
    for name in OBSERVATIONAL_DETECTORS:
        a, b = baseline_dets.get(name), intervention_dets.get(name)
        if a is None or b is None:
            continue
        af, bf = bool(a.get("fired")), bool(b.get("fired"))
        if af != bf:
            flips.append({"detector": name, "from": af, "to": bf})
    return flips


def _stat_changes(baseline_stats, intervention_stats):
    """Return `{key: {from, to, delta}}` for each `STAT_DIFF_KEYS` entry
    where both sides hold a finite numeric value. Missing or non-finite
    on either side: skip (we don't synthesize None deltas)."""
    out = {}
    for key in STAT_DIFF_KEYS:
        if key not in baseline_stats or key not in intervention_stats:
            continue
        va, vb = baseline_stats.get(key), intervention_stats.get(key)
        if not isinstance(va, (int, float)) or not isinstance(vb, (int, float)):
            continue
        try:
            fa, fb = float(va), float(vb)
        except (TypeError, ValueError):
            continue
        # Skip non-finite (e.g. inf separability serialized as null
        # would never reach here, but defensive against future schema).
        import math
        if not (math.isfinite(fa) and math.isfinite(fb)):
            continue
        out[key] = {"from": fa, "to": fb, "delta": fb - fa}
    return out


def _explain(flips, stat_changes, score_components):
    """One-line, reader-friendly justification of the score.

    Designed to read in the browser intervention list — terse sentence
    fragments separated by `; `, period-terminated.
    """
    parts = []
    if flips:
        flip_names = ", ".join(
            f"{f['detector']}: {'silent → FIRED' if f['to'] else 'FIRED → silent'}"
            for f in flips
        )
        n = len(flips)
        word = "flip" if n == 1 else "flips"
        parts.append(f"{n} detector {word} ({flip_names})")
    if "tail_1s_mean_r" in stat_changes:
        d = stat_changes["tail_1s_mean_r"]["delta"]
        if abs(d) >= 0.01:
            verb = "rose" if d > 0 else "dropped"
            parts.append(f"tail r {verb} {abs(d):.2f}")
    if not parts:
        parts.append("no detector flips and stats largely unchanged")
    return "; ".join(parts) + "."


def _intervention_label(kind, node, delta_hz):
    if kind == "ablate_node":
        return f"ablate node {node}"
    if kind == "nudge_node":
        sign = "+" if delta_hz >= 0 else "−"
        return f"nudge node {node} by {sign}{abs(delta_hz):.2f} Hz"
    return f"{kind} node {node}"


def _intervention_id(kind, node, delta_hz):
    if kind == "ablate_node":
        return f"ablate_n{node}"
    if kind == "nudge_node":
        sign = "+" if delta_hz >= 0 else "-"
        return f"nudge_n{node}_{sign}{abs(delta_hz):.2f}hz"
    return f"{kind}_n{node}"


def _score_intervention(flips, stat_changes):
    """Compute the score and its components for one intervention.

    Definition (kept dead-simple and explainable):

        score = (#flips) + |Δ tail_1s_mean_r|

    `(#flips)` is the integer count of observational detectors whose
    fired/silent state changed. `|Δ tail_1s_mean_r|` is in `[0, 1]`
    when both sides have it; absent or non-finite on either side → 0.
    """
    n_flips = len(flips)
    abs_dr = 0.0
    if "tail_1s_mean_r" in stat_changes:
        abs_dr = abs(float(stat_changes["tail_1s_mean_r"]["delta"]))
    return {
        "score": float(n_flips) + abs_dr,
        "components": {"flips": int(n_flips), "abs_tail_r_delta": float(abs_dr)},
    }


def _enumerate_interventions(N):
    """Bounded enumeration of v1 atlas interventions for an N-node scene.

    Order: all ablations first by ascending node index, then all nudges
    by ascending node index. This is the *enumeration* order, not the
    final atlas order (which sorts by score). Stable order keeps repro
    runs byte-identical.
    """
    out = []
    for k in range(N):
        out.append(("ablate_node", k, None))
    for k in range(N):
        out.append(("nudge_node", k, BRITTLE_LOCK_NUDGE_HZ))
    return out


def _run_one_intervention(cfg, kind, node, delta_hz, work_path, config_path):
    """Run one intervention through `sim.ablate`, build an observational
    summary, and return `(summary, sim_dir)`. `sim_dir` holds the full
    simulation bundle (notably `audio.wav`) which the caller can copy
    into persistent storage before the TemporaryDirectory is cleaned up.
    """
    sub = work_path / _intervention_id(kind, node, delta_hz)
    if kind == "ablate_node":
        ablate_node(cfg, sub, node=node)
    elif kind == "nudge_node":
        nudge_node(cfg, sub, node=node, delta_hz=delta_hz)
    else:
        raise ValueError(f"unknown intervention kind: {kind}")
    summary = _build_summary(cfg, config_path, sub, include_counterfactual=False)
    return summary, sub


def _build_atlas(cfg, config_path, baseline_dir):
    """Top-level atlas build. Returns the dict written to atlas.json."""
    baseline_dir = Path(baseline_dir)
    baseline_summary_path = baseline_dir / "summary.json"
    if not baseline_summary_path.exists():
        raise FileNotFoundError(
            f"baseline summary missing: {baseline_summary_path} "
            "(run scripts/run_sim.py --summary-json first)"
        )
    with baseline_summary_path.open() as f:
        baseline_full_summary = json.load(f)
    baseline_obs = _observational_subset(baseline_full_summary)
    baseline_dets = baseline_obs.get("detectors", {})
    baseline_stats = baseline_obs.get("stats", {})

    N = int(cfg["scene"]["N"])
    interventions_spec = _enumerate_interventions(N)

    # Per-intervention audio lives alongside atlas.json so the atlas is
    # self-contained. Cleared on every build to match the
    # "regenerate fully on each run" posture of run.sh and the
    # byte-identical determinism contract (a stale file from a prior
    # build would leak into the next).
    audio_dir = baseline_dir / ATLAS_AUDIO_SUBDIR
    if audio_dir.exists():
        shutil.rmtree(audio_dir)
    audio_dir.mkdir(parents=True, exist_ok=True)

    interventions = []
    with tempfile.TemporaryDirectory() as tmp:
        work_path = Path(tmp)
        for kind, node, delta_hz in interventions_spec:
            iv_id = _intervention_id(kind, node, delta_hz)
            iv_summary, sim_dir = _run_one_intervention(
                cfg, kind, node, delta_hz, work_path, config_path
            )
            iv_dets = iv_summary.get("detectors", {})
            iv_stats = iv_summary.get("stats", {})
            flips = _flip_diff(baseline_dets, iv_dets)
            stat_changes = _stat_changes(baseline_stats, iv_stats)
            scoring = _score_intervention(flips, stat_changes)
            # Persist the intervention's audio so the browser can play
            # it back. Graceful fallback if the sim, for any reason,
            # emitted no audio.wav — the entry's `audio_path` stays
            # None and the UI falls back to "audio unavailable".
            src_wav = sim_dir / "audio.wav"
            audio_path = None
            if src_wav.is_file():
                dest = audio_dir / f"{iv_id}.wav"
                shutil.copyfile(src_wav, dest)
                audio_path = f"{ATLAS_AUDIO_SUBDIR}/{iv_id}.wav"
            entry = {
                "id": iv_id,
                "kind": kind,
                "node": int(node),
                "delta_hz": (float(delta_hz) if delta_hz is not None else None),
                "label": _intervention_label(kind, node, delta_hz),
                "summary": iv_summary,
                "audio_path": audio_path,
                "deltas": {
                    "flips": flips,
                    "stat_changes": stat_changes,
                },
                "score": float(scoring["score"]),
                "score_components": scoring["components"],
                "score_explanation": _explain(flips, stat_changes, scoring["components"]),
            }
            interventions.append(entry)

    # Sort by score desc, stable on enumeration order so equal-score
    # interventions keep their kind/node ordering (ablations before
    # nudges, ascending node) — readable in the UI list.
    interventions.sort(key=lambda e: e["score"], reverse=True)

    # Baseline's own audio.wav is the sibling of summary.json / atlas.json.
    # Record the pointer so the browser's A-slot playback uses the same
    # resolution mechanism as the intervention audio.
    baseline_audio_path = None
    if (baseline_dir / "audio.wav").is_file():
        baseline_audio_path = "audio.wav"

    meta = baseline_full_summary.get("meta", {}) or {}
    return {
        "schema_version": ATLAS_SCHEMA_VERSION,
        "baseline": {
            "config_path": meta.get("config_path", str(config_path)),
            "config_name": meta.get("config_name"),
            "duration_s": meta.get("duration_s"),
            "N": int(N),
            "control_rate_hz": meta.get("control_rate_hz"),
            "summary_path": "summary.json",
            "topology_path": "topology.json",
            "audio_path": baseline_audio_path,
            "observational_summary": baseline_obs,
        },
        "intervention_families": ["ablate_node", "nudge_node"],
        "observational_detectors": list(OBSERVATIONAL_DETECTORS),
        "ranking": {
            "score_formula": "(#detector flips vs baseline) + |Δ tail_1s_mean_r|",
            "notes": (
                "Flips dominate; |Δ tail_1s_mean_r| breaks ties and ranks "
                "no-flip interventions by coherence impact. Counterfactual "
                "detectors are excluded — intervention summaries are "
                "observational-only to keep the atlas O(N) sims per fixture."
            ),
        },
        "interventions": interventions,
    }


def _write_atlas(atlas, baseline_dir):
    path = Path(baseline_dir) / "atlas.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(atlas, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")
    return path


def main():
    ap = argparse.ArgumentParser(
        description="Build the Intervention Atlas for one baseline run."
    )
    ap.add_argument("--config", required=True,
                    help="path to the baseline run's config YAML")
    ap.add_argument("--baseline-dir", required=True,
                    help="baseline run directory (contains summary.json)")
    args = ap.parse_args()

    try:
        cfg = load(args.config)
    except FileNotFoundError as e:
        sys.exit(f"error: {e}")
    except ConfigError as e:
        sys.exit(f"error: {e}")
    except yaml.YAMLError as e:
        sys.exit(f"error: YAML parse failure in {args.config}: {e}")

    atlas = _build_atlas(cfg, args.config, args.baseline_dir)
    path = _write_atlas(atlas, args.baseline_dir)
    n = len(atlas["interventions"])
    top = atlas["interventions"][0] if n else None
    print(f"ok: wrote {path} ({n} interventions)")
    if top is not None:
        print(f"  top: {top['label']} — score {top['score']:.3f} — {top['score_explanation']}")


if __name__ == "__main__":
    main()
