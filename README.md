# Resonant Instrument Lab

A local research prototype for teaching small language models to **observe, describe, diagnose, and steer** emergent musical systems — starting from the raw continuous signals they produce.

## Why

Most music AI is evaluated by taste. Taste is slow, subjective, and unfalsifiable for a single researcher. This project swaps the target: we work in a **bounded synthetic world** — a Coupled Oscillator Garden — whose dynamics are known and whose semantic labels (*phase-locked*, *groove*, *brittle lock*, *tension break*, and so on) are computable directly from state.

The headline evaluation principle is **intervention grounding**:

> When the model proposes a change, we *apply that change* in the simulator. Did the world move the way the model said it would?

That turns *"does the model understand?"* into a test a laptop can run.

## What v0 is

- A coupled-oscillator simulator with pulse-driven audio and detector-grounded labels.
- A small instruction-tuned model — encoders + frozen base LM + LoRA adapters — that learns to caption, compare, diagnose, and suggest single interventions.
- Fully local, reproducible from seeds, no third-party audio data.

## Current milestone

**Simulator + ground-truth detectors + dataset dump. No model code yet.**

Detectors are the hardest part of the project and must pass a human sniff test before any training is meaningful. Acceptance criteria for this milestone live in [`DESIGN_V0.md §9`](./DESIGN_V0.md#9-first-coding-milestone).

## Try it

Two reference regimes and one intermediate regime already run end-to-end. Each produces a full `state.npz` / `events.jsonl` / `topology.json` / `audio.wav` / frozen `config.yaml` bundle, plus — with `--summary` — a compact semantic verdict from the current detectors:

```
$ python scripts/run_sim.py --config configs/regime_locked.yaml --out runs/demo/locked --summary
ok: wrote run artifacts to runs/demo/locked

regime summary — configs/regime_locked.yaml
  6.00 s, N=8, control_rate=200 Hz

  phase_locked : FIRED   conf 0.094   longest window 5.73 s
  drifting     : silent

  mean r(t)                    0.981
  tail-1s r                    0.995
  tail 2-way velocity sep.     0.00
```

Swap the config for `configs/regime_drifting.yaml` and `drifting` fires instead; swap for `configs/regime_two_cluster.yaml` and neither global-coherence detector fires, but the tail 2-way velocity separability jumps to ~700 — the bimodal fingerprint of the intermediate regime. Detector thresholds and window sizes live inline in `sim/detectors.py`; confidence is evidence margin above/below the threshold, not a probability.

Add `--summary-json` to also write the same verdicts and stats to `summary.json` in the output directory — a machine-readable seam for programmatic / browser consumers. Both views are rendered from one shared builder so they cannot drift.

## Docs

- [`DESIGN_V0.md`](./DESIGN_V0.md) — full v0 design note.
- [`DIRECTORS_NOTES.md`](./DIRECTORS_NOTES.md) — project canon and append-only pivot log.

## Status

v0, pre-implementation. Design-only repo.
